# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference of subm convs with a stored implicit gemm rulebook.

SparseConvolution's fast path for it makes the same kernel call as the
general path with less host work, so the outputs must be the same bit for
bit. ops.implicit_gemm_rulebook_from_grid builds the rulebook spconv builds
from the index grid of the input, without a host sync: the same pairs and
sorted masks, and the same conv outputs bit for bit.
"""

import torch
from cumm import tensorview as tv
from torch.nn import functional as F

import spconv.pytorch as spconv
from spconv.pytorch import conv as spconv_conv
from spconv.pytorch import ops
from spconv.pytorch.depthwise import SubMConvDepthwise2d

DEVICE = torch.device("cuda")
CHANNELS = 32
OUT_CHANNELS = 48


def _random_input(dtype=torch.float32, size=(20, 24), batch_size: int = 2,
                  density: float = 0.5):
    """(mask, tensor): random active sites with rows in nonzero order."""
    mask = torch.rand(batch_size, *size, device=DEVICE) < density
    # nonzero() is not contiguous on CUDA, and spconv reads rows
    indices = mask.nonzero().int().contiguous()
    features = torch.randn(indices.shape[0], CHANNELS, device=DEVICE)
    return mask, spconv.SparseConvTensor(features.to(dtype), indices,
                                         list(size), batch_size)


def _parents(mask: torch.Tensor, stride: int) -> torch.Tensor:
    """The output sites of a strided subm conv, the parents p // stride of
    the active sites p, in nonzero order."""
    pooled = F.max_pool2d(mask[:, None].float(), stride, stride,
                          ceil_mode=True)[:, 0]
    return (pooled > 0).nonzero().int().contiguous()


def _with_rulebook(st: spconv.SparseConvTensor, key: str, datas):
    out = st.replace_feature(st.features)
    out.indice_dict = {key: datas}
    return out


def _site_rows(indices: torch.Tensor) -> dict:
    return {tuple(site): row for row, site in enumerate(indices.tolist())}


def _unsorted_masks(datas) -> torch.Tensor:
    """The tap mask of each output row: the stored masks are sorted, in
    mask_argsort order."""
    sorted_masks = datas.pair_mask_fwd_splits[0].view(-1)
    argsort = datas.mask_argsort_fwd_splits[0].long()
    masks = torch.empty_like(sorted_masks)
    masks[argsort] = sorted_masks
    return masks


def _check_sorted(datas):
    """Ascending masks, rows of equal masks in row order, as spconv sorts."""
    sorted_masks = datas.pair_mask_fwd_splits[0].view(-1).long()
    argsort = datas.mask_argsort_fwd_splits[0].long()
    assert torch.equal(_unsorted_masks(datas).long()[argsort], sorted_masks)
    assert bool((sorted_masks[1:] >= sorted_masks[:-1]).all())
    ties = sorted_masks[1:] == sorted_masks[:-1]
    assert bool((argsort[1:][ties] > argsort[:-1][ties]).all())


def _run(conv, st: spconv.SparseConvTensor, fast: bool):
    spconv_conv._FAST_IMPLICIT_GEMM_REUSE = fast
    try:
        with torch.no_grad():
            return conv(st)
    finally:
        spconv_conv._FAST_IMPLICIT_GEMM_REUSE = True


def _check_reuse(dtype, stride: int, bias: bool, act):
    torch.manual_seed(0)
    _, st = _random_input(dtype)
    conv = spconv.SubMConv2d(CHANNELS, OUT_CHANNELS, 3, stride, 1, bias=bias,
                             indice_key="key").to(DEVICE, dtype).eval()
    conv.act_type = act
    with torch.no_grad():
        built = conv(st)
    st = _with_rulebook(st, "key", built.indice_dict["key"])
    fast = _run(conv, st, True)
    general = _run(conv, st, False)
    case = f"{dtype} stride {stride} bias {bias} {act}"
    assert torch.equal(fast.indices, general.indices), case
    assert fast.spatial_shape == general.spatial_shape, case
    assert torch.equal(fast.features, general.features), case
    assert torch.equal(fast.features, built.features), case


def test_reuse_is_bit_identical():
    calls = []
    reuse = ops.implicit_gemm_reuse

    def counted(*args, **kwargs):
        calls.append(1)
        return reuse(*args, **kwargs)

    ops.implicit_gemm_reuse = counted
    try:
        for dtype in [torch.float32, torch.float16]:
            for stride in [1, 2]:
                for bias in [True, False]:
                    for act in [tv.gemm.Activation.None_,
                                tv.gemm.Activation.SiLU]:
                        _check_reuse(dtype, stride, bias, act)
    finally:
        ops.implicit_gemm_reuse = reuse
    assert len(calls) == 16, "the fast path did not run"


def _spconv_rulebook(st: spconv.SparseConvTensor, ksize: int, stride: int,
                     padding: int, dilation: int):
    """spconv's own rulebook of a subm conv on st, without gradients."""
    conv = spconv.SubMConv2d(CHANNELS, CHANNELS, ksize, stride, padding,
                             dilation, indice_key="ref").to(
                                 DEVICE, st.features.dtype).eval()
    with torch.no_grad():
        return conv(st).indice_dict["ref"]


def _conv_outputs(st: spconv.SparseConvTensor, datas, ksize: int, stride: int,
                  padding: int, dilation: int, dtype):
    torch.manual_seed(1)
    conv = spconv.SubMConv2d(CHANNELS, OUT_CHANNELS, ksize, stride, padding,
                             dilation, indice_key="key").to(DEVICE,
                                                            dtype).eval()
    conv.act_type = tv.gemm.Activation.SiLU
    with torch.no_grad():
        return conv(_with_rulebook(st, "key", datas))


def _check_grid_rulebook_stride1(ksize: int, dilation: int, dtype):
    torch.manual_seed(0)
    _, st = _random_input(dtype)
    padding = (ksize // 2) * dilation
    ref = _spconv_rulebook(st, ksize, 1, padding, dilation)
    grid = ops.depthwise_index_grid(st.indices, st.batch_size,
                                    st.spatial_shape)
    datas = ops.implicit_gemm_rulebook_from_grid(st.indices, grid,
                                                 [ksize] * 2, [1, 1],
                                                 [padding] * 2,
                                                 [dilation] * 2)
    case = f"k{ksize} d{dilation} {dtype}"
    assert torch.equal(datas.pair_fwd, ref.pair_fwd), case
    assert torch.equal(datas.pair_mask_fwd_splits[0],
                       ref.pair_mask_fwd_splits[0]), case
    _check_sorted(datas)
    out = _conv_outputs(st, datas, ksize, 1, padding, dilation, dtype)
    out_ref = _conv_outputs(st, ref, ksize, 1, padding, dilation, dtype)
    assert torch.equal(out.indices, out_ref.indices), case
    assert torch.equal(out.features, out_ref.features), case


def _check_grid_rulebook_strided(ksize: int, padding: int, dtype):
    """spconv's strided rulebook orders its outputs by hash: compared by
    site."""
    torch.manual_seed(0)
    mask, st = _random_input(dtype)
    ref = _spconv_rulebook(st, ksize, 2, padding, 1)
    grid = ops.depthwise_index_grid(st.indices, st.batch_size,
                                    st.spatial_shape)
    out_indices = _parents(mask, 2)
    datas = ops.implicit_gemm_rulebook_from_grid(st.indices, grid,
                                                 [ksize] * 2, [2, 2],
                                                 [padding] * 2, [1, 1],
                                                 out_indices)
    case = f"k{ksize} p{padding} {dtype}"
    rows = _site_rows(datas.out_indices)
    ref_rows = _site_rows(ref.out_indices)
    assert rows.keys() == ref_rows.keys(), case
    assert datas.out_spatial_shape == ref.out_spatial_shape, case
    order = torch.tensor([rows[site] for site in ref_rows], device=DEVICE)
    assert torch.equal(datas.pair_fwd[:, order], ref.pair_fwd), case
    assert torch.equal(_unsorted_masks(datas)[order],
                       _unsorted_masks(ref)), case
    _check_sorted(datas)
    out = _conv_outputs(st, datas, ksize, 2, padding, 1, dtype)
    out_ref = _conv_outputs(st, ref, ksize, 2, padding, 1, dtype)
    assert torch.equal(out.features[order], out_ref.features), case


def test_grid_rulebook_matches_spconv():
    for dtype in [torch.float32, torch.float16]:
        _check_grid_rulebook_stride1(3, 1, dtype)
        _check_grid_rulebook_stride1(5, 1, dtype)
        _check_grid_rulebook_stride1(3, 2, dtype)
        _check_grid_rulebook_strided(3, 1, dtype)
        _check_grid_rulebook_strided(2, 0, dtype)


def test_depthwise_shares_grid_rulebook():
    """A strided depthwise layer with the same key takes its output sites
    from the stored rulebook instead of building one."""
    torch.manual_seed(0)
    mask, st = _random_input()
    grid = ops.depthwise_index_grid(st.indices, st.batch_size,
                                    st.spatial_shape)
    datas = ops.implicit_gemm_rulebook_from_grid(st.indices, grid, [3, 3],
                                                 [2, 2], [1, 1], [1, 1],
                                                 _parents(mask, 2))
    dw = SubMConvDepthwise2d(CHANNELS, 3, 2, 1, indice_key="key").to(DEVICE)
    with torch.no_grad():
        out = dw(_with_rulebook(st, "key", datas))
        ref = dw(st)
    assert out.indices is datas.out_indices
    rows = _site_rows(out.indices)
    order = [rows[tuple(site)] for site in ref.indices.tolist()]
    assert torch.equal(out.features[order], ref.features)


def test_grid_rulebook_needs_no_host_sync():
    torch.manual_seed(0)
    mask, st = _random_input()
    grid = ops.depthwise_index_grid(st.indices, st.batch_size,
                                    st.spatial_shape)
    out_indices = _parents(mask, 2)
    builds = [
        lambda: ops.implicit_gemm_rulebook_from_grid(
            st.indices, grid, [3, 3], [1, 1], [1, 1], [1, 1]),
        lambda: ops.implicit_gemm_rulebook_from_grid(
            st.indices, grid, [3, 3], [2, 2], [1, 1], [1, 1], out_indices),
    ]
    for build in builds:
        build()  # compiles the kernel
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for build in builds:
            build()
    finally:
        torch.cuda.set_sync_debug_mode("default")


if __name__ == "__main__":
    test_reuse_is_bit_identical()
    test_grid_rulebook_matches_spconv()
    test_depthwise_shares_grid_rulebook()
    test_grid_rulebook_needs_no_host_sync()
    print("all ok")
