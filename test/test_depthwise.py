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

"""Compare SubMConvDepthwise2d and SparseConvDepthwise2d against
nn.Conv2d(groups=C), at stride 1 and 2.

A sparse convolution computes only at its output sites, and a kernel tap landing
on an inactive site contributes nothing -- which is identical to that site holding
zero. So the dense reference zeroes the inactive sites of its input, and the two
are compared only at the sparse output sites.
"""

import torch
from torch import nn

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from spconv.pytorch.depthwise import SparseConvDepthwise2d, SubMConvDepthwise2d

# we must disable tf32 to increase reference precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device("cuda")
CHANNELS = 16


def _to_sparse(x: torch.Tensor, mask: torch.Tensor) -> spconv.SparseConvTensor:
    """(B, C, H, W) + (B, H, W) bool mask -> SparseConvTensor of active sites.

    Rows are in mask.nonzero() order, the same order as dense[mask].
    """
    # (N, 3), one row per active cell: batch, y, x. On CUDA nonzero() returns
    # a non-contiguous tensor, and spconv reads indices as contiguous memory
    # without checking, so the coordinates come out scrambled without it.
    indices = mask.nonzero().contiguous()
    batch_idx = indices[:, 0]
    y_idx = indices[:, 1]
    x_idx = indices[:, 2]
    x_channels_last = x.permute(0, 2, 3, 1)  # (B, H, W, C)
    features = x_channels_last[batch_idx, y_idx, x_idx]  # (N, C)
    return spconv.SparseConvTensor(features.contiguous(),
                                   indices.int().contiguous(),
                                   list(x.shape[2:]), x.shape[0])


def _dense_reference(conv: nn.Conv2d, x: torch.Tensor, mask: torch.Tensor,
                     indices: torch.Tensor) -> torch.Tensor:
    """The dense conv on the zeroed input, read back at the output sites
    `indices` (batch, y, x), in their order."""
    # mask is (B, H, W) and x is (B, C, H, W): add a channel axis of size 1
    # so the multiply applies the same mask to every channel.
    mask_per_channel = mask.unsqueeze(1).float()  # (B, 1, H, W)
    out = conv(x * mask_per_channel)
    out_channels_last = out.permute(0, 2, 3, 1)  # (B, H, W, C)
    indices = indices.long()
    return out_channels_last[indices[:, 0], indices[:, 1], indices[:, 2]]


def _depthwise_conv(kernel_size: int, stride: int = 1, padding=None,
                    dilation: int = 1, bias: bool = True) -> nn.Conv2d:
    """padding None is the "same" padding, (k // 2) * dilation."""
    if padding is None:
        padding = (kernel_size // 2) * dilation
    return nn.Conv2d(CHANNELS, CHANNELS, kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=CHANNELS,
                     bias=bias).to(DEVICE)


def _random_input(density: float, size=(16, 16), batch: int = 2):
    x = torch.randn(batch, CHANNELS, *size, device=DEVICE)
    mask = torch.rand(batch, *size, device=DEVICE) < density
    return x, mask


def _assert_close(a: torch.Tensor, b: torch.Tensor):
    # dense and sparse sum the kernel taps in different orders
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def _check_against_dense(cls, kernel_size: int, density: float, stride: int = 1,
                         padding=None, dilation: int = 1, bias: bool = True,
                         algo=None, size=(16, 16)):
    torch.manual_seed(0)
    x, mask = _random_input(density, size)
    conv = _depthwise_conv(kernel_size, stride, padding, dilation, bias)
    dw = cls.from_conv2d(conv, algo=algo)
    out = dw(_to_sparse(x, mask))
    _assert_close(out.features, _dense_reference(conv, x, mask, out.indices))


def _sites(indices: torch.Tensor) -> set:
    sites = set()
    for site in indices.tolist():
        sites.add(tuple(site))
    return sites


def _parents(indices: torch.Tensor, stride: int) -> set:
    parents = set()
    for b, y, x in indices.tolist():
        parents.add((b, y // stride, x // stride))
    return parents


def _assert_close_by_site(out: spconv.SparseConvTensor,
                          ref: spconv.SparseConvTensor):
    """Same sites and features, in any row order: a regular rulebook orders
    its output sites by hash."""
    row_of = {}
    for i, site in enumerate(ref.indices.tolist()):
        row_of[tuple(site)] = i
    assert out.indices.shape == ref.indices.shape
    rows = []
    for site in out.indices.tolist():
        rows.append(row_of[tuple(site)])
    _assert_close(out.features, ref.features[rows])


def _check_inverse(down, down_ref: spconv.SparseConv2d, algo):
    """SparseInverseConv2d with the key "down", stored by `down`, must map
    back to the input sites. Reference: the inverse of down_ref, spconv's own
    SparseConv2d with the same pairs (key "down_ref"), fed down's features on
    the sites the two share and zeros on any extra ones, which add nothing."""
    torch.manual_seed(0)
    x, mask = _random_input(0.6)
    low = down(_to_sparse(x, mask))
    low_ref = down_ref(_to_sparse(x, mask))

    row_of = {}
    for i, site in enumerate(low.indices.tolist()):
        row_of[tuple(site)] = i
    rows_ref = []
    rows = []
    for i, site in enumerate(low_ref.indices.tolist()):
        if tuple(site) in row_of:
            rows_ref.append(i)
            rows.append(row_of[tuple(site)])
    assert len(rows) == low.indices.shape[0]
    features_ref = torch.zeros_like(low_ref.features)
    features_ref[rows_ref] = low.features[rows]

    channels = low.features.shape[1]
    up = spconv.SparseInverseConv2d(channels, x.shape[1], down_ref.kernel_size,
                                    indice_key="down", algo=algo).to(DEVICE)
    up_ref = spconv.SparseInverseConv2d(channels, x.shape[1],
                                        down_ref.kernel_size,
                                        indice_key="down_ref",
                                        algo=algo).to(DEVICE)
    up_ref.load_state_dict(up.state_dict())
    out = up(low)
    ref = up_ref(low_ref.replace_feature(features_ref))
    assert torch.equal(out.indices, ref.indices)
    _assert_close(out.features, ref.features)


def _expected_pair_fwd(kernel_size: int, num_cells: int) -> torch.Tensor:
    """pair_fwd for one row of `num_cells` active cells, spatial shape [1, n].

    Offset k is (dy, dx) = (k // kw - kw // 2, k % kw - kw // 2), row-major.
    Cell i reads cell i + dx at offset k, if that cell exists and dy == 0.
    """
    half = kernel_size // 2
    rows = []
    for k in range(kernel_size * kernel_size):
        dy = k // kernel_size - half
        dx = k % kernel_size - half
        row = []
        for i in range(num_cells):
            neighbour = i + dx
            if dy == 0 and 0 <= neighbour < num_cells:
                row.append(neighbour)
            else:
                row.append(-1)
        rows.append(row)
    return torch.tensor(rows)


def test_rulebook_direction_and_offset_order():
    """Pins the gather table on a case small enough to write out by hand.

    3 cells in a row: offset "left" must read [-1, 0, 1], "right" [1, 2, -1].
    A swapped pair direction or a column-major offset order both fail here.
    k=3 goes through the implicit gemm rulebook, k=7 (49 offsets) through the
    Native one and its conversion, so this also checks the two agree.
    """
    indices = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 0, 2]],
                           dtype=torch.int32, device=DEVICE)
    st = spconv.SparseConvTensor(torch.zeros(3, CHANNELS, device=DEVICE),
                                 indices, [1, 3], 1)
    for k in (3, 7):
        dw = SubMConvDepthwise2d(CHANNELS, k).to(DEVICE)
        _, pair_fwd, _ = dw._get_rulebook(st)
        assert torch.equal(pair_fwd.long().cpu(), _expected_pair_fwd(k, 3)), \
            f"k={k}\n{pair_fwd.cpu()}"


def test_kernel1_is_identity_mapping():
    """kernel_size=1: every site maps to itself, so the op is a per-channel scale.

    No neighbour bookkeeping at all, so this isolates the weight conversion and
    the bias from the rulebook.
    """
    torch.manual_seed(0)
    x, mask = _random_input(0.6, size=(4, 4), batch=1)
    conv = _depthwise_conv(1)
    dw = SubMConvDepthwise2d.from_conv2d(conv)
    st = _to_sparse(x, mask)
    n = st.indices.shape[0]
    _, pair_fwd, _ = dw._get_rulebook(st)
    assert pair_fwd.shape == (1, n)
    assert torch.equal(pair_fwd[0].long(), torch.arange(n, device=DEVICE))
    _assert_close(dw(st).features, _dense_reference(conv, x, mask, st.indices))


def test_kernel3_full_density():
    """Every site active: no inactive taps, so it reduces to a dense depthwise.

    A failure here is the offset order or the weight conversion, not sparsity.
    """
    _check_against_dense(SubMConvDepthwise2d, 3, density=1.0)


def test_kernel3_sparse():
    """A real mask: taps on inactive sites must contribute nothing."""
    _check_against_dense(SubMConvDepthwise2d, 3, density=0.6)
    _check_against_dense(SubMConvDepthwise2d, 3, density=0.6, bias=False)


def test_kernel7():
    """49 offsets, past the implicit gemm limit of 32: the Native rulebook."""
    _check_against_dense(SubMConvDepthwise2d, 7, density=0.6)


def test_dilation():
    _check_against_dense(SubMConvDepthwise2d, 3, density=0.6, dilation=2)


def test_every_algo():
    """The default 3x3 builds MaskImplicitGemm; force the other rulebooks,
    including Native at 3x3, where the mirrored counts matter."""
    for algo in [ConvAlgo.Native, ConvAlgo.MaskImplicitGemm,
                 ConvAlgo.MaskSplitImplicitGemm]:
        _check_against_dense(SubMConvDepthwise2d, 3, density=0.6, algo=algo)
    _check_against_dense(SubMConvDepthwise2d, 7, density=0.6,
                         algo=ConvAlgo.MaskImplicitGemm)


def test_backward_matches_dense():
    for cls, stride in [(SubMConvDepthwise2d, 1), (SubMConvDepthwise2d, 2),
                        (SparseConvDepthwise2d, 2)]:
        torch.manual_seed(0)
        x, mask = _random_input(0.6)
        conv = _depthwise_conv(3, stride)
        dw = cls.from_conv2d(conv)

        # make both inputs leaf tensors that record a gradient
        st = _to_sparse(x, mask)
        features = st.features.clone()
        features.requires_grad = True
        out_tensor = dw(st.replace_feature(features))
        out = out_tensor.features
        x_ref = x.clone()
        x_ref.requires_grad = True
        ref = _dense_reference(conv, x_ref, mask, out_tensor.indices)

        # the same random upstream gradient into both
        grad_out = torch.randn_like(ref)
        loss = (out * grad_out).sum()
        loss.backward()
        loss_ref = (ref * grad_out).sum()
        loss_ref.backward()

        _assert_close(dw.weight.grad, conv.weight.grad.permute(0, 2, 3, 1))
        _assert_close(dw.bias.grad, conv.bias.grad)
        _assert_close(features.grad, x_ref.grad.permute(0, 2, 3, 1)[mask])


def _fail_if_built(input):
    raise AssertionError("rulebook rebuilt instead of reused")


def _forbid_build(module: SubMConvDepthwise2d):
    """Make this one module raise if it tries to build a rulebook, so a test
    can prove it reused the cached one."""
    module._build_rulebook = _fail_if_built


def test_indice_key_reuse():
    """Two layers sharing a key must agree with two that build their own,
    and the second must not rebuild."""
    torch.manual_seed(0)
    x, mask = _random_input(0.6)
    shared1 = SubMConvDepthwise2d(CHANNELS, 3, indice_key="k3").to(DEVICE)
    shared2 = SubMConvDepthwise2d(CHANNELS, 3, indice_key="k3").to(DEVICE)
    # same weights, no key
    alone1 = SubMConvDepthwise2d(CHANNELS, 3).to(DEVICE)
    alone2 = SubMConvDepthwise2d(CHANNELS, 3).to(DEVICE)
    alone1.load_state_dict(shared1.state_dict())
    alone2.load_state_dict(shared2.state_dict())

    st = _to_sparse(x, mask)
    hidden = shared1(st)
    # stored on the output, input untouched, as SparseConvolution does
    assert "k3" in hidden.indice_dict
    assert "k3" not in st.indice_dict
    _forbid_build(shared2)
    out_shared = shared2(hidden)
    out_alone = alone2(alone1(_to_sparse(x, mask)))
    _assert_close(out_shared.features, out_alone.features)


def test_shares_key_with_subm_conv():
    """A key shared with a default-algo SubMConv2d, in both orders.

    3x3 shares an implicit gemm rulebook, 7x7 a Native one. Either order must
    reuse the rulebook and match layers that build their own.
    """
    for k in (3, 7):
        torch.manual_seed(0)
        x, mask = _random_input(0.6)
        subm = spconv.SubMConv2d(CHANNELS, CHANNELS, k, bias=False,
                                 indice_key="key").to(DEVICE)
        subm_alone = spconv.SubMConv2d(CHANNELS, CHANNELS, k,
                                       bias=False).to(DEVICE)
        subm_alone.load_state_dict(subm.state_dict())
        dw = SubMConvDepthwise2d(CHANNELS, k, indice_key="key").to(DEVICE)
        dw_alone = SubMConvDepthwise2d(CHANNELS, k).to(DEVICE)
        dw_alone.load_state_dict(dw.state_dict())

        # SubMConv2d first: the depthwise layer reads its rulebook
        _forbid_build(dw)
        out = dw(subm(_to_sparse(x, mask)))
        ref = dw_alone(subm_alone(_to_sparse(x, mask)))
        _assert_close(out.features, ref.features)

        # depthwise first: SubMConv2d reads the rulebook it built
        dw = SubMConvDepthwise2d(CHANNELS, k, indice_key="key").to(DEVICE)
        dw.load_state_dict(dw_alone.state_dict())
        st = _to_sparse(x, mask)
        hidden = dw(st)
        built = hidden.indice_dict["key"]
        out = subm(hidden)
        assert out.indice_dict["key"] is built
        ref = subm_alone(dw_alone(_to_sparse(x, mask)))
        _assert_close(out.features, ref.features)


def test_rejects_shared_key_across_kernel_sizes():
    """A cached rulebook describes one footprint. Reading a 3x3 rulebook as a
    7x7 must raise, as SparseConvolution does, not compute garbage."""
    torch.manual_seed(0)
    x, mask = _random_input(0.6)
    dw3 = SubMConvDepthwise2d(CHANNELS, 3, indice_key="key").to(DEVICE)
    dw7 = SubMConvDepthwise2d(CHANNELS, 7, indice_key="key").to(DEVICE)
    try:
        dw7(dw3(_to_sparse(x, mask)))
        raise AssertionError("depthwise 7x7 reused a depthwise 3x3 rulebook")
    except ValueError:
        pass

    subm3 = spconv.SubMConv2d(CHANNELS, CHANNELS, 3,
                              indice_key="key").to(DEVICE)
    try:
        dw7(subm3(_to_sparse(x, mask)))
        raise AssertionError("depthwise 7x7 reused a SubMConv2d 3x3 rulebook")
    except ValueError:
        pass


def test_in_sparse_sequential():
    """SparseSequential passes the whole SparseConvTensor only to a
    SparseModule; anything else gets bare features."""
    torch.manual_seed(0)
    x, mask = _random_input(0.6)
    dw = SubMConvDepthwise2d(CHANNELS, 3).to(DEVICE)
    st = _to_sparse(x, mask)
    _assert_close(spconv.SparseSequential(dw)(st).features, dw(st).features)


def test_subm_strided_against_dense():
    """Stride 2 with nn.Conv2d(stride=2) read at the output sites. Includes an
    odd grid with padding 0, where the last row and column have no output, and
    a 2x2 kernel, where every output reads exactly its four children."""
    for algo in [ConvAlgo.Native, ConvAlgo.MaskImplicitGemm]:
        _check_against_dense(SubMConvDepthwise2d, 3, density=0.6, stride=2,
                             padding=1, algo=algo)
        _check_against_dense(SubMConvDepthwise2d, 3, density=0.6, stride=2,
                             padding=1, bias=False, algo=algo)
        _check_against_dense(SubMConvDepthwise2d, 3, density=0.6, stride=2,
                             padding=0, algo=algo, size=(15, 15))
        _check_against_dense(SubMConvDepthwise2d, 2, density=0.6, stride=2,
                             padding=0, algo=algo)
    # kernel volume 49: Native
    _check_against_dense(SubMConvDepthwise2d, 7, density=0.6, stride=2,
                         padding=3)


def test_subm_strided_output_sites():
    """The output sites are exactly the parents p // 2 of the active input
    sites p: SparseConv2d keeps those and adds more."""
    torch.manual_seed(0)
    x, mask = _random_input(0.3)
    st = _to_sparse(x, mask)
    out = SubMConvDepthwise2d(CHANNELS, 3, 2, 1).to(DEVICE)(st)
    expected = _parents(st.indices, 2)
    assert _sites(out.indices) == expected
    assert out.indices.shape[0] == len(expected)  # no site twice
    assert out.spatial_shape == [8, 8]

    full = spconv.SparseConv2d(CHANNELS, CHANNELS, 3, 2, 1).to(DEVICE)(st)
    assert expected < _sites(full.indices)


def test_regular_against_dense():
    for algo in [ConvAlgo.Native, ConvAlgo.MaskImplicitGemm]:
        _check_against_dense(SparseConvDepthwise2d, 3, density=0.6, stride=2,
                             padding=1, algo=algo)
        _check_against_dense(SparseConvDepthwise2d, 3, density=0.6, stride=1,
                             padding=1, algo=algo)
    _check_against_dense(SparseConvDepthwise2d, 7, density=0.6, stride=2,
                         padding=3)


def test_regular_output_sites():
    """The same output sites as SparseConv2d: the active set grows."""
    torch.manual_seed(0)
    x, mask = _random_input(0.3)
    st = _to_sparse(x, mask)
    out = SparseConvDepthwise2d(CHANNELS, 3, 2, 1).to(DEVICE)(st)
    full = spconv.SparseConv2d(CHANNELS, CHANNELS, 3, 2, 1).to(DEVICE)(st)
    assert _sites(out.indices) == _sites(full.indices)
    assert out.indices.shape == full.indices.shape


def test_strided_key_inverts():
    """Both strided layers store their rulebook for SparseInverseConv2d."""
    for algo in [ConvAlgo.Native, ConvAlgo.MaskImplicitGemm]:
        for cls in [SubMConvDepthwise2d, SparseConvDepthwise2d]:
            down = cls(CHANNELS, 3, 2, 1, indice_key="down",
                       algo=algo).to(DEVICE)
            down_ref = spconv.SparseConv2d(CHANNELS, CHANNELS, 3, 2, 1,
                                           indice_key="down_ref",
                                           algo=algo).to(DEVICE)
            _check_inverse(down, down_ref, algo)


def test_strided_key_shared_with_subm_conv():
    """A strided subm key is reused when the same sites come back, as in the
    downsampling path of a U-Net or a YOLO neck: by a depthwise layer from a
    SubMConv2d, and the other way round."""
    torch.manual_seed(0)
    x, mask = _random_input(0.6)

    # SubMConv2d builds, the depthwise layer reuses
    subm = spconv.SubMConv2d(CHANNELS, CHANNELS, 3, 2, 1,
                             indice_key="down").to(DEVICE)
    up = spconv.SparseInverseConv2d(CHANNELS, CHANNELS, 3,
                                    indice_key="down").to(DEVICE)
    dw = SubMConvDepthwise2d(CHANNELS, 3, 2, 1, indice_key="down").to(DEVICE)
    dw_alone = SubMConvDepthwise2d(CHANNELS, 3, 2, 1).to(DEVICE)
    dw_alone.load_state_dict(dw.state_dict())
    back = up(subm(_to_sparse(x, mask)))  # the input sites, with the key
    _forbid_build(dw)
    _assert_close_by_site(dw(back), dw_alone(back))

    # the depthwise layer builds, SubMConv2d reuses
    dw = SubMConvDepthwise2d(CHANNELS, 3, 2, 1, indice_key="down").to(DEVICE)
    up = spconv.SparseInverseConv2d(CHANNELS, CHANNELS, 3,
                                    indice_key="down").to(DEVICE)
    subm_alone = spconv.SubMConv2d(CHANNELS, CHANNELS, 3, 2, 1).to(DEVICE)
    subm_alone.load_state_dict(subm.state_dict())
    back = up(dw(_to_sparse(x, mask)))
    out = subm(back)
    assert out.indices is back.indice_dict["down"].out_indices  # reused
    _assert_close_by_site(out, subm_alone(back))

    # a stride 1 subm layer's rulebook is a different one
    try:
        SubMConvDepthwise2d(CHANNELS, 3, indice_key="down").to(DEVICE)(back)
        raise AssertionError("stride 1 layer reused a stride 2 rulebook")
    except ValueError:
        pass


def test_rejects_unsupported():
    # an input site outside the kernel window of its parent p // stride
    for kernel_size, stride, padding in [(1, 2, 0), (3, 2, 2)]:
        try:
            SubMConvDepthwise2d(CHANNELS, kernel_size, stride, padding)
            raise AssertionError(f"accepted kernel_size {kernel_size}, stride"
                                 f" {stride}, padding {padding}")
        except ValueError:
            pass
    # strided subm implicit gemm: MaskImplicitGemm, kernel volume <= 32
    for kernel_size, algo in [(3, ConvAlgo.MaskSplitImplicitGemm),
                              (7, ConvAlgo.MaskImplicitGemm)]:
        try:
            SubMConvDepthwise2d(CHANNELS, kernel_size, 2, kernel_size // 2,
                                algo=algo)
            raise AssertionError(f"accepted {algo}, kernel_size {kernel_size}")
        except ValueError:
            pass
    # as in SparseConvolution, only subm rulebooks are reused
    torch.manual_seed(0)
    x, mask = _random_input(0.6)
    down1 = SparseConvDepthwise2d(CHANNELS, 3, 2, 1,
                                  indice_key="down").to(DEVICE)
    down2 = SparseConvDepthwise2d(CHANNELS, 3, 2, 1,
                                  indice_key="down").to(DEVICE)
    try:
        down2(down1(_to_sparse(x, mask)))
        raise AssertionError("regular layer reused a key")
    except AssertionError as e:
        if "only support reuse subm indices" not in str(e):
            raise


def test_from_conv2d_rejects_mismatch():
    not_depthwise = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=1)
    no_padding = nn.Conv2d(CHANNELS, CHANNELS, 3, groups=CHANNELS)
    child_outside = nn.Conv2d(CHANNELS, CHANNELS, 3, padding=2, stride=2,
                              groups=CHANNELS)
    for conv in [not_depthwise, no_padding, child_outside]:
        try:
            SubMConvDepthwise2d.from_conv2d(conv)
            raise AssertionError(f"accepted {conv}")
        except ValueError:
            pass


if __name__ == "__main__":
    test_rulebook_direction_and_offset_order()
    test_kernel1_is_identity_mapping()
    test_kernel3_full_density()
    test_kernel3_sparse()
    test_kernel7()
    test_dilation()
    test_every_algo()
    test_subm_strided_against_dense()
    test_subm_strided_output_sites()
    test_regular_against_dense()
    test_regular_output_sites()
    test_strided_key_inverts()
    test_strided_key_shared_with_subm_conv()
    test_rejects_unsupported()
    test_backward_matches_dense()
    test_indice_key_reuse()
    test_shares_key_with_subm_conv()
    test_rejects_shared_key_across_kernel_sizes()
    test_in_sparse_sequential()
    test_from_conv2d_rejects_mismatch()
    print("all ok")
