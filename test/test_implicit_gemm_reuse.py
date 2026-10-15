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
bit.
"""

import torch
from cumm import tensorview as tv

import spconv.pytorch as spconv
from spconv.pytorch import conv as spconv_conv
from spconv.pytorch import ops

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


def _with_rulebook(st: spconv.SparseConvTensor, key: str, datas):
    out = st.replace_feature(st.features)
    out.indice_dict = {key: datas}
    return out


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


if __name__ == "__main__":
    test_reuse_is_bit_identical()
    print("all ok")
