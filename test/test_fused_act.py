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

"""Bias and activation fused into spconv's convolutions (act_type), and kernel
size 1, which runs as a matrix multiply.

At inference a conv with act_type must equal the same conv without it followed
by the PyTorch activation, for every conv type and algo. A 1x1 conv must equal
a linear map of each site's features, with the weight in its KRSC layout.
"""

import torch
from cumm import tensorview as tv
from torch.nn import functional as F

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from spconv.pytorch.conv import SparseConvolution

DEVICE = torch.device("cuda")
ALGOS = [ConvAlgo.Native, ConvAlgo.MaskImplicitGemm]
ALPHA = 0.1
ACTS = [(tv.gemm.Activation.ReLU, F.relu),
        (tv.gemm.Activation.Sigmoid, torch.sigmoid),
        (tv.gemm.Activation.LeakyReLU, lambda x: F.leaky_relu(x, ALPHA)),
        (tv.gemm.Activation.SiLU, F.silu)]
TOLERANCE = {torch.float32: 1e-4, torch.float16: 1e-2}


def _random_tensor(ndim: int, channels: int, dtype=torch.float32,
                   device=DEVICE, size: int = 12, batch_size: int = 2,
                   density: float = 0.4) -> spconv.SparseConvTensor:
    shape = [batch_size] + [size] * ndim
    # nonzero() is not contiguous on CUDA, and spconv reads indices as
    # contiguous rows
    mask = torch.rand(shape, device=device) < density
    indices = mask.nonzero().int().contiguous()
    features = torch.randn(indices.shape[0], channels, device=device)
    return spconv.SparseConvTensor(features.to(dtype), indices, [size] * ndim,
                                   batch_size)


def _conv_types(ndim: int, algo, dtype, in_channels: int,
                out_channels: int):
    """name -> (conv, run): one conv of each type, and the forward from the
    test input to its output."""

    def conv(*args, **kwargs):
        return SparseConvolution(ndim, *args, algo=algo,
                                 **kwargs).to(DEVICE, dtype).eval()

    subm = conv(in_channels, out_channels, 3, padding=1, subm=True)
    regular = conv(in_channels, out_channels, 3, 2, 1)
    transposed = conv(in_channels, out_channels, 3, 2, 1, transposed=True)
    down = conv(in_channels, in_channels, 3, 2, 1, indice_key="down")
    inverse = conv(in_channels, out_channels, 3, indice_key="down",
                   inverse=True)
    subm1x1 = conv(in_channels, out_channels, 1, subm=True)
    regular1x1 = conv(in_channels, out_channels, 1)
    return {
        "subm": (subm, subm),
        "regular": (regular, regular),
        "transposed": (transposed, transposed),
        "inverse": (inverse, lambda x: inverse(down(x))),
        "subm 1x1": (subm1x1, subm1x1),
        "regular 1x1": (regular1x1, regular1x1),
    }


def _by_site(st: spconv.SparseConvTensor):
    """(indices, features) with the rows sorted by site: a regular
    MaskImplicitGemm rulebook orders its output sites by hash, differently in
    each run."""
    key = st.indices[:, 0].long()
    for axis, size in enumerate(st.spatial_shape):
        key = key * size + st.indices[:, axis + 1].long()
    order = torch.argsort(key)
    return st.indices[order], st.features[order]


def _check_fused_act(ndim: int, algo, dtype):
    torch.manual_seed(0)
    x = _random_tensor(ndim, 16, dtype)
    tolerance = TOLERANCE[dtype]
    for name, (conv, run) in _conv_types(ndim, algo, dtype, 16, 24).items():
        for act, act_fn in ACTS:
            with torch.no_grad():
                conv.act_type, conv.act_alpha = tv.gemm.Activation.None_, 0.0
                indices, features = _by_site(run(x))
                conv.act_type, conv.act_alpha = act, ALPHA
                out_indices, out = _by_site(run(x))
            case = f"{ndim}D {algo} {dtype} {name} {act}"
            assert torch.equal(out_indices, indices), case
            torch.testing.assert_close(out.float(), act_fn(features.float()),
                                       atol=tolerance, rtol=tolerance,
                                       msg=lambda m: f"{case}: {m}")


def test_fused_act():
    """Native fuses the activation in its bias kernel, MaskImplicitGemm in the
    gemm epilogue; a 1x1 conv applies it after the matrix multiply."""
    for ndim in [2, 3]:
        for algo in ALGOS:
            for dtype in [torch.float32, torch.float16]:
                _check_fused_act(ndim, algo, dtype)


def _check_conv1x1(subm: bool, ndim: int, device):
    torch.manual_seed(0)
    x = _random_tensor(ndim, 16, device=device)
    conv = SparseConvolution(ndim, 16, 24, 1, subm=subm).to(device)
    with torch.no_grad():
        out = conv(x)
        # KRSC: (out, *kernel_size, in)
        expected = F.linear(x.features, conv.weight.reshape(24, 16), conv.bias)
    assert torch.equal(out.indices, x.indices)
    torch.testing.assert_close(out.features, expected)


def test_conv1x1_is_linear():
    """Different in and out channels, so a transposed weight cannot pass."""
    for device in ["cpu", "cuda"]:
        for ndim in [2, 3]:
            for subm in [True, False]:
                _check_conv1x1(subm, ndim, device)


if __name__ == "__main__":
    test_conv1x1_is_linear()
    test_fused_act()
    print("all ok")
