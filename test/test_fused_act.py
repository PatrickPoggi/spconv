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

import pytest
import torch
from cumm import tensorview as tv
from torch.nn import functional as F

import spconv.pytorch as spconv
from spconv.core import ConvAlgo

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
    indices = (torch.rand(shape, device=device) < density).nonzero().int()
    features = torch.randn(indices.shape[0], channels, device=device)
    return spconv.SparseConvTensor(features.to(dtype), indices, [size] * ndim,
                                   batch_size)


def _conv_types(ndim: int, algo, dtype, in_channels: int,
                out_channels: int):
    """name -> (conv, run): one conv of each type, and the forward from the
    test input to its output."""

    def conv(*args, **kwargs):
        return spconv.SparseConvolution(ndim, *args, algo=algo,
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("algo", ALGOS)
@pytest.mark.parametrize("ndim", [2, 3])
def test_fused_act(ndim, algo, dtype):
    """Native fuses the activation in its bias kernel, MaskImplicitGemm in the
    gemm epilogue; a 1x1 conv applies it after the matrix multiply."""
    torch.manual_seed(0)
    x = _random_tensor(ndim, 16, dtype)
    tolerance = TOLERANCE[dtype]
    for name, (conv, run) in _conv_types(ndim, algo, dtype, 16, 24).items():
        for act, act_fn in ACTS:
            with torch.no_grad():
                conv.act_type, conv.act_alpha = tv.gemm.Activation.None_, 0.0
                expected = act_fn(run(x).features.float())
                conv.act_type, conv.act_alpha = act, ALPHA
                out = run(x).features.float()
            torch.testing.assert_close(
                out, expected, atol=tolerance, rtol=tolerance,
                msg=lambda m: f"{name} {act}: {m}")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("subm", [True, False])
def test_conv1x1_is_linear(subm, ndim, device):
    """Different in and out channels, so a transposed weight cannot pass."""
    torch.manual_seed(0)
    x = _random_tensor(ndim, 16, device=device)
    conv = spconv.SparseConvolution(ndim, 16, 24, 1, subm=subm).to(device)
    with torch.no_grad():
        out = conv(x)
        # KRSC: (out, *kernel_size, in)
        expected = F.linear(x.features, conv.weight.reshape(24, 16), conv.bias)
    assert torch.equal(out.indices, x.indices)
    torch.testing.assert_close(out.features, expected)
