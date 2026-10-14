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

"""Kernel size 1, which runs as a matrix multiply: a 1x1 conv must equal a
linear map of each site's features, with the weight in its KRSC layout."""

import pytest
import torch
from torch.nn import functional as F

import spconv.pytorch as spconv

DEVICE = torch.device("cuda")


def _random_tensor(ndim: int, channels: int, dtype=torch.float32,
                   device=DEVICE, size: int = 12, batch_size: int = 2,
                   density: float = 0.4) -> spconv.SparseConvTensor:
    shape = [batch_size] + [size] * ndim
    indices = (torch.rand(shape, device=device) < density).nonzero().int()
    features = torch.randn(indices.shape[0], channels, device=device)
    return spconv.SparseConvTensor(features.to(dtype), indices, [size] * ndim,
                                   batch_size)


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
