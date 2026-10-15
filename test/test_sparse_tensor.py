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

"""SparseConvTensor's handling of its inputs."""

import torch

import spconv.pytorch as spconv

DEVICE = torch.device("cuda")


def test_noncontiguous_indices():
    """Indices in any memory layout, as the column-major one nonzero()
    returns on CUDA, mean the same sites: the kernels read rows."""
    torch.manual_seed(0)
    mask = torch.rand(2, 12, 12, device=DEVICE) < 0.4
    indices = mask.nonzero().int().contiguous()
    column_major = indices.t().contiguous().t()
    assert not column_major.is_contiguous()
    features = torch.randn(indices.shape[0], 16, device=DEVICE)
    conv = spconv.SubMConv2d(16, 16, 3, padding=1).to(DEVICE)
    with torch.no_grad():
        st = spconv.SparseConvTensor(features, column_major, [12, 12], 2)
        out = conv(st)
        ref = conv(spconv.SparseConvTensor(features, indices, [12, 12], 2))
    assert st.indices.is_contiguous()
    assert torch.equal(out.indices, ref.indices)
    torch.testing.assert_close(out.features, ref.features)


if __name__ == "__main__":
    test_noncontiguous_indices()
    print("all ok")
