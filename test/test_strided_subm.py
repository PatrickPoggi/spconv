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

"""SubMConv2d with stride > 1: a submanifold conv that downsamples.

Its output sites are the parents p // stride of the active input sites p, so
the active set never grows, unlike SparseConv2d's. At those sites it must agree
with nn.Conv2d with the same stride and padding, on the zeroed input.
"""

import torch
from torch import nn

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from test_depthwise import (CHANNELS, DEVICE, _assert_close,
                            _assert_close_by_site, _check_inverse,
                            _dense_reference, _parents, _random_input, _sites,
                            _to_sparse)

OUT_CHANNELS = 32
ALGOS = [ConvAlgo.Native, ConvAlgo.MaskImplicitGemm]


def _dense_twin(conv: spconv.SubMConv2d) -> nn.Conv2d:
    """nn.Conv2d with the weights of a spconv conv: KRSC -> KCRS."""
    dense = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                      stride=conv.stride, padding=conv.padding,
                      dilation=conv.dilation,
                      bias=conv.bias is not None).to(DEVICE)
    with torch.no_grad():
        dense.weight.copy_(conv.weight.permute(0, 3, 1, 2))
        if conv.bias is not None:
            dense.bias.copy_(conv.bias)
    return dense


def _check_against_dense(kernel_size: int, stride: int, padding: int, algo,
                         bias: bool = True, size=(16, 16)):
    torch.manual_seed(0)
    x, mask = _random_input(0.6, size)
    conv = spconv.SubMConv2d(CHANNELS, OUT_CHANNELS, kernel_size, stride,
                             padding, bias=bias, algo=algo).to(DEVICE)
    out = conv(_to_sparse(x, mask))
    _assert_close(out.features,
                  _dense_reference(_dense_twin(conv), x, mask, out.indices))


def test_against_dense():
    """Includes an odd grid with padding 0, where the last row and column have
    no output, and a 2x2 kernel, where every output reads its four children."""
    for algo in ALGOS:
        _check_against_dense(3, 2, 1, algo)
        _check_against_dense(3, 2, 1, algo, bias=False)
        _check_against_dense(3, 2, 0, algo, size=(15, 15))
        _check_against_dense(2, 2, 0, algo)
    # kernel volume 49: Native
    _check_against_dense(7, 2, 3, ConvAlgo.Native)


def test_output_sites():
    """Exactly the parents p // 2 of the active sites p: SparseConv2d keeps
    those and adds more."""
    torch.manual_seed(0)
    x, mask = _random_input(0.3)
    st = _to_sparse(x, mask)
    out = spconv.SubMConv2d(CHANNELS, OUT_CHANNELS, 3, 2, 1).to(DEVICE)(st)
    expected = _parents(st.indices, 2)
    assert _sites(out.indices) == expected
    assert out.indices.shape[0] == len(expected)  # no site twice
    assert out.spatial_shape == [8, 8]

    full = spconv.SparseConv2d(CHANNELS, OUT_CHANNELS, 3, 2, 1).to(DEVICE)(st)
    assert expected < _sites(full.indices)


def test_backward_matches_dense():
    for algo in ALGOS:
        torch.manual_seed(0)
        x, mask = _random_input(0.6)
        conv = spconv.SubMConv2d(CHANNELS, OUT_CHANNELS, 3, 2, 1,
                                 algo=algo).to(DEVICE)
        dense = _dense_twin(conv)

        # make both inputs leaf tensors that record a gradient
        st = _to_sparse(x, mask)
        features = st.features.clone()
        features.requires_grad = True
        out_tensor = conv(st.replace_feature(features))
        out = out_tensor.features
        x_ref = x.clone()
        x_ref.requires_grad = True
        ref = _dense_reference(dense, x_ref, mask, out_tensor.indices)

        # the same random upstream gradient into both
        grad_out = torch.randn_like(ref)
        loss = (out * grad_out).sum()
        loss.backward()
        loss_ref = (ref * grad_out).sum()
        loss_ref.backward()

        _assert_close(conv.weight.grad, dense.weight.grad.permute(0, 2, 3, 1))
        _assert_close(conv.bias.grad, dense.bias.grad)
        _assert_close(features.grad, x_ref.grad.permute(0, 2, 3, 1)[mask])


def test_inverse_conv():
    """SparseInverseConv2d maps a strided subm key back to the input sites."""
    for algo in ALGOS:
        down = spconv.SubMConv2d(CHANNELS, OUT_CHANNELS, 3, 2, 1,
                                 indice_key="down", algo=algo).to(DEVICE)
        down_ref = spconv.SparseConv2d(CHANNELS, OUT_CHANNELS, 3, 2, 1,
                                       indice_key="down_ref",
                                       algo=algo).to(DEVICE)
        _check_inverse(down, down_ref, algo)


def test_key_reuse():
    """A strided subm key is reused when the same sites come back, as in the
    downsampling path of a U-Net or a YOLO neck, and refused by a stride 1
    subm conv, whose rulebook is a different one."""
    torch.manual_seed(0)
    x, mask = _random_input(0.6)
    down = spconv.SubMConv2d(CHANNELS, CHANNELS, 3, 2, 1,
                             indice_key="down").to(DEVICE)
    up = spconv.SparseInverseConv2d(CHANNELS, CHANNELS, 3,
                                    indice_key="down").to(DEVICE)
    down_again = spconv.SubMConv2d(CHANNELS, CHANNELS, 3, 2, 1,
                                   indice_key="down").to(DEVICE)
    down_alone = spconv.SubMConv2d(CHANNELS, CHANNELS, 3, 2, 1).to(DEVICE)
    down_alone.load_state_dict(down_again.state_dict())

    low = down(_to_sparse(x, mask))
    back = up(low)  # the input sites, with the key
    out = down_again(back)
    assert out.indices is low.indices  # reused, not rebuilt
    _assert_close_by_site(out, down_alone(back))

    try:
        spconv.SubMConv2d(CHANNELS, CHANNELS, 3,
                          indice_key="down").to(DEVICE)(back)
        raise AssertionError("stride 1 subm conv reused a stride 2 rulebook")
    except ValueError:
        pass


def test_rejects_unsupported():
    # an input site outside the kernel window of its parent p // stride
    for kernel_size, stride, padding in [(1, 2, 0), (3, 2, 2)]:
        try:
            spconv.SubMConv2d(CHANNELS, OUT_CHANNELS, kernel_size, stride,
                              padding)
            raise AssertionError(f"accepted kernel_size {kernel_size}, stride"
                                 f" {stride}, padding {padding}")
        except ValueError:
            pass
    # strided subm implicit gemm: MaskImplicitGemm, kernel volume <= 32
    for kernel_size, algo in [(3, ConvAlgo.MaskSplitImplicitGemm),
                              (7, ConvAlgo.MaskImplicitGemm)]:
        try:
            spconv.SubMConv2d(CHANNELS, OUT_CHANNELS, kernel_size, 2,
                              kernel_size // 2, algo=algo)
            raise AssertionError(f"accepted {algo}, kernel_size {kernel_size}")
        except ValueError:
            pass


if __name__ == "__main__":
    test_against_dense()
    test_output_sites()
    test_backward_matches_dense()
    test_inverse_conv()
    test_key_reuse()
    test_rejects_unsupported()
    print("all ok")
