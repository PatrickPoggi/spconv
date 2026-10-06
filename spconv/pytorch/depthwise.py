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

"""Submanifold depthwise convolution.

`SparseConvolutionBase` asserts `groups == 1`, so no grouped or depthwise
convolution exists in spconv. This provides the depthwise case, which is the one
that matters for mobile-style backbones (MobileNet, EfficientNet, YOLO's CIB
blocks) where a depthwise 3x3 or 7x7 sits between pointwise projections.

Depthwise is the easy case: no channel mixing, so the output at an active site is
a weighted sum of THAT SITE'S OWN CHANNEL across the kernel offsets.

    out[i, c] = sum_k  weight[k, c] * in[rulebook[k][i], c]   (+ bias[c])

No GEMM is needed. The rulebook is the same one a submanifold `SubMConv` builds,
so this reuses it when an `indice_key` hits and builds it otherwise.
"""

import math
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from spconv.core import ConvAlgo
from spconv.cppconstants import CPU_ONLY_BUILD
from spconv.pytorch import ops
from spconv.pytorch.core import (IndiceData, ImplicitGemmIndiceData,
                                 SparseConvTensor, expand_nd)
from spconv.pytorch.modules import SparseModule


class SubMConvDepthwise2d(SparseModule):
    """Depthwise submanifold sparse convolution.

    One kernel per channel, no channel mixing, and the active set is unchanged --
    output sites are exactly the input sites.

    Args:
        channels: both in and out; depthwise cannot change the channel count.
        kernel_size: int or [kh, kw].
        dilation: int or [dh, dw].
        bias: add a per-channel bias.
        indice_key: share the rulebook with other submanifold layers on the
            same active set and the same kernel size and dilation; a mismatch
            raises ValueError, as in SparseConvolution. This layer reads the
            rulebook of any algo; when it builds one, it uses `algo`.
        algo: the algo whose rulebook this layer builds. None picks the
            default of a SubMConv2d with the same kernel size, so the key can
            be shared with it; set it to match a SubMConv2d created with an
            explicit algo. The forward pass is the same for every algo.

    Weight layout:
        `weight` is (C, kh, kw, 1). This is spconv's KRSC layout
        (Cout, kh, kw, Cin) with Cin/groups = 1, i.e. exactly the shape a
        `SubMConv2d(C, C, k, groups=C)` would have if spconv supported groups,
        so a future grouped implementation can take these weights unchanged.
        Kernel offset k of the rulebook is weight.reshape(C, -1)[:, k], the
        same row-major flattening of (kh, kw) the Native algorithm uses for
        every other spconv weight.

        `load_from_conv2d` converts from PyTorch's (C, 1, kh, kw).
    """

    def __init__(self,
                 channels: int,
                 kernel_size: Union[int, List[int]] = 3,
                 dilation: Union[int, List[int]] = 1,
                 bias: bool = True,
                 indice_key: Optional[str] = None,
                 algo: Optional[ConvAlgo] = None):
        super().__init__()
        self.channels = channels
        self.kernel_size = expand_nd(2, kernel_size)
        self.dilation = expand_nd(2, dilation)
        self.indice_key = indice_key
        assert all(k % 2 == 1 for k in self.kernel_size), \
            "submanifold conv only supports odd kernel sizes"
        if algo is None:
            # same default as SparseConvolutionBase.__init__
            kv = 1
            for s in self.kernel_size:
                kv *= s
            if kv <= 32 and not CPU_ONLY_BUILD:
                algo = ConvAlgo.MaskImplicitGemm
            else:
                algo = ConvAlgo.Native
        if CPU_ONLY_BUILD:
            assert algo == ConvAlgo.Native, \
                "cpu only build only support native algorithm"
        self.algo = algo

        self.weight = Parameter(torch.zeros(self.channels, *self.kernel_size, 1))
        if bias:
            self.bias = Parameter(torch.zeros(self.channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_uniform_ with a=sqrt(5), as SparseConvolution uses, reduces
        # to bound = 1 / sqrt(fan_in). Depthwise sums one channel only, so
        # fan_in is kh * kw, not in_channels * kh * kw as in SparseConvolution.
        fan_in = 1
        for s in self.kernel_size:
            fan_in *= s
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = '{channels}, kernel_size={kernel_size}'
        if self.dilation != [1] * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        if self.indice_key is not None:
            s += ', indice_key={indice_key}'
        s += f', algo={self.algo}'
        return s.format(**self.__dict__)

    def _get_rulebook(self, input: SparseConvTensor):
        """Gather table for the kernel offsets, cached or freshly built.

        Returns (pair_fwd, indice_dict). pair_fwd is (kh * kw, N):
        pair_fwd[k, i] is the input row that output row i reads at offset k,
        or -1 if that neighbour is inactive. This is the implicit gemm format;
        a Native rulebook is converted to it. indice_dict is for the output
        tensor: as in SparseConvolution, a new rulebook goes into a copy of
        the input's dict, and the input is left unchanged.
        """
        indice_dict = input.indice_dict.copy()
        datas = input.find_indice_pair(self.indice_key)
        if datas is not None:
            # same checks as SparseConvolution._check_subm_reuse_valid
            self._check_subm_reuse_valid(input, datas)
        else:
            datas = self._build_rulebook(input)
            if self.indice_key is not None:
                indice_dict[self.indice_key] = datas

        if isinstance(datas, ImplicitGemmIndiceData):
            return datas.pair_fwd, indice_dict
        # convert once per rulebook, not once per layer sharing it
        pair_fwd = getattr(datas, "_depthwise_pair_fwd", None)
        if pair_fwd is None:
            pair_fwd = self._native_to_pair_fwd(datas.indice_pairs,
                                                datas.indice_pair_num,
                                                input.indices.shape[0])
            datas._depthwise_pair_fwd = pair_fwd
        return pair_fwd, indice_dict

    def _build_rulebook(
            self, input: SparseConvTensor
    ) -> Union[IndiceData, ImplicitGemmIndiceData]:
        # subm: stride 1 and padding (k // 2) * dilation are fixed by the
        # index generation kernel (indices.py, generate_subm_conv_inds).
        stride = [1, 1]
        padding = [(k // 2) * d
                   for k, d in zip(self.kernel_size, self.dilation)]
        out_padding = [0, 0]
        algo = self.algo
        if algo != ConvAlgo.Native:
            # a SubMConv reusing this in training needs pair_bwd, hence
            # is_train=self.training as in SparseConvolution.
            res = ops.get_indice_pairs_implicit_gemm(
                input.indices, input.batch_size, input.spatial_shape, algo,
                ksize=self.kernel_size, stride=stride, padding=padding,
                dilation=self.dilation, out_padding=out_padding, subm=True,
                transpose=False, is_train=self.training,
                alloc=input.thrust_allocator, timer=input._timer)
            return ImplicitGemmIndiceData(
                res[0], input.indices, res[2], res[3],
                pair_mask_fwd_splits=res[4], pair_mask_bwd_splits=res[5],
                mask_argsort_fwd_splits=res[6],
                mask_argsort_bwd_splits=res[7], masks=res[8], is_subm=True,
                spatial_shape=input.spatial_shape,
                out_spatial_shape=input.spatial_shape, algo=algo,
                ksize=self.kernel_size, stride=stride, padding=padding,
                dilation=self.dilation)
        outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
            input.indices, input.batch_size, input.spatial_shape, algo,
            self.kernel_size, stride, padding, self.dilation, out_padding,
            subm=True)
        return IndiceData(outids, input.indices, indice_pairs,
                          indice_pair_num, input.spatial_shape,
                          input.spatial_shape, is_subm=True, algo=algo,
                          ksize=self.kernel_size, stride=stride,
                          dilation=self.dilation, padding=padding)

    @staticmethod
    def _native_to_pair_fwd(indice_pairs: torch.Tensor,
                            indice_pair_num: torch.Tensor,
                            num_act: int) -> torch.Tensor:
        # indice_pairs[0, k, j] is the input row, [1, k, j] the output row
        # (indices.py, calc_subm_conv_indices). For subm the kernel runs only
        # offsets 0..centre and writes each pair's mirror at kv - 1 - k, so
        # only those counts are stored: the centre covers all num_act rows,
        # and offset k > centre has the count of kv - 1 - k. This is how
        # ops.indice_conv reads them too.
        kv = indice_pairs.shape[1]
        center = kv // 2
        counts = indice_pair_num.to(indice_pairs.device).clone()
        counts[center] = num_act
        counts[center + 1:] = counts[:center].flip(0)
        valid = (torch.arange(indice_pairs.shape[2],
                              device=indice_pairs.device)[None]
                 < counts[:, None])
        # Entries past the count are uninitialised. Send them to an extra
        # column that is dropped afterwards, rather than filtering them out
        # with nonzero(), which would sync the GPU.
        dst = torch.where(valid, indice_pairs[1].long(), num_act)
        pair_fwd = torch.full((kv, num_act + 1), -1, dtype=indice_pairs.dtype,
                              device=indice_pairs.device)
        pair_fwd.scatter_(1, dst, indice_pairs[0])
        return pair_fwd[:, :num_act]

    def _check_subm_reuse_valid(self, input: SparseConvTensor, datas):
        assert datas.is_subm, "only support reuse subm indices"
        if self.kernel_size != datas.ksize:
            raise ValueError(
                f"subm with same indice_key must have same kernel"
                f" size, expect {datas.ksize}, this layer {self.kernel_size}")
        if self.dilation != datas.dilation:
            raise ValueError(
                f"subm with same indice_key must have same dilation"
                f", expect {datas.dilation}, this layer {self.dilation}")
        if input.spatial_shape != datas.spatial_shape:
            raise ValueError(
                f"subm with same indice_key must have same spatial structure"
                f", expect {datas.spatial_shape}, input {input.spatial_shape}")
        if input.indices.shape[0] != datas.indices.shape[0]:
            raise ValueError(
                f"subm with same indice_key must have same num of indices"
                f", expect {datas.indices.shape[0]}, input {input.indices.shape[0]}")

    def forward(self, input: SparseConvTensor) -> SparseConvTensor:
        """(N, C) features in, (N, C) out, same coordinates.

        One gather per kernel offset. Each output row appears at most once per
        offset, so it accumulates with a plain add: no index_add_, no atomics.
        """
        pair_fwd, indice_dict = self._get_rulebook(input)
        pair_fwd = pair_fwd.long()  # indexing takes int64: convert once
        features = input.features
        w = self.weight.reshape(self.channels, -1)  # (C, kh, kw, 1) -> (C, kv)

        # The centre offset maps every row to itself in a submanifold conv,
        # so it needs no gather (spconv's Native forward skips it too).
        # Starting from it also replaces a zero fill and a separate bias add.
        center = pair_fwd.shape[0] // 2
        if self.bias is not None:
            output_features = torch.addcmul(self.bias, features, w[:, center])
        else:
            output_features = features * w[:, center]

        # zero row LAST, so that pair_fwd's -1 (no neighbour) reads it
        new_row = features.new_zeros((1, self.channels))
        padded_input = torch.cat([features, new_row], dim=0)
        for offset_index in range(pair_fwd.shape[0]):
            if offset_index == center:
                continue
            # multiply and add in one kernel, no temporary for the product
            output_features.addcmul_(padded_input[pair_fwd[offset_index]],
                                     w[:, offset_index])

        out_tensor = input.replace_feature(output_features)
        out_tensor.indice_dict = indice_dict
        return out_tensor

    @torch.no_grad()
    def load_from_conv2d(self, conv: nn.Conv2d) -> "SubMConvDepthwise2d":
        """Copy weights from an equivalent `nn.Conv2d(C, C, k, groups=C)`.

        The conv must compute the same function at the active sites: one
        filter per channel, stride 1 and "same" padding. Anything else raises
        ValueError rather than loading weights into a different operator.
        """
        if (conv.groups != self.channels or conv.in_channels != self.channels
                or conv.out_channels != self.channels):
            raise ValueError(f"expected depthwise Conv2d with {self.channels}"
                             f" channels, got {conv}")
        if list(conv.kernel_size) != self.kernel_size:
            raise ValueError(f"kernel_size mismatch: conv {list(conv.kernel_size)}"
                             f", this layer {self.kernel_size}")
        if list(conv.dilation) != self.dilation or list(conv.stride) != [1, 1]:
            raise ValueError(f"expected stride [1, 1] and dilation"
                             f" {self.dilation}, got stride {list(conv.stride)}"
                             f", dilation {list(conv.dilation)}")
        # submanifold output sites are the input sites: the dense equivalent
        # is the "same" padding, (k // 2) * dilation
        same_padding = [(k // 2) * d
                        for k, d in zip(self.kernel_size, self.dilation)]
        if list(conv.padding) != same_padding:
            raise ValueError(
                f"expected padding {same_padding}, got {conv.padding}")
        if (conv.bias is None) != (self.bias is None):
            raise ValueError("bias presence differs between conv and this layer")

        # (C, 1, kh, kw) -> (C, kh, kw, 1)
        self.weight.copy_(conv.weight.permute(0, 2, 3, 1))
        if self.bias is not None:
            self.bias.copy_(conv.bias)
        return self

    @classmethod
    def from_conv2d(cls,
                    conv: nn.Conv2d,
                    indice_key: Optional[str] = None,
                    algo: Optional[ConvAlgo] = None) -> "SubMConvDepthwise2d":
        """Build from a depthwise `nn.Conv2d`, on its device and dtype.

        `indice_key` and `algo` have no dense counterpart, so they are passed
        separately.
        """
        mod = cls(conv.in_channels,
                  kernel_size=list(conv.kernel_size),
                  dilation=list(conv.dilation),
                  bias=conv.bias is not None,
                  indice_key=indice_key,
                  algo=algo)
        mod = mod.to(device=conv.weight.device, dtype=conv.weight.dtype)
        return mod.load_from_conv2d(conv)

