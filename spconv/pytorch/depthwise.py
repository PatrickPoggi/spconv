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

"""Depthwise sparse convolutions.

`SparseConvolutionBase` asserts `groups == 1`, so no grouped or depthwise
convolution exists in spconv. This provides the depthwise case, which is the one
that matters for mobile-style backbones (MobileNet, EfficientNet, YOLO's CIB
blocks): a depthwise 3x3 or 7x7 between pointwise projections, and a strided
depthwise conv to downsample (YOLOv10's SCDown).

Depthwise is the easy case: no channel mixing, so an output site is a weighted
sum of ITS OWN CHANNEL across the kernel offsets.

    out[i, c] = sum_k  weight[k, c] * in[rulebook[k][i], c]   (+ bias[c])

No GEMM is needed. The rulebooks are spconv's own (ops.get_indice_pairs and
ops.get_indice_pairs_implicit_gemm) and are shared through `indice_key` as for
the other sparse convolutions. On CUDA the forward and backward run in the
kernels of spconv/csrc/sparse/depthwise.py, compiled with nvrtc on first use.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

from spconv.core import ConvAlgo
from spconv.cppconstants import CPU_ONLY_BUILD
from spconv.pytorch import functional as Fsp
from spconv.pytorch import ops
from spconv.pytorch.conv import SparseConvolutionBase
from spconv.pytorch.core import (IndiceData, ImplicitGemmIndiceData,
                                 SparseConvTensor, expand_nd)
from spconv.pytorch.modules import SparseModule


class SparseDepthwiseConvolution(SparseModule):
    """Depthwise sparse convolution: one kernel per channel, no channel mixing.

    The counterpart of SparseConvolution for groups == channels.

    Args:
        ndim: number of spatial dimensions.
        channels: both in and out; depthwise cannot change the channel count.
        kernel_size, stride, padding, dilation: as in SparseConvolution.
        bias: add a per-channel bias.
        subm: submanifold, as in SparseConvolution: the active set never
            grows. With stride 1 the output sites are exactly the input
            sites, and padding is always (k // 2) * dilation (SubMConv
            ignores the argument too). With stride > 1 an output site is
            active iff it is the parent p // stride of an active input site p
            (ops.get_indice_pairs).
            Otherwise the conv is regular, as SparseConv: an output site is
            active iff its kernel window holds an active input site, so the
            active set grows.
        indice_key: as in SparseConvolution. A submanifold layer shares its
            rulebook with the submanifold layers (SubMConv included) on the
            same active set with the same kernel size, dilation, stride and
            padding; a mismatch raises ValueError. It reads the rulebook of
            any algo. A regular or strided layer stores its rulebook for an
            inverse conv (SparseInverseConv with the same key and algo).
        algo: the algo whose rulebook this layer builds; None picks the
            default of a SparseConvolution with the same kernel size, so the
            key can be shared with it. The forward pass is the same for every
            algo.

    Weight layout:
        `weight` is (C, *kernel_size, 1). This is spconv's KRSC layout
        (Cout, *kernel_size, Cin) with Cin/groups = 1, i.e. exactly the shape
        a SparseConvolution with groups=C would have if spconv supported
        groups, so a future grouped implementation can take these weights
        unchanged. Kernel offset k of the rulebook is
        weight.reshape(C, -1)[:, k], the same row-major flattening of the
        kernel the Native algorithm uses for every other spconv weight.

        `load_from_conv2d` converts from PyTorch's (C, 1, kh, kw).
    """

    def __init__(self,
                 ndim: int,
                 channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
                 stride: Union[int, List[int], Tuple[int, ...]] = 1,
                 padding: Union[int, List[int], Tuple[int, ...]] = 0,
                 dilation: Union[int, List[int], Tuple[int, ...]] = 1,
                 bias: bool = True,
                 subm: bool = False,
                 indice_key: Optional[str] = None,
                 algo: Optional[ConvAlgo] = None):
        super().__init__()
        self.ndim = ndim
        self.channels = channels
        self.kernel_size = expand_nd(ndim, kernel_size)
        self.stride = expand_nd(ndim, stride)
        self.padding = expand_nd(ndim, padding)
        self.dilation = expand_nd(ndim, dilation)
        self.subm = subm
        self.strided_subm = subm and self.stride != [1] * ndim
        self.indice_key = indice_key
        if subm and not self.strided_subm:
            assert all(k % 2 == 1 for k in self.kernel_size), \
                "submanifold conv only supports odd kernel sizes"
            self.padding = [(k // 2) * d
                            for k, d in zip(self.kernel_size, self.dilation)]
        if algo is None:
            # same default as SparseConvolutionBase.__init__
            kv = 1
            for k in self.kernel_size:
                kv *= k
            if kv <= 32 and not CPU_ONLY_BUILD:
                algo = ConvAlgo.MaskImplicitGemm
            else:
                algo = ConvAlgo.Native
        if CPU_ONLY_BUILD:
            assert algo == ConvAlgo.Native, \
                "cpu only build only support native algorithm"
        self.algo = algo
        if self.strided_subm:
            ops.check_strided_subm(self.kernel_size, self.stride,
                                   self.padding, self.dilation, algo)

        self.weight = Parameter(torch.zeros(self.channels, *self.kernel_size, 1))
        if bias:
            self.bias = Parameter(torch.zeros(self.channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # kaiming_uniform_ with a=sqrt(5), as SparseConvolution uses, reduces
        # to bound = 1 / sqrt(fan_in). Depthwise sums one channel only, so
        # fan_in is the kernel volume, not in_channels times it.
        fan_in = 1
        for s in self.kernel_size:
            fan_in *= s
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = '{channels}, kernel_size={kernel_size}'
        if not self.subm or self.strided_subm:
            s += ', stride={stride}'
            if self.padding != [0] * len(self.padding):
                s += ', padding={padding}'
        if self.dilation != [1] * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        if self.indice_key is not None:
            s += ', indice_key={indice_key}'
        s += f', algo={self.algo}'
        return s.format(**self.__dict__)

    def _get_rulebook(self, input: SparseConvTensor):
        """The rulebook for this input, cached or freshly built.

        Returns (datas, pair_fwd, indice_dict). datas is spconv's IndiceData
        or ImplicitGemmIndiceData, with the output indices and spatial shape.
        pair_fwd is (kernel volume, num_out): pair_fwd[k, i] is the input row
        that output row i reads at offset k, or -1 if that site is inactive.
        This is the implicit gemm format; a Native rulebook is converted to
        it. indice_dict is for the output tensor: as in SparseConvolution, a
        new rulebook goes into a copy of the input's dict, and the input is
        left unchanged.
        """
        indice_dict = input.indice_dict.copy()
        datas = input.find_indice_pair(self.indice_key)
        if datas is not None:
            # as in SparseConvolution: only subm rulebooks are reused, a
            # regular conv's key is for its inverse conv
            assert self.subm, "only support reuse subm indices"
            SparseConvolutionBase._check_subm_reuse_valid(
                self, input, input.spatial_shape, datas)
        else:
            datas = self._build_rulebook(input)
            if self.indice_key is not None:
                indice_dict[self.indice_key] = datas

        if isinstance(datas, ImplicitGemmIndiceData):
            return datas, datas.pair_fwd, indice_dict
        # convert once per rulebook, not once per layer sharing it
        pair_fwd = getattr(datas, "_depthwise_pair_fwd", None)
        if pair_fwd is None:
            symmetric = datas.is_subm and datas.stride == [1] * self.ndim
            pair_fwd = ops.indice_pairs_to_pair_fwd(datas.indice_pairs,
                                                    datas.indice_pair_num,
                                                    datas.out_indices.shape[0],
                                                    subm=symmetric)
            datas._depthwise_pair_fwd = pair_fwd
        return datas, pair_fwd, indice_dict

    def _build_rulebook(
            self, input: SparseConvTensor
    ) -> Union[IndiceData, ImplicitGemmIndiceData]:
        # as in SparseConvolution._conv_forward
        if self.subm and not self.strided_subm:
            out_spatial_shape = input.spatial_shape
        else:
            out_spatial_shape = ops.get_conv_output_size(
                input.spatial_shape, self.kernel_size, self.stride,
                self.padding, self.dilation)
        out_padding = [0] * self.ndim
        if self.algo == ConvAlgo.Native:
            outids, indice_pairs, indice_pair_num = ops.get_indice_pairs(
                input.indices, input.batch_size, input.spatial_shape,
                self.algo, self.kernel_size, self.stride, self.padding,
                self.dilation, out_padding, self.subm)
            return IndiceData(outids, input.indices, indice_pairs,
                              indice_pair_num, input.spatial_shape,
                              out_spatial_shape, is_subm=self.subm,
                              algo=self.algo, ksize=self.kernel_size,
                              stride=self.stride, dilation=self.dilation,
                              padding=self.padding)
        # a regular rulebook gets the bwd pairs in any case, for an inverse
        # conv; a subm one when a SubMConv reusing it may train
        res = ops.get_indice_pairs_implicit_gemm(
            input.indices, input.batch_size, input.spatial_shape, self.algo,
            ksize=self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation, out_padding=out_padding, subm=self.subm,
            transpose=False, is_train=(not self.subm) or self.training,
            alloc=input.thrust_allocator, timer=input._timer)
        return ImplicitGemmIndiceData(
            res[0], input.indices, res[2], res[3],
            pair_mask_fwd_splits=res[4], pair_mask_bwd_splits=res[5],
            mask_argsort_fwd_splits=res[6], mask_argsort_bwd_splits=res[7],
            masks=res[8], is_subm=self.subm,
            spatial_shape=input.spatial_shape,
            out_spatial_shape=out_spatial_shape, algo=self.algo,
            ksize=self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation)

    def _get_pair_bwd(self, datas, pair_fwd: torch.Tensor,
                      num_in: int) -> torch.Tensor:
        """pair_bwd (kernel volume, num_in) of a regular or strided rulebook,
        for the input gradient: the one spconv built, or derived once per
        rulebook from pair_fwd."""
        if isinstance(datas, ImplicitGemmIndiceData) and datas.pair_bwd.numel():
            return datas.pair_bwd
        pair_bwd = getattr(datas, "_depthwise_pair_bwd", None)
        if pair_bwd is None:
            pair_bwd = ops.pair_fwd_to_pair_bwd(pair_fwd, num_in)
            datas._depthwise_pair_bwd = pair_bwd
        return pair_bwd

    def forward(self, input: SparseConvTensor) -> SparseConvTensor:
        """(N, C) features in, (num_out, C) out."""
        datas, pair_fwd, indice_dict = self._get_rulebook(input)
        features = input.features
        # a stride 1 subm rulebook is symmetric, so its bwd pairs are its
        # fwd pairs; the others need pair_bwd for the input gradient
        is_subm = self.subm and not self.strided_subm
        if features.is_cuda:
            pair_bwd = None
            if (not is_subm and features.requires_grad
                    and torch.is_grad_enabled()):
                pair_bwd = self._get_pair_bwd(datas, pair_fwd,
                                              features.shape[0])
            output_features = Fsp.indice_depthwise_conv(
                features, self.weight, self.bias, pair_fwd, pair_bwd,
                pair_fwd.shape[1], is_subm)
        else:
            output_features = self._forward_torch(features, pair_fwd,
                                                  is_subm)

        # assembled as in SparseConvolution
        out_tensor = input.shadow_copy()
        out_tensor = out_tensor.replace_feature(output_features)
        out_tensor.indices = datas.out_indices
        out_tensor.indice_dict = indice_dict
        out_tensor.spatial_shape = datas.out_spatial_shape
        return out_tensor

    def _forward_torch(self, features: torch.Tensor, pair_fwd: torch.Tensor,
                       is_subm: bool) -> torch.Tensor:
        """The forward in PyTorch ops, for CPU tensors: one gather per kernel
        offset. Each output row appears at most once per offset, so it
        accumulates with a plain add: no index_add_, no atomics."""
        pair_fwd = pair_fwd.long()  # indexing takes int64: convert once
        w = self.weight.reshape(self.channels, -1)  # (C, *k, 1) -> (C, kv)
        num_offsets, num_out = pair_fwd.shape

        bias = self.bias
        if bias is None:
            bias = features.new_zeros(self.channels)
        if is_subm:
            # The centre offset maps every row to itself in a stride 1 subm
            # conv, so it needs no gather (spconv's Native forward skips it
            # too). It starts the sum, with the bias folded in.
            center = num_offsets // 2
            output_features = torch.addcmul(bias, features, w[:, center])
        else:
            center = None
            output_features = bias.repeat(num_out, 1)

        # zero row LAST, so that pair_fwd's -1 (no neighbour) reads it
        new_row = features.new_zeros((1, self.channels))
        padded_input = torch.cat([features, new_row], dim=0)
        for offset_index in range(num_offsets):
            if offset_index == center:
                continue
            # multiply and add in one kernel, no temporary for the product
            output_features.addcmul_(padded_input[pair_fwd[offset_index]],
                                     w[:, offset_index])
        return output_features

    @torch.no_grad()
    def load_from_conv2d(self,
                         conv: nn.Conv2d) -> "SparseDepthwiseConvolution":
        """Copy weights from an equivalent `nn.Conv2d(C, C, k, groups=C)`.

        The conv must compute the same function at the output sites: one
        filter per channel, the same stride and dilation, and the same
        padding (for a stride 1 subm layer, the "same" padding). Anything
        else raises ValueError rather than loading weights into a different
        operator.
        """
        if (conv.groups != self.channels or conv.in_channels != self.channels
                or conv.out_channels != self.channels):
            raise ValueError(f"expected depthwise Conv2d with {self.channels}"
                             f" channels, got {conv}")
        if list(conv.kernel_size) != self.kernel_size:
            raise ValueError(f"kernel_size mismatch: conv {list(conv.kernel_size)}"
                             f", this layer {self.kernel_size}")
        if (list(conv.stride) != self.stride
                or list(conv.padding) != self.padding
                or list(conv.dilation) != self.dilation):
            raise ValueError(
                f"expected stride {self.stride}, padding {self.padding} and"
                f" dilation {self.dilation}, got stride {list(conv.stride)},"
                f" padding {conv.padding}, dilation {list(conv.dilation)}")
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
                    algo: Optional[ConvAlgo] = None
                    ) -> "SparseDepthwiseConvolution":
        """Build from a depthwise `nn.Conv2d`, on its device and dtype.

        `indice_key` and `algo` have no dense counterpart, so they are passed
        separately.
        """
        mod = cls(conv.in_channels,
                  kernel_size=list(conv.kernel_size),
                  stride=list(conv.stride),
                  padding=list(conv.padding),
                  dilation=list(conv.dilation),
                  bias=conv.bias is not None,
                  indice_key=indice_key,
                  algo=algo)
        mod = mod.to(device=conv.weight.device, dtype=conv.weight.dtype)
        return mod.load_from_conv2d(conv)


class SubMConvDepthwise2d(SparseDepthwiseConvolution):
    """Depthwise SubMConv2d: the active set never grows. With stride > 1 the
    output sites are the parents p // stride of the active input sites p."""

    def __init__(self,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None):
        super(SubMConvDepthwise2d,
              self).__init__(2,
                             channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             bias,
                             True,
                             indice_key=indice_key,
                             algo=algo)


class SparseConvDepthwise2d(SparseDepthwiseConvolution):
    """Depthwise SparseConv2d: an output site is active iff its kernel window
    holds an active input site, so the active set grows."""

    def __init__(self,
                 channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 indice_key=None,
                 algo: Optional[ConvAlgo] = None):
        super(SparseConvDepthwise2d,
              self).__init__(2,
                             channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             bias,
                             indice_key=indice_key,
                             algo=algo)
