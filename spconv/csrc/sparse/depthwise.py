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

import pccm
from cumm.common import GemmDTypes, TensorViewNVRTC

_DTYPE_SIZES = {"float": 4, "tv::half_t": 2}


class SparseDepthwiseConvKernel(pccm.Class):
    """CUDA kernels of the depthwise sparse convolutions.

    Rulebooks are in the implicit gemm format, as in the implicit gemm max
    pool (IndiceMaxPool): pair[k, i] is the row of `in` that row i reads at
    kernel offset k, or -1. One thread holds one vector of channels of one
    row and loops over the kernel offsets, so every output is written once,
    with no atomics.

    These are compiled at runtime with NVRTC (spconv.pytorch.ops), as the
    nvrtc gemm kernels in spconv.algo, so they run with a prebuilt spconv.
    NVRTC compiles non-template kernels only: an instance is one dtype and
    one vector length (1, or 16 bytes for 128-bit loads).
    """

    def __init__(self, dtype: str, vec_len: int):
        super().__init__()
        self.add_dependency(TensorViewNVRTC, GemmDTypes)
        self.dtype = dtype
        self.vec_len = vec_len
        self.vec_type = (f"tv::array<{dtype}, {vec_len}, "
                         f"{_DTYPE_SIZES[dtype] * vec_len}>")

    @pccm.cuda.cuda_global_function
    def forward_kernel(self):
        """out[i, c] = bias[c] + sum_k filters[k, c] * in[pair[k, i], c].

        Also the input gradient, with the bwd pairs and no bias.
        """
        code = pccm.FunctionCode()
        code.arg("out_features", f"{self.dtype}*")
        code.arg("in_features", f"const {self.dtype}*")
        code.arg("filters", f"const {self.dtype}*")  # [RS, num_features]
        code.arg("bias", f"const {self.dtype}*")  # read if has_bias
        code.arg("indice_pairs", "const int*")  # [RS, num_indices]
        code.arg("has_bias", "int")
        code.arg("num_indices", "int")
        code.arg("num_features", "int")
        code.arg("RS", "int")
        code.arg("num_blocks_x", "int")
        code.arg("num_blocks_y", "int")
        code.arg("one_dim", "int")

        code.raw(f"""
        using VecT = {self.vec_type};
        int block_idx_x = one_dim ? blockIdx.x % num_blocks_x : blockIdx.x;
        int block_idx_y = one_dim ? blockIdx.x / num_blocks_x : blockIdx.y;
        int num_vecs = num_features / {self.vec_len};
        auto out_vecs = reinterpret_cast<VecT*>(out_features);
        auto in_vecs = reinterpret_cast<const VecT*>(in_features);
        auto filter_vecs = reinterpret_cast<const VecT*>(filters);

        for (int i : tv::KernelLoopY<int>(num_indices, block_idx_y, num_blocks_y)) {{
            for (int j : tv::KernelLoopX<int>(num_vecs, block_idx_x, num_blocks_x)) {{
                // accumulate in fp32 and round once
                tv::array<float, {self.vec_len}> acc;
                if (has_bias){{
                    VecT b = reinterpret_cast<const VecT*>(bias)[j];
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < {self.vec_len}; ++v){{
                        acc[v] = float(b[v]);
                    }}
                }}else{{
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < {self.vec_len}; ++v){{
                        acc[v] = 0.0f;
                    }}
                }}
                auto indices_ptr = indice_pairs + i;
                for (int k = 0; k < RS; ++k){{
                    int in_idx = indices_ptr[0];
                    indices_ptr += num_indices;
                    if (in_idx != -1){{
                        VecT x = in_vecs[int64_t(in_idx) * num_vecs + j];
                        VecT w = filter_vecs[k * num_vecs + j];
                        TV_PRAGMA_UNROLL
                        for (int v = 0; v < {self.vec_len}; ++v){{
                            acc[v] += float(w[v]) * float(x[v]);
                        }}
                    }}
                }}
                VecT res;
                TV_PRAGMA_UNROLL
                for (int v = 0; v < {self.vec_len}; ++v){{
                    res[v] = {self.dtype}(acc[v]);
                }}
                out_vecs[int64_t(i) * num_vecs + j] = res;
            }}
        }}
        """)
        return code

    @pccm.cuda.cuda_global_function
    def backward_filter_kernel(self):
        """dfilters[k, c] += sum_i dout[i, c] * in[pair[k, i], c], in fp32.

        blockIdx.y is the kernel offset k, blockIdx.z a block of channel
        vectors, and blockIdx.x a slice of the rows; a block reduces its rows
        in shared memory, then adds its sums to dfilters.
        """
        code = pccm.FunctionCode()
        code.arg("dfilters", "float*")  # [RS, num_features], zeroed
        code.arg("dout_features", f"const {self.dtype}*")
        code.arg("in_features", f"const {self.dtype}*")
        code.arg("indice_pairs", "const int*")  # [RS, num_indices]
        code.arg("num_indices", "int")
        code.arg("num_features", "int")

        code.raw(f"""
        using VecT = {self.vec_type};
        extern __shared__ float row_sums[];  // [blockDim.y, blockDim.x, vec]
        int k = blockIdx.y;
        int j = blockIdx.z * blockDim.x + threadIdx.x;
        int num_vecs = num_features / {self.vec_len};
        auto dout_vecs = reinterpret_cast<const VecT*>(dout_features);
        auto in_vecs = reinterpret_cast<const VecT*>(in_features);

        tv::array<float, {self.vec_len}> acc;
        TV_PRAGMA_UNROLL
        for (int v = 0; v < {self.vec_len}; ++v){{
            acc[v] = 0.0f;
        }}
        if (j < num_vecs){{
            auto indices_ptr = indice_pairs + int64_t(k) * num_indices;
            for (int i = blockIdx.x * blockDim.y + threadIdx.y; i < num_indices;
                    i += gridDim.x * blockDim.y){{
                int in_idx = indices_ptr[i];
                if (in_idx != -1){{
                    VecT dout = dout_vecs[int64_t(i) * num_vecs + j];
                    VecT x = in_vecs[int64_t(in_idx) * num_vecs + j];
                    TV_PRAGMA_UNROLL
                    for (int v = 0; v < {self.vec_len}; ++v){{
                        acc[v] += float(dout[v]) * float(x[v]);
                    }}
                }}
            }}
        }}
        auto sums_ptr = row_sums + (threadIdx.y * blockDim.x + threadIdx.x) * {self.vec_len};
        TV_PRAGMA_UNROLL
        for (int v = 0; v < {self.vec_len}; ++v){{
            sums_ptr[v] = acc[v];
        }}
        __syncthreads();
        if (threadIdx.y == 0 && j < num_vecs){{
            TV_PRAGMA_UNROLL
            for (int v = 0; v < {self.vec_len}; ++v){{
                float sum = 0.0f;
                for (int y = 0; y < blockDim.y; ++y){{
                    sum += row_sums[(y * blockDim.x + threadIdx.x) * {self.vec_len} + v];
                }}
                atomicAdd(dfilters + int64_t(k) * num_features + j * {self.vec_len} + v, sum);
            }}
        }}
        """)
        return code
