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

# activations fused into the forward kernels, on the fp32 sum `a`: the cumm
# Activation ones (none, relu, sigmoid, leaky_relu), and silu
# (x * sigmoid(x)). The fast exp and division err by a few float ulps, far
# below a float16 output's rounding; for large -a, exp(-a) is inf and both go
# to 0, as they should.
_ACTIVATIONS = {
    "none": "",
    "relu": "a = a > 0.0f ? a : 0.0f;",
    "sigmoid": "a = __fdividef(1.0f, 1.0f + __expf(-a));",
    "leaky_relu": "a = a >= 0.0f ? a : a * act_alpha;",
    "silu": "a = __fdividef(a, 1.0f + __expf(-a));",
}


class SparseDepthwiseConvKernel(pccm.Class):
    """CUDA kernels of the depthwise sparse convolutions.

    Rulebooks are in the implicit gemm format, as in the implicit gemm max
    pool (IndiceMaxPool): pair[k, i] is the row of `in` that row i reads at
    kernel offset k, or -1. One thread holds one vector of channels of one
    row and loops over the kernel offsets, so every output is written once,
    with no atomics.

    These are compiled at runtime with NVRTC (spconv.pytorch.ops), as the
    nvrtc gemm kernels in spconv.algo, so they run with a prebuilt spconv.
    NVRTC compiles non-template kernels only: an instance is one dtype, one
    vector length (1, or 16 bytes for 128-bit loads) and one activation of
    forward_kernel. A kernel takes its arguments in one array, pointers
    included, as a gemm kernel takes its params struct, so a launch passes
    one argument.
    """

    def __init__(self, dtype: str, vec_len: int, act_type: str = "none"):
        super().__init__()
        self.add_dependency(TensorViewNVRTC, GemmDTypes)
        self.dtype = dtype
        self.vec_len = vec_len
        self.act = _ACTIVATIONS[act_type]
        # alignedarray: tv::array does not align to its third argument, and a
        # vector with the alignment of its element loads one element at a time
        self.vec_type = (f"tv::alignedarray<{dtype}, {vec_len}, "
                         f"{_DTYPE_SIZES[dtype] * vec_len}>")

    @pccm.cuda.cuda_global_function
    def forward_kernel(self):
        """out[i, c] = act(bias[c] + sum_k filters[k, c] * in[pair[k, i], c]).

        Also the input gradient, with the bwd pairs, no bias and no
        activation.
        """
        code = pccm.FunctionCode()
        # 0 out_features, 1 in_features, 2 filters [RS, num_features], 3 bias
        # (read if has_bias), 4 indice_pairs [RS, num_indices], 5 has_bias,
        # 6 num_indices, 7 num_features, 8 RS, 9 num_blocks_x,
        # 10 num_blocks_y, 11 one_dim, 12 act_alpha as float bits
        code.arg("args", "tv::array<int64_t, 13>")

        code.raw(f"""
        using T = {self.dtype};
        using VecT = {self.vec_type};
        T* out_features = reinterpret_cast<T*>(args[0]);
        const T* in_features = reinterpret_cast<const T*>(args[1]);
        const T* filters = reinterpret_cast<const T*>(args[2]);
        const T* bias = reinterpret_cast<const T*>(args[3]);
        const int* indice_pairs = reinterpret_cast<const int*>(args[4]);
        int has_bias = int(args[5]), num_indices = int(args[6]);
        int num_features = int(args[7]), RS = int(args[8]);
        int num_blocks_x = int(args[9]), num_blocks_y = int(args[10]);
        int one_dim = int(args[11]);
        float act_alpha = __int_as_float(int(args[12]));
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
                    float a = acc[v];
                    {self.act}
                    res[v] = {self.dtype}(a);
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
        # 0 dfilters: float [RS, num_features], zeroed, 1 dout_features,
        # 2 in_features, 3 indice_pairs [RS, num_indices], 4 num_indices,
        # 5 num_features
        code.arg("args", "tv::array<int64_t, 6>")

        code.raw(f"""
        using T = {self.dtype};
        using VecT = {self.vec_type};
        float* dfilters = reinterpret_cast<float*>(args[0]);
        const T* dout_features = reinterpret_cast<const T*>(args[1]);
        const T* in_features = reinterpret_cast<const T*>(args[2]);
        const int* indice_pairs = reinterpret_cast<const int*>(args[3]);
        int num_indices = int(args[4]), num_features = int(args[5]);
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


class SparseDepthwiseConvTileKernel(pccm.Class):
    """Forward of a 2D depthwise sparse convolution, by output tiles.

    A block takes an output tile and 32 channels. It loads the input sites
    the tile reads, halo included, into shared memory once, through an index
    grid (the row of each site, -1 where inactive), and computes the outputs
    of the tile from there: an input is read once per tile instead of once
    per kernel offset, as a dense depthwise convolution tiles its input.
    Inactive inputs are zeros, and only active outputs are written, after
    the bias and the activation.

    A thread computes 4 channels of `P` outputs of one column of the tile,
    so an input it loads serves every output of the column that reads it.
    The input tile is stored by columns modulo the stride, so the threads of
    neighbouring columns read neighbouring shared memory at any stride.

    Everything but the input sizes is a compile-time constant, so the loops
    over the taps unroll; an instance is compiled with NVRTC once per layer
    configuration (spconv.pytorch.ops). All arguments come in one array.
    """

    # threads of 4 channels on 8 threads: 32 channels per block
    CHANNELS_PER_THREAD = 4
    THREADS_PER_SITE = 8
    CHANNELS = CHANNELS_PER_THREAD * THREADS_PER_SITE

    def __init__(self, dtype: str, ksize, stride, dilation, tile,
                 outputs_per_thread: int, act_type: str):
        super().__init__()
        self.add_dependency(TensorViewNVRTC, GemmDTypes)
        assert tile[0] % outputs_per_thread == 0
        self.dtype = dtype
        self.ksize = ksize
        self.stride = stride
        self.dilation = dilation
        self.tile = tile
        self.outputs_per_thread = outputs_per_thread
        self.act = _ACTIVATIONS[act_type]
        self.num_threads = (tile[0] // outputs_per_thread) * tile[1] * \
            self.THREADS_PER_SITE
        # the input sites an output tile reads
        self.region = [(t - 1) * s + (k - 1) * d + 1
                       for t, k, s, d in zip(tile, ksize, stride, dilation)]

    @pccm.cuda.cuda_global_function
    def tile_forward_kernel(self):
        code = pccm.FunctionCode()
        # 0 out_features, 1 in_features, 2 weights: float [KH * KW + 1, C],
        # the filters and then the bias, 3 in_grid [batch, in_h, in_w],
        # 4 out_grid [batch, out_h, out_w], 5 num_features, 6 in_h, 7 in_w,
        # 8 out_h, 9 out_w, 10 pad_h, 11 pad_w, 12 channel blocks,
        # 13 tiles along x, 14 tiles along y, 15 act_alpha as float bits,
        # 16 elements from one row of in_features to the next (a view of some
        # channels of a wider tensor has more than num_features)
        code.arg("args", "tv::array<int64_t, 17>")

        T = self.dtype
        code.raw(f"""
        using T = {T};
        constexpr int KH = {self.ksize[0]}, KW = {self.ksize[1]};
        constexpr int SH = {self.stride[0]}, SW = {self.stride[1]};
        constexpr int DH = {self.dilation[0]}, DW = {self.dilation[1]};
        constexpr int TH = {self.tile[0]}, TW = {self.tile[1]};
        constexpr int P = {self.outputs_per_thread};
        constexpr int V = {self.CHANNELS_PER_THREAD}, L = {self.THREADS_PER_SITE};
        constexpr int CB = V * L;
        constexpr int THREADS = {self.num_threads};
        constexpr int RH = {self.region[0]}, RW = {self.region[1]};
        // the input tile in shared memory: [RH][SW][RWS] sites of CB
        // channels, site (ry, rx) at ry * RWP + (rx % SW) * RWS + rx / SW
        constexpr int RWS = (RW + SW - 1) / SW;
        constexpr int RWP = RWS * SW;
        constexpr int CHUNK = 16 / int(sizeof(T));
        constexpr int CPS = CB / CHUNK;  // 16-byte chunks per site
        constexpr int NCHUNK = RH * RWP * CPS;
        constexpr int ITERS = (NCHUNK + THREADS - 1) / THREADS;
        constexpr int BATCH = ITERS < 8 ? ITERS : 8;
        constexpr int OITERS = (TH * TW + THREADS - 1) / THREADS;
        constexpr int NW = (KH * KW + 1) * L;
        constexpr int WITERS = (NW + THREADS - 1) / THREADS;
        // aligned, so each is one 64- or 128-bit load or store
        using ChunkT = tv::alignedarray<T, CHUNK, 16>;
        using LaneT = tv::alignedarray<T, V, V * sizeof(T)>;
        using LaneF = tv::alignedarray<float, V, 16>;

        // bytes, as T may not be trivially constructible
        __shared__ __align__(16) unsigned char smem_in_bytes[RH * RWP * CB * sizeof(T)];
        __shared__ __align__(16) float smem_w[(KH * KW + 1) * CB];
        __shared__ int smem_out[TH * TW];
        T* smem_in = reinterpret_cast<T*>(smem_in_bytes);

        T* out_features = reinterpret_cast<T*>(args[0]);
        const T* in_features = reinterpret_cast<const T*>(args[1]);
        const float* weights = reinterpret_cast<const float*>(args[2]);
        const int* in_grid = reinterpret_cast<const int*>(args[3]);
        const int* out_grid = reinterpret_cast<const int*>(args[4]);
        int num_features = int(args[5]);
        int in_h = int(args[6]), in_w = int(args[7]);
        int out_h = int(args[8]), out_w = int(args[9]);
        int pad_h = int(args[10]), pad_w = int(args[11]);
        int num_cb = int(args[12]);
        int num_tiles_x = int(args[13]), num_tiles_y = int(args[14]);
        float act_alpha = __int_as_float(int(args[15]));
        int64_t in_row_stride = args[16];

        // the channel blocks of a tile are neighbours, so they read the rows
        // of the same sites together
        int bx = blockIdx.x;
        int cb = bx % num_cb;
        bx /= num_cb;
        int tile_x = bx % num_tiles_x;
        bx /= num_tiles_x;
        int tile_y = bx % num_tiles_y;
        int b = bx / num_tiles_y;
        int c0 = cb * CB;
        int oy0 = tile_y * TH, ox0 = tile_x * TW;
        int iy0 = oy0 * SH - pad_h, ix0 = ox0 * SW - pad_w;
        const int* in_grid_b = in_grid + int64_t(b) * in_h * in_w;
        const int* out_grid_b = out_grid + int64_t(b) * out_h * out_w;

        // loads without dependencies first: output rows and weights
        int out_rows[OITERS];
        TV_PRAGMA_UNROLL
        for (int i = 0; i < OITERS; ++i){{
            int t = threadIdx.x + i * THREADS;
            int oy = oy0 + t / TW, ox = ox0 + t % TW;
            out_rows[i] = -1;
            if (t < TH * TW && oy < out_h && ox < out_w){{
                out_rows[i] = out_grid_b[oy * out_w + ox];
            }}
        }}
        LaneF w_regs[WITERS];
        TV_PRAGMA_UNROLL
        for (int i = 0; i < WITERS; ++i){{
            int t = threadIdx.x + i * THREADS;
            int c = c0 + (t % L) * V;
            TV_PRAGMA_UNROLL
            for (int v = 0; v < V; ++v){{
                w_regs[i][v] = 0.0f;
            }}
            if (t < NW && c < num_features){{
                w_regs[i] = *reinterpret_cast<const LaneF*>(
                    weights + int64_t(t / L) * num_features + c);
            }}
        }}
        // the input tile: grid rows, then features, a batch of loads at a
        // time; inactive or outside sites are zeros
        TV_PRAGMA_UNROLL
        for (int it0 = 0; it0 < ITERS; it0 += BATCH){{
            int rows[BATCH];
            TV_PRAGMA_UNROLL
            for (int i = 0; i < BATCH; ++i){{
                int q = threadIdx.x + (it0 + i) * THREADS;
                int site = q / CPS;
                int ry = site / RWP, rem = site % RWP;
                int rx = (rem % RWS) * SW + rem / RWS;
                int iy = iy0 + ry, ix = ix0 + rx;
                rows[i] = -1;
                if (it0 + i < ITERS && q < NCHUNK && rx < RW && iy >= 0 &&
                        iy < in_h && ix >= 0 && ix < in_w){{
                    rows[i] = in_grid_b[iy * in_w + ix];
                }}
            }}
            ChunkT vals[BATCH];
            TV_PRAGMA_UNROLL
            for (int i = 0; i < BATCH; ++i){{
                int q = threadIdx.x + (it0 + i) * THREADS;
                int c = c0 + (q % CPS) * CHUNK;
                TV_PRAGMA_UNROLL
                for (int v = 0; v < CHUNK; ++v){{
                    vals[i][v] = T(0.0f);
                }}
                if (rows[i] >= 0 && c < num_features){{
                    vals[i] = *reinterpret_cast<const ChunkT*>(
                        in_features + int64_t(rows[i]) * in_row_stride + c);
                }}
            }}
            TV_PRAGMA_UNROLL
            for (int i = 0; i < BATCH; ++i){{
                int q = threadIdx.x + (it0 + i) * THREADS;
                if (it0 + i < ITERS && q < NCHUNK){{
                    reinterpret_cast<ChunkT*>(smem_in)[q] = vals[i];
                }}
            }}
        }}
        int any_active = 0;
        TV_PRAGMA_UNROLL
        for (int i = 0; i < OITERS; ++i){{
            int t = threadIdx.x + i * THREADS;
            if (t < TH * TW){{
                smem_out[t] = out_rows[i];
                any_active |= out_rows[i] >= 0;
            }}
        }}
        TV_PRAGMA_UNROLL
        for (int i = 0; i < WITERS; ++i){{
            int t = threadIdx.x + i * THREADS;
            if (t < NW){{
                *reinterpret_cast<LaneF*>(smem_w + (t / L) * CB + (t % L) * V) = w_regs[i];
            }}
        }}
        // a tile without active outputs has nothing to do
        if (!__syncthreads_or(any_active)){{
            return;
        }}

        // this thread: channels c..c+V-1 of the outputs (sy * P + p, cx) of
        // the tile; no barrier follows, so idle threads may return
        int lane = threadIdx.x % L;
        int cx = (threadIdx.x / L) % TW;
        int sy = threadIdx.x / (L * TW);
        int c = c0 + lane * V;
        int rows_p[P];
        int active = 0;
        TV_PRAGMA_UNROLL
        for (int p = 0; p < P; ++p){{
            rows_p[p] = smem_out[(sy * P + p) * TW + cx];
            active |= rows_p[p] >= 0;
        }}
        if (!active || c >= num_features){{
            return;
        }}
        // accumulate in fp32 from the bias, round once
        float acc[P][V];
        LaneF bias = *reinterpret_cast<const LaneF*>(smem_w + KH * KW * CB + lane * V);
        TV_PRAGMA_UNROLL
        for (int p = 0; p < P; ++p){{
            TV_PRAGMA_UNROLL
            for (int v = 0; v < V; ++v){{
                acc[p][v] = bias[v];
            }}
        }}
        const LaneT* in_lanes = reinterpret_cast<const LaneT*>(smem_in);
        constexpr int ROWS = (P - 1) * SH + (KH - 1) * DH + 1;
        TV_PRAGMA_UNROLL
        for (int s = 0; s < KW; ++s){{
            // column s * DW of the window of column cx, in shared memory
            int col = ((s * DW) % SW) * RWS + cx + (s * DW) / SW;
            LaneF w[KH];
            TV_PRAGMA_UNROLL
            for (int r = 0; r < KH; ++r){{
                w[r] = *reinterpret_cast<const LaneF*>(smem_w + (r * KW + s) * CB + lane * V);
            }}
            // the input rows of the P outputs: row j serves output p with
            // tap r where j == p * SH + r * DH
            TV_PRAGMA_UNROLL
            for (int j = 0; j < ROWS; ++j){{
                LaneT xh = in_lanes[((sy * P * SH + j) * RWP + col) * L + lane];
                float x[V];
                TV_PRAGMA_UNROLL
                for (int v = 0; v < V; ++v){{
                    x[v] = float(xh[v]);
                }}
                TV_PRAGMA_UNROLL
                for (int p = 0; p < P; ++p){{
                    if (j >= p * SH && (j - p * SH) % DH == 0 && (j - p * SH) / DH < KH){{
                        TV_PRAGMA_UNROLL
                        for (int v = 0; v < V; ++v){{
                            acc[p][v] += w[(j - p * SH) / DH][v] * x[v];
                        }}
                    }}
                }}
            }}
        }}
        TV_PRAGMA_UNROLL
        for (int p = 0; p < P; ++p){{
            if (rows_p[p] >= 0){{
                LaneT res;
                TV_PRAGMA_UNROLL
                for (int v = 0; v < V; ++v){{
                    float a = acc[p][v];
                    {self.act}
                    res[v] = T(a);
                }}
                *reinterpret_cast<LaneT*>(out_features + int64_t(rows_p[p]) * num_features + c) = res;
            }}
        }}
        """)
        return code


class GridPairsKernel(pccm.Class):
    """The forward rulebook of a 2D conv at given output sites, read from the
    index grid of its input (ops.depthwise_index_grid: the row of each active
    site, -1 elsewhere), in the implicit gemm format SpconvOps builds:
    pair[k, o] is the input row output o reads at kernel offset k (row-major
    over the kernel), or -1, and bit k of mask[o] is set iff that row exists.

    One thread per output and no hash table, so there is nothing to wait for
    on the host. The kernel size, stride, padding and dilation are
    compile-time constants: the loop over the taps unrolls.
    """

    def __init__(self, ksize, stride, padding, dilation):
        super().__init__()
        self.add_dependency(TensorViewNVRTC)
        self.ksize = ksize
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    @pccm.cuda.cuda_global_function
    def grid_pairs_kernel(self):
        code = pccm.FunctionCode()
        # 0 pair_fwd [KH * KW, num_out], 1 mask [num_out] (uint32 bits),
        # 2 in_grid [batch, in_h, in_w], 3 out_indices [num_out, 3] (batch, y,
        # x), 4 num_out, 5 in_h, 6 in_w
        code.arg("args", "tv::array<int64_t, 7>")
        kh, kw = self.ksize
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        code.raw(f"""
        int* pair_fwd = reinterpret_cast<int*>(args[0]);
        uint32_t* mask = reinterpret_cast<uint32_t*>(args[1]);
        const int* in_grid = reinterpret_cast<const int*>(args[2]);
        const int* out_indices = reinterpret_cast<const int*>(args[3]);
        int num_out = int(args[4]);
        int in_h = int(args[5]), in_w = int(args[6]);
        for (int o : tv::KernelLoopX<int>(num_out)) {{
            const int* site = out_indices + int64_t(o) * 3;
            const int* grid = in_grid + int64_t(site[0]) * in_h * in_w;
            int oy = site[1] * {sh} - {ph};
            int ox = site[2] * {sw} - {pw};
            uint32_t bits = 0;
            TV_PRAGMA_UNROLL
            for (int ky = 0; ky < {kh}; ++ky){{
                int iy = oy + ky * {dh};
                bool inside = iy >= 0 && iy < in_h;
                TV_PRAGMA_UNROLL
                for (int kx = 0; kx < {kw}; ++kx){{
                    int ix = ox + kx * {dw};
                    int row = -1;
                    if (inside && ix >= 0 && ix < in_w){{
                        row = grid[int64_t(iy) * in_w + ix];
                    }}
                    pair_fwd[int64_t(ky * {kw} + kx) * num_out + o] = row;
                    if (row >= 0){{
                        bits |= 1u << (ky * {kw} + kx);
                    }}
                }}
            }}
            mask[o] = bits;
        }}
        """)
        return code
