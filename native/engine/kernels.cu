// Hand-written CUDA kernels for the Idol native inference engine.
// Compiled offline to PTX (compute_86) and loaded through the NVIDIA driver.
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define HD __half

__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ---------------------------------------------------------------- GEMM
// C[M,N] = act(A[M,K] @ B[K,N] + bias[N]); fp16 in/out, fp32 accumulate.
#define GM 128
#define GN 64
#define GK 16

extern "C" __global__ void k_gemm(const HD* __restrict__ A, int lda,
                                  const HD* __restrict__ B, int ldb,
                                  const HD* __restrict__ bias,
                                  HD* __restrict__ C, int ldc,
                                  int M, int N, int K, int act) {
    __shared__ HD As[2][GM][GK];
    __shared__ HD Bs[2][GK][GN];
    __shared__ float Cs[GM][GN];
    const int tiles_n = (N + GN - 1) / GN;
    const int tile = blockIdx.x;
    const int m0 = (tile / tiles_n) * GM;
    const int n0 = (tile % tiles_n) * GN;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int row0 = wm * 32, col0 = wn * 32;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int i = tid; i < GM * GK; i += 256) {
        int r = i / GK, c = i % GK;
        const bool in = (m0 + r < M) && (c < K);
        As[0][r][c] = in ? A[(size_t)(m0 + r) * lda + c] : __float2half(0.f);
    }
    for (int i = tid; i < GK * GN; i += 256) {
        int r = i / GN, c = i % GN;
        const bool in = (r < K) && (n0 + c < N);
        Bs[0][r][c] = in ? B[(size_t)r * ldb + n0 + c] : __float2half(0.f);
    }
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += GK) {
        const int nbuf = buf ^ 1;
        if (k0 + GK < K) {
            for (int i = tid; i < GM * GK; i += 256) {
                int r = i / GK, c = i % GK;
                const bool in = (m0 + r < M) && (k0 + GK + c < K);
                As[nbuf][r][c] = in ? A[(size_t)(m0 + r) * lda + k0 + GK + c] : __float2half(0.f);
            }
            for (int i = tid; i < GK * GN; i += 256) {
                int r = i / GN, c = i % GN;
                const bool in = (k0 + GK + r < K) && (n0 + c < N);
                Bs[nbuf][r][c] = in ? B[(size_t)(k0 + GK + r) * ldb + n0 + c] : __float2half(0.f);
            }
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a[2];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::row_major> b[2];
        wmma::load_matrix_sync(a[0], &As[buf][row0][0], GK);
        wmma::load_matrix_sync(a[1], &As[buf][row0 + 16][0], GK);
        wmma::load_matrix_sync(b[0], &Bs[buf][0][col0], GN);
        wmma::load_matrix_sync(b[1], &Bs[buf][0][col0 + 16], GN);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        __syncthreads();
        buf = nbuf;
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[row0 + i * 16][col0 + j * 16], acc[i][j], GN,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < GM * GN; i += 256) {
        int r = i / GN, c = i % GN;
        int gm = m0 + r, gn = n0 + c;
        if (gm < M && gn < N) {
            float v = Cs[r][c] + (bias ? __half2float(bias[gn]) : 0.0f);
            if (act == 1) v = v > 0.0f ? v : 0.0f;
            else if (act == 2) v = gelu_tanh(v);
            else if (act == 3) v = silu(v);
            else if (act == 4) v = gelu_exact(v);
            C[(size_t)gm * ldc + gn] = __float2half(v);
        }
    }
}


// ---------------------------------------------------------------- GEMM v2
// Same maths as k_gemm, but 32-deep K stages (half the barriers) and 16-byte vector
// tile loads instead of one scalar half per thread and iteration.
#define G2M 128
#define G2N 64
#define G2K 32
#define G2SMEM ((G2M * G2K * 2 + G2K * G2N * 2) * 2 > G2M * G2N * 4 ? \
                (G2M * G2K * 2 + G2K * G2N * 2) * 2 : G2M * G2N * 4)

__device__ __forceinline__ void g2_load_a(HD* dst, const HD* __restrict__ A, int lda, int m0,
                                          int k0, int M, int K, bool vec, int tid) {
    if (vec) {
        for (int i = tid; i < G2M * G2K / 8; i += 256) {
            const int r = i >> 2, p = i & 3;
            const int mrow = m0 + r, kcol = k0 + p * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (mrow < M) {
                if (kcol + 8 <= K) {
                    v = *(const uint4*)(A + (size_t)mrow * lda + kcol);
                } else {
                    HD* h = (HD*)&v;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        h[j] = (kcol + j < K) ? A[(size_t)mrow * lda + kcol + j] : __float2half(0.f);
                }
            }
            ((uint4*)dst)[i] = v;
        }
    } else {
        for (int i = tid; i < G2M * G2K; i += 256) {
            const int r = i / G2K, c = i % G2K;
            const bool in = (m0 + r < M) && (k0 + c < K);
            dst[i] = in ? A[(size_t)(m0 + r) * lda + k0 + c] : __float2half(0.f);
        }
    }
}

__device__ __forceinline__ void g2_load_b(HD* dst, const HD* __restrict__ B, int ldb, int n0,
                                          int k0, int N, int K, bool vec, int tid) {
    if (vec) {
        for (int i = tid; i < G2K * G2N / 8; i += 256) {
            const int r = i >> 3, p = i & 7;
            const int krow = k0 + r, ncol = n0 + p * 8;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (krow < K) {
                if (ncol + 8 <= N) {
                    v = *(const uint4*)(B + (size_t)krow * ldb + ncol);
                } else {
                    HD* h = (HD*)&v;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        h[j] = (ncol + j < N) ? B[(size_t)krow * ldb + ncol + j] : __float2half(0.f);
                }
            }
            ((uint4*)dst)[i] = v;
        }
    } else {
        for (int i = tid; i < G2K * G2N; i += 256) {
            const int r = i / G2N, c = i % G2N;
            const bool in = (k0 + r < K) && (n0 + c < N);
            dst[i] = in ? B[(size_t)(k0 + r) * ldb + n0 + c] : __float2half(0.f);
        }
    }
}

extern "C" __global__ void k_gemm2(const HD* __restrict__ A, int lda,
                                   const HD* __restrict__ B, int ldb,
                                   const HD* __restrict__ bias,
                                   HD* __restrict__ C, int ldc,
                                   int M, int N, int K, int act) {
    extern __shared__ char gsmem[];
    HD* As = (HD*)gsmem;
    HD* Bs = As + 2 * G2M * G2K;
    // The tile accumulators reuse the A/B staging window: those buffers are dead once the K loop
    // drains, and overlaying them drops the block footprint from 57 KiB to 32 KiB so several
    // blocks can share an SM and hide the per-stage barrier.
    float* Cs = (float*)gsmem;

    const int tiles_n = (N + G2N - 1) / G2N;
    const int tile = blockIdx.x;
    const int m0 = (tile / tiles_n) * G2M;
    const int n0 = (tile % tiles_n) * G2N;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int row0 = wm * 32, col0 = wn * 32;
    const bool vec = ((lda & 7) == 0) && ((ldb & 7) == 0);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    g2_load_a(As, A, lda, m0, 0, M, K, vec, tid);
    g2_load_b(Bs, B, ldb, n0, 0, N, K, vec, tid);
    __syncthreads();

    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += G2K) {
        const int nb = buf ^ 1;
        if (k0 + G2K < K) {
            g2_load_a(As + nb * G2M * G2K, A, lda, m0, k0 + G2K, M, K, vec, tid);
            g2_load_b(Bs + nb * G2K * G2N, B, ldb, n0, k0 + G2K, N, K, vec, tid);
        }
        const HD* asub = As + buf * G2M * G2K;
        const HD* bsub = Bs + buf * G2K * G2N;
        #pragma unroll
        for (int kk = 0; kk < G2K / 16; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::row_major> b[2];
            wmma::load_matrix_sync(a[0], asub + row0 * G2K + kk * 16, G2K);
            wmma::load_matrix_sync(a[1], asub + (row0 + 16) * G2K + kk * 16, G2K);
            wmma::load_matrix_sync(b[0], bsub + (kk * 16) * G2N + col0, G2N);
            wmma::load_matrix_sync(b[1], bsub + (kk * 16) * G2N + col0 + 16, G2N);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
        buf = nb;
    }

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(&Cs[(row0 + i * 16) * G2N + col0 + j * 16], acc[i][j], G2N,
                                    wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < G2M * G2N; i += 256) {
        const int r = i >> 6, c = i & 63;
        const int gm = m0 + r, gn = n0 + c;
        if (gm < M && gn < N) {
            float v = Cs[i] + (bias ? __half2float(bias[gn]) : 0.0f);
            if (act == 1) v = v > 0.0f ? v : 0.0f;
            else if (act == 2) v = gelu_tanh(v);
            else if (act == 3) v = silu(v);
            else if (act == 4) v = gelu_exact(v);
            C[(size_t)gm * ldc + gn] = __float2half(v);
        }
    }
}

// ------------------------------------------------------- flash attention (D=64)
// Q,K,V token-major [L,H,64]; O token-major [L,H,64]. Two-pass online softmax.
#define FM 64
#define FB 64
#define FD 64

extern "C" __global__ void k_flash(const HD* __restrict__ Q, const HD* __restrict__ K,
                                   const HD* __restrict__ V, HD* __restrict__ O,
                                   int Lq, int Lk, int H, int sq, int so, float scale) {
    __shared__ HD Qs[FM][FD];
    __shared__ HD Ks[FB][FD];
    __shared__ HD Vs[FB][FD];
    __shared__ float Ss[FM][FB];
    __shared__ HD Ps[FM][FB];
    __shared__ float mrow[FM], lrow[FM];

    const int h = blockIdx.y;
    const int q0 = blockIdx.x * FM;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = warp * 16;
    const size_t qbase = (size_t)h * FD;

    for (int i = tid; i < FM * FD; i += blockDim.x) {
        int r = i / FD, c = i % FD;
        Qs[r][c] = (q0 + r < Lq) ? Q[(size_t)(q0 + r) * sq + qbase + c] : __float2half(0.f);
    }
    for (int r = row0 + (lane >> 2); r < row0 + 16; r += 8) { mrow[r] = -1e30f; lrow[r] = 0.f; }
    __syncthreads();

    const int nkb = (Lk + FB - 1) / FB;

    // pass 1: row max and normaliser
    for (int kb = 0; kb < nkb; ++kb) {
        const int k0 = kb * FB;
        for (int i = tid; i < FB * FD; i += blockDim.x) {
            int r = i / FD, c = i % FD;
            Ks[r][c] = (k0 + r < Lk) ? K[(size_t)(k0 + r) * sq + qbase + c] : __float2half(0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[4];
        for (int n = 0; n < 4; ++n) wmma::fill_fragment(s[n], 0.0f);
        for (int d = 0; d < 4; ++d) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Qs[row0][d * 16], FD);
            for (int n = 0; n < 4; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Ks[n * 16][d * 16], FD);
                wmma::mma_sync(s[n], a, b, s[n]);
            }
        }
        for (int n = 0; n < 4; ++n) wmma::store_matrix_sync(&Ss[row0][n * 16], s[n], FB, wmma::mem_row_major);
        __syncthreads();
        for (int r = 0; r < 16; ++r) {
            int row = row0 + r;
            float mx = -1e30f;
            for (int c = lane; c < FB; c += 32) {
                float v = (k0 + c < Lk) ? Ss[row][c] * scale : -1e30f;
                mx = fmaxf(mx, v);
            }
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            float old = mrow[row];
            float mnew = fmaxf(old, mx);
            float sum = 0.f;
            for (int c = lane; c < FB; c += 32) {
                float v = (k0 + c < Lk) ? Ss[row][c] * scale : -1e30f;
                sum += __expf(v - mnew);
            }
            for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            lrow[row] = lrow[row] * __expf(old - mnew) + sum;
            mrow[row] = mnew;
        }
        __syncthreads();
    }

    // pass 2: probabilities and output accumulation
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    for (int n = 0; n < 4; ++n) wmma::fill_fragment(acc[n], 0.0f);
    for (int kb = 0; kb < nkb; ++kb) {
        const int k0 = kb * FB;
        for (int i = tid; i < FB * FD; i += blockDim.x) {
            int r = i / FD, c = i % FD;
            Ks[r][c] = (k0 + r < Lk) ? K[(size_t)(k0 + r) * sq + qbase + c] : __float2half(0.f);
            Vs[r][c] = (k0 + r < Lk) ? V[(size_t)(k0 + r) * sq + qbase + c] : __float2half(0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[4];
        for (int n = 0; n < 4; ++n) wmma::fill_fragment(s[n], 0.0f);
        for (int d = 0; d < 4; ++d) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Qs[row0][d * 16], FD);
            for (int n = 0; n < 4; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Ks[n * 16][d * 16], FD);
                wmma::mma_sync(s[n], a, b, s[n]);
            }
        }
        for (int n = 0; n < 4; ++n) wmma::store_matrix_sync(&Ss[row0][n * 16], s[n], FB, wmma::mem_row_major);
        __syncthreads();
        for (int r = 0; r < 16; ++r) {
            int row = row0 + r;
            float m = mrow[row], inv = 1.0f / lrow[row];
            for (int c = lane; c < FB; c += 32) {
                float v = (k0 + c < Lk) ? Ss[row][c] * scale : -1e30f;
                Ps[row][c] = __float2half(__expf(v - m) * inv);
            }
        }
        __syncthreads();
        for (int kk = 0; kk < 4; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Ps[row0][kk * 16], FB);
            for (int n = 0; n < 4; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::row_major> b;
                wmma::load_matrix_sync(b, &Vs[kk * 16][n * 16], FD);
                wmma::mma_sync(acc[n], a, b, acc[n]);
            }
        }
        __syncthreads();
    }
    for (int n = 0; n < 4; ++n)
        wmma::store_matrix_sync(&Ss[row0][n * 16], acc[n], FB, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < FM * FD; i += blockDim.x) {
        int r = i / FD, c = i % FD;
        if (q0 + r < Lq) O[(size_t)(q0 + r) * so + qbase + c] = __float2half(Ss[r][c]);
    }
}


// ------------------------------------------------------------- flash v2
// Single-pass online-softmax attention, D = 64, one head per block.  BM query rows
// per block, 64 keys per tile, vectorised 16-byte tile loads, fp16 in/out with
// fp32 accumulation.  The running softmax state is rescaled only when the row max
// actually grows, which in practice is true only for the first few tiles.
template <int BM, int RESCALE_MODE>
__device__ void flash_body(const HD* __restrict__ Q, const HD* __restrict__ K,
                           const HD* __restrict__ V, HD* __restrict__ O,
                           int Lq, int Lk, int H, int sq, int so, float scale,
                           char* smem_raw) {
    constexpr int BN = 64, D = 64;
    HD* Qs = (HD*)smem_raw;
    HD* Ks = Qs + BM * D;
    HD* Vs = Ks + BN * D;
    float* Ss = (float*)(Vs + BN * D);
    HD* Ps = (HD*)(Ss + BM * BN);
    float* mrow = (float*)(Ps + BM * BN);
    float* lrow = mrow + BM;
    float* corr = lrow + BM;

    const int h = blockIdx.y;
    const int q0 = blockIdx.x * BM;
    const int tid = threadIdx.x;
    const int nthreads = (int)blockDim.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = warp * 16;
    const size_t qbase = (size_t)h * D;

    for (int i = tid; i < BM * D / 8; i += nthreads) {
        const int r = i >> 3, p = i & 7;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (q0 + r < Lq) v = *(const uint4*)(Q + (size_t)(q0 + r) * sq + qbase + p * 8);
        ((uint4*)Qs)[i] = v;
    }
    for (int r = row0 + (lane >> 2); r < row0 + 16; r += 8) {
        mrow[r] = -1e30f;
        lrow[r] = 0.f;
        corr[r] = 1.f;
    }
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
    for (int n = 0; n < 4; ++n) wmma::fill_fragment(acc[n], 0.0f);
    __syncthreads();

    const int nkb = (Lk + BN - 1) / BN;
    for (int kb = 0; kb < nkb; ++kb) {
        const int k0 = kb * BN;
        for (int i = tid; i < BN * D / 8; i += nthreads) {
            const int r = i >> 3, p = i & 7;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (k0 + r < Lk) {
                kv = *(const uint4*)(K + (size_t)(k0 + r) * sq + qbase + p * 8);
                vv = *(const uint4*)(V + (size_t)(k0 + r) * sq + qbase + p * 8);
            }
            ((uint4*)Ks)[i] = kv;
            ((uint4*)Vs)[i] = vv;
        }
        __syncthreads();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[4];
        for (int n = 0; n < 4; ++n) wmma::fill_fragment(s[n], 0.0f);
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Qs[row0 * D + d * 16], D);
            for (int n = 0; n < 4; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Ks[n * 16 * D + d * 16], D);
                wmma::mma_sync(s[n], a, b, s[n]);
            }
        }
        for (int n = 0; n < 4; ++n)
            wmma::store_matrix_sync(&Ss[row0 * BN + n * 16], s[n], BN, wmma::mem_row_major);
        __syncwarp();

        bool rescale = false;
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int row = row0 + r;
            float mx = -1e30f;
            for (int c = lane; c < BN; c += 32) {
                const float v = (k0 + c < Lk) ? Ss[row * BN + c] * scale : -1e30f;
                mx = fmaxf(mx, v);
            }
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float mnew = fmaxf(mrow[row], mx);
            const float cr = __expf(mrow[row] - mnew);
            float sum = 0.f;
            for (int c = lane; c < BN; c += 32) {
                const float v = (k0 + c < Lk) ? Ss[row * BN + c] * scale : -1e30f;
                const float e = __expf(v - mnew);
                sum += e;
                Ps[row * BN + c] = __float2half(e);
            }
            for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
            lrow[row] = lrow[row] * cr + sum;
            mrow[row] = mnew;
            corr[row] = cr;
            rescale = rescale || cr < 0.9999f;
        }

        __syncwarp();

        // Only rows whose max moved need the (shared-memory round trip) rescale, and rows are
        // warp private, so this needs no block barrier.
        if (RESCALE_MODE == 1 && __any_sync(0xffffffffu, rescale)) {
            for (int n = 0; n < 4; ++n)
                wmma::store_matrix_sync(&Ss[row0 * BN + n * 16], acc[n], BN, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 16 * BN; i += 32)
                Ss[(row0 + i / BN) * BN + (i % BN)] *= corr[row0 + i / BN];
            __syncwarp();
            for (int n = 0; n < 4; ++n)
                wmma::load_matrix_sync(acc[n], &Ss[row0 * BN + n * 16], BN, wmma::mem_row_major);
        }

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Ps[row0 * BN + kk * 16], BN);
            for (int n = 0; n < 4; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::row_major> b;
                wmma::load_matrix_sync(b, &Vs[kk * 16 * D + n * 16], D);
                wmma::mma_sync(acc[n], a, b, acc[n]);
            }
        }
        __syncthreads();
    }

    for (int n = 0; n < 4; ++n)
        wmma::store_matrix_sync(&Ss[row0 * BN + n * 16], acc[n], BN, wmma::mem_row_major);
    __syncthreads();
    for (int i = tid; i < BM * D; i += nthreads) {
        const float inv = lrow[i / D] > 0.f ? 1.0f / lrow[i / D] : 0.f;
        Ss[i] *= inv;
    }
    __syncthreads();
    for (int i = tid; i < BM * D / 8; i += nthreads) {
        const int r = i >> 3, p = i & 7;
        if (q0 + r >= Lq) continue;
        const float* src = &Ss[i * 8];
        uint4 packed;
        HD* dst = (HD*)&packed;
        #pragma unroll
        for (int k = 0; k < 8; ++k) dst[k] = __float2half(src[k]);
        *(uint4*)(O + (size_t)(q0 + r) * so + qbase + p * 8) = packed;
    }
}

extern "C" __global__ void k_flash2(const HD* __restrict__ Q, const HD* __restrict__ K,
                                    const HD* __restrict__ V, HD* __restrict__ O,
                                    int Lq, int Lk, int H, int sq, int so, float scale) {
    extern __shared__ char smem_raw[];
    flash_body<128, 1>(Q, K, V, O, Lq, Lk, H, sq, so, scale, smem_raw);
}

extern "C" __global__ void k_flash3(const HD* __restrict__ Q, const HD* __restrict__ K,
                                    const HD* __restrict__ V, HD* __restrict__ O,
                                    int Lq, int Lk, int H, int sq, int so, float scale) {
    extern __shared__ char smem_raw[];
    flash_body<64, 1>(Q, K, V, O, Lq, Lk, H, sq, so, scale, smem_raw);
}

// Debug build with the running-max rescale compiled out (only valid for a single key tile).
extern "C" __global__ void k_flash4(const HD* __restrict__ Q, const HD* __restrict__ K,
                                    const HD* __restrict__ V, HD* __restrict__ O,
                                    int Lq, int Lk, int H, int sq, int so, float scale) {
    extern __shared__ char smem_raw[];
    flash_body<64, 0>(Q, K, V, O, Lq, Lk, H, sq, so, scale, smem_raw);
}

// ------------------------------------------- generic attention stats (any D, H)
// Writes per-query row max and normaliser with scaling applied.
extern "C" __global__ void k_attn_stats(const HD* __restrict__ Q, const HD* __restrict__ K,
                                        float* __restrict__ mrow, float* __restrict__ lrow,
                                        int Lq, int Lk, int H, int D, int sq, float scale) {
    extern __shared__ HD smem_a[];
    HD* Qs = smem_a;                 // FM * 16
    HD* Ks = Qs + FM * 16;         // FB * 16
    float* Ss = (float*)(Ks + FB * 16);  // FM * FB
    const int h = blockIdx.y;
    const int q0 = blockIdx.x * FM;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = warp * 16;
    const size_t base = (size_t)h * D;
    const int nkb = (Lk + FB - 1) / FB;
    for (int r = 0; r < 16; ++r) {
        int row = q0 + row0 + r;
        if (row < Lq && lane == 0) { mrow[row] = -1e30f; lrow[row] = 0.f; }
    }
    __syncthreads();
    for (int kb = 0; kb < nkb; ++kb) {
        const int k0 = kb * FB;
        for (int n = tid; n < FM * FB; n += blockDim.x) Ss[n] = 0.f;
        for (int d0 = 0; d0 < D; d0 += 16) {
            for (int i = tid; i < FM * 16; i += blockDim.x) {
                int r = i / 16, c = i % 16;
                Qs[r * 16 + c] = (q0 + r < Lq) ? Q[(size_t)(q0 + r) * sq + base + d0 + c] : __float2half(0.f);
            }
            __syncthreads();
            for (int i = tid; i < FB * 16; i += blockDim.x) {
                int r = i / 16, c = i % 16;
                Ks[r * 16 + c] = (k0 + r < Lk) ? K[(size_t)(k0 + r) * sq + base + d0 + c] : __float2half(0.f);
            }
            __syncthreads();
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[4];
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Qs[row0 * 16], 16);
            for (int n = 0; n < 4; ++n) wmma::load_matrix_sync(s[n], &Ss[row0 * FB + n * 16], FB, wmma::mem_row_major);
            for (int n = 0; n < 4; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Ks[n * 16 * 16], 16);
                wmma::mma_sync(s[n], a, b, s[n]);
            }
            for (int n = 0; n < 4; ++n) wmma::store_matrix_sync(&Ss[row0 * FB + n * 16], s[n], FB, wmma::mem_row_major);
            __syncthreads();
        }
        for (int r = 0; r < 16; ++r) {
            int row = q0 + row0 + r;
            if (row >= Lq) continue;
            float mx = -1e30f;
            for (int c = lane; c < FB; c += 32) {
                float v = (k0 + c < Lk) ? Ss[(row0 + r) * FB + c] * scale : -1e30f;
                mx = fmaxf(mx, v);
            }
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
            float old = mrow[row];
            float mnew = fmaxf(old, mx);
            float sum = 0.f;
            for (int c = lane; c < FB; c += 32) {
                float v = (k0 + c < Lk) ? Ss[(row0 + r) * FB + c] * scale : -1e30f;
                sum += __expf(v - mnew);
            }
            for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            if (lane == 0) { lrow[row] = lrow[row] * __expf(old - mnew) + sum; mrow[row] = mnew; }
        }
        __syncthreads();
    }
}

// Compute probabilities P[Lq,Lk] (f16) for a single head from precomputed m/l.
extern "C" __global__ void k_attn_probs(const HD* __restrict__ Q, const HD* __restrict__ K,
                                        const float* __restrict__ mrow, const float* __restrict__ lrow,
                                        HD* __restrict__ P, int Lq, int Lk, int D, int sq, float scale) {
    extern __shared__ HD smem_b[];
    HD* Qs = smem_b;                 // FM * 16
    HD* Ks = Qs + FM * 16;         // FB * 16
    float* Ss = (float*)(Ks + FB * 16);
    const int q0 = blockIdx.x * FM;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int row0 = warp * 16;
    const int nkb = (Lk + FB - 1) / FB;
    for (int kb = 0; kb < nkb; ++kb) {
        const int k0 = kb * FB;
        for (int n = tid; n < FM * FB; n += blockDim.x) Ss[n] = 0.f;
        for (int d0 = 0; d0 < D; d0 += 16) {
            for (int i = tid; i < FM * 16; i += blockDim.x) {
                int r = i / 16, c = i % 16;
                Qs[r * 16 + c] = (q0 + r < Lq) ? Q[(size_t)(q0 + r) * sq + d0 + c] : __float2half(0.f);
            }
            __syncthreads();
            for (int i = tid; i < FB * 16; i += blockDim.x) {
                int r = i / 16, c = i % 16;
                Ks[r * 16 + c] = (k0 + r < Lk) ? K[(size_t)(k0 + r) * sq + d0 + c] : __float2half(0.f);
            }
            __syncthreads();
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[4];
            wmma::fragment<wmma::matrix_a, 16, 16, 16, HD, wmma::row_major> a;
            wmma::load_matrix_sync(a, &Qs[row0 * 16], 16);
            for (int n = 0; n < 4; ++n) wmma::load_matrix_sync(s[n], &Ss[row0 * FB + n * 16], FB, wmma::mem_row_major);
            for (int n = 0; n < 4; ++n) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, HD, wmma::col_major> b;
                wmma::load_matrix_sync(b, &Ks[n * 16 * 16], 16);
                wmma::mma_sync(s[n], a, b, s[n]);
            }
            for (int n = 0; n < 4; ++n) wmma::store_matrix_sync(&Ss[row0 * FB + n * 16], s[n], FB, wmma::mem_row_major);
            __syncthreads();
        }
        for (int r = 0; r < 16; ++r) {
            int row = q0 + row0 + r;
            if (row >= Lq) continue;
            float m = mrow[row], inv = 1.0f / lrow[row];
            for (int c = 0; c < FB; ++c) {
                int key = k0 + c;
                if (key >= Lk) break;
                P[(size_t)row * Lk + key] = __float2half(__expf(Ss[(row0 + r) * FB + c] * scale - m) * inv);
            }
        }
        __syncthreads();
    }
}

// ------------------------------------------------------- Swin window attention
// One block per window; Q/K/V windows [nWin, WW, D] with bias[heads,WW,WW] (optional) and mask[WW,WW].
#define WW 144
#define WW_L 12
extern "C" __global__ void k_window_attn(const HD* __restrict__ Q, const HD* __restrict__ K,
                                         const HD* __restrict__ V, const HD* __restrict__ bias,
                                         const HD* __restrict__ mask, HD* __restrict__ O,
                                         int heads, int D, float scale) {
    extern __shared__ char smem_w[];
    float* Ss = (float*)smem_w;                       // WW*WW
    float* Os = Ss + WW * WW;                       // WW*D
    HD* Vs = (HD*)(Os + WW * D);                    // WW*D
    HD* Qw = Vs + WW * D;                           // WW*D
    HD* Kw = Qw + WW * D;                           // WW*D
    const int w = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const size_t wbase = (size_t)w * WW * D;
    for (int i = tid; i < WW * D; i += blockDim.x) {
        Qw[i] = Q[wbase + i];
        Kw[i] = K[wbase + i];
        Vs[i] = V[wbase + i];
    }
    for (int i = tid; i < WW * D; ++i) Os[i] = 0.f;
    __syncthreads();
    for (int i = tid; i < WW * WW; i += blockDim.x) Ss[i] = 0.f;
    __syncthreads();
    for (int i = tid; i < WW * WW; i += blockDim.x) {
        int r = i / WW, c = i % WW;
        const HD* q = Qw + (size_t)r * D;
        const HD* k = Kw + (size_t)c * D;
        float sum = 0.f;
        for (int d = 0; d < D; ++d) sum += __half2float(q[d]) * __half2float(k[d]);
        sum *= scale;
        if (bias) sum += __half2float(bias[((size_t)h * WW + r) * WW + c]);
        if (mask) sum += __half2float(mask[(size_t)r * WW + c]);
        Ss[i] = sum;
    }
    __syncthreads();
    // softmax per row
    const int warp = tid >> 5, lane = tid & 31;
    for (int r = warp; r < WW; r += blockDim.x / 32) {
        float mx = -1e30f;
        for (int c = lane; c < WW; c += 32) mx = fmaxf(mx, Ss[r * WW + c]);
        for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));
        float sum = 0.f;
        for (int c = lane; c < WW; c += 32) {
            float v = __expf(Ss[r * WW + c] - mx);
            Ss[r * WW + c] = v;
            sum += v;
        }
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        float inv = 1.0f / sum;
        for (int c = lane; c < WW; c += 32) Ss[r * WW + c] *= inv;
    }
    __syncthreads();
    for (int i = tid; i < WW * D; i += blockDim.x) {
        int r = i / D, d = i % D;
        float sum = 0.f;
        for (int c = 0; c < WW; ++c) sum += Ss[r * WW + c] * __half2float(Vs[(size_t)c * D + d]);
        Os[i] = sum;
    }
    __syncthreads();
    for (int i = tid; i < WW * D; i += blockDim.x) O[wbase + i] = __float2half(Os[i]);
}

// ------------------------------------------------------------ normalization
extern "C" __global__ void k_ln(const HD* __restrict__ x, int ldx, const HD* __restrict__ w,
                                const HD* __restrict__ b, HD* __restrict__ y, int ldy,
                                int rows, int cols, float eps, const HD* __restrict__ ms,
                                const HD* __restrict__ mt) {
    __shared__ float red[256];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const HD* src = x + (size_t)row * ldx;
    HD* dst = y + (size_t)row * ldy;
    float sum = 0.f;
    for (int c = tid; c < cols; c += blockDim.x) sum += __half2float(src[c]);
    red[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float mean = red[0] / (float)cols;
    __syncthreads();
    float var = 0.f;
    for (int c = tid; c < cols; c += blockDim.x) {
        float d = __half2float(src[c]) - mean;
        var += d * d;
    }
    red[tid] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float rstd = rsqrtf(red[0] / (float)cols + eps);
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = (__half2float(src[c]) - mean) * rstd;
        if (w) v *= __half2float(w[c]);
        if (b) v += __half2float(b[c]);
        if (ms) {
            v = v * (1.0f + __half2float(ms[c]));
            if (mt) v += __half2float(mt[c]);
        }
        dst[c] = __float2half(v);
    }
}

extern "C" __global__ void k_ln32(const float* __restrict__ x, const float* __restrict__ w,
                                  const float* __restrict__ b, float* __restrict__ y,
                                  int rows, int cols, float eps) {
    __shared__ float red[256];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const float* src = x + (size_t)row * cols;
    float* dst = y + (size_t)row * cols;
    float sum = 0.f;
    for (int c = tid; c < cols; c += blockDim.x) sum += src[c];
    red[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float mean = red[0] / (float)cols;
    __syncthreads();
    float var = 0.f;
    for (int c = tid; c < cols; c += blockDim.x) { float d = src[c] - mean; var += d * d; }
    red[tid] = var;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    float rstd = rsqrtf(red[0] / (float)cols + eps);
    for (int c = tid; c < cols; c += blockDim.x) {
        float v = (src[c] - mean) * rstd;
        if (w) v *= w[c];
        if (b) v += b[c];
        dst[c] = v;
    }
}

// Multi-head RMS normalisation used by the decoders (F.normalize * gamma * sqrt(dim)).
extern "C" __global__ void k_rms_heads(HD* __restrict__ x, const HD* __restrict__ gamma,
                                       int rows, int heads, int dim, float scale) {
    const int row = blockIdx.x;
    const int h = blockIdx.y * (blockDim.x / 32) + (threadIdx.x >> 5);
    if (row >= rows || h >= heads) return;
    const int lane = threadIdx.x & 31;
    HD* v = x + ((size_t)row * heads + h) * dim;
    float sum = 0.f;
    for (int d = lane; d < dim; d += 32) { float t = __half2float(v[d]); sum += t * t; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    float inv = rsqrtf(sum) * scale;
    for (int d = lane; d < dim; d += 32)
        v[d] = __float2half(__half2float(v[d]) * inv * __half2float(gamma[h * dim + d]));
}

// ---------------------------------------------------------------- activations
extern "C" __global__ void k_act(HD* __restrict__ x, int n, int kind) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(x[i]);
    if (kind == 0) v = gelu_tanh(v);
    else if (kind == 1) v = silu(v);
    else if (kind == 2) v = v > 0.f ? v : 0.f;
    else if (kind == 3) v = gelu_exact(v);
    else if (kind == 4) v = 1.f / (1.f + __expf(-v));
    else if (kind == 5) v = tanhf(v);
    x[i] = __float2half(v);
}

extern "C" __global__ void k_mul_silu(const HD* __restrict__ g, const HD* __restrict__ u,
                                      HD* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(silu(__half2float(g[i])) * __half2float(u[i]));
}

extern "C" __global__ void k_add(HD* __restrict__ a, const HD* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

extern "C" __global__ void k_add_scaled(HD* __restrict__ x, const HD* __restrict__ h,
                                        const HD* __restrict__ gate, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int c = i % cols;
    x[i] = __float2half(__half2float(x[i]) + __half2float(h[i]) * __half2float(gate[c]));
}

extern "C" __global__ void k_mul_cols(HD* __restrict__ x, const HD* __restrict__ w, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    x[i] = __float2half(__half2float(x[i]) * __half2float(w[i % cols]));
}

extern "C" __global__ void k_sigmoid_mul(HD* __restrict__ x, const HD* __restrict__ gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(__half2float(x[i]) / (1.f + __expf(-__half2float(gate[i]))));
}

extern "C" __global__ void k_scale(HD* __restrict__ x, int n, float s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = __float2half(__half2float(x[i]) * s);
}

extern "C" __global__ void k_copy_h2h(HD* __restrict__ dst, const HD* __restrict__ src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

extern "C" __global__ void k_f32_to_f16(const float* __restrict__ src, HD* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

extern "C" __global__ void k_f16_to_f32(const HD* __restrict__ src, float* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __half2float(src[i]);
}

// --------------------------------------------------------------- composition
extern "C" __global__ void k_cat_channels(const HD* __restrict__ a, const HD* __restrict__ b,
                                          HD* __restrict__ out, int rows, int ca, int cb, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * (ca + cb);
    if (i >= total) return;
    int r = i / (ca + cb), c = i % (ca + cb);
    out[(size_t)r * stride + c] = (c < ca) ? a[(size_t)r * stride + c] : b[(size_t)r * stride + c - ca];
}

// Swin patch merging channel packing: (h even/odd, w even/odd) -> 4C.
extern "C" __global__ void k_patch_merge_concat(const HD* __restrict__ x, HD* __restrict__ out,
                                                int H, int W, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int H2 = H / 2, W2 = W / 2;
    int total = H2 * W2 * C * 4;
    if (i >= total) return;
    int c4 = i % (C * 4), p = i / (C * 4);
    int y = p / W2, xx = p % W2;
    int part = c4 / C, c = c4 % C;
    int sy = y * 2 + (part & 1);
    int sx = xx * 2 + (part >> 1);
    out[i] = x[((size_t)sy * W + sx) * C + c];
}

// einops b c (hg h) (wg w) -> b (c hg wg) h w
extern "C" __global__ void k_image2patches(const HD* __restrict__ x, HD* __restrict__ out,
                                           int Hf, int Wf, int Hr, int Wr, int C) {
    int hg = Hf / Hr, wg = Wf / Wr;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = Hr * Wr * C * hg * wg;
    if (i >= total) return;
    int ct = C * hg * wg, p = i / ct, cc = i % ct;
    int h = p / Wr, w = p % Wr;
    int c = cc / (hg * wg), rem = cc % (hg * wg);
    int hy = rem / wg, wx = rem % wg;
    out[i] = x[((size_t)(hy * Hr + h) * Wf + (wx * Wr + w)) * C + c];
}

extern "C" __global__ void k_gap(const HD* __restrict__ x, HD* __restrict__ out, int H, int W, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sum = 0.f;
    for (int p = 0; p < H * W; ++p) sum += __half2float(x[(size_t)p * C + c]);
    out[c] = __float2half(sum / (float)(H * W));
}

// --------------------------------------------------------------- normalisers
extern "C" __global__ void k_groupnorm_stats(const HD* __restrict__ x, float* __restrict__ mean,
                                             float* __restrict__ rstd, int H, int W, int C, int groups,
                                             float eps) {
    extern __shared__ float red_g[];
    const int g = blockIdx.x;
    const int cg = C / groups;
    const int tid = threadIdx.x;
    const int nth = (int)blockDim.x;
    const size_t pixels = (size_t)H * W;
    const HD* group = x + (size_t)g * cg;
    float s = 0.f, ss = 0.f;
    // Every thread walks its own share of the pixels and sweeps all channels of the group, so the
    // whole block stays busy even when a group only owns a handful of channels.
    if ((cg & 1) == 0) {
        const int nv = cg >> 1;
        for (size_t p = tid; p < pixels; p += nth) {
            const __half2* row = (const __half2*)(group + p * C);
            for (int c = 0; c < nv; ++c) {
                const float2 f = __half22float2(row[c]);
                s += f.x + f.y;
                ss += f.x * f.x + f.y * f.y;
            }
        }
    } else {
        for (size_t p = tid; p < pixels; p += nth) {
            const HD* row = group + p * C;
            for (int c = 0; c < cg; ++c) {
                const float v = __half2float(row[c]);
                s += v;
                ss += v * v;
            }
        }
    }
    red_g[tid] = s;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) { if (tid < st) red_g[tid] += red_g[tid + st]; __syncthreads(); }
    float total = red_g[0];
    __syncthreads();
    red_g[tid] = ss;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) { if (tid < st) red_g[tid] += red_g[tid + st]; __syncthreads(); }
    float sq = red_g[0];
    if (tid == 0) {
        float n = (float)(pixels * cg);
        float m = total / n;
        mean[g] = m;
        rstd[g] = rsqrtf(sq / n - m * m + eps);
    }
}

extern "C" __global__ void k_groupnorm_apply(const HD* __restrict__ x, const float* __restrict__ mean,
                                             const float* __restrict__ rstd, const HD* __restrict__ gamma,
                                             const HD* __restrict__ beta, HD* __restrict__ y,
                                             int HW, int C, int groups) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW * C) return;
    int c = i % C;
    int g = c / (C / groups);
    float v = (__half2float(x[i]) - mean[g]) * rstd[g];
    if (gamma) v *= __half2float(gamma[c]);
    if (beta) v += __half2float(beta[c]);
    y[i] = __float2half(v);
}

extern "C" __global__ void k_batchnorm(const HD* __restrict__ x, const HD* __restrict__ gamma,
                                       const HD* __restrict__ beta, const float* __restrict__ mean,
                                       const float* __restrict__ var, HD* __restrict__ y,
                                       int HW, int C, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HW * C) return;
    int c = i % C;
    float v = (__half2float(x[i]) - mean[c]) * rsqrtf(var[c] + eps);
    if (gamma) v *= __half2float(gamma[c]);
    if (beta) v += __half2float(beta[c]);
    y[i] = __float2half(v);
}

// --------------------------------------------------------------- resampling
extern "C" __global__ void k_resize(const HD* __restrict__ x, HD* __restrict__ y,
                                    int Hin, int Win, int Hout, int Wout, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Hout * Wout * C) return;
    int c = i % C, p = i / C, oy = p / Wout, ox = p % Wout;
    float sy = Hout > 1 ? oy * (float)(Hin - 1) / (float)(Hout - 1) : 0.f;
    float sx = Wout > 1 ? ox * (float)(Win - 1) / (float)(Wout - 1) : 0.f;
    int y0 = (int)sy, x0 = (int)sx;
    if (y0 > Hin - 1) y0 = Hin - 1;
    if (x0 > Win - 1) x0 = Win - 1;
    int y1 = y0 + 1 < Hin ? y0 + 1 : y0;
    int x1 = x0 + 1 < Win ? x0 + 1 : x0;
    float fy = sy - y0, fx = sx - x0;
    float v00 = __half2float(x[((size_t)y0 * Win + x0) * C + c]);
    float v01 = __half2float(x[((size_t)y0 * Win + x1) * C + c]);
    float v10 = __half2float(x[((size_t)y1 * Win + x0) * C + c]);
    float v11 = __half2float(x[((size_t)y1 * Win + x1) * C + c]);
    float v = (v00 * (1 - fx) + v01 * fx) * (1 - fy) + (v10 * (1 - fx) + v11 * fx) * fy;
    y[i] = __float2half(v);
}

// --------------------------------------------------------------- convolution
extern "C" __global__ void k_im2col(const HD* __restrict__ x, HD* __restrict__ A, int H, int W, int C,
                                    int KH, int KW, int stride, int pad, int Hout, int Wout) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int K = C * KH * KW;
    if (i >= Hout * Wout * K) return;
    int k = i % K, p = i / K;
    int c = k / (KH * KW), rem = k % (KH * KW);
    int ky = rem / KW, kx = rem % KW;
    int oy = p / Wout, ox = p % Wout;
    int iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
    A[i] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[((size_t)iy * W + ix) * C + c] : __float2half(0.f);
}

extern "C" __global__ void k_deform_im2col(const HD* __restrict__ x, const HD* __restrict__ offset,
                                           const HD* __restrict__ mask, HD* __restrict__ A,
                                           int H, int W, int C, int KH, int KW, int stride, int pad,
                                           int Hout, int Wout) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int T = KH * KW, K = C * T;
    if (i >= Hout * Wout * K) return;
    int k = i % K, p = i / K;
    int c = k / T, j = k % T;
    int ky = j / KW, kx = j % KW;
    int oy = p / Wout, ox = p % Wout;
    float dy = __half2float(offset[(size_t)p * (2 * T) + j]);
    float dx = __half2float(offset[(size_t)p * (2 * T) + T + j]);
    float m = __half2float(mask[(size_t)p * T + j]);
    float sy = oy * stride - pad + ky + dy;
    float sx = ox * stride - pad + kx + dx;
    if (sy <= -1.f || sx <= -1.f || sy >= (float)H || sx >= (float)W) { A[i] = __float2half(0.f); return; }
    int y0 = (int)floorf(sy), x0 = (int)floorf(sx);
    float fy = sy - y0, fx = sx - x0;
    float v = 0.f;
    for (int dyi = 0; dyi < 2; ++dyi) {
        for (int dxi = 0; dxi < 2; ++dxi) {
            int yy = y0 + dyi, xx = x0 + dxi;
            if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
            float wgt = (dyi ? fy : 1 - fy) * (dxi ? fx : 1 - fx);
            v += wgt * __half2float(x[((size_t)yy * W + xx) * C + c]);
        }
    }
    A[i] = __float2half(v * m);
}

// --------------------------------------------------------------- Swin windowing
extern "C" __global__ void k_swin_partition(const HD* __restrict__ x, HD* __restrict__ out,
                                            int Hp, int Wp, int C, int shift, int nwin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nwin * WW * C) return;
    int c = i % C, t = i / C;
    int j = t % WW, w = t / WW;
    int iy = j / WW_L, ix = j % WW_L;
    int wy = w / (Wp / WW_L), wx = w % (Wp / WW_L);
    int sy = (wy * WW_L + iy + shift) % Hp;
    int sx = (wx * WW_L + ix + shift) % Wp;
    out[i] = x[((size_t)sy * Wp + sx) * C + c];
}

extern "C" __global__ void k_swin_reverse(const HD* __restrict__ win, HD* __restrict__ out,
                                          int H, int W, int Hp, int Wp, int C, int shift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * W * C) return;
    int c = i % C, p = i / C, y = p / W, x = p % W;
    int sy = (y - shift + Hp) % Hp;
    int sx = (x - shift + Wp) % Wp;
    int wy = sy / WW_L, wx = sx / WW_L;
    int widx = wy * (Wp / WW_L) + wx;
    out[i] = win[(size_t)(widx * WW + (sy % WW_L) * WW_L + (sx % WW_L)) * C + c];
}

// --------------------------------------------------------------- RoPE / embeddings
extern "C" __global__ void k_repo_angles(const float* __restrict__ delta, const float* __restrict__ f0,
                                         const float* __restrict__ f1, const float* __restrict__ f2,
                                         float* __restrict__ cosv, float* __restrict__ sinv,
                                         int rows, int heads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads) return;
    int r = i / heads, h = i % heads;
    const float* d = delta + ((size_t)r * heads + h) * 3;
    float* cv = cosv + ((size_t)r * heads + h) * 32;
    float* sv = sinv + ((size_t)r * heads + h) * 32;
    for (int k = 0; k < 10; ++k) { float a = d[0] * f0[k] * 3.14159265358979f; cv[k] = cosf(a); sv[k] = sinf(a); }
    for (int k = 0; k < 10; ++k) { float a = d[1] * f1[k] * 3.14159265358979f; cv[10 + k] = cosf(a); sv[10 + k] = sinf(a); }
    for (int k = 0; k < 12; ++k) { float a = d[2] * f2[k] * 3.14159265358979f; cv[20 + k] = cosf(a); sv[20 + k] = sinf(a); }
}

extern "C" __global__ void k_rope_pairs(HD* __restrict__ x, const float* __restrict__ cosv,
                                        const float* __restrict__ sinv, int rows, int heads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * heads * 32) return;
    int t = i % 32, hh = i / 32;
    HD* v = x + (size_t)hh * 64 + t * 2;
    float c = cosv[(size_t)hh * 32 + t], s = sinv[(size_t)hh * 32 + t];
    float a = __half2float(v[0]), b = __half2float(v[1]);
    v[0] = __float2half(a * c - b * s);
    v[1] = __float2half(a * s + b * c);
}

extern "C" __global__ void k_rope_dino(HD* __restrict__ q, const float* __restrict__ cosv,
                                       const float* __restrict__ sinv, int L, int H, int prefix) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = L - prefix;
    if (i >= n * H * 32) return;
    int d = i % 32, hh = i / 32;
    int row = hh / H + prefix, h = hh % H;
    HD* v = q + ((size_t)row * H + h) * 64;
    int pr = row - prefix;
    float c = cosv[(size_t)pr * 64 + d];
    float s = sinv[(size_t)pr * 64 + d];
    float a = __half2float(v[d]), b = __half2float(v[d + 32]);
    v[d] = __float2half(a * c - b * s);
    v[d + 32] = __float2half(b * c + a * s);
}

extern "C" __global__ void k_pos_embed(const float* __restrict__ xyz, const float* __restrict__ freqs,
                                       float* __restrict__ out, int n, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * channels) return;
    int c = i % channels, p = i / channels;
    if (c >= 1020) { out[i] = 0.f; return; }
    int coord = c / 340, rem = c % 340;
    float v = xyz[(size_t)p * 3 + coord];
    float ang = v * freqs[rem % 170] * 3.14159265358979f;
    out[i] = rem < 170 ? sinf(ang) : cosf(ang);
}




// --------------------------------------------------------------- copies/layout
extern "C" __global__ void k_copy_rows(HD* __restrict__ dst, int dstride, const HD* __restrict__ src,
                                       int sstride, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    dst[(size_t)r * dstride + c] = src[(size_t)r * sstride + c];
}

extern "C" __global__ void k_copy_cols(HD* __restrict__ dst, int dstride, int dcol,
                                       const HD* __restrict__ src, int sstride, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    dst[(size_t)r * dstride + dcol + c] = src[(size_t)r * sstride + c];
}

extern "C" __global__ void k_add_cols_f32(HD* __restrict__ x, const float* __restrict__ w, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    x[i] = __float2half(__half2float(x[i]) + w[i % cols]);
}

extern "C" __global__ void k_scale_shift(HD* __restrict__ x, const HD* __restrict__ scale,
                                         const HD* __restrict__ shift, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int c = i % cols;
    x[i] = __float2half(__half2float(x[i]) * (1.0f + __half2float(scale[c])) + __half2float(shift[c]));
}

extern "C" __global__ void k_image_nhwc(const float* __restrict__ planar, HD* __restrict__ out,
                                        int H, int W, float s0, float s1, float s2,
                                        float b0, float b1, float b2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * W * 3) return;
    int c = i % 3, p = i / 3;
    float s = c == 0 ? s0 : (c == 1 ? s1 : s2);
    float b = c == 0 ? b0 : (c == 1 ? b1 : b2);
    out[p * 3 + c] = __float2half(planar[(size_t)c * H * W + p] * s + b);
}

// Sample the VAE posterior and pack 2x2 latent blocks into 128 channels.
extern "C" __global__ void k_pack_latent(const HD* __restrict__ mean, const HD* __restrict__ logvar,
                                         const float* __restrict__ eps, HD* __restrict__ out,
                                         int H, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (H / 2) * (W / 2) * 128) return;
    int t = i % 128, p = i / 128;
    int y = p / (W / 2), x = p % (W / 2);
    int c = t / 4, k = t % 4;
    int dy = k >> 1, dx = k & 1;
    int sy = y * 2 + dy, sx = x * 2 + dx;
    int idx = (sy * W + sx) * 64 + c;
    // `logvar` already points at channel 32 of the same row, so the channel offset must not be
    // applied twice (that read the next pixel and blew the sampled latent up ~18x).
    float lv = __half2float(logvar[idx]);
    lv = fminf(fmaxf(lv, -30.f), 20.f);
    float z = __half2float(mean[idx]) + eps[(size_t)p * 128 + t] * __expf(0.5f * lv);
    out[i] = __float2half(z);
}

// --------------------------------------------------------------- misc extras
extern "C" __global__ void k_zero(HD* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2half(0.f);
}

// C[M,N] = A[M,K] @ B[N,K]^T in fp32 (used by the VAE mid attention).
#define GT 32
extern "C" __global__ void k_gemm_nt(const HD* __restrict__ A, int lda,
                                     const HD* __restrict__ B, int ldb,
                                     float* __restrict__ C, int ldc,
                                     int M, int N, int K) {
    __shared__ HD As[GT][16];
    __shared__ HD Bs[GT][16];
    const int m0 = blockIdx.y * GT, n0 = blockIdx.x * GT;
    const int tid = threadIdx.x;
    const int tm = tid >> 4, tn = tid & 15;
    float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = tid; i < GT * 16; i += 256) {
            int r = i >> 4, c = i & 15;
            As[r][c] = A[(size_t)(m0 + r) * lda + k0 + c];
            Bs[r][c] = B[(size_t)(n0 + r) * ldb + k0 + c];
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a0 = __half2float(As[tm * 2][kk]), a1 = __half2float(As[tm * 2 + 1][kk]);
            float b0 = __half2float(Bs[tn * 2][kk]), b1 = __half2float(Bs[tn * 2 + 1][kk]);
            acc[0][0] += a0 * b0; acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0; acc[1][1] += a1 * b1;
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            int m = m0 + tm * 2 + i, n = n0 + tn * 2 + j;
            if (m < M && n < N) C[(size_t)m * ldc + n] = acc[i][j];
        }
}

// Row-wise softmax over a float matrix, in place.
extern "C" __global__ void k_softmax_rows(float* __restrict__ x, int rows, int cols) {
    __shared__ float red[256];
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int tid = threadIdx.x;
    float* row = x + (size_t)r * cols;
    float mx = -1e30f;
    for (int c = tid; c < cols; c += 256) mx = fmaxf(mx, row[c]);
    red[tid] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]); __syncthreads(); }
    mx = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int c = tid; c < cols; c += 256) { float v = __expf(row[c] - mx); row[c] = v; sum += v; }
    red[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (tid < s) red[tid] += red[tid + s]; __syncthreads(); }
    const float inv = 1.0f / red[0];
    for (int c = tid; c < cols; c += 256) row[c] *= inv;
}

extern "C" __global__ void k_mul(HD* __restrict__ a, const HD* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = __float2half(__half2float(a[i]) * __half2float(b[i]));
}

extern "C" __global__ void k_f32_to_f16_rows(const float* __restrict__ src, int sstride,
                                             HD* __restrict__ dst, int dstride, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    dst[(size_t)r * dstride + c] = __float2half(src[(size_t)r * sstride + c]);
}

extern "C" __global__ void k_f16_to_f32_rows(const HD* __restrict__ src, int sstride,
                                             float* __restrict__ dst, int dstride, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    dst[(size_t)r * dstride + c] = __half2float(src[(size_t)r * sstride + c]);
}

// im2col with an explicit destination row stride and a chunk offset (bounded memory).
extern "C" __global__ void k_im2col_s(const HD* __restrict__ x, HD* __restrict__ A, int lda,
                                      int H, int W, int C, int KH, int KW, int stride, int pad,
                                      int Hout, int Wout, int p0, int n) {
    int K = C * KH * KW;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * K) return;
    int k = i % K, pl = i / K;
    int p = pl + p0;
    int c = k / (KH * KW), rem = k % (KH * KW);
    int ky = rem / KW, kx = rem % KW;
    int oy = p / Wout, ox = p % Wout;
    int iy = oy * stride - pad + ky, ix = ox * stride - pad + kx;
    A[(size_t)pl * lda + k] =
        (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[((size_t)iy * W + ix) * C + c] : __float2half(0.f);
}

// Euler update of classifier-free guidance: x -= ((1 + w) * vp - w * vn) * dt
extern "C" __global__ void k_cfg_update(HD* __restrict__ x, const HD* __restrict__ vp,
                                        const HD* __restrict__ vn, float w, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = (1.0f + w) * __half2float(vp[i]) - w * __half2float(vn[i]);
    x[i] = __float2half(__half2float(x[i]) - v * dt);
}

// x = x + h * gate (row-wise) for an arbitrary row stride.
extern "C" __global__ void k_add_gate(HD* __restrict__ x, const HD* __restrict__ h,
                                      const HD* __restrict__ gate, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int c = i % cols;
    x[i] = __float2half(__half2float(x[i]) + __half2float(h[i]) * __half2float(gate[c]));
}

// Euler step of classifier-free guidance on the leading `cols` of a row-strided matrix.
extern "C" __global__ void k_cfg_update_strided(HD* __restrict__ x, const HD* __restrict__ vp,
                                                const HD* __restrict__ vn, int n, int stride,
                                                float w, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = i / 16, c = i % 16;
    size_t idx = (size_t)r * stride + c;
    float v = (1.0f + w) * __half2float(vp[idx]) - w * __half2float(vn[idx]);
    x[idx] = __float2half(__half2float(x[idx]) - v * dt);
}

// BM = 128 variant: eight warps per block, so the same K/V tile feeds twice the query rows
// and the block count halves.  Same maths as k_flash3.
extern "C" __global__ void k_flash5(const HD* __restrict__ Q, const HD* __restrict__ K,
                                    const HD* __restrict__ V, HD* __restrict__ O,
                                    int Lq, int Lk, int H, int sq, int so, float scale) {
    extern __shared__ char smem_raw[];
    flash_body<128, 1>(Q, K, V, O, Lq, Lk, H, sq, so, scale, smem_raw);
}

// ---------------------------------------------------------------------- BiRefNet
// Swin-L keeps 12x12 windows and head_dim 32 in every stage, so one block owns one
// (window, head) pair: K/V tiles are staged in shared memory and each warp sweeps a strip of
// query rows.  Q/K/V arrive token-major as one qkv matrix [H*W, 3C] and the results are written
// straight back into token order, so no window tensor ever has to be materialised.
#define BA_L 12
#define BA_WW 144
#define BA_D 32

extern "C" __global__ void k_swin_attn(const HD* __restrict__ qkv, HD* __restrict__ O,
                                       const HD* __restrict__ bias,
                                       int H, int W, int C, int heads, int Hp, int Wp,
                                       int shift, int use_mask, float scale) {
    __shared__ HD Ks[BA_WW * BA_D];
    __shared__ HD Vs[BA_WW * BA_D];
    __shared__ float Ps[4][BA_WW];
    __shared__ int Rid[BA_WW];
    const int nwn = Wp / BA_L;
    const int w = blockIdx.x, h = blockIdx.y;
    const int wy = w / nwn, wx = w % nwn;
    const int hlim1 = Hp - BA_L, hlim2 = Hp - shift;
    const int wlim1 = Wp - BA_L, wlim2 = Wp - shift;
    const int tid = threadIdx.x;
    const int qi = h * BA_D;
    for (int i = tid; i < BA_WW * BA_D; i += blockDim.x) {
        const int c = i / BA_D, d = i % BA_D;
        const int sy = (wy * BA_L + c / BA_L + shift) % Hp;
        const int sx = (wx * BA_L + c % BA_L + shift) % Wp;
        if (sy < H && sx < W) {
            const HD* row = qkv + (size_t)(sy * W + sx) * 3 * C;
            Ks[i] = row[C + qi + d];
            Vs[i] = row[2 * C + qi + d];
        } else {
            Ks[i] = __float2half(0.f);
            Vs[i] = __float2half(0.f);
        }
    }
    if (use_mask && tid < BA_WW) {
        // The window mask is indexed by slot position, not by the wrapped coordinate: PyTorch builds
        // it from the padded image mask before the cyclic shift, so the region ids must not be
        // rotated with the content.
        const int ay = wy * BA_L + tid / BA_L;
        const int ax = wx * BA_L + tid % BA_L;
        const int hh = ay < hlim1 ? 0 : (ay < hlim2 ? 1 : 2);
        const int ww = ax < wlim1 ? 0 : (ax < wlim2 ? 1 : 2);
        Rid[tid] = hh * 3 + ww;
    }
    __syncthreads();
    const int warp = tid >> 5, lane = tid & 31;
    const int nwarps = (int)blockDim.x >> 5;
    for (int r = warp; r < BA_WW; r += nwarps) {
        const int qy = (wy * BA_L + r / BA_L + shift) % Hp;
        const int qx = (wx * BA_L + r % BA_L + shift) % Wp;
        const int rreg = use_mask ? Rid[r] : 0;
        float q[BA_D];
        if (qy < H && qx < W) {
            const HD* qrow = qkv + (size_t)(qy * W + qx) * 3 * C + qi;
            for (int d = 0; d < BA_D; ++d) q[d] = __half2float(qrow[d]);
        } else {
            for (int d = 0; d < BA_D; ++d) q[d] = 0.f;
        }
        float sc[5];
        float mx = -1e30f;
        #pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int c = lane + t * 32;
            float s = -1e30f;
            if (c < BA_WW) {
                const HD* krow = Ks + (size_t)c * BA_D;
                s = 0.f;
                for (int d = 0; d < BA_D; ++d) s += q[d] * __half2float(krow[d]);
                s *= scale;
                if (bias) s += __half2float(bias[((size_t)h * BA_WW + r) * BA_WW + c]);
                if (use_mask && Rid[c] != rreg) s -= 100.f;
            }
            sc[t] = s;
            mx = fmaxf(mx, s);
        }
        for (int off = 16; off; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
        float sum = 0.f;
        #pragma unroll
        for (int t = 0; t < 5; ++t) {
            const int c = lane + t * 32;
            if (c < BA_WW) {
                const float e = __expf(sc[t] - mx);
                Ps[warp][c] = e;
                sum += e;
            }
        }
        for (int off = 16; off; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        const float inv = 1.f / sum;
        float acc = 0.f;
        for (int c = 0; c < BA_WW; ++c)
            acc += Ps[warp][c] * __half2float(Vs[(size_t)c * BA_D + lane]);
        if (qy < H && qx < W)
            O[(size_t)(qy * W + qx) * C + qi + lane] = __float2half(acc * inv);
        __syncwarp();
    }
}

// relative_position_bias_table [529, heads] -> dense per-head [heads][L*L][L*L] bias blocks.
extern "C" __global__ void k_rel_bias(const HD* __restrict__ table, int tstride, int heads,
                                      HD* __restrict__ out, int L) {
    const int L2 = L * L, W2 = 2 * L - 1;
    const long long total = (long long)heads * L2 * L2;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = (int)(i % L2), t = (int)(i / L2);
    const int r = t % L2, h = t / L2;
    const int idx = (r / L - c / L + L - 1) * W2 + (r % L - c % L + L - 1);
    out[i] = table[(size_t)idx * tstride + h];
}

// out = [a | b] along the channel axis of an NHWC row block; the padding tail is zeroed.
extern "C" __global__ void k_cat_ch(const HD* __restrict__ a, int sa, const HD* __restrict__ b, int sb,
                                    HD* __restrict__ out, int so, int rows, int ca, int cb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * so) return;
    const int r = i / so, c = i % so;
    HD v = __float2half(0.f);
    if (c < ca) v = a[(size_t)r * sa + c];
    else if (c < ca + cb) v = b[(size_t)r * sb + c - ca];
    out[i] = v;
}

// deformable im2col with explicit source row strides (the offset/modulator maps are padded).
extern "C" __global__ void k_deform_im2col_s(const HD* __restrict__ x, const HD* __restrict__ offset,
                                             int ostep, const HD* __restrict__ mask, int mstep,
                                             HD* __restrict__ A, int lda, int H, int W, int C,
                                             int KH, int KW, int stride, int pad, int Hout, int Wout,
                                             int p0, int n) {
    const int T = KH * KW, K = C * T;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * K) return;
    const int k = i % K, pl = i / K, p = pl + p0;
    const int c = k / T, j = k % T;
    const int ky = j / KW, kx = j % KW;
    const int oy = p / Wout, ox = p % Wout;
    // torchvision's deform_conv2d stores the pair interleaved per tap: (dy, dx, dy, dx, ...).
    const float dy = __half2float(offset[(size_t)p * ostep + 2 * j]);
    const float dx = __half2float(offset[(size_t)p * ostep + 2 * j + 1]);
    const float m = __half2float(mask[(size_t)p * mstep + j]);
    const float sy = (float)(oy * stride - pad + ky) + dy;
    const float sx = (float)(ox * stride - pad + kx) + dx;
    float v = 0.f;
    if (sy > -1.f && sx > -1.f && sy < (float)H && sx < (float)W) {
        const int y0 = (int)floorf(sy), x0 = (int)floorf(sx);
        const float fy = sy - y0, fx = sx - x0;
        for (int dyi = 0; dyi < 2; ++dyi) {
            for (int dxi = 0; dxi < 2; ++dxi) {
                const int yy = y0 + dyi, xx = x0 + dxi;
                if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
                const float wgt = (dyi ? fy : 1.f - fy) * (dxi ? fx : 1.f - fx);
                v += wgt * __half2float(x[((size_t)yy * W + xx) * C + c]);
            }
        }
        v *= m;
    }
    A[(size_t)pl * lda + k] = __float2half(v);
}

extern "C" __global__ void k_bcast_row(const HD* __restrict__ src, HD* __restrict__ dst, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    dst[i] = src[i % cols];
}

extern "C" __global__ void k_gate_mul(HD* __restrict__ x, const HD* __restrict__ gate, int gs,
                                      int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols;
    x[i] = __float2half(__half2float(x[i]) * (1.f / (1.f + __expf(-__half2float(gate[(size_t)r * gs])))));
}

// sigmoid of a strided [rows, cols] block into a dense [rows, cols] one.
extern "C" __global__ void k_sigmoid_strided(const HD* __restrict__ x, int sx, HD* __restrict__ y,
                                             int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols, c = i % cols;
    const float v = __half2float(x[(size_t)r * sx + c]);
    y[i] = __float2half(1.f / (1.f + __expf(-v)));
}

// RGBA float32 (4 channels) -> normalised, bilinearly resampled NHWC fp16 RGB (align_corners).
extern "C" __global__ void k_img_prep(const float* __restrict__ rgba, HD* __restrict__ out,
                                      int H, int W, int Hout, int Wout,
                                      float a0, float a1, float a2,
                                      float b0, float b1, float b2) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Hout * Wout * 3) return;
    const int c = i % 3, p = i / 3, oy = p / Wout, ox = p % Wout;
    const float sy = Hout > 1 ? oy * (float)(H - 1) / (float)(Hout - 1) : 0.f;
    const float sx = Wout > 1 ? ox * (float)(W - 1) / (float)(Wout - 1) : 0.f;
    const int y0 = (int)sy, x0 = (int)sx;
    const int y1 = y0 + 1 < H ? y0 + 1 : y0;
    const int x1 = x0 + 1 < W ? x0 + 1 : x0;
    const float fy = sy - y0, fx = sx - x0;
    const float* r00 = rgba + ((size_t)y0 * W + x0) * 4;
    const float* r01 = rgba + ((size_t)y0 * W + x1) * 4;
    const float* r10 = rgba + ((size_t)y1 * W + x0) * 4;
    const float* r11 = rgba + ((size_t)y1 * W + x1) * 4;
    const float v = (r00[c] * (1 - fx) + r01[c] * fx) * (1 - fy) + (r10[c] * (1 - fx) + r11[c] * fx) * fy;
    const float a = c == 0 ? a0 : (c == 1 ? a1 : a2);
    const float b = c == 0 ? b0 : (c == 1 ? b1 : b2);
    out[i] = __float2half(v * a + b);
}
