/*
 * aria_cuda.cu - CUDA backend (E8). Built with nvcc under -DARIA_CUDA.
 *
 * Correctness-first: a shared-memory tiled GEMM for the linear op, plus device
 * memory helpers. fp32 throughout (sm_61/Pascal has no fast fp16 compute; fp16
 * compute for sm_86 lands later). The kernel operates on device pointers so the
 * eventual device-resident forward (E8.5) can drop the host<->device copies.
 */

#include "aria_gpu.h"
#include "aria_sa3_dit.h"
#include "aria_sa3_dec.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(call) do {                                                        \
    cudaError_t err_ = (call);                                               \
    if (err_ != cudaSuccess)                                                 \
        fprintf(stderr, "aria_cuda: %s -> %s\n", #call, cudaGetErrorString(err_)); \
} while (0)

extern "C" int aria_cuda_available(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess && n > 0) ? 1 : 0;
}

extern "C" int aria_cuda_recommended(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 0;
    return p.major >= 7 ? 1 : 0;   /* Volta+ (tensor cores) — faster than the AVX2 CPU */
}

/* int8 tensor cores (Turing sm_75+): where the W8A8 IMMA GEMMs run */
extern "C" int aria_cuda_int8_tc(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return 0;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return 0;
    return p.major * 10 + p.minor >= 75;
}

extern "C" void aria_cuda_meminfo(size_t *used, size_t *total) {
    size_t freeb = 0, totb = 0;
    if (cudaMemGetInfo(&freeb, &totb) != cudaSuccess) { *used = 0; *total = 0; return; }
    *used = totb - freeb; *total = totb;
}

extern "C" void aria_cuda_device_info(char *buf, size_t buflen) {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        snprintf(buf, buflen, "no CUDA device");
        return;
    }
    snprintf(buf, buflen, "%s (sm_%d%d, %zu MiB)", p.name, p.major, p.minor,
             (size_t)(p.totalGlobalMem >> 20));
}

extern "C" void *aria_cuda_malloc(size_t bytes) {
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return NULL;
    return p;
}
extern "C" void aria_cuda_free(void *dptr) { if (dptr) cudaFree(dptr); }
extern "C" void aria_cuda_memcpy_h2d(void *d, const void *h, size_t n) {
    CK(cudaMemcpy(d, h, n, cudaMemcpyHostToDevice));
}
extern "C" void aria_cuda_memcpy_d2h(void *h, const void *d, size_t n) {
    CK(cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost));
}
extern "C" void aria_cuda_sync(void) { CK(cudaDeviceSynchronize()); }

/* ---- tiled GEMM: y[M,N] = x[M,K] @ W[N,K]^T + b ----
 * x and W are both row-major with K contiguous, so y[m,n] = dot(x[m,:], W[n,:]).
 * The W tile is loaded transposed into shared memory so the inner product reads
 * contiguous shared rows. TILE x TILE threads per block, one output each. */
#define TILE 16

__global__ void aria_gemm_nt(float *__restrict__ y, const float *__restrict__ x,
                             const float *__restrict__ W, const float *__restrict__ b,
                             int M, int K, int N) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty;   /* m */
    int col = blockIdx.x * TILE + tx;   /* n */
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int kx = k0 + tx, ky = k0 + ty;
        xs[ty][tx] = (row < M && kx < K) ? x[(size_t)row * K + kx] : 0.0f;
        /* ws[ty][tx] = W[col, k0+ty]  (col = blockIdx.x*TILE + tx) */
        ws[ty][tx] = (col < N && ky < K) ? W[(size_t)col * K + ky] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) acc += xs[ty][i] * ws[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) y[(size_t)row * N + col] = acc + (b ? b[col] : 0.0f);
}

extern "C" void aria_cuda_linear_dev(float *dy, const float *dx, const float *dW,
                                     const float *db, int M, int K, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    aria_gemm_nt<<<grid, block>>>(dy, dx, dW, db, M, K, N);
}

extern "C" void aria_cuda_linear(float *y, const float *x, const float *W,
                                 const float *b, int M, int K, int N) {
    float *dx = NULL, *dW = NULL, *dy = NULL, *db = NULL;
    CK(cudaMalloc(&dx, (size_t)M * K * sizeof(float)));
    CK(cudaMalloc(&dW, (size_t)N * K * sizeof(float)));
    CK(cudaMalloc(&dy, (size_t)M * N * sizeof(float)));
    CK(cudaMemcpy(dx, x, (size_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dW, W, (size_t)N * K * sizeof(float), cudaMemcpyHostToDevice));
    if (b) {
        CK(cudaMalloc(&db, (size_t)N * sizeof(float)));
        CK(cudaMemcpy(db, b, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    }
    aria_cuda_linear_dev(dy, dx, dW, db, M, K, N);
    CK(cudaGetLastError());
    CK(cudaMemcpy(y, dy, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(dx); cudaFree(dW); cudaFree(dy); if (db) cudaFree(db);
}

/* ============================ remaining hot ops ============================
 * Correctness-first kernels (thread-per-output, reductions in-thread; norms use
 * fp64 accumulation to match the CPU path). Host wrappers copy-compute-copy. */

#define THREADS 256
static int nblocks(size_t n) { return (int)((n + THREADS - 1) / THREADS); }

/* device-memory helpers for the host wrappers */
static float *d_in(const float *h, size_t n) {
    float *d = NULL; CK(cudaMalloc(&d, n * sizeof(float)));
    CK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice)); return d;
}
static float *d_new(size_t n) { float *d = NULL; CK(cudaMalloc(&d, n * sizeof(float))); return d; }
static void d_out(float *h, float *d, size_t n) {
    CK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost)); cudaFree(d);
}

/* ---- matmul C[M,N] = A[M,K] @ B[K,N] (B not transposed) ---- */
__global__ void k_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) acc += As[ty][i] * Bs[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}
static void matmul_dev(float *dC, const float *dA, const float *dB, int M, int K, int N) {
    dim3 block(TILE, TILE), grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    k_matmul<<<grid, block>>>(dC, dA, dB, M, K, N);
}
extern "C" void aria_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    float *dA = d_in(A, (size_t)M * K), *dB = d_in(B, (size_t)K * N), *dC = d_new((size_t)M * N);
    matmul_dev(dC, dA, dB, M, K, N); CK(cudaGetLastError());
    cudaFree(dA); cudaFree(dB); d_out(C, dC, (size_t)M * N);
}

/* ---- rmsnorm / gemma_rmsnorm: one block per row, fp32 shared-mem reduction ----
 * (the old thread-per-row + fp64 accum was 51% of GPU time; fp64 is 1/64 rate on
 * Ampere and 1 thread/row starves the device. fp32 is within the fp16 tolerance.) */
#define RED_TH 256
__global__ void k_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps, int gemma) {
    int r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (size_t)r * dim; float *yr = y + (size_t)r * dim;
    __shared__ float red[RED_TH];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float g = w ? (gemma ? 1.0f + w[i] : w[i]) : 1.0f;
        yr[i] = xr[i] * inv * g;
    }
}
/* fused rmsnorm (weight w) + adaLN: y = (x * rsqrt(meansq+eps) * w) * (1+scale) + shift.
 * Replaces a k_rmsnorm + k_adaln pair (saves one read+write pass of [S,dim] + a launch). */
__global__ void k_rmsnorm_adaln(float *y, const float *x, const float *w,
                                const float *scale, const float *shift, int S, int dim, float eps) {
    int r = blockIdx.x; if (r >= S) return;
    const float *xr = x + (size_t)r * dim; float *yr = y + (size_t)r * dim;
    __shared__ float red[RED_TH];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        yr[i] = (xr[i] * inv * w[i]) * (1.0f + scale[i]) + shift[i];
}
/* fp16-emitting variants of the GEMM-input producers (A2): every consumer is a tensor-
 * core GEMM whose activation was converted fp32->fp16 anyway (k_f32_to_f16 pass in
 * gemm_f16w) -- __float2half at the producer is the same RTNE bits, one [S,dim] fp32
 * round-trip less per op. Math is fp32 internally, only the store rounds. */
__global__ void k_rmsnorm_h(__half *y, const float *x, const float *w, int rows, int dim, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (size_t)r * dim; __half *yr = y + (size_t)r * dim;
    __shared__ float red[RED_TH];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        yr[i] = __float2half(xr[i] * inv * w[i]);
}
__global__ void k_rmsnorm_adaln_h(__half *y, const float *x, const float *w,
                                  const float *scale, const float *shift, int S, int dim, float eps) {
    int r = blockIdx.x; if (r >= S) return;
    const float *xr = x + (size_t)r * dim; __half *yr = y + (size_t)r * dim;
    __shared__ float red[RED_TH];
    float ss = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    red[threadIdx.x] = ss; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        yr[i] = __float2half((xr[i] * inv * w[i]) * (1.0f + scale[i]) + shift[i]);
}

extern "C" void aria_cuda_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps) {
    float *dx = d_in(x, (size_t)rows * dim), *dw = w ? d_in(w, dim) : NULL, *dy = d_new((size_t)rows * dim);
    k_rmsnorm<<<rows, RED_TH>>>(dy, dx, dw, rows, dim, eps, 0); CK(cudaGetLastError());
    cudaFree(dx); if (dw) cudaFree(dw); d_out(y, dy, (size_t)rows * dim);
}
extern "C" void aria_cuda_gemma_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps) {
    float *dx = d_in(x, (size_t)rows * dim), *dw = d_in(w, dim), *dy = d_new((size_t)rows * dim);
    k_rmsnorm<<<rows, RED_TH>>>(dy, dx, dw, rows, dim, eps, 1); CK(cudaGetLastError());
    cudaFree(dx); cudaFree(dw); d_out(y, dy, (size_t)rows * dim);
}

/* ---- dynamic tanh: y = tanh(alpha*x)*w[c] + b[c] ---- */
__global__ void k_dyt(float *y, const float *x, float alpha, const float *w, const float *b, size_t tot, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tot) return;
    int c = (int)(i % dim);
    y[i] = tanhf(alpha * x[i]) * w[c] + b[c];
}
/* fp16-emitting variant (A2): for dyt outputs whose only consumer is a GEMM. */
__global__ void k_dyt_h(__half *y, const float *x, float alpha, const float *w, const float *b, size_t tot, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tot) return;
    int c = (int)(i % dim);
    y[i] = __float2half(tanhf(alpha * x[i]) * w[c] + b[c]);
}
extern "C" void aria_cuda_dynamic_tanh(float *y, const float *x, float alpha,
                                       const float *w, const float *b, int rows, int dim) {
    size_t tot = (size_t)rows * dim;
    float *dx = d_in(x, tot), *dw = d_in(w, dim), *db = d_in(b, dim), *dy = d_new(tot);
    k_dyt<<<nblocks(tot), THREADS>>>(dy, dx, alpha, dw, db, tot, dim); CK(cudaGetLastError());
    cudaFree(dx); cudaFree(dw); cudaFree(db); d_out(y, dy, tot);
}

/* ---- elementwise activations (in place) ---- */
__global__ void k_softcap(float *s, int n, float cap) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < (size_t)n) s[i] = cap * tanhf(s[i] / cap);
}
__global__ void k_silu(float *x, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < (size_t)n) { float v = x[i]; x[i] = v / (1.0f + expf(-v)); }
}
__global__ void k_gelu(float *x, int n) {
    const float c = 0.7978845608028654f;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)n) { float v = x[i]; x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))); }
}
__global__ void k_silugate(float *o, const float *g, const float *u, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)n) { float gg = g[i]; o[i] = (gg / (1.0f + expf(-gg))) * u[i]; }
}
extern "C" void aria_cuda_softcap(float *s, int n, float cap) {
    float *d = d_in(s, n); k_softcap<<<nblocks(n), THREADS>>>(d, n, cap); CK(cudaGetLastError()); d_out(s, d, n);
}
extern "C" void aria_cuda_silu(float *x, int n) {
    float *d = d_in(x, n); k_silu<<<nblocks(n), THREADS>>>(d, n); CK(cudaGetLastError()); d_out(x, d, n);
}
extern "C" void aria_cuda_gelu_tanh(float *x, int n) {
    float *d = d_in(x, n); k_gelu<<<nblocks(n), THREADS>>>(d, n); CK(cudaGetLastError()); d_out(x, d, n);
}
extern "C" void aria_cuda_silu_gate(float *out, const float *gate, const float *up, int n) {
    float *dg = d_in(gate, n), *du = d_in(up, n), *don = d_new(n);
    k_silugate<<<nblocks(n), THREADS>>>(don, dg, du, n); CK(cudaGetLastError());
    cudaFree(dg); cudaFree(du); d_out(out, don, n);
}

/* ---- row softmax: one block per row (shared-mem max + sum), optional mask ---- */
__global__ void k_softmax(float *x, int rows, int cols, const float *mask) {
    int r = blockIdx.x; if (r >= rows) return;
    float *xr = x + (size_t)r * cols; const float *mr = mask ? mask + (size_t)r * cols : NULL;
    __shared__ float red[RED_TH];
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { float v = xr[i] + (mr ? mr[i] : 0.0f); xr[i] = v; mx = fmaxf(mx, v); }
    red[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    mx = red[0]; __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) { float e = expf(xr[i] - mx); xr[i] = e; sum += e; }
    red[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xr[i] *= inv;
}
__global__ void k_scale(float *y, const float *x, float a, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < (size_t)n) y[i] = x[i] * a;
}
/* k_softmax with fp16 output: identical fp32 math (x holds the fp32 scores), the final
 * normalize writes __float2half(e*inv) straight into the AV GEMM's half buffer -- same
 * RTNE value the old separate k_f32_to_f16 pass produced. The exp is RECOMPUTED in the
 * normalize pass instead of written back to x (A2): expf is deterministic, so the
 * result is bit-identical, and the [rows,cols] fp32 exp write -- the largest non-GEMM
 * HBM write in the DiT (206 ms/768 inst on medium 60 s) -- disappears. x stays const. */
__global__ void k_softmax_f16(const float *x, __half *out, int rows, int cols) {
    int r = blockIdx.x; if (r >= rows) return;
    const float *xr = x + (size_t)r * cols; __half *orow = out + (size_t)r * cols;
    __shared__ float red[RED_TH];
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) mx = fmaxf(mx, xr[i]);
    red[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    mx = red[0]; __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += expf(xr[i] - mx);
    red[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s]; __syncthreads(); }
    float inv = red[0] > 0.0f ? 1.0f / red[0] : 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) orow[i] = __float2half(expf(xr[i] - mx) * inv);
}
/* Warp-per-row twins of k_softmax / k_softmax_f16, 8 rows per 256-thread block (no
 * __syncthreads; short rows -- 34, 257 -- no longer idle 7/8 of a block). Lane l plays
 * the block kernel's threads t = l + 32j (j = 0..7): it keeps their 8 partials, each
 * over i = t, t+256, ... in order, and combines them in that kernel's tree pairing
 * (levels 128/64/32 in-lane, 16..1 by shuffle). fp add is commutative, so same pairing
 * = same bits; the max is order-free. Bit-identical to the block kernels. */
#define SMW_ROWS 8
__device__ __forceinline__ float smw_tree(float p[8], int is_max) {
    float a0, a1, a2, a3, b0, b1, v;
    if (is_max) {
        a0 = fmaxf(p[0], p[4]); a1 = fmaxf(p[1], p[5]); a2 = fmaxf(p[2], p[6]); a3 = fmaxf(p[3], p[7]);
        b0 = fmaxf(a0, a2); b1 = fmaxf(a1, a3); v = fmaxf(b0, b1);
        for (int s = 16; s > 0; s >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, s));
    } else {
        a0 = p[0] + p[4]; a1 = p[1] + p[5]; a2 = p[2] + p[6]; a3 = p[3] + p[7];
        b0 = a0 + a2; b1 = a1 + a3; v = b0 + b1;
        for (int s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
    }
    return __shfl_sync(0xffffffffu, v, 0);
}
__global__ void k_softmax_w(float *x, int rows, int cols, const float *mask) {
    int r = blockIdx.x * SMW_ROWS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;   /* whole warps exit together: shuffles stay full-mask */
    float *xr = x + (size_t)r * cols; const float *mr = mask ? mask + (size_t)r * cols : NULL;
    float p[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) p[j] = -INFINITY;
    for (int base = 0; base < cols; base += 256) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int i = base + 32 * j + lane;
            if (i < cols) { float v = xr[i] + (mr ? mr[i] : 0.0f); xr[i] = v; p[j] = fmaxf(p[j], v); }
        }
    }
    float mx = smw_tree(p, 1);
    #pragma unroll
    for (int j = 0; j < 8; j++) p[j] = 0.0f;
    for (int base = 0; base < cols; base += 256) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int i = base + 32 * j + lane;
            if (i < cols) { float e = expf(xr[i] - mx); xr[i] = e; p[j] += e; }
        }
    }
    float sum = smw_tree(p, 0);
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = lane; i < cols; i += 32) xr[i] *= inv;
}
__global__ void k_softmax_f16_w(const float *x, __half *out, int rows, int cols) {
    int r = blockIdx.x * SMW_ROWS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (r >= rows) return;
    const float *xr = x + (size_t)r * cols; __half *orow = out + (size_t)r * cols;
    float p[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) p[j] = -INFINITY;
    for (int base = 0; base < cols; base += 256) {
        #pragma unroll
        for (int j = 0; j < 8; j++) { int i = base + 32 * j + lane; if (i < cols) p[j] = fmaxf(p[j], xr[i]); }
    }
    float mx = smw_tree(p, 1);
    #pragma unroll
    for (int j = 0; j < 8; j++) p[j] = 0.0f;
    for (int base = 0; base < cols; base += 256) {
        #pragma unroll
        for (int j = 0; j < 8; j++) { int i = base + 32 * j + lane; if (i < cols) p[j] += expf(xr[i] - mx); }
    }
    float sum = smw_tree(p, 0);
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int i = lane; i < cols; i += 32) orow[i] = __float2half(expf(xr[i] - mx) * inv);
}
extern "C" void aria_cuda_softmax(float *x, int rows, int cols, const float *mask) {
    float *dx = d_in(x, (size_t)rows * cols), *dm = mask ? d_in(mask, (size_t)rows * cols) : NULL;
    k_softmax<<<rows, RED_TH>>>(dx, rows, cols, dm); CK(cudaGetLastError());
    if (dm) cudaFree(dm); d_out(x, dx, (size_t)rows * cols);
}

/* ---- RoPE apply on [H,N,D] (rotate first rot_dim with pairing i,i+rot/2) ---- */
__global__ void k_rope(float *x, const float *c, const float *s, int H, int N, int D, int rot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   /* h*N + p */
    if (idx >= H * N) return;
    int p = idx % N, rh = rot / 2;
    float *row = x + (size_t)idx * D;
    const float *cc = c + (size_t)p * rh, *ss = s + (size_t)p * rh;
    for (int i = 0; i < rh; i++) {
        float a = row[i], b = row[i + rh];
        row[i] = a * cc[i] - b * ss[i];
        row[i + rh] = b * cc[i] + a * ss[i];
    }
}

/* fused extract-head + per-head rmsnorm(weight w) + RoPE -> head-major out[h,s,:hd].
 * Replaces a k_extract_heads + k_rmsnorm + k_rope chain (saves two [H,S,hd] HBM
 * round-trips + two launches per q/k). One block per (h,s) row, hd threads (hd<=64).
 * rcos==NULL skips RoPE (cross-attn q). Emits fp16 directly (all consumers are the
 * tensor-core attention): __float2half(res) here is the same RTNE value the old
 * fp32-store + k_f32_to_f16 pass produced, one [H,S,hd] round-trip less. */
__global__ void k_extract_normrope(__half *out, const float *src, const float *w,
                                   const float *rcos, const float *rsin,
                                   int S, int H, int hd, int stride, int off, int rot, float eps) {
    int hs = blockIdx.x; if (hs >= H * S) return;
    int s = hs % S, h = hs / S, d = threadIdx.x;
    const float *sp = src + (size_t)s * stride + off + (size_t)h * hd;
    __shared__ float red[64], vec[64];
    float x = sp[d];
    red[d] = x * x; __syncthreads();
    for (int st = hd / 2; st > 0; st >>= 1) { if (d < st) red[d] += red[d + st]; __syncthreads(); }
    vec[d] = x * rsqrtf(red[0] / hd + eps) * w[d];
    __syncthreads();
    int rh = rot / 2; float res;
    if (rcos == NULL || d >= rot)  res = vec[d];
    else if (d < rh) res = vec[d] * rcos[(size_t)s * rh + d]      - vec[d + rh] * rsin[(size_t)s * rh + d];
    else { int i = d - rh; res = vec[d] * rcos[(size_t)s * rh + i] + vec[i] * rsin[(size_t)s * rh + i]; }
    out[(size_t)hs * hd + d] = __float2half(res);
}
extern "C" void aria_cuda_rope_apply(float *x, const float *cos_t, const float *sin_t,
                                     int H, int N, int D, int rot_dim) {
    int rh = rot_dim / 2;
    float *dx = d_in(x, (size_t)H * N * D), *dc = d_in(cos_t, (size_t)N * rh), *ds = d_in(sin_t, (size_t)N * rh);
    k_rope<<<nblocks((size_t)H * N), THREADS>>>(dx, dc, ds, H, N, D, rot_dim); CK(cudaGetLastError());
    cudaFree(dc); cudaFree(ds); d_out(x, dx, (size_t)H * N * D);
}

/* ---- multi-head SDPA: out[h]=softmax(scale*q[h]@k[h]^T + mask)@v[h] ----
 * mirrors aria_attention (scale = 1/sqrt(D)); composes gemm_nt + scale + softmax + matmul. */
extern "C" void aria_cuda_attention(float *out, const float *q, const float *k, const float *v,
                                    int H, int Nq, int Nk, int D, const float *mask) {
    float scale = 1.0f / sqrtf((float)D);
    float *dq = d_in(q, (size_t)H * Nq * D), *dk = d_in(k, (size_t)H * Nk * D);
    float *dv = d_in(v, (size_t)H * Nk * D), *dout = d_new((size_t)H * Nq * D);
    float *dscores = d_new((size_t)Nq * Nk);
    float *dmask = mask ? d_in(mask, (size_t)Nq * Nk) : NULL;
    dim3 sblk(TILE, TILE), sgrid((Nk + TILE - 1) / TILE, (Nq + TILE - 1) / TILE);
    dim3 oblk(TILE, TILE), ogrid((D + TILE - 1) / TILE, (Nq + TILE - 1) / TILE);
    for (int h = 0; h < H; h++) {
        const float *qh = dq + (size_t)h * Nq * D, *kh = dk + (size_t)h * Nk * D, *vh = dv + (size_t)h * Nk * D;
        float *oh = dout + (size_t)h * Nq * D;
        aria_gemm_nt<<<sgrid, sblk>>>(dscores, qh, kh, NULL, Nq, D, Nk);   /* scores = qh @ kh^T */
        k_scale<<<nblocks((size_t)Nq * Nk), THREADS>>>(dscores, dscores, scale, Nq * Nk);
        k_softmax<<<Nq, RED_TH>>>(dscores, Nq, Nk, dmask);
        k_matmul<<<ogrid, oblk>>>(oh, dscores, vh, Nq, Nk, D);             /* out = scores @ vh */
    }
    CK(cudaGetLastError());
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dscores); if (dmask) cudaFree(dmask);
    d_out(out, dout, (size_t)H * Nq * D);
}

/* ---- conv1d (stride 1, pre-folded weights), one thread per (cout,l) ---- */
__global__ void k_conv1d(float *out, const float *in, const float *w, const float *bias,
                         int Cin, int Cout, int K, int pad, int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   /* o*L + t */
    if (idx >= Cout * L) return;
    int o = idx / L, t = idx % L;
    const float *wo = w + (size_t)o * Cin * K;
    float acc = bias ? bias[o] : 0.0f;
    for (int i = 0; i < Cin; i++) {
        const float *wi = wo + (size_t)i * K; const float *ini = in + (size_t)i * L;
        for (int kk = 0; kk < K; kk++) { int ti = t + kk - pad; if (ti >= 0 && ti < L) acc += wi[kk] * ini[ti]; }
    }
    out[idx] = acc;
}
extern "C" void aria_cuda_conv1d(float *out, const float *in, const float *w, const float *bias,
                                 int Cin, int Cout, int K, int pad, int L) {
    float *din = d_in(in, (size_t)Cin * L), *dw = d_in(w, (size_t)Cout * Cin * K);
    float *db = bias ? d_in(bias, Cout) : NULL, *dout = d_new((size_t)Cout * L);
    k_conv1d<<<nblocks((size_t)Cout * L), THREADS>>>(dout, din, dw, db, Cin, Cout, K, pad, L); CK(cudaGetLastError());
    cudaFree(din); cudaFree(dw); if (db) cudaFree(db); d_out(out, dout, (size_t)Cout * L);
}

/* ===================== device-resident DiT (E8.5) =====================
 * Weights live on the device as fp16 (fp32 compute: loaded and converted in the
 * GEMM); activations stay fp32 in a device bump arena. The denoise loop runs on
 * the device -- only the per-step latent (x in) and velocity (v out) cross the
 * bus. Encoder/decoder stay on the CPU. Mirrors aria_sa3_dit.c step/block_core. */

__global__ void k_f32_to_f16(__half *o, const float *in, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half(in[i]);
}
extern "C" void aria_f32_to_f16_n(uint16_t *dst, const float *src, size_t n);   /* aria_cpu.c */
/* fp32 -> fp16 on the host (parallel F16C RTNE: the same bits __float2half produced on
 * the device), then a half-size copy: half the PCIe bytes, no fp32 device temp + free.
 * Chunks convert into two pinned staging buffers and DMA asynchronously, so converting
 * chunk i+1 (and faulting in its mmap'd source) overlaps chunk i's copy. */
#define UP_CHUNK ((size_t)8 << 20)   /* halves per staging chunk (16 MB) */
static __half *upload_f16(const float *h, size_t n) {
    static uint16_t *stage[2]; static cudaEvent_t ev[2]; static int pinned = -1;
    __half *d = NULL;
    if (cudaMalloc(&d, n * sizeof(__half)) != cudaSuccess) return NULL;
    if (pinned < 0) {
        pinned = cudaMallocHost((void **)&stage[0], UP_CHUNK * sizeof(uint16_t)) == cudaSuccess &&
                 cudaMallocHost((void **)&stage[1], UP_CHUNK * sizeof(uint16_t)) == cudaSuccess &&
                 cudaEventCreateWithFlags(&ev[0], cudaEventDisableTiming) == cudaSuccess &&
                 cudaEventCreateWithFlags(&ev[1], cudaEventDisableTiming) == cudaSuccess;
        cudaGetLastError();
    }
    if (!pinned) {   /* no pinned memory: one pageable copy */
        uint16_t *tmp = (uint16_t *)malloc(n * sizeof(uint16_t));
        if (!tmp) { cudaFree(d); return NULL; }
        aria_f32_to_f16_n(tmp, h, n);
        cudaMemcpy(d, tmp, n * sizeof(__half), cudaMemcpyHostToDevice);
        free(tmp);
        return d;
    }
    cudaStream_t st = cudaStreamPerThread;
    for (size_t off = 0, i = 0; off < n; off += UP_CHUNK, i++) {
        size_t c = n - off < UP_CHUNK ? n - off : UP_CHUNK;
        int b = (int)(i & 1);
        cudaEventSynchronize(ev[b]);   /* the copy out of this buffer two chunks ago is done */
        aria_f32_to_f16_n(stage[b], h + off, c);
        cudaMemcpyAsync(d + off, stage[b], c * sizeof(__half), cudaMemcpyHostToDevice, st);
        cudaEventRecord(ev[b], st);
    }
    return d;   /* later work on this stream is ordered after the copies */
}
static float *upload_f32(const float *h, size_t n) {
    float *d = NULL; if (cudaMalloc(&d, n * sizeof(float)) != cudaSuccess) return NULL;
    cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice); return d;
}

/* y[M,N] = x[M,K] @ W[N,K]^T + b ; W is fp16 (converted on load), x/y/b fp32. */
__global__ void k_gemm_f16w(float *y, const float *x, const __half *W, const float *b, int M, int K, int N) {
    __shared__ float xs[TILE][TILE], ws[TILE][TILE];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int kx = k0 + tx, ky = k0 + ty;
        xs[ty][tx] = (row < M && kx < K) ? x[(size_t)row * K + kx] : 0.0f;
        ws[ty][tx] = (col < N && ky < K) ? __half2float(W[(size_t)col * K + ky]) : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) acc += xs[ty][i] * ws[i][tx];
        __syncthreads();
    }
    if (row < M && col < N) y[(size_t)row * N + col] = acc + (b ? b[col] : 0.0f);
}
/* y[m,n] += b[n]  (broadcast bias over rows; cuBLAS GEMM has no vector-bias epilogue) */
__global__ void k_add_bias(float *y, const float *b, int M, int N) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)M * N) y[i] += b[i % N];
}
/* y[r*L + t] += b[r]  (channel-major row-broadcast: the decoders' mapping-conv bias) */
__global__ void k_add_bias_rows(float *y, const float *b, int rows, int L) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)rows * L) y[i] += b[i / L];
}

/* ---- glue kernels (device pointers) ---- */
__global__ void k_add(float *y, const float *a, const float *b, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = a[i] + b[i];
}
__global__ void k_transpose(float *dst, const float *src, int A, int B) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* a*B+b over A*B -> dst[b*A+a] */
    if (idx >= (size_t)A * B) return;
    int a = (int)(idx / B), b = (int)(idx % B);
    dst[(size_t)b * A + a] = src[idx];
}
/* fused gate + residual add: out = residual + (o + b[col]) * sigmoid(1 - gate[col]).
 * Replaces a k_gate + k_add pair (saves one read+write pass of [S,dim] + a launch).
 * b folds the producer GEMM's bias in (A2, drops that GEMM's k_add_bias pass); NULL = none. */
__global__ void k_gate_add(float *out, const float *residual, const float *o,
                           const float *gate, const float *b, int S, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * dim) return;
    int i = (int)(idx % dim);
    /* b==NULL skips the add entirely (adding literal 0.0f would flip -0.0 to +0.0) */
    float ov = b ? (o[idx] + b[i]) : o[idx];
    out[idx] = residual[idx] + ov * (1.0f / (1.0f + expf(-(1.0f - gate[i]))));
}
/* residual add with the producer GEMM's bias folded in: out = a + (o + b[col]) (A2).
 * The bias binds to o FIRST -- the exact order the old k_add_bias + k_add pair used --
 * so the fold is bit-identical. */
__global__ void k_add_bias_res(float *out, const float *a, const float *o,
                               const float *b, size_t n, int dim) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + (o[idx] + b[idx % dim]);
}
/* E12 residual steering (ADD): seq[t,c] += scale * dir[c] for c < ddim, broadcast over all S tokens.
 * scale is read from DEVICE memory so the launch can live inside the CUDA graph: the host
 * updates the effective scale (0 outside the step window) between replays; scale==0 early-exits
 * without writing -> bit-exact no-op, so replays outside the window match the un-launched case. */
__global__ void k_steer_add(float *seq, const float *dir, const float *scale_p, int S, int dim, int ddim) {
    float scale = *scale_p;
    if (scale == 0.0f) return;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * dim) return;
    int c = (int)(idx % dim);
    if (c < ddim) seq[idx] += scale * dir[c];
}
/* E12 directional ablation (PROJECT): seq[t] -= scale*(dot(dir,seq[t])/||dir||^2)*dir, per token.
 * One block per token; blockDim.x-wide reduction of the dot product. scale via device pointer
 * (graph-resident, see k_steer_add). */
__global__ void k_steer_project(float *seq, const float *dir, const float *scale_p, float inv_norm2,
                                int S, int dim, int ddim) {
    float scale = *scale_p;
    if (scale == 0.0f) return;
    int t = blockIdx.x; if (t >= S) return;
    __shared__ float red[RED_TH];
    float *xt = seq + (size_t)t * dim;
    float partial = 0.0f;
    for (int c = threadIdx.x; c < ddim; c += blockDim.x) partial += dir[c] * xt[c];
    red[threadIdx.x] = partial; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float k = scale * red[0] * inv_norm2;
    for (int c = threadIdx.x; c < ddim; c += blockDim.x) xt[c] -= k * dir[c];
}
__global__ void k_extract_heads(float *dst, const float *src, int S, int H, int hd, int stride, int off) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* (h*S+s)*hd+d */
    if (idx >= (size_t)H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)(idx / ((size_t)hd * S));
    dst[idx] = src[(size_t)s * stride + off + (size_t)h * hd + d];
}
/* fp16-emitting variant (DiT attention v: consumer is the tensor-core AV GEMM;
 * same RTNE bits as extract-fp32 + k_f32_to_f16, one round-trip less). */
__global__ void k_extract_heads_h(__half *dst, const float *src, int S, int H, int hd, int stride, int off) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* (h*S+s)*hd+d */
    if (idx >= (size_t)H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)(idx / ((size_t)hd * S));
    dst[idx] = __float2half(src[(size_t)s * stride + off + (size_t)h * hd + d]);
}
__global__ void k_merge_heads(float *dst, const float *src, int S, int H, int hd) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* src (h*S+s)*hd+d */
    if (idx >= (size_t)H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)(idx / ((size_t)hd * S));
    dst[(size_t)s * H * hd + (size_t)h * hd + d] = src[idx];
}
/* differential combine fused into the head merge: dst = merge(a - b) -- the same fp32
 * fp32 subtract the separate a[i] -= b[i] pass did, then the same conversion; saves a [H,S,hd] read-modify-write. */
__global__ void k_merge_heads_sub(float *dst, const float *a, const float *b, int S, int H, int hd) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* src (h*S+s)*hd+d */
    if (idx >= (size_t)H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)(idx / ((size_t)hd * S));
    dst[(size_t)s * H * hd + (size_t)h * hd + d] = a[idx] - b[idx];
}
__global__ void k_merge_heads_sub_h(__half *dst, const float *a, const float *b, int S, int H, int hd) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)(idx / ((size_t)hd * S));
    float v = a[idx] - b[idx];
    dst[(size_t)s * H * hd + (size_t)h * hd + d] = __float2half(v);
}
/* fp16-emitting merge (A2): dst feeds the to_out GEMM directly. */
__global__ void k_merge_heads_h(__half *dst, const float *src, int S, int H, int hd) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* src (h*S+s)*hd+d */
    if (idx >= (size_t)H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)(idx / ((size_t)hd * S));
    dst[(size_t)s * H * hd + (size_t)h * hd + d] = __float2half(src[idx]);
}
/* b2 folds the ff_in GEMM's bias ([2*inner], up half then gate half) in (A2); NULL = none. */
__global__ void k_ff_silugate(float *out, const float *proj, const float *b2, int S, int inner) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* s*inner+i */
    if (idx >= (size_t)S * inner) return;
    int s = (int)(idx / inner), i = (int)(idx % inner);
    const float *pr = proj + (size_t)s * 2 * inner;
    float up = b2 ? (pr[i] + b2[i]) : pr[i];
    float g = b2 ? (pr[inner + i] + b2[inner + i]) : pr[inner + i];
    out[idx] = (g / (1.0f + expf(-g))) * up;
}
/* fp16-emitting silu-gate (A2): out feeds the ff_out GEMM directly. */
__global__ void k_ff_silugate_h(__half *out, const float *proj, const float *b2, int S, int inner) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* s*inner+i */
    if (idx >= (size_t)S * inner) return;
    int s = (int)(idx / inner), i = (int)(idx % inner);
    const float *pr = proj + (size_t)s * 2 * inner;
    float up = b2 ? (pr[i] + b2[i]) : pr[i];
    float g = b2 ? (pr[inner + i] + b2[inner + i]) : pr[inner + i];
    out[idx] = __float2half((g / (1.0f + expf(-g))) * up);
}

/* ---- device bump arena (pointer arithmetic only) ---- */
struct darena { char *base; size_t cap, used; };
static float *da_alloc(darena *a, size_t nf) {
    size_t off = (a->used + 63) & ~((size_t)63);
    a->used = off + nf * sizeof(float);
    return (float *)(a->base + off);
}
static __half *da_alloc_half(darena *a, size_t nh) {
    size_t off = (a->used + 63) & ~((size_t)63);
    a->used = off + nh * sizeof(__half);
    return (__half *)(a->base + off);
}
static int8_t *da_alloc_i8(darena *a, size_t nb) {
    size_t off = (a->used + 63) & ~((size_t)63);
    a->used = off + nb;
    return (int8_t *)(a->base + off);
}
/* reuse a handle-owned arena across calls, growing it only when a request needs more:
 * no per-decode cudaMalloc/cudaFree (cudaFree synchronizes the device) */
static int da_reuse(darena *cache, darena *ar, size_t cap) {
    if (cache->cap < cap) {
        cudaFree(cache->base); cache->base = NULL; cache->cap = 0;
        if (cudaMalloc(&cache->base, cap) != cudaSuccess) {
            cudaGetLastError();
            fprintf(stderr, "[aria] decoder: cannot allocate a %.0f MB device decode arena\n", cap / 1048576.0);
            return -1;
        }
        cache->cap = cap;
    }
    ar->base = cache->base; ar->cap = cache->cap; ar->used = 0;
    return 0;
}
static size_t da_save(darena *a) { return a->used; }
static void da_restore(darena *a, size_t m) { a->used = m; }

/* multi-head SDPA, all heads batched (scale 1/sqrt(D)). scores holds [H,Nq,Nk].
 * The original fp32 cublasSgemm path, kept for the cheap small-music decoder
 * chunks where fp16 would only cost accuracy. */
static void attn_dev(cublasHandle_t cb, darena *ar, float *out, const float *q, const float *k,
                     const float *v, int H, int Nq, int Nk, int D, float *scores) {
    (void)ar;
    const float scale = 1.0f / sqrtf((float)D), one = 1.0f, zero = 0.0f;
    /* 1/sqrt(D) folded into the QK^T alpha (cuBLAS applies alpha to the accumulated
     * product -- same op/order as the old separate k_scale pass, one S*S*H pass less). */
    cublasSgemmStridedBatched(cb, CUBLAS_OP_T, CUBLAS_OP_N, Nk, Nq, D,
        &scale, k, D, (long long)Nk * D, q, D, (long long)Nq * D,
        &zero, scores, Nk, (long long)Nq * Nk, H);
    k_softmax_w<<<(H * Nq + SMW_ROWS - 1) / SMW_ROWS, 32 * SMW_ROWS>>>(scores, H * Nq, Nk, NULL);
    cublasSgemmStridedBatched(cb, CUBLAS_OP_N, CUBLAS_OP_N, D, Nq, Nk,
        &one, v, D, (long long)Nk * D, scores, Nk, (long long)Nq * Nk,
        &zero, out, D, (long long)Nq * D, H);
}
/* fp16-input SDPA for the DiT (the dominant attention): q/k/v are ALREADY fp16
 * (emitted by k_extract_normrope / k_extract_heads_h, or the request's cached
 * cross K/V), so the per-call k_f32_to_f16 passes are gone. QK^T/AV run on tensor
 * cores (fp16->fp32 scores, fp32 softmax, fp16 AV) -- same bits as the old
 * convert-per-call path. */
static void attn_dev_h(cublasHandle_t cb, darena *ar, float *out, const __half *q, const __half *k,
                       const __half *v, int H, int Nq, int Nk, int D, float *scores) {
    const float scale = 1.0f / sqrtf((float)D), one = 1.0f, zero = 0.0f;
    size_t ns = (size_t)H * Nq * Nk;
    size_t mark = da_save(ar);
    __half *sh = da_alloc_half(ar, ns);
    /* alpha = 1/sqrt(D) folded into QK^T (drops the k_scale S*S*H pass); softmax writes the
     * fp16 probabilities for the AV GEMM directly (drops the k_f32_to_f16 S*S*H pass). */
    cublasGemmStridedBatchedEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, Nk, Nq, D,
        &scale, k, CUDA_R_16F, D, (long long)Nk * D, q, CUDA_R_16F, D, (long long)Nq * D,
        &zero, scores, CUDA_R_32F, Nk, (long long)Nq * Nk, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    k_softmax_f16_w<<<(H * Nq + SMW_ROWS - 1) / SMW_ROWS, 32 * SMW_ROWS>>>(scores, sh, H * Nq, Nk);
    cublasGemmStridedBatchedEx(cb, CUBLAS_OP_N, CUBLAS_OP_N, D, Nq, Nk,
        &one, v, CUDA_R_16F, D, (long long)Nk * D, sh, CUDA_R_16F, Nk, (long long)Nq * Nk,
        &zero, out, CUDA_R_32F, D, (long long)Nq * D, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    da_restore(ar, mark);
}

/* ---- device quantized weight (E9.4): packed in VRAM, dequant-on-use to fp16 ----
 * Q8/Q4 weights stay packed on the device (medium fits low VRAM); each GEMM
 * dequantizes its weight into a reused fp16 scratch, then runs the tensor-core
 * cuBLAS path. dt==F32 keeps the plain fp16 weight (the default). */
struct dqw {
    int dt, N, K;
    __half *f16;     /* ARIA_F32: the fp16 weight */
    void *q;         /* ARIA_Q8: int8_t* ; ARIA_Q4: uint8_t* (packed) */
    float *scale;    /* ARIA_Q8: [N]; ARIA_Q4: [N*nblocks] */
};

__global__ void k_dequant_q8(__half *W, const int8_t *q, const float *scale, int N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * K) return;
    W[idx] = __float2half((float)q[idx] * scale[idx / K]);
}
/* asymmetric (zero-point) int4: per block [min, scale]; W = min + nib*scale. */
__global__ void k_dequant_q4(__half *W, const uint8_t *q, const float *scale, int N, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * K) return;
    int n = (int)(idx / K), k = (int)(idx % K);
    int nblk = (K + ARIA_Q4_BLOCK - 1) / ARIA_Q4_BLOCK;
    size_t rb = (size_t)((K + 1) / 2);
    uint8_t byte = q[(size_t)n * rb + (k >> 1)];
    int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
    size_t si = ((size_t)n * nblk + (k / ARIA_Q4_BLOCK)) * 2;
    W[idx] = __float2half(scale[si] + (float)nib * scale[si + 1]);
}

/* host-quantize a [N,K] f32 weight per dt and upload it (packed) to the device. */
static int upload_dqw(dqw *w, const float *W, int N, int K, aria_dtype dt) {
    w->dt = dt; w->N = N; w->K = K; w->f16 = NULL; w->q = NULL; w->scale = NULL;
    if (dt == ARIA_Q8 || dt == ARIA_Q4) {
        size_t qb = (dt == ARIA_Q8) ? aria_q8_qbytes(N, K) : aria_q4_qbytes(N, K);
        size_t ns = (dt == ARIA_Q8) ? aria_q8_nscale(N, K) : aria_q4_nscale(N, K);
        void *hq = malloc(qb); float *hs = (float *)malloc(ns * sizeof(float));
        if (dt == ARIA_Q8) aria_q8_quant((int8_t *)hq, hs, W, N, K);
        else               aria_q4_quant((uint8_t *)hq, hs, W, N, K);
        CK(cudaMalloc(&w->q, qb)); CK(cudaMemcpy(w->q, hq, qb, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&w->scale, ns * sizeof(float)));
        CK(cudaMemcpy(w->scale, hs, ns * sizeof(float), cudaMemcpyHostToDevice));
        free(hq); free(hs);
        return w->q != NULL && w->scale != NULL;
    }
    w->f16 = upload_f16(W, (size_t)N * K);
    return w->f16 != NULL;
}
static void free_dqw(dqw *w) { cudaFree(w->f16); cudaFree(w->q); cudaFree(w->scale); }

struct blk_dev {
    const float *pre_norm, *cross_norm, *ff_norm, *sa_q_norm, *sa_k_norm, *ca_q_norm, *ca_k_norm;
    const float *ssg, *ff_in_b, *ff_out_b;
    dqw sa_to_qkv, sa_to_out, ca_to_q, ca_to_out, ca_to_kv, ff_in_w, ff_out_w;
};

struct aria_cuda_dit {
    int depth, ed, H, hd, inner, io_ch, n_mem, rot;
    int differential;         /* medium: differential self+cross attention */
    aria_dtype precision;     /* block GEMM weight precision (fp32->fp16 / q8 / q4) */
    int w8a8;                 /* Q8: run int8-activation IMMA GEMMs (ARIA_W8A8=1) */
    __half *dqbuf;            /* dequant scratch: largest weight, reused per GEMM */
    __half *preprocess, *postprocess, *project_in, *project_out;
    float *memory_tokens;
    blk_dev *blocks;
    cublasHandle_t cublas;
    /* per-request */
    int T, S, n_cond, have_req;
    float *dx, *dv, *dgcond, *rope_cos, *rope_sin;
    __half **cross_k, **cross_v, **cross_kd;   /* cached per request, stored fp16 (attn input) */
    float **local_emb; int has_local;   /* [depth] each [S, ed]; inpaint local-additive cond */
    const aria_steer_set *steer;        /* E12: host steer set (borrowed); residual dirs uploaded below */
    float **d_steer_dir; int cur_step;  /* [n_steer] device copies of each steer's [dim] direction */
    int n_steer;                        /* cached steer count: h->steer is BORROWED and the caller's
                                         * set may be gone by the time free_request runs */
    float *d_steer_scale;               /* [n_steer] device effective scales (0 out-of-window); lets
                                         * the steer kernels live INSIDE the CUDA graph */
    float *h_steer_scale;               /* pinned host staging for the per-step scale upload */
    darena arena;
    /* CUDA Graph: capture the per-step compute once (fixed device pointers + deterministic
     * arena offsets), replay per denoise step -> kills per-step kernel-launch overhead. */
    cudaGraphExec_t graph_exec; cudaGraph_t graph;
    int graph_ready, graph_failed; void *cublas_ws;
    float *pin_gc, *pin_sc, *pin_nz[2], *d_noise; size_t pin_gc_n, pin_sc_n, pin_nz_n;   /* E17.2 */
    /* A3 warm-request reuse: a host copy of the request-defining inputs. When the next
     * set_request matches (same S/T/n_cond, no local cond, identical cross_ed bytes,
     * identical steer topology+dirs), the whole teardown/rebuild -- graph re-instantiate,
     * cross-K/V reprojection, arena cudaMalloc churn -- is skipped. Benches, batch jobs
     * with a repeated prompt, and steering sweeps (scales live OUTSIDE the graph) hit. */
    float *saved_ced; int saved_ced_n;      /* [n_cond*ed] host copy, NULL = nothing saved */
    aria_steer *saved_steer_items; float *saved_steer_dirs; int saved_steer_n;
};


/* y[M,N] = x[M,K] @ W[N,K]^T + b via cuBLAS GemmEx (fp16 in, fp32 accumulate ->
 * tensor cores on sm_80+). x (fp32 activations) is converted to fp16 into transient
 * arena scratch; W is already fp16. Single default stream, so reusing the scratch
 * region after the (queued) GEMM is safe by stream ordering. */
static void gemm_f16w(cublasHandle_t cb, darena *ar, float *y, const float *x, const __half *W,
                      const float *b, int M, int K, int N) {
    size_t mark = da_save(ar);
    __half *xh = da_alloc_half(ar, (size_t)M * K);
    k_f32_to_f16<<<nblocks((size_t)M * K), THREADS>>>(xh, x, (size_t)M * K);
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                 W, CUDA_R_16F, K, xh, CUDA_R_16F, K, &beta,
                 y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (b) k_add_bias<<<nblocks((size_t)M * N), THREADS>>>(y, b, M, N);
    da_restore(ar, mark);
}
/* A2: activation already fp16 (emitted by a producer kernel) -- no conversion pass. */
static void gemm_f16w_h(cublasHandle_t cb, float *y, const __half *xh, const __half *W,
                        const float *b, int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                 W, CUDA_R_16F, K, xh, CUDA_R_16F, K, &beta,
                 y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (b) k_add_bias<<<nblocks((size_t)M * N), THREADS>>>(y, b, M, N);
}

/* ---- W8A8 (E-int8, opt-in via ARIA_W8A8=1 with --precision q8): int8 tensor-core
 * GEMMs. The Q8 weights are ALREADY packed int8 + per-row scale in VRAM; instead of
 * dequantizing them to fp16 per GEMM, quantize the activations per-token (row absmax,
 * symmetric, mirrors aria_q8_quant) and run int8 x int8 -> int32 IMMA (2x fp16 tensor
 * throughput on sm_86, and the dequant pass is gone). The int32 accumulator lands in
 * the caller's fp32 buffer and is dequantized IN PLACE (y = acc*sx[m]*sw[n] + b). */
__global__ void k_quant_act_q8(int8_t *qx, float *sx, const float *x, int M, int K) {
    int m = blockIdx.x; if (m >= M) return;
    const float *row = x + (size_t)m * K;
    __shared__ float red[RED_TH];
    float mx = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) mx = fmaxf(mx, fabsf(row[i]));
    red[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = red[0];
    if (threadIdx.x == 0) sx[m] = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int8_t *qr = qx + (size_t)m * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        int v = __float2int_rn(row[i] * inv);
        qr[i] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
    }
}
/* fp16-input variant (A2): the producers emit fp16 now; quantizing from the RTNE'd
 * half adds <=0.05% relative error on top of int8's ~1%, re-gated by the -s 1 probe. */
__global__ void k_quant_act_q8_h(int8_t *qx, float *sx, const __half *x, int M, int K) {
    int m = blockIdx.x; if (m >= M) return;
    const __half *row = x + (size_t)m * K;
    __shared__ float red[RED_TH];
    float mx = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) mx = fmaxf(mx, fabsf(__half2float(row[i])));
    red[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]); __syncthreads(); }
    float amax = red[0];
    if (threadIdx.x == 0) sx[m] = amax > 0.0f ? amax / 127.0f : 0.0f;
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int8_t *qr = qx + (size_t)m * K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        int v = __float2int_rn(__half2float(row[i]) * inv);
        qr[i] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
    }
}
/* in-place int32 -> fp32 dequant epilogue: y holds the raw int32 accumulators. */
__global__ void k_deq_i32(float *y, const float *sx, const float *sw, const float *b, int M, int N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)M * N) return;
    int m = (int)(idx / N), n = (int)(idx % N);
    int32_t acc = ((const int32_t *)y)[idx];
    y[idx] = (float)acc * sx[m] * sw[n] + (b ? b[n] : 0.0f);
}
static void gemm_w8a8(cublasHandle_t cb, darena *ar, float *y, const float *x,
                      const int8_t *Wq, const float *sw, const float *b, int M, int K, int N) {
    const int ione = 1, izero = 0;
    size_t mark = da_save(ar);
    int8_t *xq = da_alloc_i8(ar, (size_t)M * K);
    float *sx = da_alloc(ar, M);
    k_quant_act_q8<<<M, RED_TH>>>(xq, sx, x, M, K);
    cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &ione,
                 Wq, CUDA_R_8I, K, xq, CUDA_R_8I, K, &izero,
                 y, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    k_deq_i32<<<nblocks((size_t)M * N), THREADS>>>(y, sx, sw, b, M, N);
    da_restore(ar, mark);
}
static void gemm_w8a8_h(cublasHandle_t cb, darena *ar, float *y, const __half *x,
                        const int8_t *Wq, const float *sw, const float *b, int M, int K, int N) {
    const int ione = 1, izero = 0;
    size_t mark = da_save(ar);
    int8_t *xq = da_alloc_i8(ar, (size_t)M * K);
    float *sx = da_alloc(ar, M);
    k_quant_act_q8_h<<<M, RED_TH>>>(xq, sx, x, M, K);
    cublasGemmEx(cb, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &ione,
                 Wq, CUDA_R_8I, K, xq, CUDA_R_8I, K, &izero,
                 y, CUDA_R_32I, N, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    k_deq_i32<<<nblocks((size_t)M * N), THREADS>>>(y, sx, sw, b, M, N);
    da_restore(ar, mark);
}

/* GEMM with a (possibly quantized) weight: dequant Q8/Q4 into the reused fp16
 * scratch, then the fp16 tensor-core path. dt==F32 uses the fp16 weight directly.
 * Q8 + w8a8 skips the dequant entirely (int8 IMMA, see gemm_w8a8 above).
 * Stream-ordered, so reusing h->dqbuf across calls in a block is safe. */
static void gemm_dqw(aria_cuda_dit *h, float *y, const float *x, const dqw *w, const float *b, int M) {
    if (w->dt == ARIA_F32) { gemm_f16w(h->cublas, &h->arena, y, x, w->f16, b, M, w->K, w->N); return; }
    if (w->dt == ARIA_Q8 && h->w8a8) {
        gemm_w8a8(h->cublas, &h->arena, y, x, (const int8_t *)w->q, w->scale, b, M, w->K, w->N);
        return;
    }
    size_t nk = (size_t)w->N * w->K;
    if (w->dt == ARIA_Q8)
        k_dequant_q8<<<nblocks(nk), THREADS>>>(h->dqbuf, (const int8_t *)w->q, w->scale, w->N, w->K);
    else
        k_dequant_q4<<<nblocks(nk), THREADS>>>(h->dqbuf, (const uint8_t *)w->q, w->scale, w->N, w->K);
    gemm_f16w(h->cublas, &h->arena, y, x, h->dqbuf, b, M, w->K, w->N);
}
/* A2: fp16-activation variant (producers emit fp16; no k_f32_to_f16 pass). */
static void gemm_dqw_h(aria_cuda_dit *h, float *y, const __half *x, const dqw *w, const float *b, int M) {
    if (w->dt == ARIA_F32) { gemm_f16w_h(h->cublas, y, x, w->f16, b, M, w->K, w->N); return; }
    if (w->dt == ARIA_Q8 && h->w8a8) {
        gemm_w8a8_h(h->cublas, &h->arena, y, x, (const int8_t *)w->q, w->scale, b, M, w->K, w->N);
        return;
    }
    size_t nk = (size_t)w->N * w->K;
    if (w->dt == ARIA_Q8)
        k_dequant_q8<<<nblocks(nk), THREADS>>>(h->dqbuf, (const int8_t *)w->q, w->scale, w->N, w->K);
    else
        k_dequant_q4<<<nblocks(nk), THREADS>>>(h->dqbuf, (const uint8_t *)w->q, w->scale, w->N, w->K);
    gemm_f16w_h(h->cublas, y, x, h->dqbuf, b, M, w->K, w->N);
}

static void dit_block_dev(aria_cuda_dit *h, float *seq, int blk) {
    int S = h->S, ed = h->ed, H = h->H, hd = h->hd, dim = ed, inner = h->inner, Sc = h->n_cond, rot = h->rot;
    blk_dev *w = &h->blocks[blk];
    darena *ar = &h->arena; size_t mark = da_save(ar);
    int nSd = (int)nblocks((size_t)S * dim);

    float *mod = da_alloc(ar, 6 * dim);
    k_add<<<nblocks((size_t)6 * dim), THREADS>>>(mod, w->ssg, h->dgcond, 6 * dim);
    float *scale_self = mod, *shift_self = mod + dim, *gate_self = mod + 2 * dim;
    float *scale_ff = mod + 3 * dim, *shift_ff = mod + 4 * dim, *gate_ff = mod + 5 * dim;
    /* hh/merged/gated are fp16 (A2): every consumer is a tensor-core GEMM that used to
     * convert them fp32->fp16 anyway; emitting fp16 at the producer is the same RTNE
     * bits with one fewer [S,dim]-class round-trip per op. EXCEPT under W8A8: its
     * activation-quant kernel always read the producers' fp32 directly, so forcing a
     * half round-trip there is a pure fidelity loss (measured 7.3%->9.6% on the -s 1
     * probe) -- the W8A8 path keeps fp32 producers (with the same A2 bias folds).
     * No residual buffer: seq is preserved across each sub-layer (norm writes to
     * hh/hhf), so the add is in-place. */
    int w8 = h->w8a8;
    __half *hh = w8 ? NULL : da_alloc_half(ar, (size_t)S * dim);
    float *hhf = w8 ? da_alloc(ar, (size_t)S * dim) : NULL;
    float *o = da_alloc(ar, (size_t)S * dim);
    __half *merged = w8 ? NULL : da_alloc_half(ar, (size_t)S * dim);
    float *mergedf = w8 ? da_alloc(ar, (size_t)S * dim) : NULL;
    /* q/k/v are fp16 end-to-end (producers emit fp16, attn_dev_h consumes it) */
    __half *qh = da_alloc_half(ar, (size_t)H * S * hd), *kh = da_alloc_half(ar, (size_t)H * S * hd);
    __half *vh = da_alloc_half(ar, (size_t)H * S * hd);
    float *ao = da_alloc(ar, (size_t)H * S * hd);
    float *scores = da_alloc(ar, (size_t)H * S * (S > Sc ? S : Sc));   /* all heads (batched attn) */
    int diff = h->differential, nHShd = (int)nblocks((size_t)H * S * hd);
    size_t nHSd = (size_t)H * S * hd;
    __half *qdh = NULL, *kdh = NULL;               /* differential: q/k_diff */
    float *aod = NULL;                             /* differential: 2nd attn out */
    if (diff) { qdh = da_alloc_half(ar, nHSd); kdh = da_alloc_half(ar, nHSd); aod = da_alloc(ar, nHSd); }

    /* self-attention (differential medium: out = attn(q,k,v) - attn(qd,kd,v)) */
    if (w8) k_rmsnorm_adaln<<<S, RED_TH>>>(hhf, seq, w->pre_norm, scale_self, shift_self, S, dim, 1e-5f);
    else    k_rmsnorm_adaln_h<<<S, RED_TH>>>(hh, seq, w->pre_norm, scale_self, shift_self, S, dim, 1e-5f);
    int nq = diff ? 5 : 3;
    float *qkv = da_alloc(ar, (size_t)S * nq * dim);
    if (w8) gemm_dqw(h, qkv, hhf, &w->sa_to_qkv, NULL, S);
    else    gemm_dqw_h(h, qkv, hh, &w->sa_to_qkv, NULL, S);
    k_extract_normrope<<<H * S, hd>>>(qh, qkv, w->sa_q_norm, h->rope_cos, h->rope_sin, S, H, hd, nq * dim, 0, rot, 1e-6f);
    k_extract_normrope<<<H * S, hd>>>(kh, qkv, w->sa_k_norm, h->rope_cos, h->rope_sin, S, H, hd, nq * dim, dim, rot, 1e-6f);
    k_extract_heads_h<<<nHShd, THREADS>>>(vh, qkv, S, H, hd, nq * dim, 2 * dim);
    attn_dev_h(h->cublas, ar, ao, qh, kh, vh, H, S, S, hd, scores);
    if (diff) {
        k_extract_normrope<<<H * S, hd>>>(qdh, qkv, w->sa_q_norm, h->rope_cos, h->rope_sin, S, H, hd, nq * dim, 3 * dim, rot, 1e-6f);
        k_extract_normrope<<<H * S, hd>>>(kdh, qkv, w->sa_k_norm, h->rope_cos, h->rope_sin, S, H, hd, nq * dim, 4 * dim, rot, 1e-6f);
        attn_dev_h(h->cublas, ar, aod, qdh, kdh, vh, H, S, S, hd, scores);
    }
    if (w8) { if (diff) k_merge_heads_sub<<<nHShd, THREADS>>>(mergedf, ao, aod, S, H, hd);
              else      k_merge_heads<<<nHShd, THREADS>>>(mergedf, ao, S, H, hd);
              gemm_dqw(h, o, mergedf, &w->sa_to_out, NULL, S); }
    else    { if (diff) k_merge_heads_sub_h<<<nHShd, THREADS>>>(merged, ao, aod, S, H, hd);
              else      k_merge_heads_h<<<nHShd, THREADS>>>(merged, ao, S, H, hd);
              gemm_dqw_h(h, o, merged, &w->sa_to_out, NULL, S); }
    k_gate_add<<<nSd, THREADS>>>(seq, seq, o, gate_self, NULL, S, dim);

    /* cross-attention (cached fp16 K/V; differential: q_diff + cross_kd) */
    if (w8) k_rmsnorm<<<S, RED_TH>>>(hhf, seq, w->cross_norm, S, dim, 1e-5f, 0);
    else    k_rmsnorm_h<<<S, RED_TH>>>(hh, seq, w->cross_norm, S, dim, 1e-5f);
    int ncq = diff ? 2 : 1;
    float *q = da_alloc(ar, (size_t)S * ncq * dim);
    if (w8) gemm_dqw(h, q, hhf, &w->ca_to_q, NULL, S);
    else    gemm_dqw_h(h, q, hh, &w->ca_to_q, NULL, S);
    k_extract_normrope<<<H * S, hd>>>(qh, q, w->ca_q_norm, NULL, NULL, S, H, hd, ncq * dim, 0, rot, 1e-6f);
    attn_dev_h(h->cublas, ar, ao, qh, h->cross_k[blk], h->cross_v[blk], H, S, Sc, hd, scores);
    if (diff) {
        k_extract_normrope<<<H * S, hd>>>(qdh, q, w->ca_q_norm, NULL, NULL, S, H, hd, ncq * dim, dim, rot, 1e-6f);
        attn_dev_h(h->cublas, ar, aod, qdh, h->cross_kd[blk], h->cross_v[blk], H, S, Sc, hd, scores);
    }
    if (w8) { if (diff) k_merge_heads_sub<<<nHShd, THREADS>>>(mergedf, ao, aod, S, H, hd);
              else      k_merge_heads<<<nHShd, THREADS>>>(mergedf, ao, S, H, hd);
              gemm_dqw(h, o, mergedf, &w->ca_to_out, NULL, S); }
    else    { if (diff) k_merge_heads_sub_h<<<nHShd, THREADS>>>(merged, ao, aod, S, H, hd);
              else      k_merge_heads_h<<<nHShd, THREADS>>>(merged, ao, S, H, hd);
              gemm_dqw_h(h, o, merged, &w->ca_to_out, NULL, S); }
    k_add<<<nSd, THREADS>>>(seq, seq, o, (size_t)S * dim);

    /* local-additive (inpaint) cond: seq += per-block left-padded local_emb (memory rows 0) */
    if (h->has_local) k_add<<<nSd, THREADS>>>(seq, seq, h->local_emb[blk], (size_t)S * dim);

    /* feed-forward (GLU); ff biases fold into the consumer kernels (A2) so the two
     * k_add_bias passes per block disappear -- bit-identical adds, one pass less each. */
    if (w8) k_rmsnorm_adaln<<<S, RED_TH>>>(hhf, seq, w->ff_norm, scale_ff, shift_ff, S, dim, 1e-5f);
    else    k_rmsnorm_adaln_h<<<S, RED_TH>>>(hh, seq, w->ff_norm, scale_ff, shift_ff, S, dim, 1e-5f);
    float *proj = da_alloc(ar, (size_t)S * 2 * inner);
    if (w8) {
        float *gatedf = da_alloc(ar, (size_t)S * inner);
        gemm_dqw(h, proj, hhf, &w->ff_in_w, NULL, S);
        k_ff_silugate<<<nblocks((size_t)S * inner), THREADS>>>(gatedf, proj, w->ff_in_b, S, inner);
        gemm_dqw(h, o, gatedf, &w->ff_out_w, NULL, S);
    } else {
        __half *gated = da_alloc_half(ar, (size_t)S * inner);
        gemm_dqw_h(h, proj, hh, &w->ff_in_w, NULL, S);
        k_ff_silugate_h<<<nblocks((size_t)S * inner), THREADS>>>(gated, proj, w->ff_in_b, S, inner);
        gemm_dqw_h(h, o, gated, &w->ff_out_w, NULL, S);
    }
    k_gate_add<<<nSd, THREADS>>>(seq, seq, o, gate_ff, w->ff_out_b, S, dim);

    /* E12 residual steering: seq += scale*dir at the block output (matches dit_block_core /
     * sf-api AdditiveInjector). Only static per-request conditions gate the LAUNCH (site, layer,
     * uploaded dir) so the kernels are identical every step and CUDA-graph capturable; the
     * per-step step-window gating lives in d_steer_scale[s] (effective scale, 0 out-of-window,
     * refreshed by aria_cuda_dit_step before each replay; the kernels no-op on scale 0). */
    if (h->steer) for (int s = 0; s < h->steer->n; s++) {
        const aria_steer *st = &h->steer->items[s];
        if (st->site != ARIA_STEER_RESIDUAL || st->layer != blk) continue;
        if (!h->d_steer_dir[s] || !h->d_steer_scale) continue;
        int ddim = st->dim < dim ? st->dim : dim;   /* clamp to the residual width (matches CPU) */
        if (st->op == ARIA_STEER_PROJECT) {
            float inv = st->dir_norm2 > 0.0f ? 1.0f / st->dir_norm2 : 0.0f;
            k_steer_project<<<S, RED_TH>>>(seq, h->d_steer_dir[s], h->d_steer_scale + s, inv, S, dim, ddim);
        } else {
            k_steer_add<<<nSd, THREADS>>>(seq, h->d_steer_dir[s], h->d_steer_scale + s, S, dim, ddim);
        }
    }

    da_restore(ar, mark);
}

extern "C" aria_cuda_dit *aria_cuda_dit_create(const aria_sa3_dit_view *v, aria_dtype precision) {
    if (!aria_cuda_available()) return NULL;
    int ed = v->ed, inner = v->inner;
    int want_a8 = (precision == ARIA_Q8A8);   /* q8 weights + int8-activation IMMA GEMMs */
    if (want_a8) precision = ARIA_Q8;
    if (precision != ARIA_Q8 && precision != ARIA_Q4) precision = ARIA_F32;  /* fp16/bf16 -> fp16 storage */
    /* Q4 is mixed precision (mirrors aria_sa3_dit_quantize): the 4 attention
     * projections (6*ed^2/block) stay Q8 and only the FFN (3*inner*ed/block) goes
     * asymmetric Q4 -- ~9% DiT velocity error vs ~23% for uniform Q4. Rough fit:
     * attention ~1 B/elem (q8), FFN ~1 B/elem (q4: 0.5 nibble + scale, rounded up),
     * fp16 = 2 B/elem, plus the dequant scratch. */
    /* differential (medium): sa_to_qkv 5*ed^2, ca_to_q 2*ed^2 (vs 3/1) -> 9*ed^2 attn */
    size_t attn_per = v->differential ? 9 : 6;
    size_t attn_elem = (size_t)v->depth * attn_per * (size_t)ed * ed;
    size_t ffn_elem  = (size_t)v->depth * 3 * (size_t)inner * ed;
    size_t wbytes = (precision == ARIA_F32) ? (attn_elem + ffn_elem) * 2 : (attn_elem + ffn_elem);
    size_t need = wbytes + (precision != ARIA_F32 ? (size_t)2 * inner * ed * sizeof(__half) : 0) + (64u << 20);
    size_t freeb = 0, totb = 0; cudaMemGetInfo(&freeb, &totb);
    if (freeb < need) { fprintf(stderr, "aria_cuda_dit: need ~%zu MiB, only %zu free\n", need >> 20, freeb >> 20); return NULL; }

    aria_cuda_dit *h = (aria_cuda_dit *)calloc(1, sizeof(aria_cuda_dit));
    h->depth = v->depth; h->ed = ed; h->H = v->num_heads; h->hd = v->head_dim;
    h->inner = inner; h->io_ch = v->io_ch; h->n_mem = v->n_mem; h->rot = v->rot_dim;
    h->precision = precision; h->differential = v->differential;
    /* W8A8 (opt-in): int8-activation IMMA GEMMs for the Q8 block weights. Measured
     * single-step fidelity: q8-dequant 3.2% -> W8A8 7.3% velocity-class error
     * (medium; between q8 and the supported q4's 9.3%) for the fastest GPU mode. */
    h->w8a8 = (precision == ARIA_Q8) && (want_a8 || getenv("ARIA_W8A8") != NULL);
    if (h->w8a8) fprintf(stderr, "[aria] DiT: W8A8 int8 tensor-core GEMMs\n");
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { free(h); return NULL; }
    /* --default-stream per-thread makes every <<<>>> launch use the (capturable) per-thread
     * stream; pin cuBLAS to it too so GEMMs and kernels share one stream. A fixed workspace
     * stops cuBLAS lazily cudaMalloc-ing mid graph-capture (which would abort the capture). */
    cublasSetStream(h->cublas, cudaStreamPerThread);
    if (cudaMalloc(&h->cublas_ws, 4u << 20) == cudaSuccess)
        cublasSetWorkspace(h->cublas, h->cublas_ws, 4u << 20);
    if (precision != ARIA_F32) { CK(cudaMalloc(&h->dqbuf, (size_t)2 * inner * ed * sizeof(__half))); }
    int C = v->io_ch;
    h->preprocess  = upload_f16(v->preprocess,  (size_t)C * C);
    h->postprocess = upload_f16(v->postprocess, (size_t)C * C);
    h->project_in  = upload_f16(v->project_in,  (size_t)ed * C);
    h->project_out = upload_f16(v->project_out, (size_t)C * ed);
    h->memory_tokens = upload_f32(v->memory_tokens, (size_t)v->n_mem * ed);
    h->blocks = (blk_dev *)calloc(v->depth, sizeof(blk_dev));
    int ok = h->preprocess && h->postprocess && h->project_in && h->project_out && h->memory_tokens;
    for (int b = 0; b < v->depth && ok; b++) {
        const aria_dit_block_w *s = &v->blocks[b]; blk_dev *d = &h->blocks[b];
        d->pre_norm   = upload_f32(s->pre_norm, ed);
        d->cross_norm = upload_f32(s->cross_norm, ed);
        d->ff_norm    = upload_f32(s->ff_norm, ed);
        d->sa_q_norm  = upload_f32(s->sa_q_norm, h->hd);
        d->sa_k_norm  = upload_f32(s->sa_k_norm, h->hd);
        d->ca_q_norm  = upload_f32(s->ca_q_norm, h->hd);
        d->ca_k_norm  = upload_f32(s->ca_k_norm, h->hd);
        d->ssg        = upload_f32(s->to_scale_shift_gate, (size_t)6 * ed);
        d->ff_in_b    = upload_f32(s->ff_in_b, (size_t)2 * inner);
        d->ff_out_b   = upload_f32(s->ff_out_b, ed);
        /* Q4 mixed precision: attention projections stay Q8 (error-sensitive).
         * differential (medium): sa_to_qkv is [5*ed,ed], ca_to_q [2*ed,ed]. */
        aria_dtype adt = (precision == ARIA_Q4) ? ARIA_Q8 : precision;
        int nq = v->differential ? 5 : 3, ncq = v->differential ? 2 : 1, nkv = v->differential ? 3 : 2;
        int u = 1;
        u &= upload_dqw(&d->sa_to_qkv, s->sa_to_qkv, nq * ed, ed, adt);
        u &= upload_dqw(&d->sa_to_out, s->sa_to_out, ed, ed, adt);
        u &= upload_dqw(&d->ca_to_q,   s->ca_to_q,   ncq * ed, ed, adt);
        u &= upload_dqw(&d->ca_to_out, s->ca_to_out, ed, ed, adt);
        /* cross K/V projection: device-resident, projected once per request from
         * cross_ed (kept f16/f32, error-sensitive -- not quantized). */
        u &= upload_dqw(&d->ca_to_kv,  s->ca_to_kv,  nkv * ed, ed, ARIA_F32);   /* cross-K/V is fidelity-sensitive: keep fp16 */
        u &= upload_dqw(&d->ff_in_w,   s->ff_in_w,   2 * inner, ed, precision);
        u &= upload_dqw(&d->ff_out_w,  s->ff_out_w,  ed, inner, precision);
        ok = u && d->ssg && d->pre_norm && d->ca_k_norm;
    }
    if (precision != ARIA_F32 && !h->dqbuf) ok = 0;
    if (!ok) { aria_cuda_dit_free(h); return NULL; }
    return h;
}

/* free per-request device buffers (so set_request can be re-called per generation) */
static void free_request(aria_cuda_dit *h) {
    if (!h->have_req) return;
    /* the captured graph baked in this request's device pointers + (S,T) shape -> drop it
     * so the next request re-captures against its fresh arena/buffers. */
    if (h->graph_ready) cudaGraphExecDestroy(h->graph_exec);
    if (h->graph) cudaGraphDestroy(h->graph);
    h->graph = NULL; h->graph_ready = 0; h->graph_failed = 0;
    cudaFree(h->dx); cudaFree(h->dv); cudaFree(h->dgcond);
    cudaFree(h->rope_cos); cudaFree(h->rope_sin); cudaFree(h->arena.base);
    if (h->cross_k) for (int b = 0; b < h->depth; b++) cudaFree(h->cross_k[b]);
    if (h->cross_v) for (int b = 0; b < h->depth; b++) cudaFree(h->cross_v[b]);
    if (h->cross_kd) for (int b = 0; b < h->depth; b++) cudaFree(h->cross_kd[b]);
    if (h->local_emb) for (int b = 0; b < h->depth; b++) cudaFree(h->local_emb[b]);
    if (h->d_steer_dir) for (int s = 0; s < h->n_steer; s++) cudaFree(h->d_steer_dir[s]);
    free(h->cross_k); free(h->cross_v); free(h->cross_kd); free(h->local_emb); free(h->d_steer_dir);
    cudaFree(h->d_steer_scale); if (h->h_steer_scale) cudaFreeHost(h->h_steer_scale);
    h->cross_k = h->cross_v = h->cross_kd = NULL; h->local_emb = NULL; h->has_local = 0;
    h->d_steer_dir = NULL; h->steer = NULL; h->cur_step = 0; h->n_steer = 0;
    h->d_steer_scale = NULL; h->h_steer_scale = NULL;
    h->dx = h->dv = h->dgcond = NULL;
    h->rope_cos = h->rope_sin = NULL; h->arena.base = NULL; h->have_req = 0;
    free(h->saved_ced); h->saved_ced = NULL; h->saved_ced_n = 0;   /* A3 warm-hit snapshot */
    free(h->saved_steer_items); free(h->saved_steer_dirs);
    h->saved_steer_items = NULL; h->saved_steer_dirs = NULL; h->saved_steer_n = 0;
}

/* A3: does the incoming request match the resident one bit-for-bit in everything the
 * device caches/graph baked in? Shape + no local cond + identical cross_ed bytes +
 * identical steer topology AND dir values (scales are per-step device state, outside
 * the graph, so they may differ freely -- the steering-sweep case). */
static int request_compatible(const aria_cuda_dit *h, const aria_sa3_dit_req_view *rv) {
    if (!h->have_req || !h->saved_ced || !rv->cross_ed) return 0;
    if (h->S != rv->S || h->T != rv->T || h->n_cond != rv->n_cond) return 0;
    if (h->has_local || rv->has_local) return 0;
    int n_new = (rv->steer && rv->steer->n > 0) ? rv->steer->n : 0;
    if (n_new != h->saved_steer_n) return 0;
    if (memcmp(h->saved_ced, rv->cross_ed, (size_t)rv->n_cond * h->ed * sizeof(float)) != 0) return 0;
    const float *dirs = h->saved_steer_dirs;
    for (int s = 0; s < n_new; s++) {
        const aria_steer *a = &rv->steer->items[s], *b = &h->saved_steer_items[s];
        if (a->site != b->site || a->layer != b->layer || a->dim != b->dim || a->op != b->op) return 0;
        if (a->dir && b->dim > 0) {
            if (memcmp(dirs, a->dir, (size_t)a->dim * sizeof(float)) != 0) return 0;
            dirs += a->dim;
        }
    }
    return 1;
}

extern "C" void aria_cuda_dit_set_request(aria_cuda_dit *h, const aria_sa3_dit_req_view *rv) {
    if (request_compatible(h, rv)) {
        /* warm hit: keep the graph, cross-K/V cache, arena and steer-dir uploads;
         * refresh only the borrowed host pointer (the caller's old steer set may be
         * freed) and the step counter. dx/dgcond are re-uploaded per step anyway. */
        h->steer = (rv->steer && rv->steer->n > 0) ? rv->steer : NULL;
        h->cur_step = 0;
        return;
    }
    free_request(h);   /* persistent weights, fresh per-request caches */
    int ed = h->ed, C = h->io_ch, S = rv->S, T = rv->T, H = h->H, hd = h->hd, n_cond = rv->n_cond, rh = h->rot / 2;
    h->T = T; h->S = S; h->n_cond = n_cond;
    cudaMalloc(&h->dx, (size_t)C * T * sizeof(float));
    cudaMalloc(&h->dv, (size_t)C * T * sizeof(float));
    cudaMalloc(&h->dgcond, (size_t)6 * ed * sizeof(float));
    h->rope_cos = upload_f32(rv->rope_cos, (size_t)S * rh);
    h->rope_sin = upload_f32(rv->rope_sin, (size_t)S * rh);
    /* device arena: one block's peak + step buffers (mirrors the CPU sizing;
     * differential needs ~32*S*ed for the extra 5-way qkv + q/k_diff + 2nd attn).
     * Allocated before the cross-K/V projection, which borrows it for GEMM scratch. */
    /* +S*ed + H*S*max(S,n_cond): fp16 q/k/v + scores scratch for tensor-core attn_dev.
     * +6*n_cond*ed: cross-K/V projection scratch (kv [n_cond,nkv*ed] + 3 fp32 temps
     * before the fp16 cache), borrowed from this arena below. */
    size_t block_floats = (size_t)6 * ed + (h->differential ? 32 : 24) * (size_t)S * ed
                          + 2 * (size_t)H * S * (S > n_cond ? S : n_cond) + 2 * (size_t)S * ed
                          + 6 * (size_t)n_cond * ed
                          + (h->w8a8 ? ((size_t)S * h->inner) / 4 + S + 64 : 0);  /* int8 act + sx */
    size_t step_floats = (size_t)S * ed + 8 * (size_t)T * C + 8 * (size_t)ed;
    h->arena.cap = (block_floats + step_floats + (1u << 20)) * sizeof(float);
    h->arena.used = 0;
    cudaMalloc(&h->arena.base, h->arena.cap);

    int nkv = h->differential ? 3 : 2;
    size_t khv = (size_t)H * n_cond * hd;
    h->cross_k = (__half **)calloc(h->depth, sizeof(__half *));
    h->cross_v = (__half **)calloc(h->depth, sizeof(__half *));
    h->cross_kd = h->differential ? (__half **)calloc(h->depth, sizeof(__half *)) : NULL;
    if (rv->cross_ed) {
        /* project cross K/V on the device from cross_ed [n_cond, ed], once per
         * request (was a per-request CPU GEMM + 24-block upload before). Cached as
         * fp16 (the attention input dtype): converting once here produces the same
         * RTNE bits attn_dev used to produce per step, 2*depth*steps passes less. */
        float *ced = upload_f32(rv->cross_ed, (size_t)n_cond * ed);
        for (int b = 0; b < h->depth; b++) {
            blk_dev *w = &h->blocks[b];
            cudaMalloc(&h->cross_k[b], khv * sizeof(__half));
            cudaMalloc(&h->cross_v[b], khv * sizeof(__half));
            if (h->cross_kd) cudaMalloc(&h->cross_kd[b], khv * sizeof(__half));
            size_t mark = da_save(&h->arena);
            float *kv = da_alloc(&h->arena, (size_t)n_cond * nkv * ed);     /* [n_cond, nkv*ed] */
            float *tk = da_alloc(&h->arena, khv), *tv = da_alloc(&h->arena, khv);
            float *tkd = h->cross_kd ? da_alloc(&h->arena, khv) : NULL;
            gemm_dqw(h, kv, ced, &w->ca_to_kv, NULL, n_cond);
            k_extract_heads<<<nblocks(khv), THREADS>>>(tk, kv, n_cond, H, hd, nkv * ed, 0);
            if (h->cross_kd) {   /* chunk order: k, k_diff, v */
                k_extract_heads<<<nblocks(khv), THREADS>>>(tkd, kv, n_cond, H, hd, nkv * ed, ed);
                k_extract_heads<<<nblocks(khv), THREADS>>>(tv,  kv, n_cond, H, hd, nkv * ed, 2 * ed);
                k_rmsnorm<<<H * n_cond, RED_TH>>>(tkd, tkd, w->ca_k_norm, H * n_cond, hd, 1e-6f, 0);
                k_f32_to_f16<<<nblocks(khv), THREADS>>>(h->cross_kd[b], tkd, khv);
            } else {
                k_extract_heads<<<nblocks(khv), THREADS>>>(tv, kv, n_cond, H, hd, nkv * ed, ed);
            }
            k_rmsnorm<<<H * n_cond, RED_TH>>>(tk, tk, w->ca_k_norm, H * n_cond, hd, 1e-6f, 0);
            k_f32_to_f16<<<nblocks(khv), THREADS>>>(h->cross_k[b], tk, khv);
            k_f32_to_f16<<<nblocks(khv), THREADS>>>(h->cross_v[b], tv, khv);
            da_restore(&h->arena, mark);
        }
        cudaFree(ced);
    } else {
        for (int b = 0; b < h->depth; b++) {   /* legacy: CPU-projected K/V uploaded */
            h->cross_k[b] = upload_f16(rv->cross_k[b], khv);
            h->cross_v[b] = upload_f16(rv->cross_v[b], khv);
            if (h->cross_kd) h->cross_kd[b] = upload_f16(rv->cross_kd[b], khv);
        }
    }
    /* continue/inpaint: upload the per-block, CPU-projected local-additive cond [S, ed]
     * (memory-token rows already zero). Added to the residual after cross-attn per block. */
    h->has_local = rv->has_local;
    if (rv->has_local) {
        h->local_emb = (float **)calloc(h->depth, sizeof(float *));
        for (int b = 0; b < h->depth; b++)
            h->local_emb[b] = upload_f32(rv->local_emb[b], (size_t)S * ed);
    }
    /* E12 steering: keep the host steer set and upload each residual direction to the device.
     * The steer kernels are recorded INTO the CUDA graph; per-step window gating happens via
     * d_steer_scale (effective scale, 0 out-of-window) refreshed before every replay, so the
     * graph stays captured while steering. */
    h->steer = (rv->steer && rv->steer->n > 0) ? rv->steer : NULL;
    if (h->steer) {
        h->n_steer = h->steer->n;   /* cached: the borrowed set may be gone by free_request time */
        h->d_steer_dir = (float **)calloc(h->n_steer, sizeof(float *));
        for (int s = 0; s < h->n_steer; s++) {
            const aria_steer *st = &h->steer->items[s];
            if (st->site == ARIA_STEER_RESIDUAL && st->dir && st->dim > 0)
                h->d_steer_dir[s] = upload_f32(st->dir, (size_t)st->dim);
        }
        if (cudaMalloc(&h->d_steer_scale, (size_t)h->n_steer * sizeof(float)) == cudaSuccess) {
            if (cudaMallocHost(&h->h_steer_scale, (size_t)h->n_steer * sizeof(float)) != cudaSuccess) {
                /* both or neither: a scale buffer without its staging would leave the kernels
                 * reading uninitialized device memory. NULL -> steer launches skipped. */
                cudaFree(h->d_steer_scale); h->d_steer_scale = NULL; h->h_steer_scale = NULL;
            }
        } else h->d_steer_scale = NULL;
    }
    /* A3: snapshot the request-defining inputs so the NEXT set_request can detect a
     * warm hit (request_compatible) and skip this whole teardown/rebuild. */
    if (rv->cross_ed && !rv->has_local) {
        h->saved_ced_n = n_cond * ed;
        h->saved_ced = (float *)malloc((size_t)h->saved_ced_n * sizeof(float));
        if (h->saved_ced) memcpy(h->saved_ced, rv->cross_ed, (size_t)h->saved_ced_n * sizeof(float));
        h->saved_steer_n = h->steer ? h->steer->n : 0;
        if (h->saved_steer_n > 0) {
            size_t dtot = 0;
            for (int s = 0; s < h->saved_steer_n; s++)
                if (h->steer->items[s].dir) dtot += (size_t)h->steer->items[s].dim;
            h->saved_steer_items = (aria_steer *)malloc((size_t)h->saved_steer_n * sizeof(aria_steer));
            h->saved_steer_dirs = dtot ? (float *)malloc(dtot * sizeof(float)) : NULL;
            float *dp = h->saved_steer_dirs;
            for (int s = 0; s < h->saved_steer_n && h->saved_steer_items; s++) {
                h->saved_steer_items[s] = h->steer->items[s];
                if (h->steer->items[s].dir && dp) {
                    memcpy(dp, h->steer->items[s].dir, (size_t)h->steer->items[s].dim * sizeof(float));
                    dp += h->steer->items[s].dim;
                }
            }
        }
    }
    h->have_req = 1;
}

extern "C" void aria_cuda_dit_set_step(aria_cuda_dit *h, int step) { h->cur_step = step; }

/* the per-step compute (reads h->dx + h->dgcond, writes h->dv). Pure async work on the
 * per-thread stream + a deterministic arena, so it can be CUDA-graph captured and replayed.
 * The D2D memory-token copy is async (a sync copy would abort a stream capture). */
static void dit_step_compute(aria_cuda_dit *h) {
    int C = h->io_ch, ed = h->ed, Mt = h->n_mem, S = h->S, T = h->T;
    darena *ar = &h->arena; size_t mark = da_save(ar);

    float *xtc = da_alloc(ar, (size_t)T * C);
    k_transpose<<<nblocks((size_t)C * T), THREADS>>>(xtc, h->dx, C, T);   /* [C,T]->[T,C] */
    float *pre = da_alloc(ar, (size_t)T * C);
    gemm_f16w(h->cublas, &h->arena, pre, xtc, h->preprocess, NULL, T, C, C);
    k_add<<<nblocks((size_t)T * C), THREADS>>>(xtc, xtc, pre, (size_t)T * C);

    float *seq = da_alloc(ar, (size_t)S * ed);
    cudaMemcpyAsync(seq, h->memory_tokens, (size_t)Mt * ed * sizeof(float), cudaMemcpyDeviceToDevice, cudaStreamPerThread);
    gemm_f16w(h->cublas, &h->arena, seq + (size_t)Mt * ed, xtc, h->project_in, NULL, T, C, ed);

    for (int b = 0; b < h->depth; b++) dit_block_dev(h, seq, b);

    float *outtc = da_alloc(ar, (size_t)T * C);
    gemm_f16w(h->cublas, &h->arena, outtc, seq + (size_t)Mt * ed, h->project_out, NULL, T, ed, C);
    float *postc = da_alloc(ar, (size_t)T * C);
    gemm_f16w(h->cublas, &h->arena, postc, outtc, h->postprocess, NULL, T, C, C);
    k_add<<<nblocks((size_t)T * C), THREADS>>>(outtc, outtc, postc, (size_t)T * C);
    k_transpose<<<nblocks((size_t)T * C), THREADS>>>(h->dv, outtc, T, C);  /* [T,C]->[C,T] */

    da_restore(ar, mark);
}

/* run one DiT step on h->dx/h->dgcond -> h->dv: one graph per (S,T) shape, captured on
 * the first step and replayed after (inline when capture is off/failed). */
static void dit_run(aria_cuda_dit *h, cudaStream_t st) {
    /* ARIA_NO_GRAPH=1 forces the inline path (A/B the graph without rebuilding) */
    if (!h->graph_ready && !h->graph_failed) {
        static int no_graph = -1;
        if (no_graph < 0) { const char *e = getenv("ARIA_NO_GRAPH"); no_graph = e && e[0] && e[0] != '0'; }
        if (no_graph) h->graph_failed = 1;
    }
    /* the H2D before each run refreshes h->dx/h->dgcond; the graph reads them at fixed pointers */
    if (h->graph_ready) {
        cudaGraphLaunch(h->graph_exec, st);
    } else if (h->graph_failed) {
        dit_step_compute(h);
    } else {
        cudaStreamSynchronize(st);   /* drain any set_request work before capturing */
        cudaGraph_t g = NULL;
        if (cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            cudaGetLastError(); h->graph_failed = 1; dit_step_compute(h);
        } else {
            dit_step_compute(h);     /* recorded, not executed */
            /* cudaGraphInstantiateWithFlags needs CUDA >= 11.4; older toolkits (dev box 11.2)
             * use the 5-arg cudaGraphInstantiate. Same semantics with flags=0. */
#if CUDART_VERSION >= 11040
            #define ARIA_GRAPH_INST(e, gr) cudaGraphInstantiateWithFlags((e), (gr), 0)
#else
            #define ARIA_GRAPH_INST(e, gr) cudaGraphInstantiate((e), (gr), NULL, NULL, 0)
#endif
            if (cudaStreamEndCapture(st, &g) == cudaSuccess && g &&
                ARIA_GRAPH_INST(&h->graph_exec, g) == cudaSuccess) {
                h->graph = g; h->graph_ready = 1;
                fprintf(stderr, "[aria] DiT: CUDA graph captured (%d blocks/step replayed)\n", h->depth);
                cudaGraphLaunch(h->graph_exec, st);   /* execute step 0 (capture didn't run it) */
            } else {
                cudaGetLastError(); h->graph_failed = 1;
                if (g) cudaGraphDestroy(g);
                dit_step_compute(h);  /* capture failed -> run inline (correctness preserved) */
            }
        }
    }
}
/* E12: per-steer effective scale at `step` (0 outside its window) */
static float steer_scale_at(const aria_cuda_dit *h, int s, int step) {
    const aria_steer *it = &h->steer->items[s];
    return (step >= it->step_lo && step <= it->step_hi) ? it->scale : 0.0f;
}

extern "C" void aria_cuda_dit_step(aria_cuda_dit *h, float *v_CT, const float *x_CT, const float *gcond) {
    int C = h->io_ch, ed = h->ed, T = h->T;
    cudaStream_t st = cudaStreamPerThread;
    cudaMemcpyAsync(h->dx, x_CT, (size_t)C * T * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(h->dgcond, gcond, (size_t)6 * ed * sizeof(float), cudaMemcpyHostToDevice, st);
    /* E12: refresh per-steer effective scales for this step (0 outside the step window) so the
     * graph-resident steer kernels see the right gating. Same fixed-pointer pattern as dx/dgcond. */
    if (h->steer && h->d_steer_scale && h->h_steer_scale) {
        for (int s = 0; s < h->steer->n; s++) h->h_steer_scale[s] = steer_scale_at(h, s, h->cur_step);
        cudaMemcpyAsync(h->d_steer_scale, h->h_steer_scale,
                        (size_t)h->steer->n * sizeof(float), cudaMemcpyHostToDevice, st);
    }
    dit_run(h, st);
    cudaMemcpyAsync(v_CT, h->dv, (size_t)C * T * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
}

/* one pingpong update, bit-identical to the host loop (aria_sampler.c, built without FMA
 * contraction): d = x - tc*v; x = (1 - tn)*d + tn*noise, every op rounded on its own. */
__global__ void k_pingpong(float *x, const float *v, const float *nz, float tc, float tn, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = __fsub_rn(x[i], __fmul_rn(tc, v[i]));
    x[i] = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, tn), d), __fmul_rn(tn, nz[i]));
}
static int pin_grow(float **p, size_t *cap, size_t n) {
    if (n <= *cap) return 0;
    if (*p) cudaFreeHost(*p);
    *p = NULL; *cap = 0;
    if (cudaMallocHost((void **)p, n * sizeof(float)) != cudaSuccess) { cudaGetLastError(); *p = NULL; return -1; }
    *cap = n;
    return 0;
}

/* E17.2 pipelined denoise: the whole pingpong loop with x resident on the device. Per step
 * the host only enqueues -- gcond (precomputed by the caller for every step), the step's
 * steer scales and its noise come from pinned buffers, the noise drawn on the host while the
 * GPU runs that step's DiT (same RNG order as the host loop) -- so the GPU never idles on a
 * host round-trip. Bit-identical to stepping aria_cuda_dit_step + the host pingpong. x_CT:
 * in = initial noise, out = final latent. Returns -1 (nothing run) if staging can't be pinned. */
extern "C" int aria_cuda_dit_sample(aria_cuda_dit *h, float *x_CT, const float *sched, int steps,
                                    const float *gcond_all, void (*noise)(float *, int, void *), void *noise_user,
                                    void (*progress)(int, int, void *), void *progress_user) {
    int C = h->io_ch, ed = h->ed, T = h->T;
    size_t n = (size_t)C * T, ng = (size_t)6 * ed;
    int ns = (h->steer && h->d_steer_scale) ? h->steer->n : 0;
    cudaStream_t st = cudaStreamPerThread;
    if (pin_grow(&h->pin_gc, &h->pin_gc_n, (size_t)steps * ng) ||
        (ns && pin_grow(&h->pin_sc, &h->pin_sc_n, (size_t)steps * ns)))
        return -1;
    if (h->pin_nz_n < n) {   /* two pinned noise buffers + the device one, grown together */
        for (int b = 0; b < 2; b++) { if (h->pin_nz[b]) cudaFreeHost(h->pin_nz[b]); h->pin_nz[b] = NULL; }
        cudaFree(h->d_noise); h->d_noise = NULL; h->pin_nz_n = 0;
        if (cudaMallocHost((void **)&h->pin_nz[0], n * sizeof(float)) != cudaSuccess ||
            cudaMallocHost((void **)&h->pin_nz[1], n * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&h->d_noise, n * sizeof(float)) != cudaSuccess) { cudaGetLastError(); return -1; }
        h->pin_nz_n = n;
    }
    memcpy(h->pin_gc, gcond_all, (size_t)steps * ng * sizeof(float));
    for (int i = 0; i < steps; i++)
        for (int s = 0; s < ns; s++) h->pin_sc[(size_t)i * ns + s] = steer_scale_at(h, s, i);
    cudaEvent_t ev_nz[2], ev_step;
    cudaEventCreateWithFlags(&ev_nz[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev_nz[1], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&ev_step, cudaEventDisableTiming);
    cudaMemcpyAsync(h->dx, x_CT, n * sizeof(float), cudaMemcpyHostToDevice, st);
    for (int i = 0; i < steps; i++) {
        cudaMemcpyAsync(h->dgcond, h->pin_gc + (size_t)i * ng, ng * sizeof(float), cudaMemcpyHostToDevice, st);
        if (ns) cudaMemcpyAsync(h->d_steer_scale, h->pin_sc + (size_t)i * ns, (size_t)ns * sizeof(float),
                                cudaMemcpyHostToDevice, st);
        dit_run(h, st);
        int b = i & 1;
        if (i >= 2) cudaEventSynchronize(ev_nz[b]);   /* step i-2's upload from this buffer is done */
        noise(h->pin_nz[b], (int)n, noise_user);       /* overlaps the DiT just enqueued */
        cudaMemcpyAsync(h->d_noise, h->pin_nz[b], n * sizeof(float), cudaMemcpyHostToDevice, st);
        cudaEventRecord(ev_nz[b], st);
        k_pingpong<<<nblocks(n), THREADS, 0, st>>>(h->dx, h->dv, h->d_noise, sched[i], sched[i + 1], n);
        if (progress) {   /* report a step once it is done; the host stays ~1 step ahead */
            cudaEventRecord(ev_step, st); cudaEventSynchronize(ev_step);
            progress(i + 1, steps, progress_user);
        }
    }
    cudaMemcpyAsync(x_CT, h->dx, n * sizeof(float), cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    cudaEventDestroy(ev_nz[0]); cudaEventDestroy(ev_nz[1]); cudaEventDestroy(ev_step);
    return 0;
}

extern "C" void aria_cuda_dit_free(aria_cuda_dit *h) {
    if (!h) return;
    if (h->pin_gc) cudaFreeHost(h->pin_gc);
    if (h->pin_sc) cudaFreeHost(h->pin_sc);
    if (h->pin_nz[0]) cudaFreeHost(h->pin_nz[0]);
    if (h->pin_nz[1]) cudaFreeHost(h->pin_nz[1]);
    cudaFree(h->d_noise);
    if (h->cublas) cublasDestroy(h->cublas);
    cudaFree(h->cublas_ws);
    cudaFree(h->dqbuf);
    cudaFree(h->preprocess); cudaFree(h->postprocess); cudaFree(h->project_in); cudaFree(h->project_out);
    cudaFree(h->memory_tokens);
    if (h->blocks) for (int b = 0; b < h->depth; b++) {
        blk_dev *d = &h->blocks[b];
        cudaFree((void *)d->pre_norm); cudaFree((void *)d->cross_norm); cudaFree((void *)d->ff_norm);
        cudaFree((void *)d->sa_q_norm); cudaFree((void *)d->sa_k_norm); cudaFree((void *)d->ca_q_norm);
        cudaFree((void *)d->ca_k_norm);
        cudaFree((void *)d->ssg); cudaFree((void *)d->ff_in_b); cudaFree((void *)d->ff_out_b);
        free_dqw(&d->sa_to_qkv); free_dqw(&d->sa_to_out); free_dqw(&d->ca_to_q);
        free_dqw(&d->ca_to_out); free_dqw(&d->ca_to_kv); free_dqw(&d->ff_in_w); free_dqw(&d->ff_out_w);
    }
    free(h->blocks);
    free_request(h);
    free(h);
}

/* ===================== device-resident taae decoder (E8.5c) =====================
 * Mirrors aria_sa3_dec.c, but batches all S=34 chunks of a pass into one workload:
 * the per-token linears become [B*S, .] GEMMs, and the per-chunk differential
 * attention runs B*H batched. Weights fp16 (GEMMs) / fp32 (norms, conv, tokens). */

#define DC_D   768
#define DC_H   12
#define DC_HD  64
#define DC_ROT 32
#define DC_INNER 2304
#define DC_QKV 3840   /* 5*768 */
#define DC_S   34
#define DC_OUT 512

/* (dynamic-tanh kernel k_dyt is defined above, shared with aria_cuda_dynamic_tanh) */
/* [B*H,S,hd] from src[B*S, stride] at offset off (per chunk b, head h, pos s) */
__global__ void k_extract_heads_b(float *dst, const float *src, int B, int S, int H, int hd, int stride, int off) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)((idx / ((size_t)hd * S)) % H);
    int b = (int)(idx / ((size_t)hd * S * H));
    dst[idx] = src[(size_t)(b * S + s) * stride + off + (size_t)h * hd + d];
}
/* [B*S, H*hd] from src[B*H,S,hd] */
__global__ void k_merge_heads_b(float *dst, const float *src, int B, int S, int H, int hd) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)B * H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)((idx / ((size_t)hd * S)) % H);
    int b = (int)(idx / ((size_t)hd * S * H));
    dst[(size_t)(b * S + s) * (H * hd) + (size_t)h * hd + d] = src[idx];
}
/* fp16-emitting variant (A2): dst feeds the to_out GEMM directly. */
__global__ void k_merge_heads_b_sub_h(__half *dst, const float *a, const float *b, int B, int S, int H, int hd) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;   /* fused a -= b + merge (exact) */
    if (idx >= (size_t)B * H * S * hd) return;
    int d = (int)(idx % hd), s = (int)((idx / hd) % S), h = (int)((idx / ((size_t)hd * S)) % H);
    int c = (int)(idx / ((size_t)hd * S * H));
    float v = a[idx] - b[idx];
    dst[(size_t)(c * S + s) * (H * hd) + (size_t)h * hd + d] = __float2half(v);
}
/* expand x[T,D] -> seq[Tp*17, D]: token t at row t*17 (zeros if padded), new_tokens at the next 16 */
__global__ void k_expand_tokens(float *seq, const float *x, const float *new_tokens, int T, int Tp, int D) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)Tp * 17 * D) return;
    int d = (int)(idx % D), p = (int)(idx / D), t = p / 17, pos = p % 17;
    if (pos == 0) seq[idx] = (t < T) ? x[(size_t)t * D + d] : 0.0f;
    else          seq[idx] = new_tokens[d];
}
/* feat[D, Lout] (channel-major) = last 16 of each 17-group: feat[d,t*16+j] = seq[(t*17+1+j)*D+d] */
__global__ void k_extract_feat(float *feat, const float *seq, int D, int Lout) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)D * Lout) return;
    int d = (int)(idx / Lout), col = (int)(idx % Lout), t = col / 16, j = col % 16;
    feat[idx] = seq[(size_t)(t * 17 + 1 + j) * D + d];
}
/* unpatch: audio[2, L*256] from dec[512, L]: audio[c, l*256+hh] = dec[c*256+hh, l] */
__global__ void k_unpatch(float *audio, const float *dec, int L) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)2 * L * 256) return;
    int c = (int)(idx / ((size_t)L * 256));
    size_t rem = idx % ((size_t)L * 256);
    int l = (int)(rem / 256), hh = (int)(rem % 256);
    audio[idx] = dec[(size_t)(c * 256 + hh) * L + l];
}

struct dec_blk_dev {
    float pre_alpha, qn_alpha, kn_alpha, ff_alpha;
    const float *pre_gamma, *pre_beta, *qn_gamma, *qn_beta, *kn_gamma, *kn_beta, *ff_gamma, *ff_beta;
    const float *ff_in_b, *ff_out_b;
    __half *to_qkv, *to_out, *ff_in_w, *ff_out_w;
};
struct aria_cuda_dec {
    float running_std;
    __half *proj_w; const float *proj_b, *new_tokens, *mapping_b;
    float *mapping_w;       /* fp32, repacked as 3 contiguous [DC_OUT, DC_D] GEMM taps */
    dec_blk_dev blocks[6];
    float *rcos, *rsin;
    cublasHandle_t cublas;
    darena ar_cache;        /* decode scratch, kept across calls */
};

/* one taae block on seq[B*S, D] (B chunks of S=34), in place, scratch from `ar` */
/* Fused small-decoder chunk attention: one block per (chunk, head) does the q/k/v/qd/kd
 * extract, the q/k DyT norms and RoPE (k_dyt / k_rope expressions), both differential
 * branches in fp32 (S = QK^T * scale, softmax, O = PV) and writes half(o - o_diff) into
 * the merged [N,768] to_out input -- replacing 13 glue launches, 4 batched SGEMMs (128x128
 * tiles for 34x34 problems), 2 softmaxes and the merge. fp32 throughout; only the dot
 * order differs from cuBLAS, so parity-gated (not byte-identical). ARIA_DEC_ATTN_GEMM=1
 * keeps the cuBLAS path for A/B. */
#define CA_T 256
__global__ void k_chunk_attn_diff(__half *merged, const float *qkv,
                                  float qa, const float *qg, const float *qbt,
                                  float ka, const float *kg, const float *kbt,
                                  const float *rcos, const float *rsin, int H, float scale) {
    const int S = DC_S, HD = DC_HD, RH = DC_ROT / 2;
    int b = blockIdx.x / H, h = blockIdx.x % H, t = threadIdx.x;
    __shared__ float sq[DC_S][DC_HD + 1], sk[DC_S][DC_HD + 1], sv[DC_S][DC_HD], so[DC_S][DC_HD];
    __shared__ float ss[DC_S][DC_S];
    const float *base = qkv + (size_t)b * S * DC_QKV + (size_t)h * HD;
    for (int i = t; i < S * HD; i += CA_T) { int r = i / HD, d = i % HD; sv[r][d] = base[(size_t)r * DC_QKV + 1536 + d]; }
    for (int br = 0; br < 2; br++) {
        int qoff = br ? 2304 : 0, koff = br ? 3072 : 768;
        for (int i = t; i < S * HD; i += CA_T) {
            int r = i / HD, d = i % HD;
            sq[r][d] = tanhf(qa * base[(size_t)r * DC_QKV + qoff + d]) * qg[d] + qbt[d];
            sk[r][d] = tanhf(ka * base[(size_t)r * DC_QKV + koff + d]) * kg[d] + kbt[d];
        }
        __syncthreads();
        for (int i = t; i < 2 * S * RH; i += CA_T) {   /* rope: rows of q then k, pairs (d, d+RH) */
            int isk = i >= S * RH, j = isk ? i - S * RH : i, r = j / RH, d = j % RH;
            float (*x)[DC_HD + 1] = isk ? sk : sq;
            float a = x[r][d], bb = x[r][d + RH], cc = rcos[r * RH + d], sn = rsin[r * RH + d];
            x[r][d] = a * cc - bb * sn;
            x[r][d + RH] = bb * cc + a * sn;
        }
        __syncthreads();
        for (int i = t; i < S * S; i += CA_T) {
            int r = i / S, c = i % S; float acc = 0.0f;
            for (int d = 0; d < HD; d++) acc += sq[r][d] * sk[c][d];
            ss[r][c] = acc * scale;
        }
        __syncthreads();
        for (int r = t >> 5; r < S; r += CA_T / 32) {   /* warp per row softmax */
            int lane = t & 31;
            float mx = -INFINITY;
            for (int c = lane; c < S; c += 32) mx = fmaxf(mx, ss[r][c]);
            for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            float sum = 0.0f;
            for (int c = lane; c < S; c += 32) { float e = expf(ss[r][c] - mx); ss[r][c] = e; sum += e; }
            for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            for (int c = lane; c < S; c += 32) ss[r][c] *= inv;
        }
        __syncthreads();
        for (int i = t; i < S * HD; i += CA_T) {
            int r = i / HD, d = i % HD; float acc = 0.0f;
            for (int c = 0; c < S; c++) acc += ss[r][c] * sv[c][d];
            if (br == 0) so[r][d] = acc; else so[r][d] -= acc;   /* differential combine, fp32 */
        }
        __syncthreads();
    }
    for (int i = t; i < S * HD; i += CA_T) {
        int r = i / HD, d = i % HD;
        merged[((size_t)b * S + r) * DC_D + (size_t)h * HD + d] = __float2half(so[r][d]);
    }
}

static void taae_block_dev(aria_cuda_dec *h, float *seq, int B, const dec_blk_dev *w, darena *ar) {
    int S = DC_S, D = DC_D, H = DC_H, hd = DC_HD;
    int N = B * S, NH = B * H * S * hd;
    size_t mark = da_save(ar);
    float *q  = da_alloc(ar, (size_t)NH), *k  = da_alloc(ar, (size_t)NH), *v = da_alloc(ar, (size_t)NH);
    float *qd = da_alloc(ar, (size_t)NH), *kd = da_alloc(ar, (size_t)NH);
    float *ob = da_alloc(ar, (size_t)NH), *od = da_alloc(ar, (size_t)NH);
    float *o = da_alloc(ar, (size_t)N * D);
    float *scores = da_alloc(ar, (size_t)B * H * S * S);

    /* differential self-attention (hh/merged/gated fp16: GEMM-input producers, A2) */
    __half *hhh = da_alloc_half(ar, (size_t)N * D);
    __half *mergedh = da_alloc_half(ar, (size_t)N * D);
    k_dyt_h<<<nblocks((size_t)N * D), THREADS>>>(hhh, seq, w->pre_alpha, w->pre_gamma, w->pre_beta, (size_t)N * D, D);
    float *qkv = da_alloc(ar, (size_t)N * DC_QKV);
    gemm_f16w_h(h->cublas, qkv, hhh, w->to_qkv, NULL, N, D, DC_QKV);
    static int dec_attn_gemm = -1;
    if (dec_attn_gemm < 0) { const char *e = getenv("ARIA_DEC_ATTN_GEMM"); dec_attn_gemm = e && e[0] && e[0] != '0'; }
    if (!dec_attn_gemm) {
        k_chunk_attn_diff<<<B * H, CA_T>>>(mergedh, qkv, w->qn_alpha, w->qn_gamma, w->qn_beta,
                                           w->kn_alpha, w->kn_gamma, w->kn_beta, h->rcos, h->rsin,
                                           H, 1.0f / sqrtf((float)hd));
    } else {
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(q,  qkv, B, S, H, hd, DC_QKV, 0);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(k,  qkv, B, S, H, hd, DC_QKV, 768);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(v,  qkv, B, S, H, hd, DC_QKV, 1536);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(qd, qkv, B, S, H, hd, DC_QKV, 2304);
    k_extract_heads_b<<<nblocks((size_t)NH), THREADS>>>(kd, qkv, B, S, H, hd, DC_QKV, 3072);
    size_t bhs = (size_t)B * H * S;
    k_dyt<<<nblocks((size_t)NH), THREADS>>>(q,  q,  w->qn_alpha, w->qn_gamma, w->qn_beta, (size_t)NH, hd);
    k_dyt<<<nblocks((size_t)NH), THREADS>>>(qd, qd, w->qn_alpha, w->qn_gamma, w->qn_beta, (size_t)NH, hd);
    k_dyt<<<nblocks((size_t)NH), THREADS>>>(k,  k,  w->kn_alpha, w->kn_gamma, w->kn_beta, (size_t)NH, hd);
    k_dyt<<<nblocks((size_t)NH), THREADS>>>(kd, kd, w->kn_alpha, w->kn_gamma, w->kn_beta, (size_t)NH, hd);
    k_rope<<<nblocks(bhs), THREADS>>>(q,  h->rcos, h->rsin, B * H, S, hd, DC_ROT);
    k_rope<<<nblocks(bhs), THREADS>>>(qd, h->rcos, h->rsin, B * H, S, hd, DC_ROT);
    k_rope<<<nblocks(bhs), THREADS>>>(k,  h->rcos, h->rsin, B * H, S, hd, DC_ROT);
    k_rope<<<nblocks(bhs), THREADS>>>(kd, h->rcos, h->rsin, B * H, S, hd, DC_ROT);
    attn_dev(h->cublas, ar, ob, q,  k,  v, B * H, S, S, hd, scores);
    attn_dev(h->cublas, ar, od, qd, kd, v, B * H, S, S, hd, scores);
    k_merge_heads_b_sub_h<<<nblocks((size_t)NH), THREADS>>>(mergedh, ob, od, B, S, H, hd);
    }
    gemm_f16w_h(h->cublas, o, mergedh, w->to_out, NULL, N, D, D);
    k_add<<<nblocks((size_t)N * D), THREADS>>>(seq, seq, o, (size_t)N * D);   /* residual in place */

    /* SwiGLU feed-forward (ff biases folded into the consumer kernels, A2) */
    k_dyt_h<<<nblocks((size_t)N * D), THREADS>>>(hhh, seq, w->ff_alpha, w->ff_gamma, w->ff_beta, (size_t)N * D, D);
    float *proj = da_alloc(ar, (size_t)N * 2 * DC_INNER);
    __half *gated = da_alloc_half(ar, (size_t)N * DC_INNER);
    gemm_f16w_h(h->cublas, proj, hhh, w->ff_in_w, NULL, N, D, 2 * DC_INNER);
    k_ff_silugate_h<<<nblocks((size_t)N * DC_INNER), THREADS>>>(gated, proj, w->ff_in_b, N, DC_INNER);
    gemm_f16w_h(h->cublas, o, gated, w->ff_out_w, NULL, N, DC_INNER, D);
    k_add_bias_res<<<nblocks((size_t)N * D), THREADS>>>(seq, seq, o, w->ff_out_b, (size_t)N * D, D);

    da_restore(ar, mark);
}

/* run 3 blocks over seq[L,768] in S-chunks; shift=1 adds the midpoint halo */
static void taae_pass_dev(aria_cuda_dec *h, float *seq, int L, const dec_blk_dev *blocks, int shift, darena *ar) {
    int S = DC_S, D = DC_D, half = S / 2;
    if (!shift) {
        int B = L / S;
        for (int b = 0; b < 3; b++) taae_block_dev(h, seq, B, &blocks[b], ar);
    } else {
        int Lp = L + S, B = Lp / S;
        size_t mark = da_save(ar);
        float *pad = da_alloc(ar, (size_t)Lp * D);
        cudaMemcpy(pad, seq, (size_t)half * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pad + (size_t)half * D, seq, (size_t)L * D * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pad + (size_t)(half + L) * D, seq + (size_t)(L - half) * D, (size_t)half * D * sizeof(float), cudaMemcpyDeviceToDevice);
        for (int b = 0; b < 3; b++) taae_block_dev(h, pad, B, &blocks[b], ar);
        cudaMemcpy(seq, pad + (size_t)half * D, (size_t)L * D * sizeof(float), cudaMemcpyDeviceToDevice);
        da_restore(ar, mark);
    }
}

extern "C" aria_cuda_dec *aria_cuda_dec_create(const aria_sa3_dec_view *v) {
    if (!aria_cuda_available()) return NULL;
    aria_cuda_dec *h = (aria_cuda_dec *)calloc(1, sizeof(aria_cuda_dec));
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { free(h); return NULL; }
    cublasSetStream(h->cublas, cudaStreamPerThread);   /* match the per-thread kernel stream */
    h->running_std = v->running_std;
    h->proj_w     = upload_f16(v->proj_w, (size_t)DC_D * 256);
    h->proj_b     = upload_f32(v->proj_b, DC_D);
    h->new_tokens = upload_f32(v->new_tokens, DC_D);
    /* repack the folded mapping-conv weight [DC_OUT, DC_D, 3] (tap-fastest) into 3
     * contiguous [DC_OUT, DC_D] tap matrices so the conv runs as accumulated GEMMs. */
    {
        size_t wsz = (size_t)DC_OUT * DC_D;
        float *taps = (float *)malloc(3 * wsz * sizeof(float));
        for (int o = 0; o < DC_OUT; o++)
            for (int i = 0; i < DC_D; i++)
                for (int k = 0; k < 3; k++)
                    taps[(size_t)k * wsz + (size_t)o * DC_D + i] = v->mapping_w[((size_t)o * DC_D + i) * 3 + k];
        h->mapping_w = upload_f32(taps, 3 * wsz);
        free(taps);
    }
    h->mapping_b  = upload_f32(v->mapping_b, DC_OUT);
    int ok = h->proj_w && h->proj_b && h->new_tokens && h->mapping_w && h->mapping_b;
    for (int i = 0; i < 6 && ok; i++) {
        const aria_taae_block_view *s = &v->blocks[i]; dec_blk_dev *d = &h->blocks[i];
        d->pre_alpha = s->pre_alpha; d->qn_alpha = s->qn_alpha; d->kn_alpha = s->kn_alpha; d->ff_alpha = s->ff_alpha;
        d->pre_gamma = upload_f32(s->pre_gamma, DC_D); d->pre_beta = upload_f32(s->pre_beta, DC_D);
        d->qn_gamma  = upload_f32(s->qn_gamma, DC_HD); d->qn_beta  = upload_f32(s->qn_beta, DC_HD);
        d->kn_gamma  = upload_f32(s->kn_gamma, DC_HD); d->kn_beta  = upload_f32(s->kn_beta, DC_HD);
        d->ff_gamma  = upload_f32(s->ff_gamma, DC_D);  d->ff_beta  = upload_f32(s->ff_beta, DC_D);
        d->ff_in_b   = upload_f32(s->ff_in_b, 2 * DC_INNER); d->ff_out_b = upload_f32(s->ff_out_b, DC_D);
        d->to_qkv    = upload_f16(s->to_qkv, (size_t)DC_QKV * DC_D);
        d->to_out    = upload_f16(s->to_out, (size_t)DC_D * DC_D);
        d->ff_in_w   = upload_f16(s->ff_in_w, (size_t)2 * DC_INNER * DC_D);
        d->ff_out_w  = upload_f16(s->ff_out_w, (size_t)DC_D * DC_INNER);
        ok = d->to_qkv && d->to_out && d->ff_in_w && d->ff_out_w && d->pre_gamma && d->ff_in_b;
    }
    /* rope tables for S=34 (computed on host, uploaded) */
    int rh = DC_ROT / 2;
    float *cs = (float *)malloc((size_t)DC_S * rh * sizeof(float)), *sn = (float *)malloc((size_t)DC_S * rh * sizeof(float));
    for (int i = 0; i < rh; i++) {
        double inv = 1.0 / pow(10000.0, (double)(2 * i) / (double)DC_ROT);
        for (int p = 0; p < DC_S; p++) { cs[p * rh + i] = (float)cos((double)p * inv); sn[p * rh + i] = (float)sin((double)p * inv); }
    }
    h->rcos = upload_f32(cs, (size_t)DC_S * rh); h->rsin = upload_f32(sn, (size_t)DC_S * rh);
    free(cs); free(sn);
    if (!ok || !h->rcos) { aria_cuda_dec_free(h); return NULL; }
    return h;
}

extern "C" void aria_cuda_dec_forward(aria_cuda_dec *h, float *audio, const float *latent, int T) {
    int D = DC_D, Tp = (T % 2 == 0) ? T : T + 1, L = Tp * 17, Lout = Tp * 16, Lt = T * 16;
    /* arena sized for the batched pass (shifted pass is the largest) */
    int Bmax = (L + DC_S) / DC_S;
    size_t bs = (size_t)Bmax * DC_S;
    size_t cap = (bs * 19200 + (size_t)Bmax * DC_H * DC_S * DC_S        /* one block's peak */
                  + (size_t)(L + DC_S) * D * 2                          /* seq + pad */
                  + (size_t)D * Lout + (size_t)DC_OUT * Lout            /* feat + mapped */
                  + (size_t)T * 256 + (size_t)T * D + (1u << 20)) * sizeof(float);
    darena ar;
    if (da_reuse(&h->ar_cache, &ar, cap) != 0) return;

    float *d_lat = upload_f32(latent, (size_t)256 * T);
    size_t mark = da_save(&ar);
    /* softnorm: z = latent * running_std ; transpose [256,T]->[T,256] ; proj ->[T,768] */
    float *z = da_alloc(&ar, (size_t)256 * T);
    k_scale<<<nblocks((size_t)256 * T), THREADS>>>(z, d_lat, h->running_std, 256 * T);
    float *ztc = da_alloc(&ar, (size_t)T * 256);
    k_transpose<<<nblocks((size_t)256 * T), THREADS>>>(ztc, z, 256, T);
    float *x = da_alloc(&ar, (size_t)T * D);
    gemm_f16w(h->cublas, &ar, x, ztc, h->proj_w, h->proj_b, T, 256, D);

    /* expand to seq[L,768], two chunked passes (unshifted, then midpoint-shift) */
    float *seq = da_alloc(&ar, (size_t)L * D);
    k_expand_tokens<<<nblocks((size_t)L * D), THREADS>>>(seq, x, h->new_tokens, T, Tp, D);
    taae_pass_dev(h, seq, L, &h->blocks[0], 0, &ar);
    taae_pass_dev(h, seq, L, &h->blocks[3], 1, &ar);

    /* extract feat[768,Lout], conv1d mapping 768->512, trim to T*16, unpatch -> audio */
    float *feat = da_alloc(&ar, (size_t)D * Lout);
    k_extract_feat<<<nblocks((size_t)D * Lout), THREADS>>>(feat, seq, D, Lout);
    float *mapped = da_alloc(&ar, (size_t)DC_OUT * Lout);
    /* mapping conv (K=3, pad=1) as 3 accumulated Sgemms over the repacked taps:
     * mapped[o,t] = b[o] + sum_k Wk[o,:] . feat[:, t+k-1]  (k=1 full range seeds with
     * beta=0; k=0/2 accumulate on the t>=1 / t<Lout-1 sub-ranges = the pad-1 zeros).
     * Row-major via the col-major identity C^T = feat^T @ Wk^T. Same fp32 dot products
     * as k_conv1d, cuBLAS reduction order -- was the top decode kernel (29 ms @ 60 s). */
    {
        const float onef = 1.0f, zerof = 0.0f;
        size_t wsz = (size_t)DC_OUT * DC_D;
        const float *wk0 = h->mapping_w, *wk1 = h->mapping_w + wsz, *wk2 = h->mapping_w + 2 * wsz;
        cublasSgemm(h->cublas, CUBLAS_OP_N, CUBLAS_OP_N, Lout, DC_OUT, D,
                    &onef, feat, Lout, wk1, D, &zerof, mapped, Lout);
        cublasSgemm(h->cublas, CUBLAS_OP_N, CUBLAS_OP_N, Lout - 1, DC_OUT, D,
                    &onef, feat, Lout, wk0, D, &onef, mapped + 1, Lout);
        cublasSgemm(h->cublas, CUBLAS_OP_N, CUBLAS_OP_N, Lout - 1, DC_OUT, D,
                    &onef, feat + 1, Lout, wk2, D, &onef, mapped, Lout);
        k_add_bias_rows<<<nblocks((size_t)DC_OUT * Lout), THREADS>>>(mapped, h->mapping_b, DC_OUT, Lout);
    }
    float *dec = da_alloc(&ar, (size_t)DC_OUT * Lt);
    CK(cudaMemcpy2D(dec, (size_t)Lt * sizeof(float), mapped, (size_t)Lout * sizeof(float),
                    (size_t)Lt * sizeof(float), DC_OUT, cudaMemcpyDeviceToDevice));
    float *d_audio = da_alloc(&ar, (size_t)2 * Lt * 256);
    k_unpatch<<<nblocks((size_t)2 * Lt * 256), THREADS>>>(d_audio, dec, Lt);
    CK(cudaGetLastError());
    CK(cudaMemcpy(audio, d_audio, (size_t)2 * Lt * 256 * sizeof(float), cudaMemcpyDeviceToHost));

    da_restore(&ar, mark);
    cudaFree(d_lat);
}

extern "C" void aria_cuda_dec_free(aria_cuda_dec *h) {
    if (!h) return;
    cudaFree(h->ar_cache.base);
    if (h->cublas) cublasDestroy(h->cublas);
    cudaFree(h->proj_w); cudaFree((void *)h->proj_b); cudaFree((void *)h->new_tokens);
    cudaFree(h->mapping_w); cudaFree((void *)h->mapping_b); cudaFree(h->rcos); cudaFree(h->rsin);
    for (int i = 0; i < 6; i++) {
        dec_blk_dev *d = &h->blocks[i];
        cudaFree((void *)d->pre_gamma); cudaFree((void *)d->pre_beta);
        cudaFree((void *)d->qn_gamma); cudaFree((void *)d->qn_beta);
        cudaFree((void *)d->kn_gamma); cudaFree((void *)d->kn_beta);
        cudaFree((void *)d->ff_gamma); cudaFree((void *)d->ff_beta);
        cudaFree((void *)d->ff_in_b); cudaFree((void *)d->ff_out_b);
        cudaFree(d->to_qkv); cudaFree(d->to_out); cudaFree(d->ff_in_w); cudaFree(d->ff_out_w);
    }
    free(h);
}

/* ===================== device-resident MEDIUM taae decoder =====================
 * dim 1536 / depth 12; the whole token sequence (T*17) runs through differential
 * DyT blocks with SLIDING-WINDOW (banded) attention (window [17,17]); blocks where
 * (12-i) < 8 use a sinusoidal FF. Weights fp16 (GEMMs) / fp32 (norms, conv). */
#define MD_D     1536
#define MD_H     24
#define MD_HD    64
#define MD_ROT   32
#define MD_INNER 4608
#define MD_QKV   7680   /* 5*1536 */
#define MD_OUT   512
#define MD_DEPTH 12
#define MD_WIN   17

/* E16.2: fuse the per-block extract+DyT+RoPE glue into one kernel per q/k/qd/kd.
 * Compile with -DMED_FUSE_GLUE=0 to restore the old 13-launch/block path for an A/B byte compare. */
#ifndef MED_FUSE_GLUE
#define MED_FUSE_GLUE 1
#endif

/* Fused sliding-window attention: one warp per query (h,i) computes only the
 * [i-W, i+W] band (<= 2W+1 keys) -- QK via shuffle-reduce, softmax over the band in
 * shared, then AV. O(N*(2W+1)) instead of the full O(N^2)-then-mask, and no N*N score
 * buffer. Exact vs the masked-softmax version (masked keys contributed 0 anyway).
 * D must be a multiple of 32 (decoder hd=64 -> 2 dims/lane). */
__global__ void k_attn_band(float *out, const float *q, const float *k, const float *v,
                            int H, int N, int D, float scale, int W) {
    __shared__ float ssc[2 * MD_WIN + 1];
    int idx = blockIdx.x; if (idx >= H * N) return;
    int h = idx / N, i = idx % N, lane = threadIdx.x, dpl = D / 32;
    const float *qh = q + (size_t)idx * D;
    float qr[8];
    for (int c = 0; c < dpl; c++) qr[c] = qh[lane + c * 32];
    int lo = i - W < 0 ? 0 : i - W, hi = i + W >= N ? N - 1 : i + W, wb = hi - lo + 1;
    for (int j = 0; j < wb; j++) {
        const float *kk = k + ((size_t)h * N + (lo + j)) * D;
        float p = 0.0f;
        for (int c = 0; c < dpl; c++) p += qr[c] * kk[lane + c * 32];
        for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
        if (lane == 0) ssc[j] = p * scale;
    }
    __syncwarp();
    if (lane == 0) {                       /* softmax over the band (<= 35 keys) */
        float mx = -INFINITY;
        for (int j = 0; j < wb; j++) mx = fmaxf(mx, ssc[j]);
        float sum = 0.0f;
        for (int j = 0; j < wb; j++) { ssc[j] = expf(ssc[j] - mx); sum += ssc[j]; }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int j = 0; j < wb; j++) ssc[j] *= inv;
    }
    __syncwarp();
    float orr[8];
    for (int c = 0; c < dpl; c++) orr[c] = 0.0f;
    for (int j = 0; j < wb; j++) {
        const float *vv = v + ((size_t)h * N + (lo + j)) * D;
        for (int c = 0; c < dpl; c++) orr[c] += ssc[j] * vv[lane + c * 32];
    }
    float *oo = out + (size_t)idx * D;
    for (int c = 0; c < dpl; c++) oo[lane + c * 32] = orr[c];
}

/* sinusoidal GLU gate: out = sin(pi*(gate+b)) * (value+b); b2 folds the ff_in GEMM's
 * bias in (A2), NULL = none. fp16-emitting (_h): out feeds the ff_out GEMM directly. */
__global__ void k_ff_singate(float *out, const float *proj, const float *b2, int S, int inner) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * inner) return;
    int s = (int)(idx / inner), i = (int)(idx % inner);
    const float *pr = proj + (size_t)s * 2 * inner;
    float up = b2 ? (pr[i] + b2[i]) : pr[i];
    float g = b2 ? (pr[inner + i] + b2[inner + i]) : pr[inner + i];
    out[idx] = sinf(3.14159265359f * g) * up;
}
__global__ void k_ff_singate_h(__half *out, const float *proj, const float *b2, int S, int inner) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)S * inner) return;
    int s = (int)(idx / inner), i = (int)(idx % inner);
    const float *pr = proj + (size_t)s * 2 * inner;
    float up = b2 ? (pr[i] + b2[i]) : pr[i];
    float g = b2 ? (pr[inner + i] + b2[inner + i]) : pr[inner + i];
    out[idx] = __float2half(sinf(3.14159265359f * g) * up);
}

/* fused extract-head + per-head DyT (dynamic tanh) + RoPE -> head-major out[h,s,:hd].
 * Medium-decoder analogue of the DiT's k_extract_normrope: replaces a k_extract_heads +
 * k_dyt + k_rope chain (saves two [H,N,hd] HBM round-trips + two launches per q/k). One
 * block per (h,s) row, hd threads (hd<=64). DyT is elementwise (no reduction), so shared
 * mem only stages the DyT output for RoPE's i,i+rot/2 pairing. Math matches exactly:
 * DyT y=tanh(alpha*x)*gamma+beta (aria_dynamic_tanh) then k_rope (same product order). */
__global__ void k_extract_dytrope(float *out, const float *src, float alpha,
                                  const float *g, const float *b,
                                  const float *rcos, const float *rsin,
                                  int S, int H, int hd, int stride, int off, int rot) {
    int hs = blockIdx.x; if (hs >= H * S) return;
    int s = hs % S, h = hs / S, d = threadIdx.x;
    const float *sp = src + (size_t)s * stride + off + (size_t)h * hd;
    __shared__ float vec[64];
    vec[d] = tanhf(alpha * sp[d]) * g[d] + b[d];   /* DyT */
    __syncthreads();
    int rh = rot / 2; float res;
    if (d >= rot)    res = vec[d];
    else if (d < rh) res = vec[d] * rcos[(size_t)s * rh + d]      - vec[d + rh] * rsin[(size_t)s * rh + d];
    else { int i = d - rh; res = vec[d] * rcos[(size_t)s * rh + i] + vec[i] * rsin[(size_t)s * rh + i]; }
    out[(size_t)hs * hd + d] = res;
}

/* E17: tensor-core sliding-window attention. One warp per (head, 16-query tile) instead
 * of per-query. The union band of a 16-query tile is [q0-W, q0+15+W] (<= 2W+1+15 keys);
 * padded to ABW_KP=64 it tiles cleanly for wmma 16x16x16. Q/K/V staged fp16 in shared,
 * QK^T and AV on tensor cores (fp32 accumulate), softmax fp32 over the exact per-row band
 * -- same key set/masking as the scalar k_attn_band, so numerically it differs only by the
 * fp16 rounding of Q/K/V/P (the same precision the DiT/FF GEMMs already run at). Requires
 * D==64 and 2W+1+15 <= 64 (medium: W=17 -> 50); the host falls back to the scalar kernel
 * otherwise (and on pre-Volta, where wmma is unavailable). */
#define ABW_QB 16   /* queries per tile (one wmma M-tile) */
#define ABW_KP 64   /* padded band width: multiple of 16, >= (2W+1)+(ABW_QB-1) */

__global__ void k_attn_band_wmma(float *out, const float *q, const float *k, const float *v,
                                 int H, int N, int D, float scale, int W) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda::wmma;
    int nqb = (N + ABW_QB - 1) / ABW_QB;
    int h = blockIdx.x / nqb, qb = blockIdx.x % nqb;
    if (h >= H) return;
    int q0 = qb * ABW_QB, kstart = q0 - W, lane = threadIdx.x;

    __shared__ __half sQ[ABW_QB * 64];
    __shared__ __half sK[ABW_KP * 64];
    __shared__ __half sV[ABW_KP * 64];
    __shared__ float  sS[ABW_QB * ABW_KP];
    __shared__ __half sP[ABW_QB * ABW_KP];
    __shared__ float  sO[ABW_QB * 64];

    const float *qb_ = q + (size_t)h * N * 64;
    const float *kb_ = k + (size_t)h * N * 64;
    const float *vb_ = v + (size_t)h * N * 64;

    for (int idx = lane; idx < ABW_QB * 64; idx += 32) {   /* Q tile [16,64], pad rows -> 0 */
        int r = idx >> 6, c = idx & 63, qg = q0 + r;
        sQ[idx] = qg < N ? __float2half(qb_[(size_t)qg * 64 + c]) : (__half)0;
    }
    for (int idx = lane; idx < ABW_KP * 64; idx += 32) {   /* K/V band [64,64], OOB keys -> 0 */
        int r = idx >> 6, c = idx & 63, kg = kstart + r, ok = kg >= 0 && kg < N;
        sK[idx] = ok ? __float2half(kb_[(size_t)kg * 64 + c]) : (__half)0;
        sV[idx] = ok ? __float2half(vb_[(size_t)kg * 64 + c]) : (__half)0;
    }
    __syncwarp();

    for (int nt = 0; nt < ABW_KP / 16; nt++) {             /* sS[16,64] = Q @ K^T (fp32 acc) */
        fragment<accumulator, 16, 16, 16, float> acc;
        fill_fragment(acc, 0.0f);
        for (int kt = 0; kt < 4; kt++) {                   /* contract over D=64 */
            fragment<matrix_a, 16, 16, 16, __half, row_major> a;
            fragment<matrix_b, 16, 16, 16, __half, col_major> b;
            load_matrix_sync(a, sQ + kt * 16, 64);
            load_matrix_sync(b, sK + nt * 16 * 64 + kt * 16, 64);
            mma_sync(acc, a, b, acc);
        }
        store_matrix_sync(sS + nt * 16, acc, ABW_KP, mem_row_major);
    }
    __syncwarp();

    if (lane < ABW_QB) {                                   /* fp32 softmax over the row's band */
        int r = lane, lo = r, hi = r + 2 * W;              /* relative band: kj in [r, r+2W] */
        float mx = -INFINITY;
        for (int kj = lo; kj <= hi; kj++) {
            int kg = kstart + kj; if (kg < 0 || kg >= N) continue;
            float s = sS[r * ABW_KP + kj] * scale; sS[r * ABW_KP + kj] = s;
            mx = fmaxf(mx, s);
        }
        float sum = 0.0f;
        for (int kj = lo; kj <= hi; kj++) {
            int kg = kstart + kj; if (kg < 0 || kg >= N) continue;
            float e = __expf(sS[r * ABW_KP + kj] - mx); sS[r * ABW_KP + kj] = e; sum += e;
        }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int kj = 0; kj < ABW_KP; kj++) {
            int kg = kstart + kj;
            float p = (kj >= lo && kj <= hi && kg >= 0 && kg < N) ? sS[r * ABW_KP + kj] * inv : 0.0f;
            sP[r * ABW_KP + kj] = __float2half(p);
        }
    }
    __syncwarp();

    for (int nt = 0; nt < 4; nt++) {                       /* sO[16,64] = P @ V (fp32 acc) */
        fragment<accumulator, 16, 16, 16, float> acc;
        fill_fragment(acc, 0.0f);
        for (int kt = 0; kt < ABW_KP / 16; kt++) {         /* contract over the 64 band keys */
            fragment<matrix_a, 16, 16, 16, __half, row_major> a;
            fragment<matrix_b, 16, 16, 16, __half, row_major> b;
            load_matrix_sync(a, sP + kt * 16, ABW_KP);
            load_matrix_sync(b, sV + kt * 16 * 64 + nt * 16, 64);
            mma_sync(acc, a, b, acc);
        }
        store_matrix_sync(sO + nt * 16, acc, 64, mem_row_major);
    }
    __syncwarp();

    for (int idx = lane; idx < ABW_QB * 64; idx += 32) {
        int r = idx >> 6, c = idx & 63, qg = q0 + r;
        if (qg < N) out[((size_t)h * N + qg) * 64 + c] = sO[idx];
    }
#else
    (void)out; (void)q; (void)k; (void)v; (void)H; (void)N; (void)D; (void)scale; (void)W;
#endif
}

/* Volta+ (tensor cores) gate for the wmma band path; ARIA_BAND_SCALAR=1 forces the scalar
 * kernel (A/B the two on one binary). Cached -- cudaGetDeviceProperties is not free. */
static int band_use_tc(void) {
    static int v = -1;
    if (v < 0) {
        const char *e = getenv("ARIA_BAND_SCALAR");
        if (e && e[0] && e[0] != '0') return (v = 0);
        cudaDeviceProp p;
        v = (cudaGetDeviceProperties(&p, 0) == cudaSuccess && p.major >= 7) ? 1 : 0;
    }
    return v;
}

/* sliding-window self-attention (Nq=Nk=N): tensor-core tile kernel where supported,
 * else the fused warp-per-query scalar kernel. */
static void attn_dev_band(float *out, const float *q, const float *k, const float *v,
                          int H, int N, int D, int W) {
    float scale = 1.0f / sqrtf((float)D);
    if (band_use_tc() && D == 64 && (2 * W + 1) + (ABW_QB - 1) <= ABW_KP) {
        int nqb = (N + ABW_QB - 1) / ABW_QB;
        k_attn_band_wmma<<<H * nqb, 32>>>(out, q, k, v, H, N, D, scale, W);
        return;
    }
    k_attn_band<<<H * N, 32>>>(out, q, k, v, H, N, D, scale, W);
}

struct aria_cuda_dec_medium {
    float running_std;
    __half *proj_w; const float *proj_b, *new_tokens, *mapping_b;
    float *mapping_w;       /* fp32 k1 [512,1536] */
    dec_blk_dev blocks[MD_DEPTH];
    cublasHandle_t cublas;
    darena ar_cache;        /* decode scratch, kept across calls */
};

static void med_block_dev(aria_cuda_dec_medium *h, float *seq, int N, const float *rcos, const float *rsin,
                          const dec_blk_dev *w, int sinusoidal, darena *ar) {
    int D = MD_D, H = MD_H, hd = MD_HD;
    size_t NH = (size_t)H * N * hd;
    size_t mark = da_save(ar);
    /* hh/o persist across the block (residuals add in place); the attention scratch is freed before the FF
     * (they're never live together) so the arena peak is the attention phase, not the
     * sum of both -- saves ~3*N*INNER (~100 MB on a 10 s medium clip). */
    float *o = da_alloc(ar, (size_t)N * D);
    __half *hhh = da_alloc_half(ar, (size_t)N * D);   /* fp16 GEMM-input producer (A2) */

    k_dyt_h<<<nblocks((size_t)N * D), THREADS>>>(hhh, seq, w->pre_alpha, w->pre_gamma, w->pre_beta, (size_t)N * D, D);
    size_t amark = da_save(ar);
    float *q  = da_alloc(ar, NH), *k = da_alloc(ar, NH), *v = da_alloc(ar, NH);
    float *qd = da_alloc(ar, NH), *kd = da_alloc(ar, NH);
    float *ob = da_alloc(ar, NH), *od = da_alloc(ar, NH);
    __half *mergedh = da_alloc_half(ar, (size_t)N * D);
    float *qkv = da_alloc(ar, (size_t)N * MD_QKV);
    gemm_f16w_h(h->cublas, qkv, hhh, w->to_qkv, NULL, N, D, MD_QKV);
#if MED_FUSE_GLUE
    /* v: plain extract (never normed/roped). q/k/qd/kd: fused extract+DyT+RoPE.
     * 13 launches (5 extract + 4 dyt + 4 rope) -> 5 (1 extract + 4 fused). */
    k_extract_heads<<<nblocks(NH), THREADS>>>(v, qkv, N, H, hd, MD_QKV, 2 * D);
    k_extract_dytrope<<<H * N, hd>>>(q,  qkv, w->qn_alpha, w->qn_gamma, w->qn_beta, rcos, rsin, N, H, hd, MD_QKV, 0,     MD_ROT);
    k_extract_dytrope<<<H * N, hd>>>(k,  qkv, w->kn_alpha, w->kn_gamma, w->kn_beta, rcos, rsin, N, H, hd, MD_QKV, D,     MD_ROT);
    k_extract_dytrope<<<H * N, hd>>>(qd, qkv, w->qn_alpha, w->qn_gamma, w->qn_beta, rcos, rsin, N, H, hd, MD_QKV, 3 * D, MD_ROT);
    k_extract_dytrope<<<H * N, hd>>>(kd, qkv, w->kn_alpha, w->kn_gamma, w->kn_beta, rcos, rsin, N, H, hd, MD_QKV, 4 * D, MD_ROT);
#else
    k_extract_heads<<<nblocks(NH), THREADS>>>(q,  qkv, N, H, hd, MD_QKV, 0);
    k_extract_heads<<<nblocks(NH), THREADS>>>(k,  qkv, N, H, hd, MD_QKV, D);
    k_extract_heads<<<nblocks(NH), THREADS>>>(v,  qkv, N, H, hd, MD_QKV, 2 * D);
    k_extract_heads<<<nblocks(NH), THREADS>>>(qd, qkv, N, H, hd, MD_QKV, 3 * D);
    k_extract_heads<<<nblocks(NH), THREADS>>>(kd, qkv, N, H, hd, MD_QKV, 4 * D);
    k_dyt<<<nblocks(NH), THREADS>>>(q,  q,  w->qn_alpha, w->qn_gamma, w->qn_beta, NH, hd);
    k_dyt<<<nblocks(NH), THREADS>>>(qd, qd, w->qn_alpha, w->qn_gamma, w->qn_beta, NH, hd);
    k_dyt<<<nblocks(NH), THREADS>>>(k,  k,  w->kn_alpha, w->kn_gamma, w->kn_beta, NH, hd);
    k_dyt<<<nblocks(NH), THREADS>>>(kd, kd, w->kn_alpha, w->kn_gamma, w->kn_beta, NH, hd);
    size_t hn = (size_t)H * N;
    k_rope<<<nblocks(hn), THREADS>>>(q,  rcos, rsin, H, N, hd, MD_ROT);
    k_rope<<<nblocks(hn), THREADS>>>(qd, rcos, rsin, H, N, hd, MD_ROT);
    k_rope<<<nblocks(hn), THREADS>>>(k,  rcos, rsin, H, N, hd, MD_ROT);
    k_rope<<<nblocks(hn), THREADS>>>(kd, rcos, rsin, H, N, hd, MD_ROT);
#endif
    attn_dev_band(ob, q,  k,  v, H, N, hd, MD_WIN);
    attn_dev_band(od, qd, kd, v, H, N, hd, MD_WIN);
    k_merge_heads_sub_h<<<nblocks(NH), THREADS>>>(mergedh, ob, od, N, H, hd);
    gemm_f16w_h(h->cublas, o, mergedh, w->to_out, NULL, N, D, D);
    k_add<<<nblocks((size_t)N * D), THREADS>>>(seq, seq, o, (size_t)N * D);   /* residual in place */
    da_restore(ar, amark);   /* free q/k/v/qd/kd/ob/od/merged/qkv before the FF */

    k_dyt_h<<<nblocks((size_t)N * D), THREADS>>>(hhh, seq, w->ff_alpha, w->ff_gamma, w->ff_beta, (size_t)N * D, D);
    float *proj = da_alloc(ar, (size_t)N * 2 * MD_INNER);
    __half *gated = da_alloc_half(ar, (size_t)N * MD_INNER);
    gemm_f16w_h(h->cublas, proj, hhh, w->ff_in_w, NULL, N, D, 2 * MD_INNER);
    if (sinusoidal) k_ff_singate_h<<<nblocks((size_t)N * MD_INNER), THREADS>>>(gated, proj, w->ff_in_b, N, MD_INNER);
    else            k_ff_silugate_h<<<nblocks((size_t)N * MD_INNER), THREADS>>>(gated, proj, w->ff_in_b, N, MD_INNER);
    gemm_f16w_h(h->cublas, o, gated, w->ff_out_w, NULL, N, MD_INNER, D);
    k_add_bias_res<<<nblocks((size_t)N * D), THREADS>>>(seq, seq, o, w->ff_out_b, (size_t)N * D, D);

    da_restore(ar, mark);
}

extern "C" aria_cuda_dec_medium *aria_cuda_dec_medium_create(const aria_sa3_dec_medium_view *v) {
    if (!aria_cuda_available()) return NULL;
    aria_cuda_dec_medium *h = (aria_cuda_dec_medium *)calloc(1, sizeof(*h));
    if (cublasCreate(&h->cublas) != CUBLAS_STATUS_SUCCESS) { free(h); return NULL; }
    cublasSetStream(h->cublas, cudaStreamPerThread);   /* match the per-thread kernel stream */
    h->running_std = v->running_std;
    h->proj_w     = upload_f16(v->proj_w, (size_t)MD_D * 256);
    h->proj_b     = upload_f32(v->proj_b, MD_D);
    h->new_tokens = upload_f32(v->new_tokens, MD_D);
    h->mapping_w  = upload_f32(v->mapping_w, (size_t)MD_OUT * MD_D);   /* k1 */
    h->mapping_b  = upload_f32(v->mapping_b, MD_OUT);
    int ok = h->proj_w && h->proj_b && h->new_tokens && h->mapping_w && h->mapping_b;
    for (int i = 0; i < MD_DEPTH && ok; i++) {
        const aria_taae_block_view *s = &v->blocks[i]; dec_blk_dev *d = &h->blocks[i];
        d->pre_alpha = s->pre_alpha; d->qn_alpha = s->qn_alpha; d->kn_alpha = s->kn_alpha; d->ff_alpha = s->ff_alpha;
        d->pre_gamma = upload_f32(s->pre_gamma, MD_D); d->pre_beta = upload_f32(s->pre_beta, MD_D);
        d->qn_gamma  = upload_f32(s->qn_gamma, MD_HD); d->qn_beta  = upload_f32(s->qn_beta, MD_HD);
        d->kn_gamma  = upload_f32(s->kn_gamma, MD_HD); d->kn_beta  = upload_f32(s->kn_beta, MD_HD);
        d->ff_gamma  = upload_f32(s->ff_gamma, MD_D);  d->ff_beta  = upload_f32(s->ff_beta, MD_D);
        d->ff_in_b   = upload_f32(s->ff_in_b, 2 * MD_INNER); d->ff_out_b = upload_f32(s->ff_out_b, MD_D);
        d->to_qkv    = upload_f16(s->to_qkv, (size_t)MD_QKV * MD_D);
        d->to_out    = upload_f16(s->to_out, (size_t)MD_D * MD_D);
        d->ff_in_w   = upload_f16(s->ff_in_w, (size_t)2 * MD_INNER * MD_D);
        d->ff_out_w  = upload_f16(s->ff_out_w, (size_t)MD_D * MD_INNER);
        ok = d->to_qkv && d->to_out && d->ff_in_w && d->ff_out_w && d->pre_gamma && d->ff_in_b;
    }
    if (!ok) { aria_cuda_dec_medium_free(h); return NULL; }
    return h;
}

extern "C" void aria_cuda_dec_medium_forward(aria_cuda_dec_medium *h, float *audio, const float *latent, int T) {
    int D = MD_D, N = T * 17, Lt = T * 16, rh = MD_ROT / 2;
    /* rope tables for N (host -> device) */
    float *cs = (float *)malloc((size_t)N * rh * sizeof(float)), *sn = (float *)malloc((size_t)N * rh * sizeof(float));
    for (int i = 0; i < rh; i++) {
        double inv = 1.0 / pow(10000.0, (double)(2 * i) / (double)MD_ROT);
        for (int p = 0; p < N; p++) { cs[(size_t)p * rh + i] = (float)cos((double)p * inv); sn[(size_t)p * rh + i] = (float)sin((double)p * inv); }
    }
    float *rcos = upload_f32(cs, (size_t)N * rh), *rsin = upload_f32(sn, (size_t)N * rh);
    free(cs); free(sn);

    /* no N*N score buffer (fused band attn), and the FF scratch no longer co-resides
     * with the attention scratch (freed between phases) -- peak is the attention phase
     * (16*N*D + a GEMM-scratch margin); 17*N*D bounds it and the FF (3.5*N*INNER < 17*D). */
    size_t blockpk = (size_t)17 * N * D;
    size_t cap = ((size_t)N * D + blockpk + (size_t)D * Lt + (size_t)MD_OUT * Lt
                  + (size_t)T * 256 + (size_t)T * D + (size_t)2 * Lt * 256 + (1u << 20)) * sizeof(float);
    darena ar;
    if (da_reuse(&h->ar_cache, &ar, cap) != 0) return;

    float *d_lat = upload_f32(latent, (size_t)256 * T);
    size_t mark = da_save(&ar);
    float *z = da_alloc(&ar, (size_t)256 * T);
    k_scale<<<nblocks((size_t)256 * T), THREADS>>>(z, d_lat, h->running_std, 256 * T);
    float *ztc = da_alloc(&ar, (size_t)T * 256);
    k_transpose<<<nblocks((size_t)256 * T), THREADS>>>(ztc, z, 256, T);
    float *x = da_alloc(&ar, (size_t)T * D);
    gemm_f16w(h->cublas, &ar, x, ztc, h->proj_w, h->proj_b, T, 256, D);

    /* seq[N, 1536]: token t at row t*17, new_tokens at the next 16 (no even-pad) */
    float *seq = da_alloc(&ar, (size_t)N * D);
    k_expand_tokens<<<nblocks((size_t)N * D), THREADS>>>(seq, x, h->new_tokens, T, T, D);
    static int med_glue_reported = 0;
    if (!med_glue_reported) {
        med_glue_reported = 1;
#if MED_FUSE_GLUE
        fprintf(stderr, "[aria] med decoder head-glue FUSED: 13->5 launches/block "
                "(5x extract+4x dyt+4x rope -> 1x extract + 4x extract_dytrope); "
                "%d blocks/decode: %d->%d (-%d)\n", MD_DEPTH, 13 * MD_DEPTH, 5 * MD_DEPTH, 8 * MD_DEPTH);
#else
        fprintf(stderr, "[aria] med decoder head-glue UNFUSED: 13 launches/block "
                "(%d blocks/decode: %d)\n", MD_DEPTH, 13 * MD_DEPTH);
#endif
    }
    for (int b = 0; b < MD_DEPTH; b++)
        med_block_dev(h, seq, N, rcos, rsin, &h->blocks[b], (MD_DEPTH - b) < 8, &ar);

    float *feat = da_alloc(&ar, (size_t)D * Lt);
    k_extract_feat<<<nblocks((size_t)D * Lt), THREADS>>>(feat, seq, D, Lt);
    float *mapped = da_alloc(&ar, (size_t)MD_OUT * Lt);
    /* the K=1 mapping conv IS a GEMM: mapped[MD_OUT, Lt] = W[MD_OUT, D] @ feat[D, Lt] + b.
     * Row-major via the col-major identity C^T = feat^T @ W^T. Same fp32 dot products as
     * k_conv1d, cuBLAS reduction order -- was a 42 ms single kernel at 60 s. */
    {
        const float onef = 1.0f, zerof = 0.0f;
        cublasSgemm(h->cublas, CUBLAS_OP_N, CUBLAS_OP_N, Lt, MD_OUT, D,
                    &onef, feat, Lt, h->mapping_w, D, &zerof, mapped, Lt);
        k_add_bias_rows<<<nblocks((size_t)MD_OUT * Lt), THREADS>>>(mapped, h->mapping_b, MD_OUT, Lt);
    }
    float *d_audio = da_alloc(&ar, (size_t)2 * Lt * 256);
    k_unpatch<<<nblocks((size_t)2 * Lt * 256), THREADS>>>(d_audio, mapped, Lt);
    CK(cudaGetLastError());
    CK(cudaMemcpy(audio, d_audio, (size_t)2 * Lt * 256 * sizeof(float), cudaMemcpyDeviceToHost));

    da_restore(&ar, mark);
    cudaFree(d_lat); cudaFree(rcos); cudaFree(rsin);
}

extern "C" void aria_cuda_dec_medium_free(aria_cuda_dec_medium *h) {
    if (!h) return;
    cudaFree(h->ar_cache.base);
    if (h->cublas) cublasDestroy(h->cublas);
    cudaFree(h->proj_w); cudaFree((void *)h->proj_b); cudaFree((void *)h->new_tokens);
    cudaFree(h->mapping_w); cudaFree((void *)h->mapping_b);
    for (int i = 0; i < MD_DEPTH; i++) {
        dec_blk_dev *d = &h->blocks[i];
        cudaFree((void *)d->pre_gamma); cudaFree((void *)d->pre_beta);
        cudaFree((void *)d->qn_gamma); cudaFree((void *)d->qn_beta);
        cudaFree((void *)d->kn_gamma); cudaFree((void *)d->kn_beta);
        cudaFree((void *)d->ff_gamma); cudaFree((void *)d->ff_beta);
        cudaFree((void *)d->ff_in_b); cudaFree((void *)d->ff_out_b);
        cudaFree(d->to_qkv); cudaFree(d->to_out); cudaFree(d->ff_in_w); cudaFree(d->ff_out_w);
    }
    free(h);
}
