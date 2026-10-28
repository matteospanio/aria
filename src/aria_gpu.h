/*
 * aria_gpu.h - CUDA backend surface (E8).
 *
 * Built only under -DARIA_CUDA (aria_cuda.cu). Device-pointer kernels are the
 * building blocks for an eventual device-resident forward (E8.5); the host-pointer
 * wrappers (allocate + copy + compute + copy back) are for op-level parity tests
 * and standalone use. All symbols are C-callable.
 */

#ifndef ARIA_GPU_H
#define ARIA_GPU_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if a usable CUDA device is present (0 in a CPU-only build). */
int aria_cuda_available(void);
/* 1 if the device is worth auto-selecting over the CPU (sm_70+ / tensor cores).
 * Weaker cards (e.g. GT 1030, sm_61) are slower than the AVX2 CPU, so `--device
 * auto` skips them; `--device cuda` still forces any available device. */
int aria_cuda_recommended(void);
int aria_cuda_int8_tc(void);   /* sm_75+: int8 tensor cores for W8A8 */
/* short device description into buf (name, compute capability, VRAM). */
void aria_cuda_device_info(char *buf, size_t buflen);
/* current device memory: *used and *total in bytes (used = total - free). */
void aria_cuda_meminfo(size_t *used, size_t *total);

/* raw device memory */
void *aria_cuda_malloc(size_t bytes);
void  aria_cuda_free(void *dptr);
void  aria_cuda_memcpy_h2d(void *ddst, const void *hsrc, size_t bytes);
void  aria_cuda_memcpy_d2h(void *hdst, const void *dsrc, size_t bytes);
void  aria_cuda_sync(void);

/* y[M,N] = x[M,K] @ W[N,K]^T + b   (W is PyTorch Linear layout; b may be NULL).
 * _dev: all pointers are device memory. host wrapper: all pointers are host. */
void aria_cuda_linear_dev(float *dy, const float *dx, const float *dW, const float *db,
                          int M, int K, int N);
void aria_cuda_linear(float *y, const float *x, const float *W, const float *b,
                      int M, int K, int N);

/* ---- remaining hot ops (E8.3/E8.4); host-pointer wrappers mirror aria_ops.h ---- */
void aria_cuda_matmul(float *C, const float *A, const float *B, int M, int K, int N);
void aria_cuda_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps);
void aria_cuda_gemma_rmsnorm(float *y, const float *x, const float *w, int rows, int dim, float eps);
void aria_cuda_dynamic_tanh(float *y, const float *x, float alpha,
                            const float *w, const float *b, int rows, int dim);
void aria_cuda_softcap(float *s, int n, float cap);
void aria_cuda_silu(float *x, int n);
void aria_cuda_gelu_tanh(float *x, int n);
void aria_cuda_silu_gate(float *out, const float *gate, const float *up, int n);
void aria_cuda_softmax(float *x, int rows, int cols, const float *mask);
void aria_cuda_rope_apply(float *x, const float *cos_t, const float *sin_t,
                          int H, int N, int D, int rot_dim);
void aria_cuda_attention(float *out, const float *q, const float *k, const float *v,
                         int H, int Nq, int Nk, int D, const float *mask);
void aria_cuda_conv1d(float *out, const float *in, const float *w, const float *bias,
                      int Cin, int Cout, int K, int pad, int L);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_GPU_H */
