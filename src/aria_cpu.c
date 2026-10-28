/*
 * aria_cpu.c - CPU implementations of the aria_ops primitive surface.
 *
 * Scalar + compiler-autovectorized (-O3 -march=native), OpenMP across rows.
 * SIMD-intrinsic hot paths land in Phase 3; correctness comes first.
 */

#include "aria_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifndef _WIN32
#include <unistd.h>   /* sysconf (aria_win_compat.h shims it on Windows) */
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

/* persistent per-thread scratch: grown as needed, never per-call malloc'd. Safe
 * whether the caller is single-threaded (DiT) or each OpenMP worker owns its copy. */
static float *grow_tls(float **buf, size_t *cap, size_t need) {
    if (need > *cap) { free(*buf); *buf = (float *)malloc(need * sizeof(float)); *cap = *buf ? need : 0; }
    return *buf;
}


#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define ARIA_AVX2 1
/* register-tile dimensions (output rows x cols held in ymm accumulators):
 * regs used = NR (weight vecs) + 1 (x vec) + MR*NR (accumulators) <= 16.
 * 3x3 (3+1+9=13 ymm) tuned via `make bench-sweep`: it matches/beats 4x3 (which
 * saturates all 16 ymm) on the SA3 shapes while leaving register headroom. */
#ifndef ARIA_MR
#define ARIA_MR 3
#endif
#ifndef ARIA_NR
#define ARIA_NR 3
#endif
static inline float aria_hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x1));
    return _mm_cvtss_f32(lo);
}
static inline double aria_hsum256d(__m256d v) {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    lo = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(lo, _mm_unpackhi_pd(lo, lo)));
}
/* sum of x[i]^2 over dim, accumulated in double (8-wide), to match the scalar
 * double-precision reduction used for RMS norms within float tolerance. */
static inline double aria_sumsq_f(const float *x, int dim) {
    __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256d xl = _mm256_cvtps_pd(_mm256_castps256_ps128(xv));
        __m256d xh = _mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1));
        a0 = _mm256_fmadd_pd(xl, xl, a0);
        a1 = _mm256_fmadd_pd(xh, xh, a1);
    }
    double ss = aria_hsum256d(a0) + aria_hsum256d(a1);
    for (; i < dim; i++) ss += (double)x[i] * x[i];
    return ss;
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define ARIA_NEON 1
/* sum of x[i]^2 accumulated in double (2-wide f64x2 NEON), matching the AVX2/scalar
 * double reduction the RMS norms rely on, within float tolerance. AArch64 f64 NEON. */
static inline double aria_sumsq_f(const float *x, int dim) {
    float64x2_t a0 = vdupq_n_f64(0.0), a1 = vdupq_n_f64(0.0);
    int i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float64x2_t xl = vcvt_f64_f32(vget_low_f32(xv));
        float64x2_t xh = vcvt_f64_f32(vget_high_f32(xv));
        a0 = vfmaq_f64(a0, xl, xl);
        a1 = vfmaq_f64(a1, xh, xh);
    }
    double ss = vaddvq_f64(a0) + vaddvq_f64(a1);
    for (; i < dim; i++) ss += (double)x[i] * x[i];
    return ss;
}
#else
/* scalar sum of x[i]^2 in double -- matches the AVX2 double-accumulation reduction
 * (used by the RMS norms) within float tolerance when built without AVX2/FMA. */
static inline double aria_sumsq_f(const float *x, int dim) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    return ss;
}
#endif

/* y[M,N] = x[M,K] @ W^T + b, W=[N,K] (PyTorch Linear).
 * Blocked over output columns (NB) so each weight panel (NB*K) stays in L2 and is
 * reused across all M rows; the inner dot is vectorized via omp-simd reduction
 * (the float reduction won't auto-vectorize under strict FP otherwise). The
 * vectorized summation order differs slightly from a scalar sweep -- within the
 * parity tolerances. */
/* simd dot for edge tiles */
static inline float aria_dot(const float *a, const float *w, int K) {
    float acc = 0.0f;
    #ifdef _OPENMP
    #pragma omp simd reduction(+:acc)
    #endif
    for (int k = 0; k < K; k++) acc += a[k] * w[k];
    return acc;
}

#ifdef ARIA_AVX2
/* MR x NR output tile, vectorized over K (8 lanes), MR*NR ymm accumulators for ILP.
 * y[(m+i),(n+j)] = dot(x[m+i], W[n+j]) + b[n+j], for i<MR, j<NR. */
static inline void aria_microkernel(float *y, const float *x, const float *W, const float *b,
                                    int m, int n, int K, int N) {
    __m256 acc[ARIA_MR][ARIA_NR];
    const float *xp[ARIA_MR], *wp[ARIA_NR];
    for (int i = 0; i < ARIA_MR; i++) { xp[i] = x + (size_t)(m + i) * K;
        for (int j = 0; j < ARIA_NR; j++) acc[i][j] = _mm256_setzero_ps(); }
    for (int j = 0; j < ARIA_NR; j++) wp[j] = W + (size_t)(n + j) * K;
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 wv[ARIA_NR];
        for (int j = 0; j < ARIA_NR; j++) wv[j] = _mm256_loadu_ps(wp[j] + k);
        for (int i = 0; i < ARIA_MR; i++) {
            __m256 xv = _mm256_loadu_ps(xp[i] + k);
            for (int j = 0; j < ARIA_NR; j++) acc[i][j] = _mm256_fmadd_ps(xv, wv[j], acc[i][j]);
        }
    }
    for (int i = 0; i < ARIA_MR; i++)
        for (int j = 0; j < ARIA_NR; j++) {
            float s = aria_hsum256(acc[i][j]);
            for (int kk = k; kk < K; kk++) s += xp[i][kk] * wp[j][kk];  /* K%8 tail */
            if (b) s += b[n + j];
            y[(size_t)(m + i) * N + (n + j)] = s;
        }
}
#endif  /* ARIA_AVX2 dot-product microkernel */

#if defined(ARIA_AVX2) || defined(ARIA_NEON)
/* ---- packed outer-product GEMM (BLIS-style) ----
 * A (x) is packed once into k-major MR-panels; each NR-panel of B (W) is packed on
 * the fly. The microkernel does pure outer products (broadcast A, fma into NR-wide B
 * accumulators) -- no horizontal sums, fully contiguous reads -- beating the
 * dot-product kernel's ~235 GFLOP/s ceiling on the SA3 shapes. Each output is a
 * sequential-k accumulation (within parity tolerance of the reference). The packing
 * layer is ISA-agnostic; only the microkernel differs (AVX2 6x16 ymm / NEON 6x16 q). */
#if defined(ARIA_AVX2)
#define PMR 6
#define PNR 16
#elif defined(ARIA_NEON)
#ifndef PMR
#define PMR 6
#endif
#ifndef PNR
#define PNR 16
#endif
#endif
static void pack_a_panel(float *dst, const float *x, int m0, int M, int K) {
    for (int i = 0; i < PMR; i++) {
        int m = m0 + i;
        if (m < M) { const float *xr = x + (size_t)m * K; for (int k = 0; k < K; k++) dst[k * PMR + i] = xr[k]; }
        else       { for (int k = 0; k < K; k++) dst[k * PMR + i] = 0.0f; }
    }
}
static void pack_b_panel(float *dst, const float *W, int n0, int N, int pc, int kc, int K) {
    for (int j = 0; j < PNR; j++) {
        int n = n0 + j;
        if (n < N) { const float *wr = W + (size_t)n * K + pc; for (int k = 0; k < kc; k++) dst[k * PNR + j] = wr[k]; }
        else       { for (int k = 0; k < kc; k++) dst[k * PNR + j] = 0.0f; }
    }
}
/* B2: half-precision B-source packing -- fp16 (IEEE) or bf16 weights are widened to
 * f32 WHILE packing, so the microkernel stays the fp32 packed kernel and the weight
 * stream from memory halves. Widening happens once per (panel, K-block), amortized
 * exactly like the fp32 copy the pack already did. */
static void pack_b_panel_h(float *dst, const uint16_t *W, int n0, int N, int pc, int kc, int K, int bf16) {
    for (int j = 0; j < PNR; j++) {
        int n = n0 + j;
        if (n < N) {
            const uint16_t *wr = W + (size_t)n * K + pc;
            if (bf16) {
                for (int k = 0; k < kc; k++) {
                    union { uint32_t u; float f; } v; v.u = (uint32_t)wr[k] << 16;
                    dst[k * PNR + j] = v.f;
                }
            } else {
#if defined(__F16C__)
                for (int k = 0; k < kc; k++) dst[k * PNR + j] = _cvtsh_ss(wr[k]);   /* exact */
#else
                for (int k = 0; k < kc; k++) dst[k * PNR + j] = aria_half_to_float(wr[k]);
#endif
            }
        } else for (int k = 0; k < kc; k++) dst[k * PNR + j] = 0.0f;
    }
}
#if defined(ARIA_AVX2)
/* K-blocked microkernel: accumulates a KC-slice into y (C in memory across K-blocks).
 * full-N tile (n0+PNR<=N), edge-M guarded; bias added on the last K-block. */
static inline void ukern_acc(float *y, int N, const float *Ap, const float *Bp, int kc,
                             const float *bias, int m0, int n0, int M, int first, int last) {
    __m256 c0[PMR], c1[PMR];
    int em = m0 + PMR > M;
    for (int i = 0; i < PMR; i++) {
        if (first || (em && m0 + i >= M)) { c0[i] = _mm256_setzero_ps(); c1[i] = _mm256_setzero_ps(); }
        else { float *yr = y + (size_t)(m0 + i) * N + n0; c0[i] = _mm256_loadu_ps(yr); c1[i] = _mm256_loadu_ps(yr + 8); }
    }
    for (int k = 0; k < kc; k++) {
        __m256 b0 = _mm256_loadu_ps(Bp + (size_t)k * PNR), b1 = _mm256_loadu_ps(Bp + (size_t)k * PNR + 8);
        const float *ap = Ap + (size_t)k * PMR;
        for (int i = 0; i < PMR; i++) {
            __m256 a = _mm256_set1_ps(ap[i]);
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
        }
    }
    if (last && bias) {
        __m256 vb0 = _mm256_loadu_ps(bias + n0), vb1 = _mm256_loadu_ps(bias + n0 + 8);
        for (int i = 0; i < PMR; i++) { c0[i] = _mm256_add_ps(c0[i], vb0); c1[i] = _mm256_add_ps(c1[i], vb1); }
    }
    for (int i = 0; i < PMR && m0 + i < M; i++) {
        float *yr = y + (size_t)(m0 + i) * N + n0;
        _mm256_storeu_ps(yr, c0[i]); _mm256_storeu_ps(yr + 8, c1[i]);
    }
}
static inline void ukern_6x16(float *y, int N, const float *Ap, const float *Bp, int K,
                              const float *bias, int m0, int n0, int M) {
    __m256 c0[PMR], c1[PMR];
    for (int i = 0; i < PMR; i++) { c0[i] = _mm256_setzero_ps(); c1[i] = _mm256_setzero_ps(); }
    for (int k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(Bp + (size_t)k * PNR), b1 = _mm256_loadu_ps(Bp + (size_t)k * PNR + 8);
        const float *ap = Ap + (size_t)k * PMR;
        for (int i = 0; i < PMR; i++) {
            __m256 a = _mm256_set1_ps(ap[i]);
            c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
            c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
        }
    }
    if (n0 + PNR <= N && m0 + PMR <= M) {                 /* full tile: vector store */
        __m256 vb0 = bias ? _mm256_loadu_ps(bias + n0) : _mm256_setzero_ps();
        __m256 vb1 = bias ? _mm256_loadu_ps(bias + n0 + 8) : _mm256_setzero_ps();
        for (int i = 0; i < PMR; i++) {
            float *yr = y + (size_t)(m0 + i) * N + n0;
            _mm256_storeu_ps(yr,     _mm256_add_ps(c0[i], vb0));
            _mm256_storeu_ps(yr + 8, _mm256_add_ps(c1[i], vb1));
        }
    } else {                                              /* edge tile: scalar store */
        float t0[8], t1[8];
        for (int i = 0; i < PMR && m0 + i < M; i++) {
            _mm256_storeu_ps(t0, c0[i]); _mm256_storeu_ps(t1, c1[i]);
            float *yr = y + (size_t)(m0 + i) * N;
            for (int j = 0; j < PNR; j++) {
                int n = n0 + j;
                if (n < N) yr[n] = (j < 8 ? t0[j] : t1[j - 8]) + (bias ? bias[n] : 0.0f);
            }
        }
    }
}
#elif defined(ARIA_NEON)
/* NEON port of the two microkernels above. Same PMR x PNR register tile (6x16 =
 * 24 float32x4 accumulators, + PNR/4 B-vecs + 1 broadcast A ~= 29 of 32 q-regs),
 * same pure outer product (vld1q_dup broadcast of A, vfmaq into PNR-wide B
 * accumulators), same sequential-k accumulation order as the AVX2 path. */
#define NBV (PNR / 4)
static inline void ukern_acc(float *y, int N, const float *Ap, const float *Bp, int kc,
                             const float *bias, int m0, int n0, int M, int first, int last) {
    float32x4_t c[PMR][NBV];
    int em = m0 + PMR > M;
    for (int i = 0; i < PMR; i++) {
        if (first || (em && m0 + i >= M)) { for (int j = 0; j < NBV; j++) c[i][j] = vdupq_n_f32(0.0f); }
        else { float *yr = y + (size_t)(m0 + i) * N + n0; for (int j = 0; j < NBV; j++) c[i][j] = vld1q_f32(yr + 4 * j); }
    }
    for (int k = 0; k < kc; k++) {
        float32x4_t bb[NBV];
        for (int j = 0; j < NBV; j++) bb[j] = vld1q_f32(Bp + (size_t)k * PNR + 4 * j);
        const float *ap = Ap + (size_t)k * PMR;
        for (int i = 0; i < PMR; i++) {
            float32x4_t a = vld1q_dup_f32(ap + i);
            for (int j = 0; j < NBV; j++) c[i][j] = vfmaq_f32(c[i][j], a, bb[j]);
        }
    }
    if (last && bias)
        for (int i = 0; i < PMR; i++)
            for (int j = 0; j < NBV; j++) c[i][j] = vaddq_f32(c[i][j], vld1q_f32(bias + n0 + 4 * j));
    for (int i = 0; i < PMR && m0 + i < M; i++) {
        float *yr = y + (size_t)(m0 + i) * N + n0;
        for (int j = 0; j < NBV; j++) vst1q_f32(yr + 4 * j, c[i][j]);
    }
}
static inline void ukern_6x16(float *y, int N, const float *Ap, const float *Bp, int K,
                              const float *bias, int m0, int n0, int M) {
    float32x4_t c[PMR][NBV];
    for (int i = 0; i < PMR; i++) for (int j = 0; j < NBV; j++) c[i][j] = vdupq_n_f32(0.0f);
    for (int k = 0; k < K; k++) {
        float32x4_t bb[NBV];
        for (int j = 0; j < NBV; j++) bb[j] = vld1q_f32(Bp + (size_t)k * PNR + 4 * j);
        const float *ap = Ap + (size_t)k * PMR;
        for (int i = 0; i < PMR; i++) {
            float32x4_t a = vld1q_dup_f32(ap + i);
            for (int j = 0; j < NBV; j++) c[i][j] = vfmaq_f32(c[i][j], a, bb[j]);
        }
    }
    if (n0 + PNR <= N && m0 + PMR <= M) {                 /* full tile: vector store */
        for (int i = 0; i < PMR; i++) {
            float *yr = y + (size_t)(m0 + i) * N + n0;
            for (int j = 0; j < NBV; j++) {
                float32x4_t o = c[i][j];
                if (bias) o = vaddq_f32(o, vld1q_f32(bias + n0 + 4 * j));
                vst1q_f32(yr + 4 * j, o);
            }
        }
    } else {                                              /* edge tile: scalar store */
        float t[PNR];
        for (int i = 0; i < PMR && m0 + i < M; i++) {
            for (int j = 0; j < NBV; j++) vst1q_f32(t + 4 * j, c[i][j]);
            float *yr = y + (size_t)(m0 + i) * N;
            for (int j = 0; j < PNR; j++) {
                int n = n0 + j;
                if (n < N) yr[n] = t[j] + (bias ? bias[n] : 0.0f);
            }
        }
    }
}
#endif  /* ARIA_AVX2 / ARIA_NEON microkernel */
/* edge-N tile (n0 + PNR > N): run the full-width kernel on a zero-padded copy and
 * store only the valid columns -- each kept element is the same FMA chain. */
static void ukern_edge(float *y, int N, const float *Ap, const float *Bp, int kc,
                       const float *bias, int m0, int n0, int M, int first, int last) {
    float t[PMR * PNR], tb[PNR];
    int nv = N - n0 < PNR ? N - n0 : PNR, mv = M - m0 < PMR ? M - m0 : PMR;
    for (int i = 0; i < PMR; i++)
        for (int j = 0; j < PNR; j++)
            t[i * PNR + j] = (!first && i < mv && j < nv) ? y[(size_t)(m0 + i) * N + n0 + j] : 0.0f;
    for (int j = 0; j < PNR; j++) tb[j] = (bias && j < nv) ? bias[n0 + j] : 0.0f;
    ukern_acc(t, PNR, Ap, Bp, kc, bias ? tb : NULL, 0, 0, PMR, 0, last);   /* t preloaded */
    for (int i = 0; i < mv; i++)
        for (int j = 0; j < nv; j++) y[(size_t)(m0 + i) * N + n0 + j] = t[i * PNR + j];
}

/* GEMM cache blocking, sized once from the running CPU's caches. It never changes bits:
 * every output stays one sequential-k FMA chain (C round-trips memory exactly between
 * K-blocks; the accumulator starts at +0, so it is never -0 and a bias-less store needs
 * no +0 add). KC: the KC x 16 B micro-panel fills ~half of L1; MC: the MC x KC A block
 * fills ~1/4 of L2 (HT siblings share it). ARIA_GEMM_KC / ARIA_GEMM_MC override. */
static int gemm_kc = 0, gemm_mc = 0;
static void gemm_blocking(void) {
    if (gemm_kc) return;
    long l1 = sysconf(_SC_LEVEL1_DCACHE_SIZE), l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l1 <= 0) l1 = 32 * 1024;
    if (l2 <= 0) l2 = 256 * 1024;
    int kc = (int)(l1 / 2 / (PNR * (long)sizeof(float))) / 8 * 8;
    kc = kc < 128 ? 128 : kc > 512 ? 512 : kc;
    int mc = (int)(l2 / 4 / (kc * (long)sizeof(float))) / PMR * PMR;
    mc = mc < 4 * PMR ? 4 * PMR : mc > 64 * PMR ? 64 * PMR : mc;
    const char *e;
    if ((e = getenv("ARIA_GEMM_KC")) && atoi(e) > 0) kc = atoi(e);
    if ((e = getenv("ARIA_GEMM_MC")) && atoi(e) > 0) mc = (atoi(e) + PMR - 1) / PMR * PMR;
    gemm_mc = mc;
    gemm_kc = kc;   /* written last: it is the "initialized" flag */
}

static void pack_b_any(float *dst, const void *W, int wt, int n0, int N, int pc, int kc, int K) {
    if (wt == 0) pack_b_panel(dst, (const float *)W, n0, N, pc, kc, K);
    else         pack_b_panel_h(dst, (const uint16_t *)W, n0, N, pc, kc, K, wt == 2);
}

/* Two drivers for y[M,N] = x[M,K] . W[N,K]^T (+ b), W fp32 (wt 0) / fp16 (1) / bf16 (2).
 * Both keep every output one sequential-k FMA chain, so either gives the same bits.
 *  - panels: each thread packs a 16-column weight panel and sweeps all of A with it
 *    (weights stream once, packing overlaps compute; A re-read per panel). Best while
 *    packed A is cache-sized -- the DiT, T5.
 *  - blocked: per KC K-block, pack B~ once (shared) and sweep (MC x NB-panel) tiles, so A
 *    is read per tile from L2 instead of per panel from L3/DRAM -- big-A GEMMs (decoders).
 * K>2048 GEMMs keep their edge-N columns (N % 16) on aria_dot in both (no SA3 shape). */
static void gemm_panels(float *y, const float *x, const void *W, int wt,
                        const float *b, int M, int K, int N, const float *Ap) {
    const int np_m = (M + PMR - 1) / PMR, np_n = (N + PNR - 1) / PNR;
    /* large K: block K so the NR-wide B-panel stays L1/L2-resident across the M loop
     * (a single NR*K panel overflows L2 for K>=4096, halving throughput). C is then
     * accumulated across K-blocks in memory. */
    const int KC = 1024, kblk = (K > 2048), Nfull = (N / PNR) * PNR;
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        /* pack A in parallel into the caller's buffer; the for's barrier makes every
         * panel visible before any microkernel reads it (same copy, same layout) */
        #ifdef _OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int p = 0; p < np_m; p++) pack_a_panel((float *)Ap + (size_t)p * PMR * K, x, p * PMR, M, K);
        static __thread float *Bbuf = NULL; static __thread size_t Bcap = 0;
        float *Bp = grow_tls(&Bbuf, &Bcap, (size_t)PNR * (kblk ? KC : K));
        if (Bp && !kblk) {
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic)
            #endif
            for (int pn = 0; pn < np_n; pn++) {
                int n0 = pn * PNR;
                pack_b_any(Bp, W, wt, n0, N, 0, K, K);
                for (int pm = 0; pm < np_m; pm++)
                    ukern_6x16(y, N, Ap + (size_t)pm * PMR * K, Bp, K, b, pm * PMR, n0, M);
            }
        } else if (Bp) {
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic)
            #endif
            for (int pn = 0; pn < Nfull / PNR; pn++) {   /* full N-panels (large-K N is /16) */
                int n0 = pn * PNR;
                for (int pc = 0; pc < K; pc += KC) {
                    int kc = pc + KC <= K ? KC : K - pc;
                    pack_b_any(Bp, W, wt, n0, N, pc, kc, K);
                    for (int pm = 0; pm < np_m; pm++)
                        ukern_acc(y, N, Ap + (size_t)pm * PMR * K + pc * PMR, Bp, kc, b,
                                  pm * PMR, n0, M, pc == 0, pc + kc >= K);
                }
            }
        }
    }
}

static void gemm_blocked(float *y, const float *x, const void *W, int wt,
                         const float *b, int M, int K, int N, float *Ap) {
    gemm_blocking();
    const int KC = gemm_kc, MC = gemm_mc;
    static __thread float *Bbuf = NULL; static __thread size_t Bcap = 0;
    const int Nk = K > 2048 ? (N / PNR) * PNR : N;    /* columns the microkernel owns */
    const int np_m = (M + PMR - 1) / PMR, npk = (Nk + PNR - 1) / PNR;
    float *Bt = grow_tls(&Bbuf, &Bcap, (size_t)npk * PNR * (K < KC ? K : KC) + 1);
    if (!Bt) return;
    int nth = 1;
    #ifdef _OPENMP
    nth = omp_in_parallel() ? 1 : omp_get_max_threads();
    #endif
    const int n_mb = (M + MC - 1) / MC;
    int NB = npk * n_mb / (4 * nth);                  /* aim for >= 4 tiles per thread */
    NB = NB < 1 ? 1 : NB > 8 ? 8 : NB;
    const int n_nb = (npk + NB - 1) / NB, ntiles = n_mb * n_nb;
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        #ifdef _OPENMP
        #pragma omp for schedule(static)
        #endif
        for (int p = 0; p < np_m; p++) pack_a_panel(Ap + (size_t)p * PMR * K, x, p * PMR, M, K);
        for (int pc = 0; pc < K; pc += KC) {          /* K-blocks in order: the chain order */
            int kc = pc + KC <= K ? KC : K - pc;
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (int pn = 0; pn < npk; pn++)
                pack_b_any(Bt + (size_t)pn * PNR * kc, W, wt, pn * PNR, Nk, pc, kc, K);
            #ifdef _OPENMP
            #pragma omp for schedule(dynamic, 1)
            #endif
            for (int t = 0; t < ntiles; t++) {
                int m0 = (t / n_nb) * MC, m1 = m0 + MC < M ? m0 + MC : M;
                int pn0 = (t % n_nb) * NB, pn1 = pn0 + NB < npk ? pn0 + NB : npk;
                for (int pn = pn0; pn < pn1; pn++) {
                    const float *Bp = Bt + (size_t)pn * PNR * kc;
                    int n0 = pn * PNR;
                    for (int m = m0; m < m1; m += PMR) {
                        const float *Apm = Ap + (size_t)(m / PMR) * PMR * K + (size_t)pc * PMR;
                        if (n0 + PNR <= Nk) ukern_acc(y, N, Apm, Bp, kc, b, m, n0, M, pc == 0, pc + kc >= K);
                        else                ukern_edge(y, N, Apm, Bp, kc, b, m, n0, M, pc == 0, pc + kc >= K);
                    }
                }
            }
        }
    }
}

/* packed A above this many bytes takes the blocked driver (ARIA_GEMM_BLOCK_A=<MB>) */
static size_t gemm_block_a = 0;
static void aria_linear_packed_w(float *y, const float *x, const void *W, int wt,
                                 const float *b, int M, int K, int N) {
    if (!gemm_block_a) {
        const char *e = getenv("ARIA_GEMM_BLOCK_A");
        gemm_block_a = (e && atof(e) > 0 ? (size_t)(atof(e) * 1048576.0) : (size_t)8 << 20);
    }
    static __thread float *Abuf = NULL; static __thread size_t Acap = 0;
    const int np_m = (M + PMR - 1) / PMR;
    size_t abytes = (size_t)np_m * PMR * K * sizeof(float);
    float *Ap = grow_tls(&Abuf, &Acap, abytes / sizeof(float));
    if (!Ap) return;
    if (abytes >= gemm_block_a) gemm_blocked(y, x, W, wt, b, M, K, N, Ap);
    else                        gemm_panels(y, x, W, wt, b, M, K, N, Ap);
    const int Nk = K > 2048 ? (N / PNR) * PNR : N;
    if (Nk < N)   /* rare edge-N for K>2048 GEMMs (N not a multiple of 16) */
        for (int m = 0; m < M; m++)
            for (int n = Nk; n < N; n++) {
                float acc;
                if (wt == 0) acc = aria_dot(x + (size_t)m * K, (const float *)W + (size_t)n * K, K);
                else {
                    const uint16_t *wr = (const uint16_t *)W + (size_t)n * K;
                    acc = 0.0f;
                    for (int k = 0; k < K; k++)
                        acc += x[(size_t)m * K + k] * (wt == 2 ? aria_bf16_to_float(wr[k]) : aria_half_to_float(wr[k]));
                }
                y[(size_t)m * N + n] = acc + (b ? b[n] : 0.0f);
            }
}
static void aria_linear_packed(float *y, const float *x, const float *W, const float *b,
                               int M, int K, int N) {
    aria_linear_packed_w(y, x, W, 0, b, M, K, N);
}
static void aria_linear_packed_h(float *y, const float *x, const uint16_t *W, int bf16,
                                 const float *b, int M, int K, int N) {
    aria_linear_packed_w(y, x, W, bf16 ? 2 : 1, b, M, K, N);
}
#endif  /* packed GEMM (ARIA_AVX2 || ARIA_NEON) */

void aria_f32_to_f16_n(uint16_t *dst, const float *src, size_t n) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i0 = 0; i0 < n; i0 += 4096) {
        size_t i1 = i0 + 4096 < n ? i0 + 4096 : n, i = i0;
#if defined(__F16C__)
        for (; i + 8 <= i1; i += 8)
            _mm_storeu_si128((__m128i *)(dst + i),
                             _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#endif
        for (; i < i1; i++) dst[i] = aria_float_to_half(src[i]);
    }
}

void aria_linear_packed_only(float *y, const float *x, const float *W, const float *b,
                             int M, int K, int N) {
#if defined(ARIA_AVX2) || defined(ARIA_NEON)
    aria_linear_packed(y, x, W, b, M, K, N);
#else
    aria_linear(y, x, W, b, M, K, N);   /* scalar build: one kernel for every size */
#endif
}

void aria_linear_hw(float *y, const float *x, const uint16_t *W, int bf16,
                    const float *b, int M, int K, int N) {
#if defined(ARIA_AVX2) || defined(ARIA_NEON)
    if ((size_t)M * N * K >= (1u << 18)) {
        aria_linear_packed_h(y, x, W, bf16, b, M, K, N);
        return;
    }
#endif
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {   /* small-GEMM fallback: widen-on-the-fly dot */
        const float *xr = x + (size_t)m * K;
        float *yr = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const uint16_t *wr = W + (size_t)n * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += xr[k] * (bf16 ? aria_bf16_to_float(wr[k]) : aria_half_to_float(wr[k]));
            yr[n] = acc + (b ? b[n] : 0.0f);
        }
    }
}

void aria_linear(float *y, const float *x, const float *W, const float *b,
                 int M, int K, int N) {
#if defined(ARIA_AVX2) || defined(ARIA_NEON)
    /* packed outer-product GEMM amortizes A-packing over the M*N*K work; for tiny
     * GEMMs the packing/alloc overhead isn't worth it, so use the dot-product (AVX2)
     * or scalar (NEON/other) kernel below. */
    if ((size_t)M * N * K >= (1u << 18)) {
        aria_linear_packed(y, x, W, b, M, K, N);
        return;
    }
#endif
#ifdef ARIA_AVX2
    {
        const int MR = ARIA_MR, NR = ARIA_NR;
        int Mfull = M - (M % MR);
        int Nfull = N - (N % NR);
        int NBN = ((48 + NR - 1) / NR) * NR;       /* ~L2 weight panel, multiple of NR */
        int n_chunks = (Nfull > 0) ? (Nfull + NBN - 1) / NBN : 0;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic)
        #endif
        for (int c = 0; c < n_chunks; c++) {
            int cn0 = c * NBN;
            int cn1 = cn0 + NBN < Nfull ? cn0 + NBN : Nfull;
            for (int mm = 0; mm < Mfull; mm += MR)
                for (int nn = cn0; nn < cn1; nn += NR)
                    aria_microkernel(y, x, W, b, mm, nn, K, N);
        }
        /* N-edge: cols [Nfull,N) for rows [0,Mfull) */
        for (int m = 0; m < Mfull; m++)
            for (int n = Nfull; n < N; n++)
                y[(size_t)m * N + n] = aria_dot(x + (size_t)m * K, W + (size_t)n * K, K) + (b ? b[n] : 0.0f);
        /* M-edge: rows [Mfull,M) for all cols */
        for (int m = Mfull; m < M; m++)
            for (int n = 0; n < N; n++)
                y[(size_t)m * N + n] = aria_dot(x + (size_t)m * K, W + (size_t)n * K, K) + (b ? b[n] : 0.0f);
        return;
    }
#else
    const int NB = 32;                 /* NB*K*4 bytes ~ L2-resident weight panel */
    int n_tiles = (N + NB - 1) / NB;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int tile = 0; tile < n_tiles; tile++) {
        int n0 = tile * NB;
        int n1 = (n0 + NB < N) ? n0 + NB : N;
        for (int m = 0; m < M; m++) {
            const float *xr = x + (size_t)m * K;
            float *yr = y + (size_t)m * N;
            for (int n = n0; n < n1; n++) {
                float acc = b ? b[n] : 0.0f;
                acc += aria_dot(xr, W + (size_t)n * K, K);
                yr[n] = acc;
            }
        }
    }
#endif
}

void aria_matmul(float *C, const float *A, const float *B, int M, int K, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const float *ar = A + (size_t)m * K;
        float *cr = C + (size_t)m * N;
        for (int n = 0; n < N; n++) cr[n] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = ar[k];
            const float *br = B + (size_t)k * N;
            for (int n = 0; n < N; n++) cr[n] += a * br[n];
        }
    }
}

void aria_rmsnorm(float *y, const float *x, const float *weight,
                  int rows, int dim, float eps) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        float inv = (float)(1.0 / sqrt(aria_sumsq_f(xr, dim) / dim + eps));
#ifdef ARIA_AVX2
        __m256 vi = _mm256_set1_ps(inv);
        int i = 0;
        if (weight) {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i), vi),
                                                       _mm256_loadu_ps(weight + i)));
            for (; i < dim; i++) yr[i] = xr[i] * inv * weight[i];
        } else {
            for (; i + 8 <= dim; i += 8)
                _mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_loadu_ps(xr + i), vi));
            for (; i < dim; i++) yr[i] = xr[i] * inv;
        }
#elif defined(ARIA_NEON)
        float32x4_t vi = vdupq_n_f32(inv);
        int i = 0;
        if (weight) {
            for (; i + 4 <= dim; i += 4)
                vst1q_f32(yr + i, vmulq_f32(vmulq_f32(vld1q_f32(xr + i), vi), vld1q_f32(weight + i)));
            for (; i < dim; i++) yr[i] = xr[i] * inv * weight[i];
        } else {
            for (; i + 4 <= dim; i += 4)
                vst1q_f32(yr + i, vmulq_f32(vld1q_f32(xr + i), vi));
            for (; i < dim; i++) yr[i] = xr[i] * inv;
        }
#else
        if (weight) { for (int i = 0; i < dim; i++) yr[i] = xr[i] * inv * weight[i]; }
        else        { for (int i = 0; i < dim; i++) yr[i] = xr[i] * inv; }
#endif
    }
}

void aria_dynamic_tanh(float *y, const float *x, float alpha,
                       const float *weight, const float *bias,
                       int rows, int dim) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        for (int i = 0; i < dim; i++) {
            float t = tanhf(alpha * xr[i]);
            yr[i] = t * (weight ? weight[i] : 1.0f) + (bias ? bias[i] : 0.0f);
        }
    }
}

void aria_gemma_rmsnorm(float *y, const float *x, const float *weight,
                        int rows, int dim, float eps) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        float inv = (float)(1.0 / sqrt(aria_sumsq_f(xr, dim) / dim + eps));
#ifdef ARIA_AVX2
        const __m256 one = _mm256_set1_ps(1.0f);
        __m256 vi = _mm256_set1_ps(inv);
        int i = 0;
        for (; i + 8 <= dim; i += 8)
            _mm256_storeu_ps(yr + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(xr + i), vi),
                                                   _mm256_add_ps(one, _mm256_loadu_ps(weight + i))));
        for (; i < dim; i++) yr[i] = xr[i] * inv * (1.0f + weight[i]);
#elif defined(ARIA_NEON)
        const float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t vi = vdupq_n_f32(inv);
        int i = 0;
        for (; i + 4 <= dim; i += 4)
            vst1q_f32(yr + i, vmulq_f32(vmulq_f32(vld1q_f32(xr + i), vi),
                                        vaddq_f32(one, vld1q_f32(weight + i))));
        for (; i < dim; i++) yr[i] = xr[i] * inv * (1.0f + weight[i]);
#else
        for (int i = 0; i < dim; i++) yr[i] = xr[i] * inv * (1.0f + weight[i]);
#endif
    }
}

void aria_softcap(float *s, int n, float cap) {
    for (int i = 0; i < n; i++) s[i] = cap * tanhf(s[i] / cap);
}

void aria_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v));
    }
}

void aria_gelu_tanh(float *x, int n) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float inner = c * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

void aria_silu_gate(float *out, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void aria_softmax_inplace(float *x, int rows, int cols, const float *mask) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * cols;
        const float *mr = mask ? mask + (size_t)r * cols : NULL;
        float maxv = -INFINITY;
        for (int i = 0; i < cols; i++) {
            float v = xr[i] + (mr ? mr[i] : 0.0f);
            xr[i] = v;
            if (v > maxv) maxv = v;
        }
        float sum = 0.0f;
        for (int i = 0; i < cols; i++) { float e = expf(xr[i] - maxv); xr[i] = e; sum += e; }
        float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int i = 0; i < cols; i++) xr[i] *= inv;
    }
}

void aria_rope_freqs(float *cos_t, float *sin_t, int n, int rot_dim, float base) {
    int rh = rot_dim / 2;
    for (int i = 0; i < rh; i++) {
        double inv = 1.0 / pow((double)base, (double)(2 * i) / (double)rot_dim);
        for (int p = 0; p < n; p++) {
            double ang = (double)p * inv;
            cos_t[(size_t)p * rh + i] = (float)cos(ang);
            sin_t[(size_t)p * rh + i] = (float)sin(ang);
        }
    }
}

void aria_rope_apply(float *x, const float *cos_t, const float *sin_t,
                     int H, int N, int D, int rot_dim) {
    int rh = rot_dim / 2;
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int h = 0; h < H; h++) {
        for (int p = 0; p < N; p++) {
            float *row = x + ((size_t)h * N + p) * D;
            const float *c = cos_t + (size_t)p * rh;
            const float *s = sin_t + (size_t)p * rh;
            for (int i = 0; i < rh; i++) {
                float a = row[i], b = row[i + rh];
                row[i]      = a * c[i] - b * s[i];
                row[i + rh] = b * c[i] + a * s[i];
            }
            /* dims >= rot_dim left unchanged */
        }
    }
}

void aria_attention(float *out, const float *q, const float *k, const float *v,
                    int H, int Nq, int Nk, int D, const float *mask, float *scratch) {
    float scale = 1.0f / sqrtf((float)D);
    (void)scratch;   /* parallel-over-heads: each head uses its own thread-local scores */
    /* Parallelize over HEADS (H >= cores), so each per-head QK/softmax/AV runs serially
     * on one thread -- the inner aria_linear/aria_matmul/softmax OpenMP regions become
     * nested and run single-threaded. Avoids spawning an 8-thread region per tiny K=D
     * head GEMM, and parallelizes the softmax across heads.
     * Within a head, process queries in BQ-row blocks so the [BQ,Nk] score tile stays
     * L2-resident through QK->softmax->AV instead of streaming a full [Nq,Nk] matrix
     * (2 MB/head at Nq=Nk=710) through DRAM three times. Rows are independent, so the
     * output is identical to the single-shot path. */
    /* A4: parallelize over (head, query-block) pairs instead of heads only -- H=16
     * heads on a 20-thread part left 4 cores idle plus tail imbalance; H x ceil(Nq/BQ)
     * units fill the machine. Each (h,qb) computation is unchanged -> bit-identical. */
    const int BQ = 64;
    const int nqb = (Nq + BQ - 1) / BQ;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) collapse(2)
    #endif
    for (int h = 0; h < H; h++) {
        for (int qbi = 0; qbi < nqb; qbi++) {
            static __thread float *sc = NULL; static __thread size_t sccap = 0;
            int bqcap = Nq < BQ ? Nq : BQ;
            float *scores = grow_tls(&sc, &sccap, (size_t)bqcap * Nk);
            if (!scores) continue;
            const float *qh = q + (size_t)h * Nq * D;
            const float *kh = k + (size_t)h * Nk * D;
            const float *vh = v + (size_t)h * Nk * D;
            float *oh = out + (size_t)h * Nq * D;
            int qb = qbi * BQ;
            int bq = (Nq - qb < BQ) ? (Nq - qb) : BQ;
            aria_linear(scores, qh + (size_t)qb * D, kh, NULL, bq, D, Nk);  /* [bq,Nk] = qh . kh^T */
            for (size_t i = 0; i < (size_t)bq * Nk; i++) scores[i] *= scale;
            aria_softmax_inplace(scores, bq, Nk, mask ? mask + (size_t)qb * Nk : NULL);
            aria_matmul(oh + (size_t)qb * D, scores, vh, bq, Nk, D);
        }
    }
}

void aria_attention_band(float *out, const float *q, const float *k, const float *v,
                         int H, int N, int D, int W) {
    /* sliding-window self-attention: query i attends only to keys [i-W, i+W].
     * Identical result to aria_attention with a [-W,W] additive band mask, but
     * O(N*(2W+1)*D) instead of O(N^2*D) -- the band is ~3% of the full matrix at
     * the medium decoder's sequence lengths. Parallel over (head, 64-query block).
     * AVX2: each block transposes its key range once, then a query's scores are
     * computed 8 keys at a time with the d-loop kept sequential as mul-then-add --
     * the same rounding sequence as the scalar `s += q[d]*k[d]`, so bit-identical
     * (the max is order-free; exp + the sequential sum stay scalar). AV vectorizes
     * over d with the same mul-then-add per element. */
    float scale = 1.0f / sqrtf((float)D);
    const int BQ = 64;
    const int nqb = (N + BQ - 1) / BQ;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) collapse(2)
    #endif
    for (int h = 0; h < H; h++) {
        for (int qbi = 0; qbi < nqb; qbi++) {
            static __thread float *sc = NULL; static __thread size_t sccap = 0;
            float *scores = grow_tls(&sc, &sccap, (size_t)(2 * W + 1 + 8));
            if (!scores) continue;
            const float *qh = q + (size_t)h * N * D;
            const float *kh = k + (size_t)h * N * D;
            const float *vh = v + (size_t)h * N * D;
            float *oh = out + (size_t)h * N * D;
            int i0 = qbi * BQ, i1 = i0 + BQ < N ? i0 + BQ : N;
#if defined(ARIA_AVX2)
            int j0 = i0 - W < 0 ? 0 : i0 - W, j1 = i1 + W < N ? i1 + W : N;
            int ldk = ((j1 - j0) + 7) / 8 * 8 + 8;               /* padded, so 8-wide loads stay inside */
            static __thread float *kt = NULL; static __thread size_t ktcap = 0;
            float *kT = grow_tls(&kt, &ktcap, (size_t)D * ldk);   /* kT[d][j - j0] */
            if (!kT) continue;
            for (int d = 0; d < D; d++) {
                float *kr = kT + (size_t)d * ldk;
                for (int j = j0; j < j1; j++) kr[j - j0] = kh[(size_t)j * D + d];
                for (int j = j1 - j0; j < ldk; j++) kr[j] = 0.0f;
            }
            const __m256 vscale = _mm256_set1_ps(scale);
#endif
            for (int i = i0; i < i1; i++) {
                int lo = i - W < 0 ? 0 : i - W;
                int hi = i + W >= N ? N - 1 : i + W;
                const float *qi = qh + (size_t)i * D;
                float maxv = -INFINITY;
#if defined(ARIA_AVX2)
                for (int jj = lo; jj <= hi; jj += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    const float *kc = kT + (jj - j0);
                    for (int d = 0; d < D; d++)
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_set1_ps(qi[d]), _mm256_loadu_ps(kc + (size_t)d * ldk)));
                    _mm256_storeu_ps(scores + (jj - lo), _mm256_mul_ps(acc, vscale));
                }
                for (int j = lo; j <= hi; j++) if (scores[j - lo] > maxv) maxv = scores[j - lo];
#else
                for (int j = lo; j <= hi; j++) {
                    const float *kj = kh + (size_t)j * D;
                    float s = 0.0f;
                    for (int d = 0; d < D; d++) s += qi[d] * kj[d];
                    s *= scale;
                    scores[j - lo] = s;
                    if (s > maxv) maxv = s;
                }
#endif
                float sum = 0.0f;
                for (int j = lo; j <= hi; j++) { float e = expf(scores[j - lo] - maxv); scores[j - lo] = e; sum += e; }
                float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
                float *oi = oh + (size_t)i * D;
                for (int d = 0; d < D; d++) oi[d] = 0.0f;
                for (int j = lo; j <= hi; j++) {
                    float wgt = scores[j - lo] * inv;
                    const float *vj = vh + (size_t)j * D;
#if defined(ARIA_AVX2)
                    int d = 0;
                    const __m256 vw = _mm256_set1_ps(wgt);
                    for (; d + 8 <= D; d += 8)
                        _mm256_storeu_ps(oi + d, _mm256_add_ps(_mm256_loadu_ps(oi + d), _mm256_mul_ps(vw, _mm256_loadu_ps(vj + d))));
                    for (; d < D; d++) oi[d] += wgt * vj[d];
#else
                    for (int d = 0; d < D; d++) oi[d] += wgt * vj[d];
#endif
                }
            }
        }
    }
}

void aria_conv1d(float *out, const float *in, const float *w, const float *bias,
                 int Cin, int Cout, int K, int pad, int L) {
    /* A4: loop-interchanged -- taps outer, t inner. Each output element accumulates
     * its (i,k) terms in the SAME ascending order as the old per-t scalar loop, so
     * the result is bit-identical; but the inner loop is now two contiguous streams
     * (outo[t] += wik * ini[t+k-pad]) that auto-vectorize instead of a branchy
     * strided gather. The old form was the top decode kernel on the CPU/Pi. */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int o = 0; o < Cout; o++) {
        const float *wo = w + (size_t)o * Cin * K;
        float *outo = out + (size_t)o * L;
        float b = bias ? bias[o] : 0.0f;
        for (int t = 0; t < L; t++) outo[t] = b;
        for (int i = 0; i < Cin; i++) {
            const float *wi = wo + (size_t)i * K;
            const float *ini = in + (size_t)i * L;
            for (int k = 0; k < K; k++) {
                float wik = wi[k];
                int t0 = pad - k > 0 ? pad - k : 0;            /* ti = t+k-pad in [0,L) */
                int t1 = L + pad - k < L ? L + pad - k : L;
                const float *src = ini + t0 + k - pad;
                #ifdef _OPENMP
                #pragma omp simd
                #endif
                for (int t = t0; t < t1; t++) outo[t] += wik * src[t - t0];
            }
        }
    }
}

void aria_ff_glu(float *out, const float *x, int N, int dim, int inner, int dim_out,
                 const float *W_in, const float *b_in,
                 const float *W_out, const float *b_out, float *scratch) {
    float *proj  = scratch ? scratch : malloc((size_t)N * 2 * inner * sizeof(float));
    float *gated = scratch ? scratch + (size_t)N * 2 * inner : malloc((size_t)N * inner * sizeof(float));
    if (!proj || !gated) { if (!scratch) { free(proj); free(gated); } return; }
    aria_linear(proj, x, W_in, b_in, N, dim, 2 * inner);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; n++) {
        const float *pr = proj + (size_t)n * 2 * inner;
        /* GLU: value = first half, gate = second half; out = value * silu(gate) */
        aria_silu_gate(gated + (size_t)n * inner, pr + inner, pr, inner);
    }
    aria_linear(out, gated, W_out, b_out, N, inner, dim_out);
    if (!scratch) { free(proj); free(gated); }
}

void aria_add(float *y, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) y[i] = a[i] + b[i];
}
void aria_addto(float *y, const float *a, int n) {
    for (int i = 0; i < n; i++) y[i] += a[i];
}
void aria_scale(float *y, const float *x, float s, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] * s;
}
void aria_axpy(float *y, const float *x, float s, int n) {
    for (int i = 0; i < n; i++) y[i] += s * x[i];
}
