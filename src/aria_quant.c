/*
 * aria_quant.c - weight quantization + dequant-on-use GEMM (CPU). See aria_quant.h.
 * Correctness-first scalar kernels (SIMD is a later optimization); the GEMM shape
 * mirrors aria_linear so the model can dispatch on dtype with no other change.
 */

#include "aria_quant.h"
#include "aria_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
/* E15.2b: NEON dequant-on-use GEMM. int8/int4 weights are widened to f32 and
 * multiplied by the f32 activations (vld1q_s8 -> vmovl -> vcvtq_f32_s32 -> vfmaq),
 * so the math is identical to the scalar dequant path (relerr matches, not the
 * int8xint8 vdotq_s32 route which would need activation quantization and raise the
 * error above the scalar tolerance). AArch64 horizontal reduce via vaddvq. */
#elif defined(__AVX2__)
#include <immintrin.h>
/* B4: x86 was scalar for every quant GEMM (the NEON branches above are ARM-only), so
 * q8 -- and especially W8A8 -- ran SLOWER than the packed fp32 kernel on x86. Two
 * AVX2 paths fix that: (1) q8 widening (cvtepi8->cvtdq2ps->fma, same fp32 math class
 * as the NEON widening); (2) q8a8 int8xint8 via the maddubs sign trick --
 * maddubs(|x|, sign(w,x)) pairs into int16 (max 127*127*2 = 32258 < 32767: safe
 * because BOTH sides clamp to +-127) then madd(.,1) into exact int32, bit-identical
 * to the scalar/NEON-sdot accumulation. */
static inline int32_t aria_avx2_hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    return _mm_cvtsi128_si32(s);
}
static inline float aria_avx2_hsum_f32(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 0x1));
    return _mm_cvtss_f32(lo);
}
#endif

const char *aria_dtype_name(aria_dtype dt) {
    switch (dt) {
        case ARIA_F32:  return "fp32";
        case ARIA_F16:  return "fp16";
        case ARIA_BF16: return "bf16";
        case ARIA_Q8:   return "q8";
        case ARIA_Q4:   return "q4";
        case ARIA_Q8A8: return "q8a8";
        case ARIA_DTYPE_AUTO: return "auto";
        default:        return "?";
    }
}

int aria_dtype_parse(const char *s, aria_dtype *out) {
    if (!s) return -1;
    if (!strcmp(s, "fp32") || !strcmp(s, "f32") || !strcmp(s, "float32")) *out = ARIA_F32;
    else if (!strcmp(s, "fp16") || !strcmp(s, "f16") || !strcmp(s, "half")) *out = ARIA_F16;
    else if (!strcmp(s, "bf16") || !strcmp(s, "bfloat16")) *out = ARIA_BF16;
    else if (!strcmp(s, "q8") || !strcmp(s, "int8")) *out = ARIA_Q8;
    else if (!strcmp(s, "q4") || !strcmp(s, "int4")) *out = ARIA_Q4;
    else if (!strcmp(s, "q8a8") || !strcmp(s, "w8a8")) *out = ARIA_Q8A8;
    else if (!strcmp(s, "auto")) *out = ARIA_DTYPE_AUTO;
    else return -1;
    return 0;
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* ---- Q8 ---- */

void aria_q8_quant(int8_t *q, float *scale, const float *W, int N, int K) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; n++) {
        const float *wr = W + (size_t)n * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) { float a = fabsf(wr[k]); if (a > amax) amax = a; }
        float s = amax / 127.0f;
        scale[n] = s;
        int8_t *qr = q + (size_t)n * K;
        if (s > 0.0f) {
            float inv = 1.0f / s;
            for (int k = 0; k < K; k++) qr[k] = (int8_t)clampi((int)lrintf(wr[k] * inv), -127, 127);
        } else {
            memset(qr, 0, (size_t)K);
        }
    }
}

void aria_linear_q8(float *y, const float *x, const int8_t *q, const float *scale,
                    const float *b, int M, int K, int N) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        float *yr = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const int8_t *qn = q + (size_t)n * K;
            float acc;
#if defined(__ARM_NEON)
            /* widen 16 int8 weights/iter to f32, fma with x; scale[n] factored out. */
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                int8x16_t qv = vld1q_s8(qn + k);
                int16x8_t lo = vmovl_s8(vget_low_s8(qv));
                int16x8_t hi = vmovl_s8(vget_high_s8(qv));
                a0 = vfmaq_f32(a0, vld1q_f32(xr + k),      vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
                a1 = vfmaq_f32(a1, vld1q_f32(xr + k + 4),  vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
                a2 = vfmaq_f32(a2, vld1q_f32(xr + k + 8),  vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
                a3 = vfmaq_f32(a3, vld1q_f32(xr + k + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
            }
            acc = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            for (; k < K; k++) acc += xr[k] * (float)qn[k];
#elif defined(__AVX2__)
            /* widen 16 int8 weights/iter to f32, fma with x (B4; mirrors the NEON path) */
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                __m128i q16 = _mm_loadu_si128((const __m128i *)(qn + k));
                __m256i w0 = _mm256_cvtepi8_epi32(q16);
                __m256i w1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q16, 8));
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + k),     _mm256_cvtepi32_ps(w0), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(xr + k + 8), _mm256_cvtepi32_ps(w1), a1);
            }
            acc = aria_avx2_hsum_f32(_mm256_add_ps(a0, a1));
            for (; k < K; k++) acc += xr[k] * (float)qn[k];
#else
            acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xr[k] * (float)qn[k];
#endif
            yr[n] = acc * scale[n] + (b ? b[n] : 0.0f);
        }
    }
}

/* ---- W8A8 (E15.2c): int8 activations x int8 weights, exact int32 accumulation ----
 * The activation row is quantized like a q8 weight row (symmetric absmax/127,
 * round-to-nearest -- mirrors the GPU k_quant_act_q8 exactly), then the dot runs in
 * integer math: int32 accumulation is exact (|acc| <= 127*127*K < 2^31 for K < 133k),
 * so the NEON sdot path and the scalar path are BIT-IDENTICAL. Opt-in via ARIA_W8A8
 * (checked in aria_linear_qw): trades ~q4-class fidelity for the fastest CPU q8 mode
 * (no per-element widening, 1/4 the fp32 weight traffic). */

/* per-thread activation scratch (same pattern as aria_cpu.c grow_tls) */
static int8_t *q8a8_tls(size_t need) {
    static __thread int8_t *buf = NULL; static __thread size_t cap = 0;
    if (need > cap) { free(buf); buf = (int8_t *)malloc(need); cap = buf ? need : 0; }
    return buf;
}

#if defined(__AVX2__) && (defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__)))
/* AVX-VNNI: vpdpbusd multiplies unsigned x signed bytes and adds groups of 4 straight
 * into int32. Activations become unsigned by +128 (xu = x ^ 0x80) and the extra
 * 128 * sum(w) is subtracted per column, so every output is the same exact int32 as the
 * scalar / maddubs / sdot paths (bit-identical) -- with 4x the multiply-adds per
 * instruction and a register-blocked 4x3 tile: each weight vector loaded once feeds 4
 * activation rows. Threads split the columns, so each streams its weights once while
 * the int8 activations stay in cache. */
#define ARIA_VNNI 1
#if defined(__AVXVNNI__)
#define DPBUSD(acc, u, s) _mm256_dpbusd_avx_epi32((acc), (u), (s))
#else
#define DPBUSD(acc, u, s) _mm256_dpbusd_epi32((acc), (u), (s))
#endif
#define VMR 4   /* activation rows per tile */
#define VNR 3   /* weight rows (output columns) per tile */

static void q8a8_vnni(float *y, const uint8_t *xu, const float *sx, const int8_t *q,
                      const float *scale, const float *b, int M, int K, int N, int32_t *cs) {
    const int K32 = K & ~31;
    int nb = (N + VNR - 1) / VNR;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 2)
    #endif
    for (int j = 0; j < nb; j++) {
        int n0 = j * VNR, nr = N - n0 < VNR ? N - n0 : VNR;
        for (int n = n0; n < n0 + nr; n++) {   /* 128 * sum(w) over the vector part */
            int32_t sum = 0; const int8_t *qn = q + (size_t)n * K;
            for (int k = 0; k < K32; k++) sum += qn[k];
            cs[n] = 128 * sum;
        }
        for (int m0 = 0; m0 < M; m0 += VMR) {
            int mr = M - m0 < VMR ? M - m0 : VMR;
            int32_t acc[VMR][VNR];
            if (mr == VMR && nr == VNR) {
                const int8_t *q0 = q + (size_t)n0 * K, *q1 = q0 + K, *q2 = q1 + K;
                const uint8_t *x0 = xu + (size_t)m0 * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
                __m256i a00 = _mm256_setzero_si256(), a01 = a00, a02 = a00, a10 = a00, a11 = a00, a12 = a00;
                __m256i a20 = a00, a21 = a00, a22 = a00, a30 = a00, a31 = a00, a32 = a00;
                for (int k = 0; k < K32; k += 32) {
                    __m256i w0 = _mm256_loadu_si256((const __m256i *)(q0 + k));
                    __m256i w1 = _mm256_loadu_si256((const __m256i *)(q1 + k));
                    __m256i w2 = _mm256_loadu_si256((const __m256i *)(q2 + k));
                    __m256i v = _mm256_loadu_si256((const __m256i *)(x0 + k));
                    a00 = DPBUSD(a00, v, w0); a01 = DPBUSD(a01, v, w1); a02 = DPBUSD(a02, v, w2);
                    v = _mm256_loadu_si256((const __m256i *)(x1 + k));
                    a10 = DPBUSD(a10, v, w0); a11 = DPBUSD(a11, v, w1); a12 = DPBUSD(a12, v, w2);
                    v = _mm256_loadu_si256((const __m256i *)(x2 + k));
                    a20 = DPBUSD(a20, v, w0); a21 = DPBUSD(a21, v, w1); a22 = DPBUSD(a22, v, w2);
                    v = _mm256_loadu_si256((const __m256i *)(x3 + k));
                    a30 = DPBUSD(a30, v, w0); a31 = DPBUSD(a31, v, w1); a32 = DPBUSD(a32, v, w2);
                }
                acc[0][0] = aria_avx2_hsum_i32(a00); acc[0][1] = aria_avx2_hsum_i32(a01); acc[0][2] = aria_avx2_hsum_i32(a02);
                acc[1][0] = aria_avx2_hsum_i32(a10); acc[1][1] = aria_avx2_hsum_i32(a11); acc[1][2] = aria_avx2_hsum_i32(a12);
                acc[2][0] = aria_avx2_hsum_i32(a20); acc[2][1] = aria_avx2_hsum_i32(a21); acc[2][2] = aria_avx2_hsum_i32(a22);
                acc[3][0] = aria_avx2_hsum_i32(a30); acc[3][1] = aria_avx2_hsum_i32(a31); acc[3][2] = aria_avx2_hsum_i32(a32);
            } else {   /* edge tile: one (m, n) pair at a time */
                for (int mi = 0; mi < mr; mi++)
                    for (int ni = 0; ni < nr; ni++) {
                        const int8_t *qn = q + (size_t)(n0 + ni) * K;
                        const uint8_t *xr = xu + (size_t)(m0 + mi) * K;
                        __m256i a = _mm256_setzero_si256();
                        for (int k = 0; k < K32; k += 32)
                            a = DPBUSD(a, _mm256_loadu_si256((const __m256i *)(xr + k)), _mm256_loadu_si256((const __m256i *)(qn + k)));
                        acc[mi][ni] = aria_avx2_hsum_i32(a);
                    }
            }
            for (int mi = 0; mi < mr; mi++) {
                int m = m0 + mi;
                const uint8_t *xr = xu + (size_t)m * K;
                for (int ni = 0; ni < nr; ni++) {
                    int n = n0 + ni;
                    const int8_t *qn = q + (size_t)n * K;
                    int32_t a = acc[mi][ni] - cs[n];
                    for (int k = K32; k < K; k++) a += ((int32_t)xr[k] - 128) * qn[k];   /* tail, signed */
                    y[(size_t)m * N + n] = (float)a * sx[m] * scale[n] + (b ? b[n] : 0.0f);
                }
            }
        }
    }
}
#endif

void aria_linear_q8a8(float *y, const float *x, const int8_t *q, const float *scale,
                      const float *b, int M, int K, int N) {
    int8_t *xq = q8a8_tls((size_t)M * K + (size_t)M * sizeof(float) + 64 + (size_t)N * sizeof(int32_t) + 64);
    float *sx = (float *)(void *)(xq + (((size_t)M * K + 63) & ~(size_t)63));
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {   /* quantize the activation rows (symmetric int8) */
        const float *xr = x + (size_t)m * K;
        int8_t *qr = xq + (size_t)m * K;
        float amax = 0.0f;
        for (int k = 0; k < K; k++) { float a = fabsf(xr[k]); if (a > amax) amax = a; }
        sx[m] = amax / 127.0f;
        if (amax > 0.0f) {
            float inv = 127.0f / amax;
            for (int k = 0; k < K; k++) qr[k] = (int8_t)clampi((int)lrintf(xr[k] * inv), -127, 127);
        } else {
            memset(qr, 0, (size_t)K);
        }
    }
#ifdef ARIA_VNNI
    {
        int32_t *cs = (int32_t *)(void *)((char *)sx + (((size_t)M * sizeof(float) + 63) & ~(size_t)63));
        uint8_t *xu = (uint8_t *)xq;   /* x + 128: flip the sign bit in place */
        for (size_t i = 0; i < (size_t)M * K; i++) xu[i] ^= 0x80;
        q8a8_vnni(y, xu, sx, q, scale, b, M, K, N, cs);
        return;
    }
#endif
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const int8_t *xr = xq + (size_t)m * K;
        float *yr = y + (size_t)m * N;
        float sm = sx[m];
        for (int n = 0; n < N; n++) {
            const int8_t *qn = q + (size_t)n * K;
            int32_t acc;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int32x4_t a2 = vdupq_n_s32(0), a3 = vdupq_n_s32(0);
            int k = 0;
            for (; k + 64 <= K; k += 64) {   /* 4x sdot: 64 int8 MACs/iter */
                a0 = vdotq_s32(a0, vld1q_s8(xr + k),      vld1q_s8(qn + k));
                a1 = vdotq_s32(a1, vld1q_s8(xr + k + 16), vld1q_s8(qn + k + 16));
                a2 = vdotq_s32(a2, vld1q_s8(xr + k + 32), vld1q_s8(qn + k + 32));
                a3 = vdotq_s32(a3, vld1q_s8(xr + k + 48), vld1q_s8(qn + k + 48));
            }
            for (; k + 16 <= K; k += 16) a0 = vdotq_s32(a0, vld1q_s8(xr + k), vld1q_s8(qn + k));
            acc = vaddvq_s32(vaddq_s32(vaddq_s32(a0, a1), vaddq_s32(a2, a3)));
            for (; k < K; k++) acc += (int32_t)xr[k] * qn[k];
#elif defined(__ARM_NEON)
            /* no-dotprod NEON: int8xint8 -> int16 (vmull), pairwise-add into int32.
             * Still exact integer math == the scalar/sdot result. */
            int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                int8x16_t xv = vld1q_s8(xr + k), qv = vld1q_s8(qn + k);
                a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(xv),  vget_low_s8(qv)));
                a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(xv), vget_high_s8(qv)));
            }
            acc = vaddvq_s32(vaddq_s32(a0, a1));
            for (; k < K; k++) acc += (int32_t)xr[k] * qn[k];
#elif defined(__AVX2__)
            /* B4: maddubs sign trick -- exact int32 (see header comment), 64 int8/iter */
            __m256i c0 = _mm256_setzero_si256(), c1 = _mm256_setzero_si256();
            const __m256i ones = _mm256_set1_epi16(1);
            int k = 0;
            for (; k + 64 <= K; k += 64) {
                __m256i x0 = _mm256_loadu_si256((const __m256i *)(xr + k));
                __m256i w0 = _mm256_loadu_si256((const __m256i *)(qn + k));
                __m256i x1 = _mm256_loadu_si256((const __m256i *)(xr + k + 32));
                __m256i w1 = _mm256_loadu_si256((const __m256i *)(qn + k + 32));
                __m256i p0 = _mm256_maddubs_epi16(_mm256_sign_epi8(x0, x0), _mm256_sign_epi8(w0, x0));
                __m256i p1 = _mm256_maddubs_epi16(_mm256_sign_epi8(x1, x1), _mm256_sign_epi8(w1, x1));
                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(p0, ones));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(p1, ones));
            }
            for (; k + 32 <= K; k += 32) {
                __m256i x0 = _mm256_loadu_si256((const __m256i *)(xr + k));
                __m256i w0 = _mm256_loadu_si256((const __m256i *)(qn + k));
                __m256i p0 = _mm256_maddubs_epi16(_mm256_sign_epi8(x0, x0), _mm256_sign_epi8(w0, x0));
                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(p0, ones));
            }
            acc = aria_avx2_hsum_i32(_mm256_add_epi32(c0, c1));
            for (; k < K; k++) acc += (int32_t)xr[k] * qn[k];
#else
            acc = 0;
            for (int k = 0; k < K; k++) acc += (int32_t)xr[k] * qn[k];
#endif
            yr[n] = (float)acc * sm * scale[n] + (b ? b[n] : 0.0f);
        }
    }
}

/* ---- Q4 (block asymmetric / zero-point; 2 nibbles/byte; per block [min,scale]) ----
 * nibble in [0,15]; W ~= min + nib*scale, scale = (max-min)/15. Affine quant fits
 * skewed weight blocks far better than symmetric int4. */

void aria_q4_quant(uint8_t *q, float *scale, const float *W, int N, int K) {
    int nblk = aria_q4_nblocks(K);
    size_t rb = aria_q4_rowbytes(K);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; n++) {
        const float *wr = W + (size_t)n * K;
        uint8_t *qr = q + (size_t)n * rb;
        float *sr = scale + (size_t)n * 2 * nblk;   /* [min,scale] per block */
        memset(qr, 0, rb);
        for (int bi = 0; bi < nblk; bi++) {
            int k0 = bi * ARIA_Q4_BLOCK;
            int k1 = k0 + ARIA_Q4_BLOCK; if (k1 > K) k1 = K;
            float mn = wr[k0], mx = wr[k0];
            for (int k = k0 + 1; k < k1; k++) { float v = wr[k]; if (v < mn) mn = v; if (v > mx) mx = v; }
            float s = (mx - mn) / 15.0f;
            sr[2 * bi] = mn; sr[2 * bi + 1] = s;
            float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
            for (int k = k0; k < k1; k++) {
                uint8_t nib = (s > 0.0f) ? (uint8_t)clampi((int)lrintf((wr[k] - mn) * inv), 0, 15) : 0;
                if (k & 1) qr[k >> 1] |= (uint8_t)(nib << 4);
                else       qr[k >> 1] |= nib;
            }
        }
    }
}

/* ---- aria_qweight dispatch ---- */

void aria_qweight_set(aria_qweight *w, const float *W, int N, int K, aria_dtype dt) {
    w->a8 = (dt == ARIA_Q8A8);   /* Q8A8 = Q8 weights + int8 activations at GEMM time */
    if (dt == ARIA_Q8A8) dt = ARIA_Q8;
    w->dt = dt; w->N = N; w->K = K; w->f32 = NULL; w->q = NULL; w->scale = NULL;
    if (dt == ARIA_Q8) {
        w->q = malloc(aria_q8_qbytes(N, K));
        w->scale = malloc(aria_q8_nscale(N, K) * sizeof(float));
        aria_q8_quant((int8_t *)w->q, w->scale, W, N, K);
    } else if (dt == ARIA_Q4) {
        w->q = malloc(aria_q4_qbytes(N, K));
        w->scale = malloc(aria_q4_nscale(N, K) * sizeof(float));
        aria_q4_quant((uint8_t *)w->q, w->scale, W, N, K);
    } else if (dt == ARIA_F16 || dt == ARIA_BF16) {
        /* B2: half-precision STORAGE, fp32 compute -- the packed GEMM widens the
         * weight panels while packing, so the weight memory stream halves. */
        uint16_t *h = malloc((size_t)N * K * sizeof(uint16_t));
        w->q = h;
        if (h) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (size_t i = 0; i < (size_t)N * K; i++)
                h[i] = (dt == ARIA_BF16) ? aria_float_to_bf16(W[i]) : aria_float_to_half(W[i]);
        } else { w->dt = ARIA_F32; w->f32 = W; }   /* OOM: fall back to borrowing f32 */
    } else {
        w->dt = ARIA_F32;
        w->f32 = W;
    }
}

void aria_qweight_free(aria_qweight *w) {
    if (!w) return;
    free(w->q); free(w->scale);
    w->q = NULL; w->scale = NULL; w->f32 = NULL;
}

void aria_linear_qw(float *y, const float *x, const aria_qweight *w, const float *b, int M) {
    static int w8a8 = -1;   /* opt-in int8-activation path, same env knob as the GPU */
    if (w8a8 < 0) w8a8 = getenv("ARIA_W8A8") != NULL;
    if (w->dt == ARIA_Q8) {
        if (w8a8 || w->a8) aria_linear_q8a8(y, x, (const int8_t *)w->q, w->scale, b, M, w->K, w->N);
        else      aria_linear_q8(y, x, (const int8_t *)w->q, w->scale, b, M, w->K, w->N);
    }
    else if (w->dt == ARIA_Q4) aria_linear_q4(y, x, (const uint8_t *)w->q, w->scale, b, M, w->K, w->N);
    else if (w->dt == ARIA_F16 || w->dt == ARIA_BF16)
        aria_linear_hw(y, x, (const uint16_t *)w->q, w->dt == ARIA_BF16, b, M, w->K, w->N);
    else                       aria_linear(y, x, w->f32, b, M, w->K, w->N);
}

size_t aria_qweight_bytes(const aria_qweight *w) {
    if (w->dt == ARIA_Q8) return aria_q8_qbytes(w->N, w->K) + aria_q8_nscale(w->N, w->K) * sizeof(float);
    if (w->dt == ARIA_Q4) return aria_q4_qbytes(w->N, w->K) + aria_q4_nscale(w->N, w->K) * sizeof(float);
    if (w->dt == ARIA_F16 || w->dt == ARIA_BF16) return (size_t)w->N * w->K * sizeof(uint16_t);
    return (size_t)w->N * w->K * sizeof(float);
}

int aria_qweight_write(FILE *f, const aria_qweight *w) {
    if (w->dt != ARIA_Q8 && w->dt != ARIA_Q4) return -1;
    int32_t hdr[3] = { (int32_t)w->dt, w->N, w->K };
    size_t qb = (w->dt == ARIA_Q8) ? aria_q8_qbytes(w->N, w->K) : aria_q4_qbytes(w->N, w->K);
    size_t ns = (w->dt == ARIA_Q8) ? aria_q8_nscale(w->N, w->K) : aria_q4_nscale(w->N, w->K);
    uint64_t qb64 = qb, ns64 = ns;
    if (fwrite(hdr, sizeof(int32_t), 3, f) != 3) return -1;
    if (fwrite(&qb64, sizeof(uint64_t), 1, f) != 1) return -1;
    if (fwrite(w->q, 1, qb, f) != qb) return -1;
    if (fwrite(&ns64, sizeof(uint64_t), 1, f) != 1) return -1;
    if (fwrite(w->scale, sizeof(float), ns, f) != ns) return -1;
    return 0;
}

int aria_qweight_read(FILE *f, aria_qweight *w) {
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, f) != 3) return -1;
    w->dt = (aria_dtype)hdr[0]; w->N = hdr[1]; w->K = hdr[2]; w->f32 = NULL; w->q = NULL; w->scale = NULL;
    w->a8 = 0;
    if (w->dt != ARIA_Q8 && w->dt != ARIA_Q4) return -1;
    uint64_t qb64, ns64;
    if (fread(&qb64, sizeof(uint64_t), 1, f) != 1) return -1;
    w->q = malloc(qb64);
    if (!w->q || fread(w->q, 1, qb64, f) != qb64) return -1;
    if (fread(&ns64, sizeof(uint64_t), 1, f) != 1) return -1;
    w->scale = (float *)malloc(ns64 * sizeof(float));
    if (!w->scale || fread(w->scale, sizeof(float), ns64, f) != ns64) return -1;
    return 0;
}

void aria_linear_q4(float *y, const float *x, const uint8_t *q, const float *scale,
                    const float *b, int M, int K, int N) {
    int nblk = aria_q4_nblocks(K);
    size_t rb = aria_q4_rowbytes(K);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const float *xr = x + (size_t)m * K;
        float *yr = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const uint8_t *qn = q + (size_t)n * rb;
            const float *sn = scale + (size_t)n * 2 * nblk;   /* [min,scale] per block */
            float acc = 0.0f;
            for (int bi = 0; bi < nblk; bi++) {
                int k0 = bi * ARIA_Q4_BLOCK;
                int k1 = k0 + ARIA_Q4_BLOCK; if (k1 > K) k1 = K;
                float mn = sn[2 * bi], sc = sn[2 * bi + 1];
                /* sum_k x*(min + nib*scale) = min*sum(x) + scale*sum(nib*x) */
                float xsum = 0.0f, nxsum = 0.0f;
#if defined(__ARM_NEON)
                if (k1 - k0 == ARIA_Q4_BLOCK) {   /* full 32-wide block: NEON */
                    /* 16 bytes -> low/high nibbles (evens/odds) -> zip back to element
                     * order (elem0..15, elem16..31), widen to f32, fma with contiguous x. */
                    uint8x16_t bytes = vld1q_u8(qn + (k0 >> 1));
                    uint8x16_t evens = vandq_u8(bytes, vdupq_n_u8(0x0F));
                    uint8x16_t odds  = vshrq_n_u8(bytes, 4);
                    uint8x16x2_t z = vzipq_u8(evens, odds);
                    float32x4_t xsv = vdupq_n_f32(0.0f), nxv = vdupq_n_f32(0.0f);
                    for (int c = 0; c < 2; c++) {
                        uint8x16_t nb = c ? z.val[1] : z.val[0];
                        uint16x8_t l = vmovl_u8(vget_low_u8(nb));
                        uint16x8_t h = vmovl_u8(vget_high_u8(nb));
                        float32x4_t n0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(l)));
                        float32x4_t n1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(l)));
                        float32x4_t n2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(h)));
                        float32x4_t n3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(h)));
                        const float *xp = xr + k0 + c * 16;
                        float32x4_t x0 = vld1q_f32(xp),     x1 = vld1q_f32(xp + 4);
                        float32x4_t x2 = vld1q_f32(xp + 8), x3 = vld1q_f32(xp + 12);
                        xsv = vaddq_f32(xsv, vaddq_f32(vaddq_f32(x0, x1), vaddq_f32(x2, x3)));
                        nxv = vfmaq_f32(nxv, n0, x0);
                        nxv = vfmaq_f32(nxv, n1, x1);
                        nxv = vfmaq_f32(nxv, n2, x2);
                        nxv = vfmaq_f32(nxv, n3, x3);
                    }
                    xsum = vaddvq_f32(xsv);
                    nxsum = vaddvq_f32(nxv);
                } else {                          /* partial tail block: scalar */
                    for (int k = k0; k < k1; k++) {
                        uint8_t byte = qn[k >> 1];
                        int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                        xsum += xr[k];
                        nxsum += xr[k] * (float)nib;
                    }
                }
#else
                for (int k = k0; k < k1; k++) {
                    uint8_t byte = qn[k >> 1];
                    int nib = (k & 1) ? (byte >> 4) : (byte & 0x0F);
                    xsum += xr[k];
                    nxsum += xr[k] * (float)nib;
                }
#endif
                acc += mn * xsum + sc * nxsum;
            }
            yr[n] = acc + (b ? b[n] : 0.0f);
        }
    }
}
