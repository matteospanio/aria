/*
 * aria_ops.h - Backend-agnostic primitive op surface
 *
 * Every model module (DiT, taae_v2 autoencoder, T5Gemma encoder) is built
 * from these primitives. CPU implementations live in aria_cpu.c; CUDA
 * implementations (Phase 3) live in aria_cuda.cu behind #ifdef ARIA_CUDA.
 *
 * Convention: CPU ops operate on flat row-major float arrays with explicit
 * dimensions (iris.c style) -- simple to test and to parity-check against
 * PyTorch. Weight tensors follow PyTorch's nn.Linear layout: W is [out, in].
 */

#ifndef ARIA_OPS_H
#define ARIA_OPS_H

#include <stddef.h>
#include <stdint.h>

/* ---- half-precision storage helpers (B2) ---- */

/* IEEE fp16 <-> fp32, scalar, RTNE on the way down. Storage-only: all compute stays
 * fp32 (the packed GEMM widens while packing weight panels). */
static inline float aria_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, f;
    if (exp == 0) {
        if (man == 0) f = sign;
        else {  /* subnormal: renormalize */
            int e = 0;
            while (!(man & 0x400u)) { man <<= 1; e++; }
            f = sign | ((uint32_t)(113 - e) << 23) | ((man & 0x3FFu) << 13);
        }
    } else if (exp == 31) f = sign | 0x7F800000u | (man << 13);
    else f = sign | ((exp + 112u) << 23) | (man << 13);
    union { uint32_t u; float fl; } v; v.u = f; return v.fl;
}
static inline uint16_t aria_float_to_half(float x) {
    union { float fl; uint32_t u; } v; v.fl = x;
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xFFu) - 127 + 15;
    uint32_t man = v.u & 0x7FFFFFu;
    if (exp >= 31) return (uint16_t)(sign | (((v.u >> 23) & 0xFF) == 0xFF && man
                                            ? 0x7E00u : 0x7C00u));   /* inf/nan */
    if (exp <= 0) {   /* subnormal or zero: shift with RTNE */
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        int shift = 14 - exp;
        uint32_t q = man >> shift, rem = man & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        return (uint16_t)(sign | q);
    }
    uint32_t q = man >> 13, rem = man & 0x1FFFu;
    uint16_t out = (uint16_t)(sign | ((uint32_t)exp << 10) | q);
    if (rem > 0x1000u || (rem == 0x1000u && (out & 1))) out++;   /* RTNE (may carry into exp) */
    return out;
}
/* bulk fp32 -> fp16 (RTNE; F16C where built, else aria_float_to_half -- same bits for every
 * finite value, and the same as CUDA's __float2half), parallel. For device weight upload. */
void aria_f32_to_f16_n(uint16_t *dst, const float *src, size_t n);
static inline float aria_bf16_to_float(uint16_t h) {
    union { uint32_t u; float fl; } v; v.u = (uint32_t)h << 16; return v.fl;
}
static inline uint16_t aria_float_to_bf16(float x) {   /* RTNE on bit 16 */
    union { float fl; uint32_t u; } v; v.fl = x;
    uint32_t lsb = (v.u >> 16) & 1u;
    return (uint16_t)((v.u + 0x7FFFu + lsb) >> 16);
}

/* ---- linear algebra ---- */

/* y[M,N] = x[M,K] @ W^T + b ; W is [N,K] (PyTorch Linear weight), b is [N] or NULL. */
void aria_linear(float *y, const float *x, const float *W, const float *b,
                 int M, int K, int N);
/* aria_linear without the small-GEMM shortcut (always the packed kernel where built),
 * so each output's math does not depend on M/N -- for callers whose sizes vary per
 * request but must stay byte-identical to a larger run (T5 real-token attention). */
void aria_linear_packed_only(float *y, const float *x, const float *W, const float *b,
                             int M, int K, int N);

/* B2: as aria_linear but W stored as fp16 (bf16=0) or bf16 (bf16=1) uint16; widened
 * to f32 while packing weight panels -- the compute path is the same packed kernel,
 * the weight memory stream halves. */
void aria_linear_hw(float *y, const float *x, const uint16_t *W, int bf16,
                    const float *b, int M, int K, int N);

/* C[M,N] = A[M,K] @ B[K,N] (both row-major). */
void aria_matmul(float *C, const float *A, const float *B, int M, int K, int N);

/* ---- normalization ---- */

/* RMSNorm over the last dim. y and x are [rows, dim]. weight is [dim] or NULL.
 * y = x / sqrt(mean(x^2) + eps) * (weight ? weight : 1). Computed in fp32. */
void aria_rmsnorm(float *y, const float *x, const float *weight,
                  int rows, int dim, float eps);

/* DynamicTanh (taae_v2): y = tanh(alpha * x) * weight + bias, per-channel
 * weight/bias of length dim. alpha is a learned scalar. */
void aria_dynamic_tanh(float *y, const float *x, float alpha,
                       const float *weight, const float *bias,
                       int rows, int dim);

/* Gemma RMSNorm: y = x / sqrt(mean(x^2)+eps) * (1 + weight). Note the (1+weight):
 * Gemma stores zero-centered norm gains. Computed in fp32. */
void aria_gemma_rmsnorm(float *y, const float *x, const float *weight,
                        int rows, int dim, float eps);

/* Attention-logit softcapping: s[i] = cap * tanh(s[i] / cap), n elements. */
void aria_softcap(float *s, int n, float cap);

/* ---- activations (in-place over n elements) ---- */
void aria_silu(float *x, int n);          /* x * sigmoid(x) */
void aria_gelu_tanh(float *x, int n);     /* gelu_pytorch_tanh */

/* SiLU-gated FFN combine: out[i] = silu(gate[i]) * up[i], n elements. */
void aria_silu_gate(float *out, const float *gate, const float *up, int n);

/* ---- rotary position embedding (rotate-half, GPT-NeoX style) ---- */

/* Precompute cos/sin tables [n, rot_dim/2] for integer positions 0..n-1.
 * inv_freq[i] = 1 / base^(2i/rot_dim). Matches stable_audio_tools RotaryEmbedding
 * (base 10000, interpolation_factor 1). cos_t/sin_t hold n*(rot_dim/2) floats. */
void aria_rope_freqs(float *cos_t, float *sin_t, int n, int rot_dim, float base);

/* Apply RoPE in place to x laid out [H, N, D] (head-major). Rotates the first
 * rot_dim (<= D) dims with pairing (i, i+rot_dim/2); dims >= rot_dim untouched. */
void aria_rope_apply(float *x, const float *cos_t, const float *sin_t,
                     int H, int N, int D, int rot_dim);

/* ---- attention ---- */

/* Multi-head scaled-dot-product attention, bidirectional (non-causal).
 * q [H,Nq,D], k/v [H,Nk,D], out [H,Nq,D]. scale = 1/sqrt(D) (SDPA default).
 * Optional additive mask [Nq,Nk] (e.g. -inf for padding) or NULL.
 * scratch: [Nq*Nk] floats, or NULL to malloc internally. */
void aria_attention(float *out, const float *q, const float *k, const float *v,
                    int H, int Nq, int Nk, int D, const float *mask, float *scratch);

/* Sliding-window (banded) self-attention: query i attends to keys [i-W, i+W] only.
 * Same result as aria_attention with a [-W,W] band mask, but O(N*(2W+1)*D) instead
 * of O(N^2*D). q/k/v/out [H,N,D]. */
void aria_attention_band(float *out, const float *q, const float *k, const float *v,
                         int H, int N, int D, int W);

/* ---- conv1d (stride 1, pre-folded weights) ---- */

/* out[Cout,L] = conv1d(in[Cin,L]) with weight w[Cout,Cin,K], zero-padded `pad`
 * each side, stride 1 (length preserving when 2*pad == K-1). bias [Cout] or NULL.
 * Weight-normalized convs must have weight_g/weight_v folded before calling. */
void aria_conv1d(float *out, const float *in, const float *w, const float *bias,
                 int Cin, int Cout, int K, int pad, int L);

/* ---- GLU/SwiGLU feed-forward ---- */

/* FeedForward: proj x[N,dim] -> [N,2*inner] via W_in[2*inner,dim](+b_in),
 * GLU combine value*SiLU(gate) (value = first half), then W_out[dim_out,inner]
 * (+b_out) -> out[N,dim_out]. scratch: [N*3*inner] floats, or NULL to malloc. */
void aria_ff_glu(float *out, const float *x, int N, int dim, int inner, int dim_out,
                 const float *W_in, const float *b_in,
                 const float *W_out, const float *b_out, float *scratch);

/* ---- softmax over rows ---- */
/* In-place row softmax. x is [rows, cols]. Optional additive mask [rows,cols]
 * (e.g. -inf for padding) when mask != NULL. */
void aria_softmax_inplace(float *x, int rows, int cols, const float *mask);

/* ---- elementwise ---- */
void aria_add(float *y, const float *a, const float *b, int n);   /* y = a + b */
void aria_addto(float *y, const float *a, int n);                 /* y += a */
void aria_scale(float *y, const float *x, float s, int n);        /* y = x * s */
void aria_axpy(float *y, const float *x, float s, int n);         /* y += s * x */

#endif /* ARIA_OPS_H */
