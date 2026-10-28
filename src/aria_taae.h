/*
 * aria_taae.h - shared taae_v2 resampling transformer block + chunk machinery.
 *
 * The encoder and decoder resampling stacks (SAMEEncoder / SAMEDecoder) use the
 * identical TransformerBlock (transformer_dim 768, 12 heads, head_dim 64,
 * differential self-attention, DynamicTanh norms, SwiGLU FF) chunked into S=34
 * windows with a midpoint-shift halo on the second half. Only the surrounding
 * resampling glue (new-token insertion, mapping placement, extraction) differs
 * between encode and decode, so the block + chunk pass live here, used by both.
 */

#ifndef ARIA_TAAE_H
#define ARIA_TAAE_H

#include <stddef.h>
#include "aria_arena.h"
#include "aria_quant.h"   /* B3: optional q8/fp16 decoder-weight overlay */

#define TAAE_D     768   /* transformer model dim */
#define TAAE_H     12    /* heads */
#define TAAE_HD    64    /* head dim */
#define TAAE_ROT   32    /* rotated dims */
#define TAAE_INNER 2304  /* FF inner (ff_mult 3) */
#define TAAE_QKV   3840  /* 5 * 768 (q,k,v + qd,kd for differential attn) */
#define TAAE_S     34    /* effective chunk = 2 * 17 */

/* one resampling TransformerBlock's weights, borrowed from the mmap (zero-copy). */
typedef struct {
    float pre_alpha;  const float *pre_gamma, *pre_beta;
    const float *to_qkv;                 /* [3840,768] */
    float qn_alpha;   const float *qn_gamma, *qn_beta;
    float kn_alpha;   const float *kn_gamma, *kn_beta;
    const float *to_out;                 /* [768,768] */
    float ff_alpha;   const float *ff_gamma, *ff_beta;
    const float *ff_in_w, *ff_in_b;      /* [4608,768],[4608] */
    const float *ff_out_w, *ff_out_b;    /* [768,2304],[768] */
    /* B3 (opt-in, ARIA_DEC_Q8): quantized/half overlay of the 4 big GEMM matrices.
     * qon=0 -> the fp32 mmap pointers above are used (default). When on, the block
     * forward dispatches through aria_linear_qw (q8 sdot/AVX2 on ARM/x86, or fp16). */
    int qon;
    aria_qweight q_to_qkv, q_to_out, q_ff_in_w, q_ff_out_w;
} taae_block_w;

/* B3: build/free the decoder-block overlay (dt = ARIA_Q8/F16/BF16) for the 4 big
 * GEMM matrices; releases the fp32 mmap source (page-aligned) once copied. `dim`/
 * `inner` are the block's runtime dims (small: TAAE_D/TAAE_INNER; medium: 1536/...). */
void taae_block_quantize(taae_block_w *w, int dim, int inner, aria_dtype dt);
void taae_block_overlay_free(taae_block_w *w);

/* run one block in place on xc[nch*S,768] = nch independent chunks of S tokens
 * (attention stays inside each chunk; nch = 1: one sequence of S). rope tables
 * rcos/rsin[S,16]. All scratch comes from `ar` (>= taae_block_arena_bytes(nch*S)). */
void taae_block_forward(float *xc, int nch, int S, const taae_block_w *w,
                        const float *rcos, const float *rsin, aria_arena *ar);

/* run 3 blocks over seq[L,768] in S=34 chunks; shift=1 adds a midpoint-shift halo.
 * Chunks are independent, so blocks run over groups of chunks (batched GEMMs). */
void taae_chunk_pass(float *x, int L, const taae_block_w *blocks,
                     const float *rcos, const float *rsin, int shift);

/* scratch arena bytes sufficient for one block over `rows` tokens. */
size_t taae_block_arena_bytes(int rows);

/* ---- medium decoder block (runtime dim; sliding-window + sinusoidal FF) ----
 * Same DyT + differential-attention TransformerBlock as the small-music decoder,
 * but generalized: arbitrary `dim`/`inner`, sliding-window attention of half-width
 * `win` (query i attends to keys [i-win, i+win]), and a sinusoidal FF gate
 * sin(pi*x) instead of SiLU. xc[N,dim] in place; rope tables rcos/rsin[N,16]. */
void taae_med_block_forward(float *xc, int N, int dim, int H, int hd, int inner,
                            const taae_block_w *w, const float *rcos, const float *rsin,
                            int win, int sinusoidal, aria_arena *ar);
/* scratch floats for one medium block at sequence length N (dim/inner runtime). */
size_t taae_med_block_floats(int N, int dim, int inner);

#endif /* ARIA_TAAE_H */
