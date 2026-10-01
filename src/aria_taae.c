/*
 * aria_taae.c - shared taae_v2 resampling block + chunk pass. See aria_taae.h.
 * Extracted verbatim from the decoder so the encoder and decoder share one
 * (tested) implementation of the differential-attention block.
 */

#include "aria_taae.h"
#include "aria_win_compat.h"
#include "aria_ops.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

/* ---- B3: opt-in q8/fp16 decoder-weight overlay ----
 * Dispatch one GEMM through the overlay (q8 sdot/AVX2 or fp16 widening) when the
 * block is quantized, else the fp32 mmap weight. */
static inline void taae_gemm(float *y, const float *x, const float *Wf32,
                             const aria_qweight *qw, int qon, const float *b, int M, int K, int N) {
    if (qon && qw->q) aria_linear_qw(y, x, qw, b, M);
    else              aria_linear(y, x, Wf32, b, M, K, N);
}
static void taae_release_source(const void *p, size_t bytes) {   /* mirrors dit_release_source */
    if (!p || !bytes) return;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) return;
    uintptr_t a = (uintptr_t)p, lo = (a + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    uintptr_t hi = (a + bytes) & ~((uintptr_t)pg - 1);
    if (hi > lo) madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
}
void taae_block_quantize(taae_block_w *w, int dim, int inner, aria_dtype dt) {
    if (dt == ARIA_F32) return;
    int qkv = (dim == TAAE_D) ? TAAE_QKV : 5 * dim;   /* small: fixed 3840; medium: 5*dim */
    aria_qweight_set(&w->q_to_qkv,  w->to_qkv,   qkv, dim, dt);
    aria_qweight_set(&w->q_to_out,  w->to_out,   dim, dim, dt);
    aria_qweight_set(&w->q_ff_in_w, w->ff_in_w,  2 * inner, dim, dt);
    aria_qweight_set(&w->q_ff_out_w,w->ff_out_w, dim, inner, dt);
    taae_release_source(w->to_qkv,   (size_t)qkv * dim * sizeof(float));
    taae_release_source(w->to_out,   (size_t)dim * dim * sizeof(float));
    taae_release_source(w->ff_in_w,  (size_t)2 * inner * dim * sizeof(float));
    taae_release_source(w->ff_out_w, (size_t)dim * inner * sizeof(float));
    w->qon = 1;
}
void taae_block_overlay_free(taae_block_w *w) {
    if (!w->qon) return;
    aria_qweight_free(&w->q_to_qkv); aria_qweight_free(&w->q_to_out);
    aria_qweight_free(&w->q_ff_in_w); aria_qweight_free(&w->q_ff_out_w);
    w->qon = 0;
}

/* ---- head transpose helpers (chunk-local) ---- */
static void extract_heads(float *dst, const float *src, int N, int H, int hd, int stride, int off) {
    for (int h = 0; h < H; h++)
        for (int s = 0; s < N; s++)
            memcpy(dst + ((size_t)h * N + s) * hd, src + (size_t)s * stride + off + (size_t)h * hd,
                   (size_t)hd * sizeof(float));
}
static void merge_heads(float *dst, const float *src, int N, int H, int hd) {
    for (int h = 0; h < H; h++)
        for (int s = 0; s < N; s++)
            memcpy(dst + (size_t)s * (H * hd) + (size_t)h * hd, src + ((size_t)h * N + s) * hd,
                   (size_t)hd * sizeof(float));
}

void taae_block_forward(float *xc, int N, const taae_block_w *w,
                        const float *rcos, const float *rsin, aria_arena *ar) {
    const int D = TAAE_D, H = TAAE_H, hd = TAAE_HD;
    size_t mark = aria_arena_save(ar);
    float *h = aria_arena_floats(ar, (size_t)N * D);
    float *res = aria_arena_floats(ar, (size_t)N * D);
    float *qkv = aria_arena_floats(ar, (size_t)N * TAAE_QKV);
    float *q = aria_arena_floats(ar, (size_t)H * N * hd);
    float *k = aria_arena_floats(ar, (size_t)H * N * hd);
    float *v = aria_arena_floats(ar, (size_t)H * N * hd);
    float *qd = aria_arena_floats(ar, (size_t)H * N * hd);
    float *kd = aria_arena_floats(ar, (size_t)H * N * hd);
    float *ob = aria_arena_floats(ar, (size_t)H * N * hd);
    float *od = aria_arena_floats(ar, (size_t)H * N * hd);
    float *merged = aria_arena_floats(ar, (size_t)N * D);
    float *o = aria_arena_floats(ar, (size_t)N * D);
    float *scores = aria_arena_floats(ar, (size_t)N * N);
    float *ffs = aria_arena_floats(ar, (size_t)N * 3 * TAAE_INNER);

    /* self-attention (differential) */
    memcpy(res, xc, (size_t)N * D * sizeof(float));
    aria_dynamic_tanh(h, xc, w->pre_alpha, w->pre_gamma, w->pre_beta, N, D);
    taae_gemm(qkv, h, w->to_qkv, &w->q_to_qkv, w->qon, NULL, N, D, TAAE_QKV);
    extract_heads(q,  qkv, N, H, hd, TAAE_QKV, 0);
    extract_heads(k,  qkv, N, H, hd, TAAE_QKV, 768);
    extract_heads(v,  qkv, N, H, hd, TAAE_QKV, 1536);
    extract_heads(qd, qkv, N, H, hd, TAAE_QKV, 2304);
    extract_heads(kd, qkv, N, H, hd, TAAE_QKV, 3072);
    aria_dynamic_tanh(q,  q,  w->qn_alpha, w->qn_gamma, w->qn_beta, H * N, hd);
    aria_dynamic_tanh(qd, qd, w->qn_alpha, w->qn_gamma, w->qn_beta, H * N, hd);
    aria_dynamic_tanh(k,  k,  w->kn_alpha, w->kn_gamma, w->kn_beta, H * N, hd);
    aria_dynamic_tanh(kd, kd, w->kn_alpha, w->kn_gamma, w->kn_beta, H * N, hd);
    aria_rope_apply(q,  rcos, rsin, H, N, hd, TAAE_ROT);
    aria_rope_apply(qd, rcos, rsin, H, N, hd, TAAE_ROT);
    aria_rope_apply(k,  rcos, rsin, H, N, hd, TAAE_ROT);
    aria_rope_apply(kd, rcos, rsin, H, N, hd, TAAE_ROT);
    aria_attention(ob, q,  k,  v, H, N, N, hd, NULL, scores);
    aria_attention(od, qd, kd, v, H, N, N, hd, NULL, scores);
    for (size_t i = 0; i < (size_t)H * N * hd; i++) ob[i] -= od[i];
    merge_heads(merged, ob, N, H, hd);
    taae_gemm(o, merged, w->to_out, &w->q_to_out, w->qon, NULL, N, D, D);
    for (size_t i = 0; i < (size_t)N * D; i++) xc[i] = res[i] + o[i];

    /* feed-forward (SwiGLU). Fused aria_ff_glu on the fp32 path; when the block is
     * quantized we unfuse into two aria_linear_qw GEMMs + the SiLU gate (same math,
     * lets the q8/fp16 overlay drive both FF matrices). */
    memcpy(res, xc, (size_t)N * D * sizeof(float));
    aria_dynamic_tanh(h, xc, w->ff_alpha, w->ff_gamma, w->ff_beta, N, D);
    if (w->qon) {
        /* reuse ffs[N, 3*inner]: proj = ffs[N, 2*inner], gated = ffs + N*2*inner
         * (same layout as the medium path -- no extra arena alloc). */
        float *gated = ffs + (size_t)N * 2 * TAAE_INNER;
        aria_linear_qw(ffs, h, &w->q_ff_in_w, w->ff_in_b, N);            /* [N, 2*inner] */
        for (int nn = 0; nn < N; nn++)
            aria_silu_gate(gated + (size_t)nn * TAAE_INNER,
                           ffs + (size_t)nn * 2 * TAAE_INNER + TAAE_INNER,
                           ffs + (size_t)nn * 2 * TAAE_INNER, TAAE_INNER);
        aria_linear_qw(o, gated, &w->q_ff_out_w, w->ff_out_b, N);
    } else {
        aria_ff_glu(o, h, N, D, TAAE_INNER, D, w->ff_in_w, w->ff_in_b, w->ff_out_w, w->ff_out_b, ffs);
    }
    for (size_t i = 0; i < (size_t)N * D; i++) xc[i] = res[i] + o[i];

    aria_arena_restore(ar, mark);
}

void taae_chunk_pass(float *x, int L, const taae_block_w *blocks,
                     const float *rcos, const float *rsin, int shift, aria_arena *arenas) {
    const int S = TAAE_S, half = S / 2;  /* 17 */
    float *base = x, *pad = NULL;
    int n;
    if (!shift) {
        n = L / S;
    } else {
        int Lp = L + S;  /* [x[:17], x, x[-17:]] */
        pad = malloc((size_t)Lp * TAAE_D * sizeof(float));
        memcpy(pad, x, (size_t)half * TAAE_D * sizeof(float));
        memcpy(pad + (size_t)half * TAAE_D, x, (size_t)L * TAAE_D * sizeof(float));
        memcpy(pad + (size_t)(half + L) * TAAE_D, x + (size_t)(L - half) * TAAE_D,
               (size_t)half * TAAE_D * sizeof(float));
        base = pad; n = Lp / S;
    }
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int c = 0; c < n; c++) {
        #ifdef _OPENMP
        aria_arena *ar = &arenas[omp_get_thread_num()];
        #else
        aria_arena *ar = &arenas[0];
        #endif
        float *chunk = base + (size_t)c * S * TAAE_D;
        for (int b = 0; b < 3; b++) taae_block_forward(chunk, S, &blocks[b], rcos, rsin, ar);
    }
    if (shift) {
        memcpy(x, pad + (size_t)half * TAAE_D, (size_t)L * TAAE_D * sizeof(float));
        free(pad);
    }
}

size_t taae_block_arena_bytes(void) {
    size_t N = TAAE_S;
    size_t fl = 12 * N * TAAE_D + N * TAAE_QKV + N * N + 3 * N * TAAE_INNER + 4096;
    return fl * sizeof(float);
}

/* ---- medium decoder block: runtime dim, sliding-window band mask, sin/SiLU FF ---- */
#include <math.h>

size_t taae_med_block_floats(int N, int dim, int inner) {
    /* peak = attention phase: h+res+o(3) + qkv(5) + q,k,v,qd,kd,ob,od(7) + merged(1)
     * = 16*N*dim. Banded attention keeps its tiny score window thread-local, so no
     * N*N scores buffer. The FF (3*N*dim + 3*N*inner) is freed-and-reused, and
     * 3*N*inner < 13*N*dim for the medium decoder, so this bound covers it. */
    (void)inner;
    return (size_t)16 * N * dim + 4096;
}

void taae_med_block_forward(float *xc, int N, int dim, int H, int hd, int inner,
                            const taae_block_w *w, const float *rcos, const float *rsin,
                            int win, int sinusoidal, aria_arena *ar) {
    size_t mark = aria_arena_save(ar);
    /* h/res/o persist; the attention scratch is freed before the FF (never live
     * together) so the arena peak is the attention phase, not the sum -- saves
     * ~3*N*inner (~100 MB on a 10 s medium clip). */
    float *h   = aria_arena_floats(ar, (size_t)N * dim);
    float *res = aria_arena_floats(ar, (size_t)N * dim);
    float *o   = aria_arena_floats(ar, (size_t)N * dim);
    size_t amark = aria_arena_save(ar);
    float *qkv = aria_arena_floats(ar, (size_t)N * 5 * dim);
    float *q  = aria_arena_floats(ar, (size_t)H * N * hd);
    float *k  = aria_arena_floats(ar, (size_t)H * N * hd);
    float *v  = aria_arena_floats(ar, (size_t)H * N * hd);
    float *qd = aria_arena_floats(ar, (size_t)H * N * hd);
    float *kd = aria_arena_floats(ar, (size_t)H * N * hd);
    float *ob = aria_arena_floats(ar, (size_t)H * N * hd);
    float *od = aria_arena_floats(ar, (size_t)H * N * hd);
    float *merged = aria_arena_floats(ar, (size_t)N * dim);

    /* self-attention (differential): out = attn(q,k,v) - attn(qd,kd,v), banded mask */
    memcpy(res, xc, (size_t)N * dim * sizeof(float));
    aria_dynamic_tanh(h, xc, w->pre_alpha, w->pre_gamma, w->pre_beta, N, dim);
    taae_gemm(qkv, h, w->to_qkv, &w->q_to_qkv, w->qon, NULL, N, dim, 5 * dim);
    extract_heads(q,  qkv, N, H, hd, 5 * dim, 0);
    extract_heads(k,  qkv, N, H, hd, 5 * dim, dim);
    extract_heads(v,  qkv, N, H, hd, 5 * dim, 2 * dim);
    extract_heads(qd, qkv, N, H, hd, 5 * dim, 3 * dim);
    extract_heads(kd, qkv, N, H, hd, 5 * dim, 4 * dim);
    aria_dynamic_tanh(q,  q,  w->qn_alpha, w->qn_gamma, w->qn_beta, H * N, hd);
    aria_dynamic_tanh(qd, qd, w->qn_alpha, w->qn_gamma, w->qn_beta, H * N, hd);
    aria_dynamic_tanh(k,  k,  w->kn_alpha, w->kn_gamma, w->kn_beta, H * N, hd);
    aria_dynamic_tanh(kd, kd, w->kn_alpha, w->kn_gamma, w->kn_beta, H * N, hd);
    aria_rope_apply(q,  rcos, rsin, H, N, hd, TAAE_ROT);
    aria_rope_apply(qd, rcos, rsin, H, N, hd, TAAE_ROT);
    aria_rope_apply(k,  rcos, rsin, H, N, hd, TAAE_ROT);
    aria_rope_apply(kd, rcos, rsin, H, N, hd, TAAE_ROT);
    aria_attention_band(ob, q,  k,  v, H, N, hd, win);
    aria_attention_band(od, qd, kd, v, H, N, hd, win);
    for (size_t i = 0; i < (size_t)H * N * hd; i++) ob[i] -= od[i];
    merge_heads(merged, ob, N, H, hd);
    taae_gemm(o, merged, w->to_out, &w->q_to_out, w->qon, NULL, N, dim, dim);
    for (size_t i = 0; i < (size_t)N * dim; i++) xc[i] = res[i] + o[i];
    aria_arena_restore(ar, amark);   /* free qkv/q/k/v/qd/kd/ob/od/merged/scores before the FF */

    /* feed-forward (GLU): out = value * act(gate), act = SiLU or sin(pi*x) */
    float *ffs = aria_arena_floats(ar, (size_t)N * 3 * inner);
    memcpy(res, xc, (size_t)N * dim * sizeof(float));
    aria_dynamic_tanh(h, xc, w->ff_alpha, w->ff_gamma, w->ff_beta, N, dim);
    taae_gemm(ffs, h, w->ff_in_w, &w->q_ff_in_w, w->qon, w->ff_in_b, N, dim, 2 * inner);  /* [N, 2*inner] */
    float *gated = ffs + (size_t)N * 2 * inner;
    for (int n = 0; n < N; n++) {
        const float *pr = ffs + (size_t)n * 2 * inner;
        float *g = gated + (size_t)n * inner;
        for (int i = 0; i < inner; i++) {
            float val = pr[i], gate = pr[inner + i];
            float a = sinusoidal ? sinf(3.14159265359f * gate) : gate / (1.0f + expf(-gate));
            g[i] = val * a;
        }
    }
    taae_gemm(o, gated, w->ff_out_w, &w->q_ff_out_w, w->qon, w->ff_out_b, N, inner, dim);
    for (size_t i = 0; i < (size_t)N * dim; i++) xc[i] = res[i] + o[i];

    aria_arena_restore(ar, mark);
}
