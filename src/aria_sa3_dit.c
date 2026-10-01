/*
 * aria_sa3_dit.c - SA3 conditioning embedders + DiT transformer block (CPU).
 */

#include "aria_sa3_dit.h"
#include "aria_win_compat.h"
#include "aria_ops.h"
#include "aria_quant.h"
#include "aria_cond.h"
#include "aria_arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

static inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

void aria_sa3_mlp2(float *out, const float *in, int N, int in_dim, int mid, int out_dim,
                   const float *W0, const float *b0, const float *W2, const float *b2) {
    float *h = malloc((size_t)N * mid * sizeof(float));
    aria_linear(h, in, W0, b0, N, in_dim, mid);
    aria_silu(h, N * mid);
    aria_linear(out, h, W2, b2, N, mid, out_dim);
    free(h);
}

void aria_sa3_timestep_embed(float *out, float t, int feat_dim, int ed,
                             const float *W0, const float *b0, const float *W2, const float *b2) {
    float *fourier = malloc((size_t)feat_dim * sizeof(float));
    aria_expo_fourier(fourier, t, feat_dim, 0.5f, 10000.0f);
    aria_sa3_mlp2(out, fourier, 1, feat_dim, ed, ed, W0, b0, W2, b2);
    free(fourier);
}

/* dst[H,S,hd] from src[S, stride] reading hd-blocks at `offset` per head:
 * dst[h,s,d] = src[s*stride + offset + h*hd + d] */
/* A4: the block's elementwise glue ran serial between the parallel GEMMs -- ~2-4 s of
 * a 60 s small CPU generation. Rows are independent, so plain `omp parallel for` is
 * bit-exact (same per-element arithmetic, just spread over threads). */
static void extract_heads(float *dst, const float *src, int S, int H, int hd,
                          int stride, int offset) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int h = 0; h < H; h++)
        for (int s = 0; s < S; s++) {
            const float *sp = src + (size_t)s * stride + offset + (size_t)h * hd;
            float *dp = dst + ((size_t)h * S + s) * hd;
            memcpy(dp, sp, (size_t)hd * sizeof(float));
        }
}

/* dst[S, H*hd] from src[H,S,hd]: dst[s, h*hd+d] = src[h,s,d] */
static void merge_heads(float *dst, const float *src, int S, int H, int hd) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) collapse(2)
    #endif
    for (int h = 0; h < H; h++)
        for (int s = 0; s < S; s++) {
            const float *sp = src + ((size_t)h * S + s) * hd;
            float *dp = dst + (size_t)s * (H * hd) + (size_t)h * hd;
            memcpy(dp, sp, (size_t)hd * sizeof(float));
        }
}

/* y = y*(1+scale)+shift over rows; scale/shift are [dim] broadcast over S rows. */
static void adaln_modulate(float *y, const float *scale, const float *shift, int S, int dim) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int s = 0; s < S; s++) {
        float *yr = y + (size_t)s * dim;
        #ifdef _OPENMP
        #pragma omp simd
        #endif
        for (int i = 0; i < dim; i++) yr[i] = yr[i] * (1.0f + scale[i]) + shift[i];
    }
}

/* y *= sigmoid(1 - gate), gate [dim] broadcast over S rows. */
static void gate_sigmoid(float *y, const float *gate, int S, int dim) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int s = 0; s < S; s++) {
        float *yr = y + (size_t)s * dim;
        for (int i = 0; i < dim; i++) yr[i] *= sigmoidf(1.0f - gate[i]);
    }
}

/* Quantized mirror of a block's per-step GEMMs, built on load when precision != f32
 * (additive overlay -- the f32 path and the CUDA view are untouched). ca_to_kv is
 * deliberately left f32: it projects the cross-attention K/V once per request (not
 * per step), so quantizing it saves no per-step time and only slows the setup. */
typedef struct {
    aria_qweight sa_to_qkv, sa_to_out, ca_to_q, ca_to_out, ff_in_w, ff_out_w;
} dit_block_q;

/* quant-aware GLU feed-forward (mirrors aria_ff_glu, but the two GEMMs dispatch
 * across precisions via aria_linear_qw). scratch: [N*3*inner] floats. */
static void ff_glu_qw(float *out, const float *x, int N, int inner, int dim_out,
                      const aria_qweight *Win, const float *bin,
                      const aria_qweight *Wout, const float *bout, float *scratch) {
    float *proj = scratch, *gated = scratch + (size_t)N * 2 * inner;
    aria_linear_qw(proj, x, Win, bin, N);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; n++)
        aria_silu_gate(gated + (size_t)n * inner, proj + (size_t)n * 2 * inner + inner,
                       proj + (size_t)n * 2 * inner, inner);
    aria_linear_qw(out, gated, Wout, bout, N);
    (void)dim_out;
}

/* E12.9: apply a matching LoRA residual on top of a just-computed base GEMM output
 * (y[M,out] = x[M,in] @ W^T). No-op when no adapter matches or scale==0 (base is
 * left bit-identical). Scratch t[M*rank] comes from the block arena. */
static void lora_hook(const aria_lora_adapter *lora, int layer, aria_lora_proj proj,
                      float *y, const float *x, int M, int in, int out, aria_arena *ar) {
    if (!lora) return;
    const aria_lora_item *it = aria_lora_find(lora, layer, proj);
    if (!it) return;
    float *t = aria_arena_floats(ar, (size_t)M * it->rank);
    aria_lora_linear(y, x, M, in, out, it->rank, it->scale, it->down, it->up, t);
}

/* Block forward with the cross-attention K/V already projected and head-split
 * (cross_k/cross_v are [H, Sc, hd]; cross_k is post-k_norm). All temporaries come
 * from `ar` (save/restore-scoped), so the hot loop never malloc/frees. `bq` is the
 * quantized weight overlay (NULL = f32; that branch is bit-identical to before).
 * `lora` (E12.9) adds low-rank residuals to the 6 per-step block GEMMs; NULL = none
 * (ca_to_kv is folded into the cached cross K/V by the request, not here). */
static void dit_block_core(float *x, int S, int dim, int H, int hd, int inner,
                           const float *cross_k, const float *cross_v, const float *cross_kd, int Sc,
                           const float *global_cond,
                           const float *rope_cos, const float *rope_sin, int rot_dim,
                           const float *local_emb, const aria_dit_block_w *w,
                           const dit_block_q *bq, int diff,
                           const aria_steer_set *steer, const aria_lora_adapter *lora,
                           int layer_idx, int step, aria_arena *ar) {
    const float eps_norm = 1e-5f, eps_qk = 1e-6f;
    size_t mark = aria_arena_save(ar);

    /* modulation = to_scale_shift_gate + global_cond, chunked into 6 [dim] slices */
    float *mod = aria_arena_floats(ar, (size_t)6 * dim);
    for (int i = 0; i < 6 * dim; i++) mod[i] = w->to_scale_shift_gate[i] + global_cond[i];
    const float *scale_self = mod, *shift_self = mod + dim, *gate_self = mod + 2 * dim;
    const float *scale_ff = mod + 3 * dim, *shift_ff = mod + 4 * dim, *gate_ff = mod + 5 * dim;

    /* no residual buffer: every sub-layer normalizes into h (not x), so x is preserved
     * across the sub-layer and the residual add is in-place (x += o, see below). */
    float *h        = aria_arena_floats(ar, (size_t)S * dim);
    float *o        = aria_arena_floats(ar, (size_t)S * dim);
    float *merged   = aria_arena_floats(ar, (size_t)S * dim);
    float *qh       = aria_arena_floats(ar, (size_t)H * S * hd);
    float *kh       = aria_arena_floats(ar, (size_t)H * S * hd);
    float *vh       = aria_arena_floats(ar, (size_t)H * S * hd);
    float *ao       = aria_arena_floats(ar, (size_t)H * S * hd);
    /* differential attention (medium): extra diff q/k + a second attention output.
     * out = attn(q,k,v) - attn(q_diff,k_diff,v) (no lambda; v shared). */
    float *qdh = diff ? aria_arena_floats(ar, (size_t)H * S * hd) : NULL;
    float *kdh = diff ? aria_arena_floats(ar, (size_t)H * S * hd) : NULL;
    float *aod = diff ? aria_arena_floats(ar, (size_t)H * S * hd) : NULL;
    float *scores   = aria_arena_floats(ar, (size_t)S * (S > Sc ? S : Sc));

    /* ---------- self-attention ---------- */
    aria_rmsnorm(h, x, w->pre_norm, S, dim, eps_norm);
    adaln_modulate(h, scale_self, shift_self, S, dim);
    {
        int nq = diff ? 5 : 3;
        float *qkv = aria_arena_floats(ar, (size_t)S * nq * dim);
        if (bq) aria_linear_qw(qkv, h, &bq->sa_to_qkv, NULL, S);  /* bq sized nq*dim */
        else    aria_linear(qkv, h, w->sa_to_qkv, NULL, S, dim, nq * dim);
        lora_hook(lora, layer_idx, ARIA_LORA_SA_TO_QKV, qkv, h, S, dim, nq * dim, ar);
        extract_heads(qh, qkv, S, H, hd, nq * dim, 0);
        extract_heads(kh, qkv, S, H, hd, nq * dim, dim);
        extract_heads(vh, qkv, S, H, hd, nq * dim, 2 * dim);
        if (diff) {
            extract_heads(qdh, qkv, S, H, hd, nq * dim, 3 * dim);
            extract_heads(kdh, qkv, S, H, hd, nq * dim, 4 * dim);
        }
    }
    aria_rmsnorm(qh, qh, w->sa_q_norm, H * S, hd, eps_qk);
    aria_rmsnorm(kh, kh, w->sa_k_norm, H * S, hd, eps_qk);
    aria_rope_apply(qh, rope_cos, rope_sin, H, S, hd, rot_dim);
    aria_rope_apply(kh, rope_cos, rope_sin, H, S, hd, rot_dim);
    aria_attention(ao, qh, kh, vh, H, S, S, hd, NULL, scores);
    if (diff) {
        aria_rmsnorm(qdh, qdh, w->sa_q_norm, H * S, hd, eps_qk);
        aria_rmsnorm(kdh, kdh, w->sa_k_norm, H * S, hd, eps_qk);
        aria_rope_apply(qdh, rope_cos, rope_sin, H, S, hd, rot_dim);
        aria_rope_apply(kdh, rope_cos, rope_sin, H, S, hd, rot_dim);
        aria_attention(aod, qdh, kdh, vh, H, S, S, hd, NULL, scores);
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < (size_t)H * S * hd; i++) ao[i] -= aod[i];
    }
    merge_heads(merged, ao, S, H, hd);
    if (bq) aria_linear_qw(o, merged, &bq->sa_to_out, NULL, S);
    else    aria_linear(o, merged, w->sa_to_out, NULL, S, dim, dim);
    lora_hook(lora, layer_idx, ARIA_LORA_SA_TO_OUT, o, merged, S, dim, dim, ar);
    gate_sigmoid(o, gate_self, S, dim);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < (size_t)S * dim; i++) x[i] += o[i];   /* in-place residual add */

    /* ---------- cross-attention (cached K/V, no rope, no gate) ---------- */
    aria_rmsnorm(h, x, w->cross_norm, S, dim, eps_norm);
    {
        int nq = diff ? 2 : 1;
        float *q = aria_arena_floats(ar, (size_t)S * nq * dim);
        if (bq) aria_linear_qw(q, h, &bq->ca_to_q, NULL, S);  /* bq sized nq*dim */
        else    aria_linear(q, h, w->ca_to_q, NULL, S, dim, nq * dim);
        lora_hook(lora, layer_idx, ARIA_LORA_CA_TO_Q, q, h, S, dim, nq * dim, ar);
        extract_heads(qh, q, S, H, hd, nq * dim, 0);
        aria_rmsnorm(qh, qh, w->ca_q_norm, H * S, hd, eps_qk);
        aria_attention(ao, qh, cross_k, cross_v, H, S, Sc, hd, NULL, scores);
        if (diff) {
            extract_heads(qdh, q, S, H, hd, nq * dim, dim);
            aria_rmsnorm(qdh, qdh, w->ca_q_norm, H * S, hd, eps_qk);
            aria_attention(aod, qdh, cross_kd, cross_v, H, S, Sc, hd, NULL, scores);
            #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < (size_t)H * S * hd; i++) ao[i] -= aod[i];
        }
        merge_heads(merged, ao, S, H, hd);
    }
    if (bq) aria_linear_qw(o, merged, &bq->ca_to_out, NULL, S);
    else    aria_linear(o, merged, w->ca_to_out, NULL, S, dim, dim);
    lora_hook(lora, layer_idx, ARIA_LORA_CA_TO_OUT, o, merged, S, dim, dim, ar);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < (size_t)S * dim; i++) x[i] += o[i];   /* in-place residual add */

    /* ---------- local-additive (inpaint) cond: x += left-padded local_emb ---------- */
    if (local_emb)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (size_t i = 0; i < (size_t)S * dim; i++) x[i] += local_emb[i];

    /* ---------- feed-forward (GLU) ---------- */
    aria_rmsnorm(h, x, w->ff_norm, S, dim, eps_norm);
    adaln_modulate(h, scale_ff, shift_ff, S, dim);
    {
        const aria_lora_item *lff_in  = lora ? aria_lora_find(lora, layer_idx, ARIA_LORA_FF_IN)  : NULL;
        const aria_lora_item *lff_out = lora ? aria_lora_find(lora, layer_idx, ARIA_LORA_FF_OUT) : NULL;
        if (!lff_in && !lff_out) {
            float *ffs = aria_arena_floats(ar, (size_t)S * 3 * inner);
            if (bq) ff_glu_qw(o, h, S, inner, dim, &bq->ff_in_w, w->ff_in_b, &bq->ff_out_w, w->ff_out_b, ffs);
            else    aria_ff_glu(o, h, S, dim, inner, dim, w->ff_in_w, w->ff_in_b, w->ff_out_w, w->ff_out_b, ffs);
        } else {
            /* explicit GLU so the ff_in/ff_out LoRA residuals slot between the two GEMMs
             * (base weights untouched; GLU order matches aria_ff_glu / ff_glu_qw). */
            float *proj  = aria_arena_floats(ar, (size_t)S * 2 * inner);
            float *gated = aria_arena_floats(ar, (size_t)S * inner);
            if (bq) aria_linear_qw(proj, h, &bq->ff_in_w, w->ff_in_b, S);
            else    aria_linear(proj, h, w->ff_in_w, w->ff_in_b, S, dim, 2 * inner);
            lora_hook(lora, layer_idx, ARIA_LORA_FF_IN, proj, h, S, dim, 2 * inner, ar);
            for (int n = 0; n < S; n++)
                aria_silu_gate(gated + (size_t)n * inner,
                               proj + (size_t)n * 2 * inner + inner, proj + (size_t)n * 2 * inner, inner);
            if (bq) aria_linear_qw(o, gated, &bq->ff_out_w, w->ff_out_b, S);
            else    aria_linear(o, gated, w->ff_out_w, w->ff_out_b, S, inner, dim);
            lora_hook(lora, layer_idx, ARIA_LORA_FF_OUT, o, gated, S, inner, dim, ar);
        }
    }
    gate_sigmoid(o, gate_ff, S, dim);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (size_t i = 0; i < (size_t)S * dim; i++) x[i] += o[i];   /* in-place residual add */

    /* ---------- E12 residual steering at the block output, broadcast over all tokens, gated
     * by layer + step-window. ADD = x += scale*dir (AdditiveInjector). PROJECT = x -= scale *
     * (x.u) u  (Arditi/ds4 directional ablation; u = dir/||dir||, so per token subtract
     * scale*(dot(dir,x)/||dir||^2)*dir). scale==0 / out-of-window skip -> bit-exact. ---------- */
    if (steer) for (int s = 0; s < steer->n; s++) {
        const aria_steer *st = &steer->items[s];
        if (st->site != ARIA_STEER_RESIDUAL || st->layer != layer_idx) continue;
        if (st->scale == 0.0f || step < st->step_lo || step > st->step_hi) continue;
        int d = st->dim < dim ? st->dim : dim;
        if (st->op == ARIA_STEER_PROJECT) {
            float inv = st->dir_norm2 > 0.0f ? 1.0f / st->dir_norm2 : 0.0f;
            for (int t = 0; t < S; t++) {
                float *xt = x + (size_t)t * dim;
                float dot = 0.0f;
                for (int c = 0; c < d; c++) dot += st->dir[c] * xt[c];
                float k = st->scale * dot * inv;
                for (int c = 0; c < d; c++) xt[c] -= k * st->dir[c];
            }
        } else {
            for (int t = 0; t < S; t++)
                for (int c = 0; c < d; c++)
                    x[(size_t)t * dim + c] += st->scale * st->dir[c];
        }
    }

    aria_arena_restore(ar, mark);
}

/* Standalone block forward (used by the block parity test): projects the cross
 * K/V from `context` here, then runs the shared core on a private arena. */
void aria_dit_block_forward(float *x, int S, int dim, int num_heads, int head_dim, int inner,
                            const float *context, int Sc, int dim_ctx,
                            const float *global_cond,
                            const float *rope_cos, const float *rope_sin, int rot_dim,
                            const aria_dit_block_w *w, int differential) {
    int H = num_heads, hd = head_dim;
    int nkv = differential ? 3 : 2;   /* to_kv = [k,v] or [k,k_diff,v] */
    float *kv = malloc((size_t)Sc * nkv * dim * sizeof(float));
    float *cross_k = malloc((size_t)H * Sc * hd * sizeof(float));
    float *cross_v = malloc((size_t)H * Sc * hd * sizeof(float));
    float *cross_kd = differential ? malloc((size_t)H * Sc * hd * sizeof(float)) : NULL;
    aria_linear(kv, context, w->ca_to_kv, NULL, Sc, dim_ctx, nkv * dim);
    if (differential) {                /* chunk order: k, k_diff, v */
        extract_heads(cross_k,  kv, Sc, H, hd, nkv * dim, 0);
        extract_heads(cross_kd, kv, Sc, H, hd, nkv * dim, dim);
        extract_heads(cross_v,  kv, Sc, H, hd, nkv * dim, 2 * dim);
        aria_rmsnorm(cross_k,  cross_k,  w->ca_k_norm, H * Sc, hd, 1e-6f);
        aria_rmsnorm(cross_kd, cross_kd, w->ca_k_norm, H * Sc, hd, 1e-6f);
    } else {
        extract_heads(cross_k, kv, Sc, H, hd, nkv * dim, 0);
        extract_heads(cross_v, kv, Sc, H, hd, nkv * dim, dim);
        aria_rmsnorm(cross_k, cross_k, w->ca_k_norm, H * Sc, hd, 1e-6f);
    }
    free(kv);

    aria_arena ar;
    size_t cap = ((size_t)6 * dim + 30 * (size_t)S * dim + (size_t)S * (S > Sc ? S : Sc)
                  + (1u << 18)) * sizeof(float);
    aria_arena_init(&ar, cap);
    dit_block_core(x, S, dim, H, hd, inner, cross_k, cross_v, cross_kd, Sc, global_cond,
                   rope_cos, rope_sin, rot_dim, NULL, w, NULL, differential, NULL, NULL, 0, 0, &ar);
    aria_arena_free(&ar);
    free(cross_k); free(cross_v); free(cross_kd);
}

/* ---------------- full DiT model ---------------- */

struct aria_sa3_dit {
    int depth, ed, num_heads, head_dim, inner, io_ch, n_mem, rot_dim, cond_dim, ts_feat_dim;
    int differential;   /* medium: differential self+cross attention */
    const float *preprocess, *postprocess, *project_in, *project_out, *memory_tokens;
    const float *to_cond0, *to_cond2, *to_global0, *to_global2;
    const float *to_ts0_w, *to_ts0_b, *to_ts2_w, *to_ts2_b;
    const float *gce0_w, *gce0_b, *gce2_w, *gce2_b;
    aria_dit_block_w *blocks;
    aria_dtype precision;   /* weight precision for the block GEMMs (default F32) */
    dit_block_q *bq;        /* quantized overlay [depth], or NULL when precision==F32 */
    int failed;
};

/* zero-copy: borrow the F32 weight directly from the mmap (valid while sf open) */
static const float *track(aria_sa3_dit *m, safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) { fprintf(stderr, "aria_sa3_dit_load: missing tensor %s\n", name); m->failed = 1; return NULL; }
    const float *p = safetensors_f32_ptr(sf, t);
    if (!p) { fprintf(stderr, "aria_sa3_dit_load: %s is not F32\n", name); m->failed = 1; return NULL; }
    return p;
}

aria_sa3_dit *aria_sa3_dit_load(safetensors_file_t *sf, const aria_sa3_config *cfg) {
    aria_sa3_dit *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->depth = cfg->depth;
    m->ed = cfg->embed_dim;
    m->num_heads = cfg->num_heads;
    m->head_dim = cfg->head_dim;
    m->inner = (int)(cfg->embed_dim * cfg->ff_mult);
    m->io_ch = cfg->io_channels;
    m->n_mem = cfg->num_memory_tokens;
    m->rot_dim = cfg->head_dim / 2 < 32 ? 32 : cfg->head_dim / 2;  /* max(head_dim//2, 32) */
    m->cond_dim = cfg->cond_token_dim;
    m->ts_feat_dim = 256;
    m->differential = cfg->differential_attn;
    m->blocks = calloc((size_t)m->depth, sizeof(aria_dit_block_w));

    m->preprocess    = track(m, sf, "model.model.preprocess_conv.weight");
    m->postprocess   = track(m, sf, "model.model.postprocess_conv.weight");
    m->project_in    = track(m, sf, "model.model.transformer.project_in.weight");
    m->project_out   = track(m, sf, "model.model.transformer.project_out.weight");
    m->memory_tokens = track(m, sf, "model.model.transformer.memory_tokens");
    m->to_cond0      = track(m, sf, "model.model.to_cond_embed.0.weight");
    m->to_cond2      = track(m, sf, "model.model.to_cond_embed.2.weight");
    m->to_global0    = track(m, sf, "model.model.to_global_embed.0.weight");
    m->to_global2    = track(m, sf, "model.model.to_global_embed.2.weight");
    m->to_ts0_w      = track(m, sf, "model.model.to_timestep_embed.0.weight");
    m->to_ts0_b      = track(m, sf, "model.model.to_timestep_embed.0.bias");
    m->to_ts2_w      = track(m, sf, "model.model.to_timestep_embed.2.weight");
    m->to_ts2_b      = track(m, sf, "model.model.to_timestep_embed.2.bias");
    m->gce0_w        = track(m, sf, "model.model.transformer.global_cond_embedder.0.weight");
    m->gce0_b        = track(m, sf, "model.model.transformer.global_cond_embedder.0.bias");
    m->gce2_w        = track(m, sf, "model.model.transformer.global_cond_embedder.2.weight");
    m->gce2_b        = track(m, sf, "model.model.transformer.global_cond_embedder.2.bias");

    char nm[256];
    for (int i = 0; i < m->depth; i++) {
        aria_dit_block_w *b = &m->blocks[i];
        #define BW(field, suffix) do { \
            snprintf(nm, sizeof(nm), "model.model.transformer.layers.%d." suffix, i); \
            b->field = track(m, sf, nm); } while (0)
        BW(pre_norm, "pre_norm.gamma");
        BW(cross_norm, "cross_attend_norm.gamma");
        BW(ff_norm, "ff_norm.gamma");
        BW(sa_to_qkv, "self_attn.to_qkv.weight");
        BW(sa_q_norm, "self_attn.q_norm.gamma");
        BW(sa_k_norm, "self_attn.k_norm.gamma");
        BW(sa_to_out, "self_attn.to_out.weight");
        BW(ca_to_q, "cross_attn.to_q.weight");
        BW(ca_to_kv, "cross_attn.to_kv.weight");
        BW(ca_q_norm, "cross_attn.q_norm.gamma");
        BW(ca_k_norm, "cross_attn.k_norm.gamma");
        BW(ca_to_out, "cross_attn.to_out.weight");
        BW(ff_in_w, "ff.ff.0.proj.weight");
        BW(ff_in_b, "ff.ff.0.proj.bias");
        BW(ff_out_w, "ff.ff.2.weight");
        BW(ff_out_b, "ff.ff.2.bias");
        BW(to_scale_shift_gate, "to_scale_shift_gate");
        BW(to_local0_w, "to_local_embed.0.weight");
        BW(to_local0_b, "to_local_embed.0.bias");
        BW(to_local2_w, "to_local_embed.2.weight");
        BW(to_local2_b, "to_local_embed.2.bias");
        #undef BW
    }

    if (m->failed) { aria_sa3_dit_free(m); return NULL; }
    return m;
}

/* free the quantized overlay (if any). */
static void dit_free_bq(aria_sa3_dit *m) {
    if (!m->bq) return;
    for (int i = 0; i < m->depth; i++) {
        dit_block_q *q = &m->bq[i];
        aria_qweight_free(&q->sa_to_qkv); aria_qweight_free(&q->sa_to_out);
        aria_qweight_free(&q->ca_to_q);   aria_qweight_free(&q->ca_to_out);
        aria_qweight_free(&q->ff_in_w);   aria_qweight_free(&q->ff_out_w);
    }
    free(m->bq); m->bq = NULL;
}

/* (Re)build the quantized weight overlay for the block GEMMs. dt==F32 drops the
 * overlay (back to the zero-copy mmap path). Idempotent: no-op if dt unchanged. */
/* B2: drop the mmap'd fp32 source of a block weight now that the overlay (q8/q4/fp16/
 * bf16) holds its own copy and the CPU hot path reads the overlay, not this tensor.
 * Only the fully-interior pages are advised away, so a page shared with an adjacent
 * still-needed tensor (a norm gamma, a bias) is never touched; the source is file-
 * backed and never written, so a stray later read (e.g. a GPU get_view on the same
 * resident model) just re-faults from disk. This is what turns the CPU overlay from
 * a footprint ADD (source + copy resident) into the promised footprint REPLACE. */
static void dit_release_source(const void *p, size_t bytes) {
    if (!p || !bytes) return;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) return;
    uintptr_t a = (uintptr_t)p, lo = (a + (uintptr_t)pg - 1) & ~((uintptr_t)pg - 1);
    uintptr_t hi = (a + bytes) & ~((uintptr_t)pg - 1);
    if (hi > lo) madvise((void *)lo, (size_t)(hi - lo), MADV_DONTNEED);
}

void aria_sa3_dit_quantize(aria_sa3_dit *m, aria_dtype dt) {
    if (!m || dt == m->precision) return;
    dit_free_bq(m);
    m->precision = dt;
    if (dt == ARIA_F32) return;
    int ed = m->ed, inner = m->inner;
    /* Mixed precision for Q4: the four attention projections feed qk-rmsnorm +
     * softmax, where quantization error is amplified through the nonlinearity, so
     * keep them Q8; only the FFN (the bulk of the params) goes Q4. This is what
     * makes Q4 usable -- ~9% DiT velocity error vs ~23% for uniform asym-Q4. The
     * Q8 path stays uniform Q8. (The GPU mirrors this in aria_cuda_dit_create.) */
    aria_dtype adt = (dt == ARIA_Q4) ? ARIA_Q8 : dt;   /* attention precision */
    int nq = m->differential ? 5 : 3, ncq = m->differential ? 2 : 1;  /* medium: 5/2-way */
    m->bq = calloc((size_t)m->depth, sizeof(dit_block_q));
    for (int i = 0; i < m->depth; i++) {
        const aria_dit_block_w *w = &m->blocks[i];
        dit_block_q *q = &m->bq[i];
        aria_qweight_set(&q->sa_to_qkv, w->sa_to_qkv, nq * ed, ed, adt);
        aria_qweight_set(&q->sa_to_out, w->sa_to_out, ed, ed, adt);
        aria_qweight_set(&q->ca_to_q,   w->ca_to_q,   ncq * ed, ed, adt);
        aria_qweight_set(&q->ca_to_out, w->ca_to_out, ed, ed, adt);
        aria_qweight_set(&q->ff_in_w,   w->ff_in_w,   2 * inner, ed, dt);
        aria_qweight_set(&q->ff_out_w,  w->ff_out_w,  ed, inner, dt);
        /* the overlay owns its copy now: reclaim the fp32 mmap source of the six big
         * GEMM matrices (ca_to_kv stays f32 -- projected per request -- so it is NOT
         * released; norms/biases are tiny and shared-page, left resident). */
        dit_release_source(w->sa_to_qkv, (size_t)nq * ed * ed * sizeof(float));
        dit_release_source(w->sa_to_out, (size_t)ed * ed * sizeof(float));
        dit_release_source(w->ca_to_q,   (size_t)ncq * ed * ed * sizeof(float));
        dit_release_source(w->ca_to_out, (size_t)ed * ed * sizeof(float));
        dit_release_source(w->ff_in_w,   (size_t)2 * inner * ed * sizeof(float));
        dit_release_source(w->ff_out_w,  (size_t)ed * inner * sizeof(float));
    }
}

/* ---- offline packed quantized DiT (E9.2) ---- */
#define ARIA_DIT_QMAGIC "ARIAQNT1"

/* the 6 packed per-block GEMM weights, in a stable order for (de)serialization. */
static aria_qweight *dit_block_q_weights(dit_block_q *q, aria_qweight **out6) {
    out6[0] = &q->sa_to_qkv; out6[1] = &q->sa_to_out; out6[2] = &q->ca_to_q;
    out6[3] = &q->ca_to_out; out6[4] = &q->ff_in_w;  out6[5] = &q->ff_out_w;
    return NULL;
}

/* Quantize `m` to `dt` and write the packed block overlay to `path`. 0 on success. */
int aria_sa3_dit_quant_save(aria_sa3_dit *m, aria_dtype dt, const char *path) {
    if (!m || (dt != ARIA_Q8 && dt != ARIA_Q4)) return -1;
    aria_sa3_dit_quantize(m, dt);
    if (!m->bq) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int32_t hdr[4] = { (int32_t)dt, m->depth, m->ed, m->inner };
    int ok = (fwrite(ARIA_DIT_QMAGIC, 1, 8, f) == 8) && (fwrite(hdr, sizeof(int32_t), 4, f) == 4);
    for (int i = 0; ok && i < m->depth; i++) {
        aria_qweight *w6[6]; dit_block_q_weights(&m->bq[i], w6);
        for (int j = 0; j < 6; j++) if (aria_qweight_write(f, w6[j]) != 0) { ok = 0; break; }
    }
    fclose(f);
    return ok ? 0 : -1;
}

/* Load a packed block overlay (from aria_sa3_dit_quant_save) into m->bq. The
 * model keeps its f32 weights for the non-block parts; only the per-step GEMMs
 * come from the file. Returns 0 on success, <0 on error / shape mismatch. */
int aria_sa3_dit_quant_load(aria_sa3_dit *m, const char *path) {
    if (!m) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[8]; int32_t hdr[4];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, ARIA_DIT_QMAGIC, 8) != 0 ||
        fread(hdr, sizeof(int32_t), 4, f) != 4 ||
        hdr[1] != m->depth || hdr[2] != m->ed || hdr[3] != m->inner) { fclose(f); return -1; }
    dit_free_bq(m);
    m->precision = (aria_dtype)hdr[0];
    m->bq = calloc((size_t)m->depth, sizeof(dit_block_q));
    int ok = (m->bq != NULL);
    for (int i = 0; ok && i < m->depth; i++) {
        aria_qweight *w6[6]; dit_block_q_weights(&m->bq[i], w6);
        for (int j = 0; j < 6; j++) if (aria_qweight_read(f, w6[j]) != 0) { ok = 0; break; }
    }
    fclose(f);
    if (!ok) { dit_free_bq(m); m->precision = ARIA_F32; return -1; }
    return 0;
}

/* total bytes of the block GEMM weights at the current precision (for reporting). */
size_t aria_sa3_dit_weight_bytes(const aria_sa3_dit *m) {
    if (!m) return 0;
    if (!m->bq) {
        int ed = m->ed, inner = m->inner;   /* the 6 per-step GEMMs (ca_to_kv excluded) */
        int nq = m->differential ? 5 : 3, ncq = m->differential ? 2 : 1;
        size_t per = ((size_t)nq * ed * ed + (size_t)ed * ed + (size_t)ncq * ed * ed
                      + (size_t)ed * ed + (size_t)2 * inner * ed + (size_t)ed * inner) * sizeof(float);
        return per * m->depth;
    }
    size_t tot = 0;
    for (int i = 0; i < m->depth; i++) {
        const dit_block_q *q = &m->bq[i];
        tot += aria_qweight_bytes(&q->sa_to_qkv) + aria_qweight_bytes(&q->sa_to_out)
             + aria_qweight_bytes(&q->ca_to_q)   + aria_qweight_bytes(&q->ca_to_out)
             + aria_qweight_bytes(&q->ff_in_w)   + aria_qweight_bytes(&q->ff_out_w);
    }
    return tot;
}

void aria_sa3_dit_free(aria_sa3_dit *m) {
    if (!m) return;
    dit_free_bq(m);
    free(m->blocks);   /* f32 weights are borrowed from the mmap; nothing else to free */
    free(m);
}

/* transpose [C,T] -> [T,C] */
static void transpose_ct_tc(float *dst, const float *src, int C, int T) {
    for (int c = 0; c < C; c++)
        for (int t = 0; t < T; t++)
            dst[(size_t)t * C + c] = src[(size_t)c * T + t];
}
/* transpose [T,C] -> [C,T] */
static void transpose_tc_ct(float *dst, const float *src, int T, int C) {
    for (int t = 0; t < T; t++)
        for (int c = 0; c < C; c++)
            dst[(size_t)c * T + t] = src[(size_t)t * C + c];
}

/* ---------------- per-request context ----------------
 * Immutable model state (weights) lives in aria_sa3_dit; everything that is
 * scratch or step-invariant for one generation lives here, so the N denoising
 * steps reuse a single arena and never reproject the constant prompt context. */
struct aria_sa3_dit_req {
    const aria_sa3_dit *m;
    int T, S, n_cond;
    aria_arena arena;
    float *cross_ed;        /* [n_cond, ed]  to_cond_embed(prompt) */
    float *global_seconds;  /* [ed]          to_global_embed(seconds) (timestep added per step) */
    float *rope_cos, *rope_sin;             /* [S, rot/2] */
    float **cross_k, **cross_v;             /* [depth] each [H, n_cond, hd]; cross_k post k_norm */
    float **cross_kd;                       /* [depth] differential cross k_diff (post k_norm), or NULL */
    float **local_emb; int has_local;       /* [depth] each [S, ed] (left-padded) or NULL */
    const aria_steer_set *steer; int cur_step;  /* E12: residual-site steering + current denoise step */
    const aria_lora_adapter *lora;          /* E12.9: runtime LoRA adapter (CPU), or NULL */
};

/* Project block b's cross-attention K/V from cross_ed into the (pre-allocated)
 * r->cross_k[b]/cross_v[b]/cross_kd[b], folding k_norm into K/K_diff. `kv` is caller
 * scratch [n_cond, nkv*dim]. With a matching ca_to_kv LoRA item its low-rank residual
 * is added to the kv projection before the head split (base weight untouched). */
static void req_project_cross_kv(aria_sa3_dit_req *r, int b, float *kv,
                                 const aria_lora_adapter *lora) {
    const aria_sa3_dit *m = r->m;
    int ed = m->ed, H = m->num_heads, hd = m->head_dim, dim = ed, diff = m->differential;
    int nkv = diff ? 3 : 2, n_cond = r->n_cond;
    const aria_dit_block_w *w = &m->blocks[b];
    aria_linear(kv, r->cross_ed, w->ca_to_kv, NULL, n_cond, ed, nkv * dim);   /* f32 (once/request) */
    if (lora) {
        const aria_lora_item *it = aria_lora_find(lora, b, ARIA_LORA_CA_TO_KV);
        if (it) aria_lora_linear(kv, r->cross_ed, n_cond, ed, nkv * dim,
                                 it->rank, it->scale, it->down, it->up, NULL);
    }
    extract_heads(r->cross_k[b], kv, n_cond, H, hd, nkv * dim, 0);
    if (diff) {                                      /* chunk order: k, k_diff, v */
        extract_heads(r->cross_kd[b], kv, n_cond, H, hd, nkv * dim, dim);
        extract_heads(r->cross_v[b],  kv, n_cond, H, hd, nkv * dim, 2 * dim);
        aria_rmsnorm(r->cross_kd[b], r->cross_kd[b], w->ca_k_norm, H * n_cond, hd, 1e-6f);
    } else {
        extract_heads(r->cross_v[b], kv, n_cond, H, hd, nkv * dim, dim);
    }
    aria_rmsnorm(r->cross_k[b], r->cross_k[b], w->ca_k_norm, H * n_cond, hd, 1e-6f);
}

aria_sa3_dit_req *aria_sa3_dit_req_begin(const aria_sa3_dit *m, int T,
                                         const float *cross_768, int n_cond,
                                         const float *global_768, int gpu) {
    int ed = m->ed, H = m->num_heads, hd = m->head_dim, dim = ed;
    int S = m->n_mem + T, rot = m->rot_dim, depth = m->depth;
    aria_sa3_dit_req *r = calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->m = m; r->T = T; r->S = S; r->n_cond = n_cond;

    /* step-invariant conditioning */
    r->cross_ed = malloc((size_t)n_cond * ed * sizeof(float));
    aria_sa3_mlp2(r->cross_ed, cross_768, n_cond, m->cond_dim, ed, ed, m->to_cond0, NULL, m->to_cond2, NULL);
    r->global_seconds = malloc((size_t)ed * sizeof(float));
    aria_sa3_mlp2(r->global_seconds, global_768, 1, m->cond_dim, ed, ed, m->to_global0, NULL, m->to_global2, NULL);

    r->rope_cos = malloc((size_t)S * (rot / 2) * sizeof(float));
    r->rope_sin = malloc((size_t)S * (rot / 2) * sizeof(float));
    aria_rope_freqs(r->rope_cos, r->rope_sin, S, rot, 10000.0f);

    /* per-block cross-attention K/V: projected once from cross_ed (constant across
     * all steps), head-split, with k_norm folded into the cached K. The GPU backend
     * defers this (gpu=1): it projects on-device from cross_ed instead. */
    int diff = m->differential;
    if (!gpu) {
    int nkv = diff ? 3 : 2;   /* to_kv = [k,v] or [k,k_diff,v] */
    r->cross_k = calloc((size_t)depth, sizeof(float *));
    r->cross_v = calloc((size_t)depth, sizeof(float *));
    r->cross_kd = diff ? calloc((size_t)depth, sizeof(float *)) : NULL;
    float *kv = malloc((size_t)n_cond * nkv * dim * sizeof(float));
    for (int b = 0; b < depth; b++) {
        r->cross_k[b] = malloc((size_t)H * n_cond * hd * sizeof(float));
        r->cross_v[b] = malloc((size_t)H * n_cond * hd * sizeof(float));
        if (diff) r->cross_kd[b] = malloc((size_t)H * n_cond * hd * sizeof(float));
        req_project_cross_kv(r, b, kv, NULL);   /* base only; ca_to_kv LoRA folded in set_lora */
    }
    free(kv);
    }  /* !gpu */

    /* arena holds the persistent residual stream + step input scratch + one block's
     * worth of temporaries (blocks save/restore, so only one is live at a time). */
    int inner = m->inner;
    /* one block's live scratch (no intra-block restore): ~20*S*dim non-diff,
     * ~30*S*dim with differential (extra 5-way qkv + q/k_diff + 2nd attn out). */
    size_t block_floats = (size_t)6 * dim + (diff ? 32 : 24) * (size_t)S * dim
                          + (size_t)S * (S > n_cond ? S : n_cond);
    (void)inner;  /* 3*S*inner == 12*S*dim is already inside the bound */
    size_t step_floats  = (size_t)S * ed + 8 * (size_t)T * m->io_ch + 8 * (size_t)ed;
    aria_arena_init(&r->arena, (block_floats + step_floats + (1u << 20)) * sizeof(float));
    return r;
}

void aria_sa3_dit_req_set_local(aria_sa3_dit_req *r, const float *local_raw,
                                int n_local, int local_dim) {
    const aria_sa3_dit *m = r->m;
    int ed = m->ed, Mt = m->n_mem, S = r->S, depth = m->depth;
    if (!local_raw || n_local <= 0) return;  /* no-op (text->audio) */
    /* each block: emb = to_local_embed(local_raw) [n_local, ed]; left-pad past the
     * Mt memory tokens into [S, ed] (the cond aligns to the audio-latent positions). */
    r->local_emb = calloc((size_t)depth, sizeof(float *));
    float *emb = malloc((size_t)n_local * ed * sizeof(float));
    for (int b = 0; b < depth; b++) {
        const aria_dit_block_w *w = &m->blocks[b];
        aria_sa3_mlp2(emb, local_raw, n_local, local_dim, ed, ed,
                      w->to_local0_w, w->to_local0_b, w->to_local2_w, w->to_local2_b);
        r->local_emb[b] = calloc((size_t)S * ed, sizeof(float));  /* zeros for memory tokens */
        int rows = n_local < (S - Mt) ? n_local : (S - Mt);
        memcpy(r->local_emb[b] + (size_t)Mt * ed, emb, (size_t)rows * ed * sizeof(float));
    }
    free(emb);
    r->has_local = 1;
}

void aria_sa3_dit_req_set_steer(aria_sa3_dit_req *r, const aria_steer_set *steer) { r->steer = steer; }
void aria_sa3_dit_req_set_step(aria_sa3_dit_req *r, int step) { r->cur_step = step; }

void aria_sa3_dit_req_set_lora(aria_sa3_dit_req *r, const aria_lora_adapter *lora) {
    r->lora = lora;
    if (!lora || !r->cross_k) return;   /* GPU path: host cross K/V deferred (E12.9b) */
    /* re-fold ca_to_kv adapters into the cached cross K/V (projected base-only in
     * req_begin). Only blocks that carry a ca_to_kv item are recomputed. */
    int nkv = r->m->differential ? 3 : 2;
    float *kv = NULL;
    for (int b = 0; b < r->m->depth; b++) {
        if (!aria_lora_find(lora, b, ARIA_LORA_CA_TO_KV)) continue;
        if (!kv) kv = malloc((size_t)r->n_cond * nkv * r->m->ed * sizeof(float));
        req_project_cross_kv(r, b, kv, lora);
    }
    free(kv);
}

void aria_sa3_dit_step(const aria_sa3_dit *m, aria_sa3_dit_req *r,
                       float *out_CT, const float *x_CT, float t) {
    int C = m->io_ch, ed = m->ed, Mt = m->n_mem, S = r->S, T = r->T, rot = m->rot_dim;
    aria_arena *ar = &r->arena;
    size_t mark = aria_arena_save(ar);

    /* global_cond = global_cond_embedder(global_seconds + timestep_embed(t)) */
    float *g = aria_arena_floats(ar, (size_t)ed);
    aria_sa3_timestep_embed(g, t, m->ts_feat_dim, ed, m->to_ts0_w, m->to_ts0_b, m->to_ts2_w, m->to_ts2_b);
    for (int i = 0; i < ed; i++) g[i] += r->global_seconds[i];
    float *gcond = aria_arena_floats(ar, (size_t)6 * ed);
    aria_sa3_mlp2(gcond, g, 1, ed, ed, 6 * ed, m->gce0_w, m->gce0_b, m->gce2_w, m->gce2_b);

    /* preprocess_conv(x) + x  (1x1 conv over channels), x: [C,T] -> [T,C] */
    float *xtc = aria_arena_floats(ar, (size_t)T * C);
    transpose_ct_tc(xtc, x_CT, C, T);
    float *pre = aria_arena_floats(ar, (size_t)T * C);
    aria_linear(pre, xtc, m->preprocess, NULL, T, C, C);
    for (size_t i = 0; i < (size_t)T * C; i++) xtc[i] += pre[i];

    /* project_in [T,C] -> [T,ed], prepend memory tokens -> seq [S,ed] */
    float *seq = aria_arena_floats(ar, (size_t)S * ed);
    memcpy(seq, m->memory_tokens, (size_t)Mt * ed * sizeof(float));
    aria_linear(seq + (size_t)Mt * ed, xtc, m->project_in, NULL, T, C, ed);

    for (int b = 0; b < m->depth; b++)
        dit_block_core(seq, S, ed, m->num_heads, m->head_dim, m->inner,
                       r->cross_k[b], r->cross_v[b], r->cross_kd ? r->cross_kd[b] : NULL, r->n_cond, gcond,
                       r->rope_cos, r->rope_sin, rot,
                       r->has_local ? r->local_emb[b] : NULL, &m->blocks[b],
                       m->bq ? &m->bq[b] : NULL, m->differential, r->steer, r->lora, b, r->cur_step, ar);

    /* strip memory tokens, project_out [T,ed] -> [T,C] */
    float *outtc = aria_arena_floats(ar, (size_t)T * C);
    aria_linear(outtc, seq + (size_t)Mt * ed, m->project_out, NULL, T, ed, C);

    /* postprocess_conv(out) + out */
    float *postc = aria_arena_floats(ar, (size_t)T * C);
    aria_linear(postc, outtc, m->postprocess, NULL, T, C, C);
    for (size_t i = 0; i < (size_t)T * C; i++) outtc[i] += postc[i];

    transpose_tc_ct(out_CT, outtc, T, C);
    aria_arena_restore(ar, mark);
}

void aria_sa3_dit_req_end(aria_sa3_dit_req *r) {
    if (!r) return;
    aria_arena_free(&r->arena);
    free(r->cross_ed); free(r->global_seconds); free(r->rope_cos); free(r->rope_sin);
    if (r->cross_k) for (int b = 0; b < r->m->depth; b++) free(r->cross_k[b]);
    if (r->cross_v) for (int b = 0; b < r->m->depth; b++) free(r->cross_v[b]);
    if (r->cross_kd) for (int b = 0; b < r->m->depth; b++) free(r->cross_kd[b]);
    free(r->cross_k); free(r->cross_v); free(r->cross_kd);
    if (r->local_emb) for (int b = 0; b < r->m->depth; b++) free(r->local_emb[b]);
    free(r->local_emb);
    free(r);
}

void aria_sa3_dit_forward(const aria_sa3_dit *m, float *out_CT, const float *x_CT, int T,
                          float t, const float *cross_768, int n_cond, const float *global_768) {
    aria_sa3_dit_req *r = aria_sa3_dit_req_begin(m, T, cross_768, n_cond, global_768, 0);
    aria_sa3_dit_step(m, r, out_CT, x_CT, t);
    aria_sa3_dit_req_end(r);
}

void aria_sa3_dit_get_view(const aria_sa3_dit *m, aria_sa3_dit_view *v) {
    v->depth = m->depth; v->ed = m->ed; v->num_heads = m->num_heads; v->head_dim = m->head_dim;
    v->inner = m->inner; v->io_ch = m->io_ch; v->n_mem = m->n_mem; v->rot_dim = m->rot_dim;
    v->differential = m->differential;
    v->preprocess = m->preprocess; v->postprocess = m->postprocess;
    v->project_in = m->project_in; v->project_out = m->project_out; v->memory_tokens = m->memory_tokens;
    v->blocks = m->blocks;
}

void aria_sa3_dit_req_get_view(const aria_sa3_dit_req *r, aria_sa3_dit_req_view *v) {
    v->T = r->T; v->S = r->S; v->n_cond = r->n_cond; v->depth = r->m->depth;
    v->global_seconds = r->global_seconds; v->rope_cos = r->rope_cos; v->rope_sin = r->rope_sin;
    v->cross_k = r->cross_k; v->cross_v = r->cross_v; v->cross_kd = r->cross_kd;
    v->cross_ed = r->cross_ed;
    v->local_emb = r->local_emb; v->has_local = r->has_local;
    v->steer = r->steer;
}

void aria_sa3_dit_global_cond(const aria_sa3_dit *m, const float *global_seconds, float t, float *gcond) {
    int ed = m->ed;
    float *g = malloc((size_t)ed * sizeof(float));
    aria_sa3_timestep_embed(g, t, m->ts_feat_dim, ed, m->to_ts0_w, m->to_ts0_b, m->to_ts2_w, m->to_ts2_b);
    for (int i = 0; i < ed; i++) g[i] += global_seconds[i];
    aria_sa3_mlp2(gcond, g, 1, ed, ed, 6 * ed, m->gce0_w, m->gce0_b, m->gce2_w, m->gce2_b);
    free(g);
}
