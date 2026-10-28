/*
 * aria_sa3_dec_medium.c - taae_v2 decoder for the MEDIUM model (latent -> audio).
 *
 * Differs from small-music (aria_sa3_dec.c): transformer_dim 1536, depth 12,
 * inner 4608, mapping WNConv1d k1 (not k3), and -- the big one -- the resampling
 * runs the whole token sequence through SLIDING-WINDOW (banded) attention instead
 * of chunked attention with a midpoint-shift halo. The later blocks use a
 * sinusoidal FF. The block kernel is taae_med_block_forward (aria_taae.c).
 *
 * Flow (latent[256,T] -> audio[2, T*4096]):
 *   1. softnorm: z = latent * running_std
 *   2. transpose + Linear 256->1536 (decoder.layers.1)
 *   3. per latent token: [token, 16x new_tokens] -> seq[T*17, 1536]
 *   4. 12 blocks over the full seq, band mask window [17,17]; blocks with
 *      (12-i) < 8 use the sinusoidal FF
 *   5. take the last 16 of each 17-group -> feat[1536, T*16]
 *   6. WNConv1d mapping 1536->512 (k1), unpatch (256) -> stereo audio
 */

#include "aria_sa3_dec.h"
#include "aria_taae.h"
#include "aria_ops.h"
#include "aria_arena.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MED_D     1536
#define MED_DEPTH 12
#define MED_H     24
#define MED_HD    64
#define MED_INNER 4608   /* ff_mult 3 */
#define MED_OUT   512
#define MED_SEG   17     /* 1 token + 16 new */
#define MED_WIN   17     /* sliding-window half-width [17,17] */

struct aria_sa3_dec_medium {
    float running_std;
    const float *proj_w, *proj_b;     /* layers.1 [1536,256],[1536] */
    const float *new_tokens;          /* [1536] */
    taae_block_w blocks[MED_DEPTH];
    float *mapping_w;                 /* folded [512,1536] (k1, owned) */
    const float *mapping_b;           /* [512] */
    int failed;
};

static const float *track(aria_sa3_dec_medium *m, safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) { fprintf(stderr, "dec_medium: missing %s\n", name); m->failed = 1; return NULL; }
    const float *p = safetensors_f32_ptr(sf, t);
    if (!p) { fprintf(stderr, "dec_medium: %s not F32\n", name); m->failed = 1; return NULL; }
    return p;
}
static float track_scalar(aria_sa3_dec_medium *m, safetensors_file_t *sf, const char *name) {
    const float *p = track(m, sf, name); return p ? p[0] : 0.0f;
}

aria_sa3_dec_medium *aria_sa3_dec_medium_load(safetensors_file_t *sf) {
    aria_sa3_dec_medium *m = calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->running_std = track_scalar(m, sf, "pretransform.model.bottleneck.running_std");
    m->proj_w = track(m, sf, "pretransform.model.decoder.layers.1.weight");
    m->proj_b = track(m, sf, "pretransform.model.decoder.layers.1.bias");
    m->new_tokens = track(m, sf, "pretransform.model.decoder.layers.3.new_tokens");

    char nm[256];
    #define P "pretransform.model.decoder.layers.3.transformers.%d."
    for (int i = 0; i < MED_DEPTH; i++) {
        taae_block_w *b = &m->blocks[i];
        #define SCA(dst, suf) do { snprintf(nm, sizeof(nm), P suf, i); dst = track_scalar(m, sf, nm); } while (0)
        #define PTR(dst, suf) do { snprintf(nm, sizeof(nm), P suf, i); dst = track(m, sf, nm); } while (0)
        SCA(b->pre_alpha, "pre_norm.alpha"); PTR(b->pre_gamma, "pre_norm.gamma"); PTR(b->pre_beta, "pre_norm.beta");
        PTR(b->to_qkv, "self_attn.to_qkv.weight");
        SCA(b->qn_alpha, "self_attn.q_norm.alpha"); PTR(b->qn_gamma, "self_attn.q_norm.gamma"); PTR(b->qn_beta, "self_attn.q_norm.beta");
        SCA(b->kn_alpha, "self_attn.k_norm.alpha"); PTR(b->kn_gamma, "self_attn.k_norm.gamma"); PTR(b->kn_beta, "self_attn.k_norm.beta");
        PTR(b->to_out, "self_attn.to_out.weight");
        SCA(b->ff_alpha, "ff_norm.alpha"); PTR(b->ff_gamma, "ff_norm.gamma"); PTR(b->ff_beta, "ff_norm.beta");
        PTR(b->ff_in_w, "ff.ff.0.proj.weight"); PTR(b->ff_in_b, "ff.ff.0.proj.bias");
        PTR(b->ff_out_w, "ff.ff.2.weight"); PTR(b->ff_out_b, "ff.ff.2.bias");
        #undef SCA
        #undef PTR
    }
    #undef P

    /* fold the weight-normalized mapping conv (k1): fw[o,i] = v[o,i] * g[o]/||v[o]|| */
    const float *g = track(m, sf, "pretransform.model.decoder.layers.3.mapping.weight_g");   /* [512,1,1] */
    const float *vv = track(m, sf, "pretransform.model.decoder.layers.3.mapping.weight_v");  /* [512,1536,1] */
    m->mapping_b = track(m, sf, "pretransform.model.decoder.layers.3.mapping.bias");
    if (!m->failed) {
        int rowlen = MED_D;
        float *fw = malloc((size_t)MED_OUT * rowlen * sizeof(float));
        for (int o = 0; o < MED_OUT; o++) {
            double ss = 0.0;
            const float *vr = vv + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) ss += (double)vr[idx] * vr[idx];
            float scale = (float)((double)g[o] / sqrt(ss));
            float *fr = fw + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) fr[idx] = vr[idx] * scale;
        }
        m->mapping_w = fw;
    }
    if (m->failed) { aria_sa3_dec_medium_free(m); return NULL; }
    /* B3 (opt-in): quantize the medium decoder block GEMMs (ARIA_DEC_Q8). Same fidelity
     * caveat as the small decoder -- final-audio GEMMs, gate before relying on it. */
    const char *dq = getenv("ARIA_DEC_Q8");
    aria_dtype ddt;
    if (dq && aria_dtype_parse(dq[0] == '1' ? "q8" : dq, &ddt) == 0 && ddt != ARIA_F32)
        for (int i = 0; i < MED_DEPTH; i++) taae_block_quantize(&m->blocks[i], MED_D, MED_INNER, ddt);
    return m;
}

void aria_sa3_dec_medium_free(aria_sa3_dec_medium *m) {
    if (!m) return;
    for (int i = 0; i < MED_DEPTH; i++) taae_block_overlay_free(&m->blocks[i]);
    free(m->mapping_w);
    free(m);
}

void aria_sa3_dec_medium_get_view(const aria_sa3_dec_medium *m, aria_sa3_dec_medium_view *v) {
    v->running_std = m->running_std;
    v->proj_w = m->proj_w; v->proj_b = m->proj_b; v->new_tokens = m->new_tokens;
    v->mapping_w = m->mapping_w; v->mapping_b = m->mapping_b;
    for (int i = 0; i < MED_DEPTH; i++) {
        const taae_block_w *s = &m->blocks[i];
        aria_taae_block_view *d = &v->blocks[i];
        d->pre_alpha = s->pre_alpha; d->qn_alpha = s->qn_alpha;
        d->kn_alpha = s->kn_alpha;   d->ff_alpha = s->ff_alpha;
        d->pre_gamma = s->pre_gamma; d->pre_beta = s->pre_beta;
        d->qn_gamma = s->qn_gamma;   d->qn_beta = s->qn_beta;
        d->kn_gamma = s->kn_gamma;   d->kn_beta = s->kn_beta;
        d->ff_gamma = s->ff_gamma;   d->ff_beta = s->ff_beta;
        d->to_qkv = s->to_qkv; d->to_out = s->to_out;
        d->ff_in_w = s->ff_in_w; d->ff_in_b = s->ff_in_b;
        d->ff_out_w = s->ff_out_w; d->ff_out_b = s->ff_out_b;
    }
}

/* rope cos/sin for ABSOLUTE token positions [pos0, pos0+n). Same formula as
 * aria_rope_freqs (base 10000, TAAE_ROT rotated dims); byte-identical to it when
 * pos0 == 0. The offset lets a decode window carry its tokens' true global rotary
 * phase so band attention matches the monolithic decode bit-for-bit (E16.1). */
static void med_rope_freqs_at(float *cos_t, float *sin_t, int n, int pos0) {
    const int rh = TAAE_ROT / 2;
    for (int i = 0; i < rh; i++) {
        double inv = 1.0 / pow(10000.0, (double)(2 * i) / (double)TAAE_ROT);
        for (int p = 0; p < n; p++) {
            double ang = (double)(pos0 + p) * inv;
            cos_t[(size_t)p * rh + i] = (float)cos(ang);
            sin_t[(size_t)p * rh + i] = (float)sin(ang);
        }
    }
}

/* decode latent[256,T] -> audio[2, T*4096]; the seq's first token sits at absolute
 * rope position pos0 (= 0 for a monolithic decode, = A*17 for the window [A,B)). */
static void med_decode_slice(const aria_sa3_dec_medium *m, float *audio,
                             const float *latent, int T, int pos0) {
    const int D = MED_D;
    int N = T * MED_SEG;                  /* full token sequence length */

    /* 1. softnorm + transpose [256,T] -> [T,256], proj -> seq token slots */
    float *ztc = malloc((size_t)T * 256 * sizeof(float));
    if (!ztc) return;
    for (int t = 0; t < T; t++)
        for (int c = 0; c < 256; c++) ztc[(size_t)t * 256 + c] = latent[(size_t)c * T + t] * m->running_std;
    float *proj = malloc((size_t)T * D * sizeof(float));
    /* gcc -O3 emits a spurious -Wmaybe-uninitialized for `ztc` here: it cannot
     * prove the double loop above writes every element. ztc is fully initialized
     * (t in [0,T), c in [0,256)). Left as-is rather than papering over it. */
    aria_linear(proj, ztc, m->proj_w, m->proj_b, T, 256, D);
    free(ztc);

    /* 3. seq[T*17, 1536]: per token [proj_token, 16x new_tokens] */
    float *seq = malloc((size_t)N * D * sizeof(float));
    for (int t = 0; t < T; t++) {
        memcpy(seq + (size_t)(t * MED_SEG) * D, proj + (size_t)t * D, (size_t)D * sizeof(float));
        for (int j = 1; j < MED_SEG; j++)
            memcpy(seq + (size_t)(t * MED_SEG + j) * D, m->new_tokens, (size_t)D * sizeof(float));
    }
    free(proj);

    /* rope (absolute pos0 offset); sliding-window attention (band half-width [17,17]) */
    float *rc = malloc((size_t)N * (TAAE_ROT / 2) * sizeof(float));
    float *rs = malloc((size_t)N * (TAAE_ROT / 2) * sizeof(float));
    med_rope_freqs_at(rc, rs, N, pos0);

    /* 4. 12 blocks over the full sequence; blocks with (depth-i) < 8 are sinusoidal */
    aria_arena ar;
    aria_arena_init(&ar, taae_med_block_floats(N, D, MED_INNER) * sizeof(float));
    for (int b = 0; b < MED_DEPTH; b++) {
        int sinusoidal = (MED_DEPTH - b) < 8;
        taae_med_block_forward(seq, N, D, MED_H, MED_HD, MED_INNER, &m->blocks[b], rc, rs, MED_WIN, sinusoidal, &ar);
    }
    aria_arena_free(&ar);
    free(rc); free(rs);

    /* 5-6. the k1 WNConv1d mapping 1536->512 over the last 16 of each 17-group is a GEMM
     * on those token rows (copied contiguously), then unpatch the [Lt,512] rows -> stereo */
    int Lt = T * 16;
    float *feat = malloc((size_t)Lt * D * sizeof(float));   /* [Lt, 1536] token-major */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int t = 0; t < T; t++)
        memcpy(feat + (size_t)t * 16 * D, seq + (size_t)(t * MED_SEG + 1) * D, (size_t)16 * D * sizeof(float));
    free(seq);
    float *mapped = malloc((size_t)Lt * MED_OUT * sizeof(float));
    aria_linear(mapped, feat, m->mapping_w, m->mapping_b, Lt, D, MED_OUT);
    free(feat);
    aria_unpatch_stereo_tm(audio, mapped, Lt);   /* [2, Lt*256] = [2, T*4096] */
    free(mapped);
}

/* latent[256,T] -> audio[2, T*4096] */
void aria_sa3_dec_medium_forward(const aria_sa3_dec_medium *m, float *audio, const float *latent, int T) {
    med_decode_slice(m, audio, latent, T, 0);
}

/* ---- E16.1 windowed decode (medium) ----------------------------------------
 * Medium mixes frames ONLY through 12 blocks of sliding-window (banded) attention,
 * half-width win=17 TOKENS per block. Everything else (proj, extract, WNConv1d k1,
 * unpatch) is per-frame. After 12 blocks a token has spread +/-12*17 = +/-204 tokens;
 * at 17 tokens/frame that is +/-12 latent frames -- the exact receptive field.
 *   Recomputing each window over its whole slice, block b corrupts tokens within
 *   b*17 of the window edge (their band hits the slice clamp instead of the true
 *   neighbour), so after 12 blocks the outer 204 tokens = 12 frames of each edge are
 *   wrong. Keeping only [start,end) with a 12-frame halo discards them.
 *   RoPE is translation-covariant but not bit-exact under a position shift, so each
 *   window's rope is generated at its ABSOLUTE offset (A*17); band attention over an
 *   interior token then reuses identical q/k/v as the monolithic decode. At the true
 *   edges (A=0 / B=T) the slice clamp equals the monolithic clamp, so no halo needed.
 *   => H = 12 latent frames; frame alignment is arbitrary (band + absolute rope).
 *      Byte-identical output; peak arena O(window) (taae_med_block_floats(N=wf*17)). */
#define MED_WIN_HALO 12   /* latent frames = 12 blocks * 17-token band / 17 tokens/frame */

int aria_sa3_dec_medium_window_for_budget(size_t budget_bytes) {
    /* per decoded frame: the block arena (taae_med_block_floats = 16*N*dim) + seq */
    size_t per_frame = (size_t)18 * MED_SEG * MED_D * sizeof(float);
    long w = (long)(budget_bytes / per_frame) - 2 * MED_WIN_HALO;
    return w > 86 ? (int)(w > 1 << 20 ? 1 << 20 : w) : 86;   /* never under ~8 s */
}

void aria_sa3_dec_medium_forward_windowed(const aria_sa3_dec_medium *m, float *audio,
                                          const float *latent, int T, int window_frames) {
    int W = window_frames > 0 ? window_frames : 86;   /* ~8 s at 44.1 kHz / 4096 */
    const int H = MED_WIN_HALO;
    int maxwf = W + 2 * H;
    /* one slice would cover the clip: decode it whole -- same bytes, same peak */
    if (T <= maxwf) { med_decode_slice(m, audio, latent, T, 0); return; }
    float *lat_slice = malloc((size_t)256 * maxwf * sizeof(float));
    float *aud_slice = malloc((size_t)2 * maxwf * 4096 * sizeof(float));
    for (int start = 0; start < T; start += W) {
        int end = start + W; if (end > T) end = T;
        int A = start - H; if (A < 0) A = 0;      /* any frame aligns (band + absolute rope) */
        int B = end + H;   if (B > T) B = T;
        int wf = B - A;
        for (int c = 0; c < 256; c++)
            memcpy(lat_slice + (size_t)c * wf, latent + (size_t)c * T + A, (size_t)wf * sizeof(float));
        med_decode_slice(m, aud_slice, lat_slice, wf, A * MED_SEG);   /* audio[2, wf*4096] */
        for (int c = 0; c < 2; c++)
            memcpy(audio + (size_t)c * T * 4096 + (size_t)start * 4096,
                   aud_slice + (size_t)c * wf * 4096 + (size_t)(start - A) * 4096,
                   (size_t)(end - start) * 4096 * sizeof(float));
    }
    free(lat_slice); free(aud_slice);
}

/* B5 (E14.1): decode ONLY latent frames [rStart, rEnd) of the medium latent into
 * out_audio (full [2, T*4096] buffer; rest untouched). One haloed window; keep region
 * byte-identical to the same span of the full/windowed decode (H = MED_WIN_HALO). */
void aria_sa3_dec_medium_forward_range(const aria_sa3_dec_medium *m, float *audio,
                                       const float *latent, int T, int rStart, int rEnd) {
    if (rStart < 0) rStart = 0;
    if (rEnd > T) rEnd = T;
    if (rEnd <= rStart) return;
    const int H = MED_WIN_HALO;
    int A = rStart - H; if (A < 0) A = 0;
    int B = rEnd + H;   if (B > T) B = T;
    int wf = B - A;
    float *lat_slice = malloc((size_t)256 * wf * sizeof(float));
    float *aud_slice = malloc((size_t)2 * wf * 4096 * sizeof(float));
    for (int c = 0; c < 256; c++)
        memcpy(lat_slice + (size_t)c * wf, latent + (size_t)c * T + A, (size_t)wf * sizeof(float));
    med_decode_slice(m, aud_slice, lat_slice, wf, A * MED_SEG);
    for (int c = 0; c < 2; c++)
        memcpy(audio + (size_t)c * T * 4096 + (size_t)rStart * 4096,
               aud_slice + (size_t)c * wf * 4096 + (size_t)(rStart - A) * 4096,
               (size_t)(rEnd - rStart) * 4096 * sizeof(float));
    free(lat_slice); free(aud_slice);
}
