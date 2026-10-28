/*
 * aria_sa3_dec.c - taae_v2 decoder (CPU). See aria_sa3_dec.h for the spec.
 */

#include "aria_sa3_dec.h"
#include "aria_ops.h"
#include "aria_arena.h"
#include "aria_taae.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DEC_D   TAAE_D    /* transformer model dim (768) */
#define DEC_ROT TAAE_ROT  /* rotated dims (32) */
#define DEC_S   TAAE_S    /* effective chunk = 2 * 17 (34) */
#define DEC_OUT 512        /* decoder out channels */

struct aria_sa3_dec {
    float running_std;
    const float *proj_w, *proj_b;        /* layers.1 [768,256],[768] */
    const float *new_tokens;             /* [768] */
    taae_block_w blocks[6];
    float *mapping_w;                    /* folded [512,768,3] (owned) */
    const float *mapping_b;              /* [512] */
    int failed;
};

/* ---- channel<->token transpose (decoder-local) ---- */
static void transpose_ct_tc(float *dst, const float *src, int C, int T) {
    for (int c = 0; c < C; c++) for (int t = 0; t < T; t++) dst[(size_t)t * C + c] = src[(size_t)c * T + t];
}

void aria_sa3_softnorm_decode(const aria_sa3_dec *m, float *z, const float *latent, int T) {
    for (size_t i = 0; i < (size_t)256 * T; i++) z[i] = latent[i] * m->running_std;
}

void aria_unpatch_stereo_tm(float *audio, const float *dec_tm, int L) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int l = 0; l < L; l++)   /* row l = [ch0 patch (256) | ch1 patch (256)] */
        for (int c = 0; c < 2; c++)
            memcpy(audio + (size_t)c * L * 256 + (size_t)l * 256, dec_tm + (size_t)l * 512 + c * 256,
                   256 * sizeof(float));
}

void aria_unpatch_stereo(float *audio, const float *dec, int L) {
    for (int c = 0; c < 2; c++)
        for (int l = 0; l < L; l++)
            for (int hh = 0; hh < 256; hh++)
                audio[(size_t)c * L * 256 + (size_t)l * 256 + hh] = dec[(size_t)(c * 256 + hh) * L + l];
}

void aria_sa3_same_decode(const aria_sa3_dec *m, float *dec, const float *z, int T) {
    const int D = DEC_D;
    /* transpose [256,T] -> [T,256], proj -> [T,768] */
    float *ztc = calloc((size_t)T * 256, sizeof(float));
    transpose_ct_tc(ztc, z, 256, T);
    float *x = malloc((size_t)T * D * sizeof(float));
    aria_linear(x, ztc, m->proj_w, m->proj_b, T, 256, D);
    free(ztc);

    /* zero-pad T to even Tp, expand with new_tokens -> seq[L,768], L = Tp*17 */
    int Tp = (T % 2 == 0) ? T : T + 1;
    int L = Tp * 17;
    float *seq = malloc((size_t)L * D * sizeof(float));
    for (int t = 0; t < Tp; t++) {
        const float *xt = (t < T) ? x + (size_t)t * D : NULL;  /* padded token = zeros */
        if (xt) memcpy(seq + (size_t)(t * 17) * D, xt, (size_t)D * sizeof(float));
        else    memset(seq + (size_t)(t * 17) * D, 0, (size_t)D * sizeof(float));
        for (int j = 1; j <= 16; j++)
            memcpy(seq + (size_t)(t * 17 + j) * D, m->new_tokens, (size_t)D * sizeof(float));
    }
    free(x);

    /* rope tables for S=34 */
    float rcos[DEC_S * (DEC_ROT / 2)], rsin[DEC_S * (DEC_ROT / 2)];
    aria_rope_freqs(rcos, rsin, DEC_S, DEC_ROT, 10000.0f);

    taae_chunk_pass(seq, L, &m->blocks[0], rcos, rsin, 0);  /* blocks 0,1,2 */
    taae_chunk_pass(seq, L, &m->blocks[3], rcos, rsin, 1);  /* blocks 3,4,5 midpoint-shift */

    /* WNConv1d mapping 768 -> 512 (k3 p1) over the last 16 of each 17-group, as one GEMM:
     * im2col row t = [x(t-1)[i], x(t)[i], x(t+1)[i]] interleaved per channel i (zero past
     * the edges), which is exactly the folded weight's [512][768][3] layout. */
    int Lout = Tp * 16;
    float *xcat = malloc((size_t)Lout * 3 * D * sizeof(float));
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int t = 0; t < Lout; t++) {
        float *xr = xcat + (size_t)t * 3 * D;
        for (int k = 0; k < 3; k++) {
            int tt = t + k - 1;
            const float *src = (tt >= 0 && tt < Lout) ? seq + (size_t)((tt / 16) * 17 + 1 + tt % 16) * D : NULL;
            for (int i = 0; i < D; i++) xr[i * 3 + k] = src ? src[i] : 0.0f;
        }
    }
    free(seq);
    float *mapped = malloc((size_t)Lout * DEC_OUT * sizeof(float));   /* [Lout, 512] token-major */
    aria_linear(mapped, xcat, m->mapping_w, m->mapping_b, Lout, 3 * D, DEC_OUT);
    free(xcat);

    /* trim to T*16 and transpose to dec[512, T*16] */
    int Lt = T * 16;
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int c = 0; c < DEC_OUT; c++)
        for (int l = 0; l < Lt; l++) dec[(size_t)c * Lt + l] = mapped[(size_t)l * DEC_OUT + c];
    free(mapped);
}

void aria_sa3_dec_forward(const aria_sa3_dec *m, float *out_audio, const float *latent, int T) {
    float *z = malloc((size_t)256 * T * sizeof(float));
    aria_sa3_softnorm_decode(m, z, latent, T);
    int Lt = T * 16;
    float *dec = malloc((size_t)DEC_OUT * Lt * sizeof(float));
    aria_sa3_same_decode(m, dec, z, T);
    free(z);
    aria_unpatch_stereo(out_audio, dec, Lt);  /* [2, Lt*256] = [2, T*4096] */
    free(dec);
}

/* ---- E16.1 windowed decode (small-music) ------------------------------------
 * The decoder's receptive field in LATENT-FRAME space is small, so we can decode
 * frames [start, end) exactly by feeding the slice [A, B) = [start-H, end+H) and
 * keeping only [start, end). Derivation (from aria_sa3_same_decode + taae_chunk_pass):
 *
 *   Each latent frame f expands to a 17-token group (1 proj token + 16 new_tokens),
 *   so seq position p belongs to frame p/17. The two chunk passes are the ONLY
 *   cross-frame mixing (proj, extract, WNConv1d k3, unpatch are per-frame or +/-1):
 *     - pass 1 (shift=0): disjoint S=34 chunks = frame pairs {0,1},{2,3},...  No
 *       leakage across a pair.
 *     - pass 2 (shift=1): pad 17 front/back then S=34 chunks = frame pairs
 *       {1,2},{3,4},...  i.e. the midpoint shift straddles the pass-1 pairs.
 *   Composing the two: an output frame f's pass-2 chunk is {f-1,f} (f even) or
 *   {f,f+1} (f odd); each of those frames' pass-1 chunk pulls in one more neighbour,
 *   giving a receptive field of [f-2,f+1] (even) / [f-1,f+2] (odd) -- at most +/-2
 *   frames. Pass 1 has no padding (exact for every in-slice frame once A is even so
 *   the local pairing matches the global one); pass 2's edge padding corrupts only
 *   the slice's first (A) and last (B-1) frame, both discarded in the halo. The k3
 *   mapping conv adds only a +/-1 feat-position (<< 1 frame) dependency.
 *   => H = 2 latent frames, with A snapped DOWN and B snapped UP to the 2-frame
 *      (S=34) chunk grid. At the true edges (A=0 / B=T) the slice padding matches
 *      the monolithic padding, so no halo is needed there. Result is byte-identical.
 * Peak arena is O(window): seq/feat/mapped/audio-slice all size to B-A, not T. */
#define DEC_WIN_HALO 2   /* latent frames; see derivation above */

int aria_sa3_dec_window_for_budget(size_t budget_bytes) {
    /* per decoded frame: seq + shift-pass copy (17 tokens) + mapping im2col + mapped + audio;
     * the chunk-pass arena is fixed-size (one group of chunks), not per-frame */
    size_t per_frame = (size_t)(2 * 17 * DEC_D + 16 * 3 * DEC_D + 16 * DEC_OUT + 2 * 4096) * sizeof(float);
    long w = (long)(budget_bytes / per_frame) - 2 * DEC_WIN_HALO - 2;
    return w > 86 ? (int)(w > 1 << 20 ? 1 << 20 : w) : 86;   /* never under ~8 s */
}

void aria_sa3_dec_forward_windowed(const aria_sa3_dec *m, float *out_audio,
                                   const float *latent, int T, int window_frames) {
    int W = window_frames > 0 ? window_frames : 86;   /* ~8 s at 44.1 kHz / 4096 */
    const int H = DEC_WIN_HALO;
    int maxwf = W + 2 * H + 2;                          /* upper bound on slice frames */
    /* one slice would cover the clip: decode it whole -- same bytes, same peak */
    if (T <= maxwf) { aria_sa3_dec_forward(m, out_audio, latent, T); return; }
    float *lat_slice = malloc((size_t)256 * maxwf * sizeof(float));
    float *aud_slice = malloc((size_t)2 * maxwf * 4096 * sizeof(float));
    for (int start = 0; start < T; start += W) {
        int end = start + W; if (end > T) end = T;
        int A = start - H; if (A < 0) A = 0; A -= (A & 1);        /* snap down to even */
        int B = end + H;   if (B > T) B = T; else if (B & 1) B++; /* interior: snap up to even */
        int wf = B - A;
        for (int c = 0; c < 256; c++)
            memcpy(lat_slice + (size_t)c * wf, latent + (size_t)c * T + A, (size_t)wf * sizeof(float));
        aria_sa3_dec_forward(m, aud_slice, lat_slice, wf);       /* audio[2, wf*4096] */
        for (int c = 0; c < 2; c++)
            memcpy(out_audio + (size_t)c * T * 4096 + (size_t)start * 4096,
                   aud_slice + (size_t)c * wf * 4096 + (size_t)(start - A) * 4096,
                   (size_t)(end - start) * 4096 * sizeof(float));
    }
    free(lat_slice); free(aud_slice);
}

/* B5 (E14.1): decode ONLY latent frames [rStart, rEnd) into out_audio (the full
 * [2, T*4096] buffer; the rest is left untouched by this call). One haloed window --
 * the keep region is byte-identical to the same span of a full/windowed decode, the
 * halo derivation is identical to aria_sa3_dec_forward_windowed above. Lets --stream
 * decode ~one emit chunk instead of the whole rolling window. */
void aria_sa3_dec_forward_range(const aria_sa3_dec *m, float *out_audio,
                                const float *latent, int T, int rStart, int rEnd) {
    if (rStart < 0) rStart = 0;
    if (rEnd > T) rEnd = T;
    if (rEnd <= rStart) return;
    const int H = DEC_WIN_HALO;
    int A = rStart - H; if (A < 0) A = 0; A -= (A & 1);
    int B = rEnd + H;   if (B > T) B = T; else if (B & 1) B++;
    int wf = B - A;
    float *lat_slice = malloc((size_t)256 * wf * sizeof(float));
    float *aud_slice = malloc((size_t)2 * wf * 4096 * sizeof(float));
    for (int c = 0; c < 256; c++)
        memcpy(lat_slice + (size_t)c * wf, latent + (size_t)c * T + A, (size_t)wf * sizeof(float));
    aria_sa3_dec_forward(m, aud_slice, lat_slice, wf);
    for (int c = 0; c < 2; c++)
        memcpy(out_audio + (size_t)c * T * 4096 + (size_t)rStart * 4096,
               aud_slice + (size_t)c * wf * 4096 + (size_t)(rStart - A) * 4096,
               (size_t)(rEnd - rStart) * 4096 * sizeof(float));
    free(lat_slice); free(aud_slice);
}

void aria_sa3_dec_block_test(const aria_sa3_dec *m, int idx, float *xc, int N) {
    float *rcos = malloc((size_t)N * (DEC_ROT / 2) * sizeof(float));
    float *rsin = malloc((size_t)N * (DEC_ROT / 2) * sizeof(float));
    aria_rope_freqs(rcos, rsin, N, DEC_ROT, 10000.0f);
    aria_arena ar;
    aria_arena_init(&ar, taae_block_arena_bytes(N));
    taae_block_forward(xc, 1, N, &m->blocks[idx], rcos, rsin, &ar);
    aria_arena_free(&ar);
    free(rcos); free(rsin);
}

/* ---- loader ---- */
/* zero-copy: borrow the F32 weight directly from the mmap (valid while sf open) */
static const float *track(aria_sa3_dec *m, safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) { fprintf(stderr, "aria_sa3_dec_load: missing %s\n", name); m->failed = 1; return NULL; }
    const float *p = safetensors_f32_ptr(sf, t);
    if (!p) { fprintf(stderr, "aria_sa3_dec_load: %s is not F32\n", name); m->failed = 1; return NULL; }
    return p;
}
static float track_scalar(aria_sa3_dec *m, safetensors_file_t *sf, const char *name) {
    const float *p = track(m, sf, name);
    return p ? p[0] : 0.0f;
}

aria_sa3_dec *aria_sa3_dec_load(safetensors_file_t *sf) {
    aria_sa3_dec *m = calloc(1, sizeof(*m));
    if (!m) return NULL;

    m->running_std = track_scalar(m, sf, "pretransform.model.bottleneck.running_std");
    m->proj_w = track(m, sf, "pretransform.model.decoder.layers.1.weight");
    m->proj_b = track(m, sf, "pretransform.model.decoder.layers.1.bias");
    m->new_tokens = track(m, sf, "pretransform.model.decoder.layers.3.new_tokens");

    char nm[256];
    #define P "pretransform.model.decoder.layers.3.transformers.%d."
    for (int i = 0; i < 6; i++) {
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

    /* fold the weight-normalized mapping conv */
    const float *g = track(m, sf, "pretransform.model.decoder.layers.3.mapping.weight_g");   /* [512,1,1] */
    const float *vv = track(m, sf, "pretransform.model.decoder.layers.3.mapping.weight_v");  /* [512,768,3] */
    m->mapping_b = track(m, sf, "pretransform.model.decoder.layers.3.mapping.bias");
    if (!m->failed) {
        int rowlen = 768 * 3;
        float *fw = malloc((size_t)512 * rowlen * sizeof(float));
        for (int o = 0; o < 512; o++) {
            double ss = 0.0;
            const float *vr = vv + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) ss += (double)vr[idx] * vr[idx];
            float scale = (float)((double)g[o] / sqrt(ss));
            float *fr = fw + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) fr[idx] = vr[idx] * scale;
        }
        m->mapping_w = fw;  /* owned via dec_free's mapping_w branch */
    }

    if (m->failed) { aria_sa3_dec_free(m); return NULL; }
    /* B3 (opt-in): quantize the decoder block GEMMs (ARIA_DEC_Q8=q8|fp16|bf16). q8 uses
     * the CPU sdot/AVX2 int8 path; source is released. Fidelity-gate before relying on it
     * -- the decoder is the final audio reconstruction, no error-correction downstream. */
    const char *dq = getenv("ARIA_DEC_Q8");
    aria_dtype ddt;
    if (dq && aria_dtype_parse(dq[0] == '1' ? "q8" : dq, &ddt) == 0 && ddt != ARIA_F32)
        for (int i = 0; i < 6; i++) taae_block_quantize(&m->blocks[i], TAAE_D, TAAE_INNER, ddt);
    return m;
}

void aria_sa3_dec_free(aria_sa3_dec *m) {
    if (!m) return;
    for (int i = 0; i < 6; i++) taae_block_overlay_free(&m->blocks[i]);
    free(m->mapping_w);   /* the folded conv weight is the only owned buffer */
    free(m);
}

void aria_sa3_dec_get_view(const aria_sa3_dec *m, aria_sa3_dec_view *v) {
    v->running_std = m->running_std;
    v->proj_w = m->proj_w; v->proj_b = m->proj_b; v->new_tokens = m->new_tokens;
    v->mapping_w = m->mapping_w; v->mapping_b = m->mapping_b;
    for (int i = 0; i < 6; i++) {
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
