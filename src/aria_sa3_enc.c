/*
 * aria_sa3_enc.c - taae_v2 encoder (CPU). See aria_sa3_enc.h for the spec.
 * Mirror of aria_sa3_dec.c; the transformer block + chunk pass are shared
 * (aria_taae.{c,h}). The encode is a one-time op (continue/inpaint), not the
 * denoise hot loop, so it favors clarity over the decoder's arena plumbing.
 */

#include "aria_sa3_enc.h"
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

#define ENC_IN  512   /* patched audio channels (2 * patch_size/256... = 2*256) */
#define ENC_LAT 256   /* latent dim */
#define PATCH   256   /* patch_size */

struct aria_sa3_enc {
    const float *scaling_factor, *bias;  /* softnorm, [256] each (borrowed) */
    float running_std;
    float *mapping_w;                    /* folded WNConv1d [768,512] (owned) */
    const float *mapping_b;              /* [768] */
    const float *new_tokens;             /* [768] */
    taae_block_w blocks[6];
    const float *proj_w, *proj_b;        /* layers.2 Linear [256,768],[256] */
    int failed;
};

static void transpose_ct_tc(float *dst, const float *src, int C, int T) {
    for (int c = 0; c < C; c++) for (int t = 0; t < T; t++) dst[(size_t)t * C + c] = src[(size_t)c * T + t];
}
static void transpose_tc_ct(float *dst, const float *src, int T, int C) {
    for (int t = 0; t < T; t++) for (int c = 0; c < C; c++) dst[(size_t)c * T + t] = src[(size_t)t * C + c];
}

int aria_sa3_patch_len(int L)        { return (L + PATCH - 1) / PATCH; }
int aria_sa3_latent_len(int T_patch) { return ((T_patch + 31) / 32 * 32) / 16; }

void aria_sa3_patchify(float *patches, const float *audio, int L, int T_patch) {
    /* patches[c*256+h, t] = audio[c, t*256+h]; channels-first audio[2,L]; pad tail with 0 */
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < T_patch; t++)
            for (int hh = 0; hh < PATCH; hh++) {
                int s = t * PATCH + hh;
                patches[(size_t)(c * PATCH + hh) * T_patch + t] =
                    (s < L) ? audio[(size_t)c * L + s] : 0.0f;
            }
}

void aria_sa3_softnorm_encode(const aria_sa3_enc *m, float *latent, const float *enc, int T) {
    for (int c = 0; c < ENC_LAT; c++) {
        float sc = m->scaling_factor[c], bs = m->bias[c];
        for (int t = 0; t < T; t++)
            latent[(size_t)c * T + t] = (enc[(size_t)c * T + t] * sc + bs) / m->running_std;
    }
}

void aria_sa3_same_encode(const aria_sa3_enc *m, float *enc, const float *patches, int T_patch, int T_lat) {
    const int D = TAAE_D;
    int Tpp = T_lat * 16;   /* T_patch zero-padded to a multiple of 32 */

    /* 1. zero-pad patches[512,T_patch] -> [512,Tpp] (channel-major) */
    float *pp = calloc((size_t)ENC_IN * Tpp, sizeof(float));
    for (int c = 0; c < ENC_IN; c++)
        memcpy(pp + (size_t)c * Tpp, patches + (size_t)c * T_patch, (size_t)T_patch * sizeof(float));

    /* 2. WNConv1d mapping 512->768, k1 p0 (channel-major) */
    float *mapped = malloc((size_t)D * Tpp * sizeof(float));
    aria_conv1d(mapped, pp, m->mapping_w, m->mapping_b, ENC_IN, D, 1, 0, Tpp);
    free(pp);

    /* 3. token-major [Tpp,768] */
    float *x = malloc((size_t)Tpp * D * sizeof(float));
    transpose_ct_tc(x, mapped, D, Tpp);
    free(mapped);

    /* 4. group every 16 tokens + 1 new_token -> 17; seq[L,768], L = (Tpp/16)*17 */
    int G = Tpp / 16;        /* = T_lat */
    int L = G * 17;
    float *seq = malloc((size_t)L * D * sizeof(float));
    for (int g = 0; g < G; g++) {
        memcpy(seq + (size_t)(g * 17) * D, x + (size_t)(g * 16) * D, (size_t)16 * D * sizeof(float));
        memcpy(seq + (size_t)(g * 17 + 16) * D, m->new_tokens, (size_t)D * sizeof(float));
    }
    free(x);

    /* 5. two chunked (S=34) transformer passes: blocks 0-2 unshifted, 3-5 shifted */
    float rcos[TAAE_S * (TAAE_ROT / 2)], rsin[TAAE_S * (TAAE_ROT / 2)];
    aria_rope_freqs(rcos, rsin, TAAE_S, TAAE_ROT, 10000.0f);
    taae_chunk_pass(seq, L, &m->blocks[0], rcos, rsin, 0);
    taae_chunk_pass(seq, L, &m->blocks[3], rcos, rsin, 1);

    /* 6. take the last token of each 17-group -> feat[G,768] (token-major) */
    float *feat = malloc((size_t)G * D * sizeof(float));
    for (int g = 0; g < G; g++)
        memcpy(feat + (size_t)g * D, seq + (size_t)(g * 17 + 16) * D, (size_t)D * sizeof(float));
    free(seq);

    /* 7. Linear 768->256 (token-major), transpose to channel-major enc[256,G] */
    float *proj = malloc((size_t)G * ENC_LAT * sizeof(float));
    aria_linear(proj, feat, m->proj_w, m->proj_b, G, D, ENC_LAT);
    free(feat);
    transpose_tc_ct(enc, proj, G, ENC_LAT);
    free(proj);
}

int aria_sa3_enc_forward(const aria_sa3_enc *m, float *latent, const float *audio, int L) {
    int T_patch = aria_sa3_patch_len(L);
    int T_lat = aria_sa3_latent_len(T_patch);
    float *patches = malloc((size_t)ENC_IN * T_patch * sizeof(float));
    aria_sa3_patchify(patches, audio, L, T_patch);
    float *enc = malloc((size_t)ENC_LAT * T_lat * sizeof(float));
    aria_sa3_same_encode(m, enc, patches, T_patch, T_lat);
    free(patches);
    aria_sa3_softnorm_encode(m, latent, enc, T_lat);
    free(enc);
    return T_lat;
}

/* ---- loader (zero-copy: borrow F32 weights from the mmap) ---- */
static const float *track(aria_sa3_enc *m, safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) { fprintf(stderr, "aria_sa3_enc_load: missing %s\n", name); m->failed = 1; return NULL; }
    const float *p = safetensors_f32_ptr(sf, t);
    if (!p) { fprintf(stderr, "aria_sa3_enc_load: %s is not F32\n", name); m->failed = 1; return NULL; }
    return p;
}
static float track_scalar(aria_sa3_enc *m, safetensors_file_t *sf, const char *name) {
    const float *p = track(m, sf, name);
    return p ? p[0] : 0.0f;
}

aria_sa3_enc *aria_sa3_enc_load(safetensors_file_t *sf) {
    /* only the 768-d SAME-S encoder is implemented; medium ships the 1536-d SAME-L one,
     * whose tensors share these names -- reading them as 768-d would encode garbage */
    const safetensor_t *nt = safetensors_find(sf, "pretransform.model.encoder.layers.0.new_tokens");
    if (nt && nt->ndim >= 1 && nt->shape[nt->ndim - 1] != TAAE_D) {
        fprintf(stderr, "aria_sa3_enc_load: encoder is %lld-d; only the %d-d SAME-S encoder is implemented\n",
                (long long)nt->shape[nt->ndim - 1], TAAE_D);
        return NULL;
    }
    aria_sa3_enc *m = calloc(1, sizeof(*m));
    if (!m) return NULL;

    m->scaling_factor = track(m, sf, "pretransform.model.bottleneck.scaling_factor");  /* [1,256,1] */
    m->bias = track(m, sf, "pretransform.model.bottleneck.bias");                       /* [1,256,1] */
    m->running_std = track_scalar(m, sf, "pretransform.model.bottleneck.running_std");
    m->new_tokens = track(m, sf, "pretransform.model.encoder.layers.0.new_tokens");     /* [1,1,768] */
    m->proj_w = track(m, sf, "pretransform.model.encoder.layers.2.weight");             /* [256,768] */
    m->proj_b = track(m, sf, "pretransform.model.encoder.layers.2.bias");               /* [256] */

    char nm[256];
    #define P "pretransform.model.encoder.layers.0.transformers.%d."
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

    /* fold the weight-normalized mapping conv (k1): fw[o,i] = v[o,i] * g[o]/||v[o]|| */
    const float *g = track(m, sf, "pretransform.model.encoder.layers.0.mapping.weight_g");   /* [768,1,1] */
    const float *vv = track(m, sf, "pretransform.model.encoder.layers.0.mapping.weight_v");  /* [768,512,1] */
    m->mapping_b = track(m, sf, "pretransform.model.encoder.layers.0.mapping.bias");          /* [768] */
    if (!m->failed) {
        int rowlen = ENC_IN * 1;
        float *fw = malloc((size_t)TAAE_D * rowlen * sizeof(float));
        for (int o = 0; o < TAAE_D; o++) {
            double ss = 0.0;
            const float *vr = vv + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) ss += (double)vr[idx] * vr[idx];
            float scale = (float)((double)g[o] / sqrt(ss));
            float *fr = fw + (size_t)o * rowlen;
            for (int idx = 0; idx < rowlen; idx++) fr[idx] = vr[idx] * scale;
        }
        m->mapping_w = fw;
    }

    if (m->failed) { aria_sa3_enc_free(m); return NULL; }
    return m;
}

void aria_sa3_enc_free(aria_sa3_enc *m) {
    if (!m) return;
    free(m->mapping_w);   /* the folded conv weight is the only owned buffer */
    free(m);
}
