/* test_enc_medium.c - parity for the medium model's SAME-L encoder vs the official
 * stable_audio_3 (CPU fp32 reference: scripts/dump_phase1.py <medium> <dumps> encoder_medium).
 * Stages: the SAME-L encoder (patches -> enc), full audio -> latent on a clean, a padded
 * and an odd length (17 frames), and windowed == monolithic (byte-identical). Needs
 * ARIA_MODEL (the medium model) and ARIA_DUMPS (with enc_medium/); SKIPs otherwise. */
#include "../src/aria_safetensors.h"
#include "../src/aria_sa3_enc.h"
#include "../src/aria_parity.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static int load(const char *name, aria_parity_tensor *t) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/enc_medium/%s", DUMPS, name);
    if (aria_parity_load(path, t) != 0) { printf("FAIL: load %s\n", path); fails++; return -1; }
    return 0;
}
static void compare(const char *what, const float *got, const float *ref, int64_t n, float atol, float rtol) {
    float md = aria_parity_maxabsdiff(got, ref, n), maxref = 0.0f;
    double se = 0, sr = 0;
    for (int64_t i = 0; i < n; i++) {
        float a = fabsf(ref[i]); if (a > maxref) maxref = a;
        se += (double)(got[i] - ref[i]) * (got[i] - ref[i]); sr += (double)ref[i] * ref[i];
    }
    float thr = atol + rtol * maxref;
    printf("%s %s: maxdiff=%.3e thr=%.3e rel_rmse=%.2e (max|ref|=%.3f)\n", md > thr ? "FAIL" : "ok  ",
           what, md, thr, sqrt(se / (sr + 1e-30)), maxref);
    if (md > thr) fails++;
}
static void full(const aria_sa3_enc *m, const char *what, const aria_parity_tensor *audio, const aria_parity_tensor *ref) {
    int L = (int)audio->shape[1], T = (int)ref->shape[1];
    float *lat = malloc((size_t)256 * T * sizeof(float));
    int t = aria_sa3_enc_forward(m, lat, audio->data, L);
    if (t != T) { printf("FAIL %s: %d latent frames, expected %d\n", what, t, T); fails++; }
    else compare(what, lat, ref->data, ref->numel, 1e-3f, 5e-3f);
    free(lat);
}

int main(void) {
    const char *model = getenv("ARIA_MODEL");
    DUMPS = getenv("ARIA_DUMPS");
    if (!model || !DUMPS) { printf("test_enc_medium: SKIP (set ARIA_MODEL=<medium> and ARIA_DUMPS)\n"); return 0; }
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) { printf("FAIL: open safetensors\n"); return 1; }
    aria_sa3_enc *m = aria_sa3_enc_load(sf);
    if (!m) { printf("FAIL: encoder load\n"); return 1; }

    aria_parity_tensor patches, encref, audio, latref, audpad, latpad, audodd, latodd;
    if (load("patches.atns", &patches) || load("enc.atns", &encref) || load("audio.atns", &audio) ||
        load("latent.atns", &latref) || load("audio_pad.atns", &audpad) || load("latent_pad.atns", &latpad) ||
        load("audio_odd.atns", &audodd) || load("latent_odd.atns", &latodd)) return 1;

    int T_patch = (int)patches.shape[1], T = (int)encref.shape[1];
    if (aria_sa3_enc_latent_len(m, T_patch) != T) { printf("FAIL latent length\n"); fails++; }
    float *enc = malloc((size_t)256 * T * sizeof(float));
    aria_sa3_same_encode(m, enc, patches.data, T_patch, T);
    compare("SAME-L encoder", enc, encref.data, encref.numel, 1e-3f, 5e-3f);
    free(enc);
    full(m, "full (16 frames)", &audio, &latref);
    full(m, "full padded (T_patch 250)", &audpad, &latpad);
    full(m, "full odd (17 frames)", &audodd, &latodd);

    /* windowed == monolithic: 40 frames in windows of 8 (+12-frame halos) */
    {
        int Tw = 40, Tp = Tw * 16;
        float *p = malloc((size_t)512 * Tp * sizeof(float));
        unsigned s = 1;
        for (int i = 0; i < 512 * Tp; i++) { s = s * 1664525u + 1013904223u; p[i] = ((float)(s >> 8) / 16777216.0f - 0.5f) * 0.2f; }
        float *a = malloc((size_t)256 * Tw * sizeof(float)), *b = malloc((size_t)256 * Tw * sizeof(float));
        aria_sa3_same_encode_win(m, a, p, Tp, Tw, 1 << 20);   /* one window = monolithic */
        aria_sa3_same_encode_win(m, b, p, Tp, Tw, 8);
        int same = !memcmp(a, b, (size_t)256 * Tw * sizeof(float));
        printf("%s windowed (8 + halo) == monolithic, 40 frames\n", same ? "ok  " : "FAIL");
        if (!same) fails++;
        free(p); free(a); free(b);
    }
    aria_sa3_enc_free(m);
    safetensors_close(sf);
    if (fails) { printf("test_enc_medium: FAILED\n"); return 1; }
    printf("test_enc_medium: OK\n");
    return 0;
}
