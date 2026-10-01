/*
 * aria_hpss.c - median-filtering harmonic/percussive separation. See aria_hpss.h.
 */
#include "aria_hpss.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define HP_N    2048      /* FFT size (~46 ms @ 44.1k) */
#define HP_HOP  512       /* 75% overlap */
#define HP_TMED 17        /* horizontal (time) median width -> harmonic */
#define HP_FMED 17        /* vertical (freq) median width   -> percussive */

/* in-place iterative radix-2 FFT; inv=0 forward, inv=1 inverse (1/n scaled). */
static void hp_fft(float *re, float *im, int n, int inv) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inv ? 1.0 : -1.0);
        float wr = (float)cos(ang), wi = (float)sin(ang);
        for (int i = 0; i < n; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int k = 0; k < len / 2; k++) {
                int a = i + k, b = a + len / 2;
                float vr = re[b] * cr - im[b] * ci;
                float vi = re[b] * ci + im[b] * cr;
                re[b] = re[a] - vr; im[b] = im[a] - vi;
                re[a] += vr;        im[a] += vi;
                float ncr = cr * wr - ci * wi, nci = cr * wi + ci * wr;
                cr = ncr; ci = nci;
            }
        }
    }
    if (inv) for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

/* median of `w` values (small w; insertion-style partial sort on a scratch copy). */
static float hp_median(const float *v, int w, float *scratch) {
    for (int i = 0; i < w; i++) scratch[i] = v[i];
    for (int i = 1; i < w; i++) {           /* insertion sort */
        float x = scratch[i]; int j = i - 1;
        while (j >= 0 && scratch[j] > x) { scratch[j + 1] = scratch[j]; j--; }
        scratch[j + 1] = x;
    }
    return scratch[w / 2];
}

/* separate one channel `in`[nf] (stride `st`) into harmonic/percussive (same stride). */
static void hp_channel(const float *in, int64_t nf, int st, float *harm, float *perc) {
    int N = HP_N, hop = HP_HOP;
    int nfr = (int)((nf + hop - 1) / hop) + 1;        /* frames (zero-padded tail) */
    float *win = malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (N - 1)));

    /* STFT -> re/im[nfr][N], magnitude mag[nfr][N] */
    float *re = calloc((size_t)nfr * N, sizeof(float));
    float *im = calloc((size_t)nfr * N, sizeof(float));
    float *mag = malloc((size_t)nfr * N * sizeof(float));
    float *fr = malloc(sizeof(float) * N), *fi = malloc(sizeof(float) * N);
    for (int f = 0; f < nfr; f++) {
        int64_t base = (int64_t)f * hop - N / 2;       /* centered frames */
        for (int i = 0; i < N; i++) {
            int64_t s = base + i;
            fr[i] = (s >= 0 && s < nf) ? in[s * st] * win[i] : 0.0f;
            fi[i] = 0.0f;
        }
        hp_fft(fr, fi, N, 0);
        float *R = re + (size_t)f * N, *I = im + (size_t)f * N, *M = mag + (size_t)f * N;
        for (int b = 0; b < N; b++) { R[b] = fr[b]; I[b] = fi[b]; M[b] = sqrtf(fr[b] * fr[b] + fi[b] * fi[b]); }
    }

    /* harmonic = time-median (per bin), percussive = freq-median (per frame) */
    float *scratch = malloc(sizeof(float) * (HP_TMED > HP_FMED ? HP_TMED : HP_FMED));
    float *col = malloc(sizeof(float) * (nfr > N ? nfr : N));
    float *H = malloc((size_t)nfr * N * sizeof(float));
    float *P = malloc((size_t)nfr * N * sizeof(float));
    int th = HP_TMED / 2, fh = HP_FMED / 2;
    for (int b = 0; b < N; b++) {                      /* horizontal median over time */
        for (int f = 0; f < nfr; f++) {
            for (int k = 0; k < HP_TMED; k++) {
                int ff = f - th + k; if (ff < 0) ff = 0; if (ff >= nfr) ff = nfr - 1;
                col[k] = mag[(size_t)ff * N + b];
            }
            H[(size_t)f * N + b] = hp_median(col, HP_TMED, scratch);
        }
    }
    for (int f = 0; f < nfr; f++) {                    /* vertical median over freq */
        float *M = mag + (size_t)f * N;
        for (int b = 0; b < N; b++) {
            for (int k = 0; k < HP_FMED; k++) {
                int bb = b - fh + k; if (bb < 0) bb = 0; if (bb >= N) bb = N - 1;
                col[k] = M[bb];
            }
            P[(size_t)f * N + b] = hp_median(col, HP_FMED, scratch);
        }
    }

    /* soft (power-2) masks, applied to the complex STFT, then ISTFT + overlap-add.
     * harm/perc are pre-zeroed by the caller (strided per channel) and accumulated here. */
    float *norm = calloc(nf, sizeof(float));
    for (int f = 0; f < nfr; f++) {
        float *R = re + (size_t)f * N, *I = im + (size_t)f * N;
        float *Hm = H + (size_t)f * N, *Pm = P + (size_t)f * N;
        float hr[HP_N], hi[HP_N], pr[HP_N], pi[HP_N];
        for (int b = 0; b < N; b++) {
            float h2 = Hm[b] * Hm[b], p2 = Pm[b] * Pm[b], d = h2 + p2 + 1e-9f;
            float mh = h2 / d, mp = p2 / d;
            hr[b] = R[b] * mh; hi[b] = I[b] * mh;
            pr[b] = R[b] * mp; pi[b] = I[b] * mp;
        }
        hp_fft(hr, hi, N, 1); hp_fft(pr, pi, N, 1);
        int64_t base = (int64_t)f * hop - N / 2;
        for (int i = 0; i < N; i++) {
            int64_t s = base + i;
            if (s < 0 || s >= nf) continue;
            harm[s * st] += hr[i] * win[i];
            perc[s * st] += pr[i] * win[i];
            norm[s] += win[i] * win[i];
        }
    }
    for (int64_t s = 0; s < nf; s++) {
        float g = norm[s] > 1e-6f ? 1.0f / norm[s] : 0.0f;
        harm[s * st] *= g; perc[s * st] *= g;
    }

    free(win); free(re); free(im); free(mag); free(fr); free(fi);
    free(scratch); free(col); free(H); free(P); free(norm);
}

void aria_hpss_separate(const float *in, int64_t nframes, int ch,
                        float *harmonic, float *percussive) {
    memset(harmonic, 0, (size_t)nframes * ch * sizeof(float));
    memset(percussive, 0, (size_t)nframes * ch * sizeof(float));
    for (int c = 0; c < ch; c++)
        hp_channel(in + c, nframes, ch, harmonic + c, percussive + c);
}
