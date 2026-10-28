/*
 * aria_sampler.c - RNG, LogSNR schedule, pingpong sampler.
 */

#include "aria_sampler.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ---- xoshiro256** ---- */
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

void aria_rng_seed(aria_rng *r, uint64_t seed) {
    r->mode = ARIA_RNG_XOSHIRO;
    /* SplitMix64 to fill the state */
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}

/* ---- PyTorch at::mt19937 (CPU randn engine) ---- */
#define MT_N 624
#define MT_M 397
static void mt_seed(aria_rng *r, uint64_t seed) {
    uint32_t *s = r->mt;
    s[0] = (uint32_t)(seed & 0xffffffffULL);
    for (uint32_t j = 1; j < MT_N; j++)
        s[j] = (uint32_t)(1812433253u * (s[j - 1] ^ (s[j - 1] >> 30)) + j);
    r->mt_left = 1; r->mt_next = 0;
}
static inline uint32_t mt_twist(uint32_t u, uint32_t v) {
    uint32_t mixed = (u & 0x80000000u) | (v & 0x7fffffffu);
    return (mixed >> 1) ^ ((v & 1u) ? 0x9908b0dfu : 0u);
}
static void mt_next_state(aria_rng *r) {
    uint32_t *p = r->mt;
    r->mt_left = MT_N; r->mt_next = 0;
    for (int j = MT_N - MT_M + 1; --j; p++) *p = p[MT_M] ^ mt_twist(p[0], p[1]);
    for (int j = MT_M; --j; p++)            *p = p[MT_M - MT_N] ^ mt_twist(p[0], p[1]);
    *p = p[MT_M - MT_N] ^ mt_twist(p[0], r->mt[0]);
}
static uint32_t mt_next(aria_rng *r) {
    if (--r->mt_left <= 0) mt_next_state(r);
    uint32_t y = r->mt[r->mt_next++];
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}
/* torch uniform_real_distribution<float>(0,1): low 24 bits / 2^24 */
static inline float mt_uniform_f(aria_rng *r) {
    return (float)(mt_next(r) & ((1u << 24) - 1)) * (1.0f / (float)(1u << 24));
}
/* torch normal_fill_16: 8 Box-Muller pairs in place (data holds uniforms).
 * Matches torch's exact precision: radius = sqrtf(-2*logf(u1)) (float); theta is
 * computed in double (`2.0f * c10::pi<double> * u2`) but stored to a float, then
 * cosf/sinf. Residual vs torch is libm's last-ULP rounding of logf/cosf/sinf. */
static void mt_normal_fill_16(float *d) {
    for (int j = 0; j < 8; j++) {
        float u1 = 1.0f - d[j];          /* (0,1] avoids log(0) */
        float u2 = d[j + 8];
        float radius = sqrtf(-2.0f * logf(u1));
        float theta = (float)(2.0 * 3.14159265358979323846 * (double)u2);
        d[j]     = radius * cosf(theta);
        d[j + 8] = radius * sinf(theta);
    }
}
/* torch normal_fill (float, mean 0 std 1), for size >= 16 (generation's n=256*T) */
static void mt_normal_fill(aria_rng *r, float *data, int n) {
    for (int i = 0; i < n; i++) data[i] = mt_uniform_f(r);
    for (int i = 0; i < n - 15; i += 16) mt_normal_fill_16(data + i);
    if (n % 16 != 0) {
        float *tail = data + n - 16;     /* re-draw + overwrite the last block */
        for (int i = 0; i < 16; i++) tail[i] = mt_uniform_f(r);
        mt_normal_fill_16(tail);
    }
}

void aria_rng_init(aria_rng *r, uint64_t seed, aria_rng_mode mode) {
    if (mode == ARIA_RNG_TORCH) { r->mode = ARIA_RNG_TORCH; mt_seed(r, seed); }
    else aria_rng_seed(r, seed);
}

static uint64_t rng_next(aria_rng *r) {
    uint64_t *s = r->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}

double aria_rng_next_double(aria_rng *r) {
    /* 53-bit mantissa in [0,1) */
    return (rng_next(r) >> 11) * (1.0 / 9007199254740992.0);
}

float aria_rng_gaussian(aria_rng *r) {
    double u1 = aria_rng_next_double(r);
    double u2 = aria_rng_next_double(r);
    if (u1 < 1e-300) u1 = 1e-300;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
}

void aria_rng_randn(aria_rng *r, float *out, int n) {
    if (r->mode == ARIA_RNG_TORCH) {
        /* always the MT path so torch mode never reads uninitialized xoshiro state.
         * n>=16 (every generation case, n=256*T) bit-matches torch.randn; n<16 is a
         * deterministic MT block (torch uses a different scalar path there, so it is
         * not bit-exact for n<16 -- never hit in generation). */
        if (n >= 16) { mt_normal_fill(r, out, n); return; }
        float tmp[16];
        for (int i = 0; i < 16; i++) tmp[i] = mt_uniform_f(r);
        mt_normal_fill_16(tmp);
        for (int i = 0; i < n; i++) out[i] = tmp[i];
        return;
    }
    /* draw the 2n uniforms in stream order (as n aria_rng_gaussian calls would), then
     * run the double-precision Box-Muller transforms in parallel: identical values, with
     * the libm-heavy part off the serial path (the GPU waits on this every step) */
    uint64_t *u = malloc((size_t)2 * n * sizeof(uint64_t));
    if (!u) { for (int i = 0; i < n; i++) out[i] = aria_rng_gaussian(r); return; }
    for (size_t i = 0; i < (size_t)2 * n; i++) u[i] = rng_next(r);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < n; i++) {
        double u1 = (u[2 * (size_t)i] >> 11) * (1.0 / 9007199254740992.0);
        double u2 = (u[2 * (size_t)i + 1] >> 11) * (1.0 / 9007199254740992.0);
        if (u1 < 1e-300) u1 = 1e-300;
        out[i] = (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
    }
    free(u);
}

/* ---- schedule ---- */
void aria_logsnr_schedule(float *out, int steps, float sigma_max,
                          float anchor_logsnr, float anchor_length, float rate,
                          float logsnr_end, float seq_len) {
    int N = steps + 1;
    double logsnr_start = (double)anchor_logsnr - (double)rate * log2((double)seq_len / (double)anchor_length);
    for (int i = 0; i < N; i++) {
        double ti = (N > 1) ? (double)sigma_max * (1.0 - (double)i / (double)(N - 1)) : (double)sigma_max;
        double logsnr = (double)logsnr_end - ti * ((double)logsnr_end - logsnr_start);
        double t_out = 1.0 / (1.0 + exp(logsnr));  /* sigmoid(-logsnr) */
        if (ti <= 0.0) t_out = 0.0;
        if (ti >= 1.0) t_out = 1.0;
        out[i] = (float)t_out;
    }
    out[0] = sigma_max;  /* pin first to sigma_max */
}

void aria_dist_shift_defaults(aria_dist_shift *s, int type) {
    memset(s, 0, sizeof(*s));
    s->type = type;
    s->anchor_length = 2000.0f; s->anchor_logsnr = -6.2f; s->rate = 1.0f; s->logsnr_end = 2.0f;
    s->base_shift = 0.5f; s->max_shift = 1.15f;
    s->alpha_min = 1.0f; s->alpha_max = 1.0f;
    s->min_length = 256.0f; s->max_length = 4096.0f;
}

void aria_build_schedule(float *out, int steps, float sigma_max,
                         const aria_dist_shift *s, int seq_len) {
    if (seq_len < 1) seq_len = 1;
    if (s->type == ARIA_SHIFT_LOGSNR) {
        aria_logsnr_schedule(out, steps, sigma_max, s->anchor_logsnr, s->anchor_length,
                             s->rate, s->logsnr_end, (float)seq_len);
        return;
    }
    double L = seq_len < s->min_length ? s->min_length : seq_len > s->max_length ? s->max_length : seq_len;
    double em = 1.0, alpha = 1.0;   /* FULL: e^mu; FLUX: alpha (log-linear in seq_len) */
    if (s->type == ARIA_SHIFT_FULL)
        em = exp(-(s->base_shift + (s->max_shift - s->base_shift) * (L - s->min_length) /
                                   (s->max_length - s->min_length)));
    if (s->type == ARIA_SHIFT_FLUX) {
        double l0 = log(s->min_length), l1 = log(s->max_length);
        if (l1 == l0) l1 += 1e-8;
        double a0 = log(fmax(s->alpha_min, 1e-8)), a1 = log(fmax(s->alpha_max, 1e-8));
        alpha = exp(a0 + (log(L) - l0) / (l1 - l0) * (a1 - a0));
    }
    int N = steps + 1;
    for (int i = 0; i < N; i++) {
        double t = (N > 1) ? (double)sigma_max * (1.0 - (double)i / (double)(N - 1)) : (double)sigma_max;
        double o = t;
        if (s->type == ARIA_SHIFT_FULL) {
            /* 1 - e^mu / (e^mu + t/(1-t)); the t->1 limit is 1, t=0 gives 0 */
            o = t >= 1.0 ? 1.0 : t <= 0.0 ? 0.0 : 1.0 - em / (em + (1.0 / (1.0 - t) - 1.0));
            if (s->use_sine) o = sin(o * 3.14159265358979323846 / 2.0);
        } else if (s->type == ARIA_SHIFT_FLUX) {
            o = alpha * t / (1.0 + (alpha - 1.0) * t);
        }
        out[i] = (float)o;
    }
    out[0] = sigma_max;
}

/* ---- pingpong ---- */
void aria_pingpong_cb(float *x, int n, const float *sigmas, int steps,
                      aria_denoiser_fn denoise, void *ctx,
                      aria_rng *rng, const float *injected_noise,
                      void (*progress)(int step, int total, void *user), void *user,
                      void (*post_step)(int step, float *x, int n, void *puser), void *puser) {
    float *v = malloc((size_t)n * sizeof(float));
    float *denoised = malloc((size_t)n * sizeof(float));
    /* draw step noise as a whole vector (matches torch randn_like; identical
     * sequence to per-element draws for xoshiro). NULL when noise is injected. */
    float *noisebuf = injected_noise ? NULL : malloc((size_t)n * sizeof(float));
    for (int i = 0; i < steps; i++) {
        float tc = sigmas[i], tn = sigmas[i + 1];
        denoise(ctx, x, tc, v, n);
        for (int j = 0; j < n; j++) denoised[j] = x[j] - tc * v[j];
        const float *noise;
        if (injected_noise) noise = injected_noise + (size_t)i * n;
        else { aria_rng_randn(rng, noisebuf, n); noise = noisebuf; }
        for (int j = 0; j < n; j++) x[j] = (1.0f - tn) * denoised[j] + tn * noise[j];
        /* E12.2: latent steer AFTER this step's update, BEFORE the next DiT call
         * (and for i==steps-1, before decode). Bit-exact no-op when unused. */
        if (post_step) post_step(i, x, n, puser);
        if (progress) progress(i + 1, steps, user);
    }
    free(v);
    free(denoised);
    free(noisebuf);
}

void aria_pingpong(float *x, int n, const float *sigmas, int steps,
                   aria_denoiser_fn denoise, void *ctx,
                   aria_rng *rng, const float *injected_noise) {
    aria_pingpong_cb(x, n, sigmas, steps, denoise, ctx, rng, injected_noise, NULL, NULL, NULL, NULL);
}
