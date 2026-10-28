/*
 * aria_sampler.h - RNG, LogSNR schedule, and the pingpong rectified-flow sampler.
 *
 * Reference: stable_audio_tools inference/sampling.py (build_schedule, LogSNRShift,
 * sample_flow_pingpong). Pingpong update per step:
 *   denoised = x - t_curr * v ;  x = (1 - t_next) * denoised + t_next * noise
 * where v = denoiser(x, t_curr). Noise is drawn every step (zeroed when t_next=0).
 */

#ifndef ARIA_SAMPLER_H
#define ARIA_SAMPLER_H

#include <stdint.h>

/* PRNG with two engines selectable per instance:
 *  - ARIA_RNG_XOSHIRO: xoshiro256** (seeded via SplitMix64), N(0,1) via Box-Muller.
 *  - ARIA_RNG_TORCH:   PyTorch's at::mt19937 + the float `normal_fill` (block-of-16
 *    Box-Muller), so aria_rng_randn bit-matches `torch.manual_seed(s); torch.randn(n)`
 *    on CPU for n >= 16 (the only case generation hits; n = 256*T). */
typedef enum { ARIA_RNG_XOSHIRO = 0, ARIA_RNG_TORCH = 1 } aria_rng_mode;
typedef struct {
    int mode;
    uint64_t s[4];        /* xoshiro256** state */
    uint32_t mt[624];     /* at::mt19937 state */
    int mt_left, mt_next;
} aria_rng;
void  aria_rng_seed(aria_rng *r, uint64_t seed);                       /* xoshiro (back-compat) */
void  aria_rng_init(aria_rng *r, uint64_t seed, aria_rng_mode mode);   /* select the engine */
double aria_rng_next_double(aria_rng *r);   /* xoshiro uniform [0,1) */
float aria_rng_gaussian(aria_rng *r);       /* xoshiro N(0,1) */
void  aria_rng_randn(aria_rng *r, float *out, int n);  /* engine-aware N(0,1) fill */

/* build_schedule + LogSNRShift. out holds steps+1 values, descending from
 * sigma_max to 0 (first pinned to sigma_max). logsnr_start = anchor_logsnr -
 * rate*log2(seq_len/anchor_length); logsnr = logsnr_end - t*(logsnr_end -
 * logsnr_start); t_out = sigmoid(-logsnr). */
void aria_logsnr_schedule(float *out, int steps, float sigma_max,
                          float anchor_logsnr, float anchor_length, float rate,
                          float logsnr_end, float seq_len);

/* Timestep shifts of stable-audio-3 inference/distribution_shift.py, as named by a
 * model config's {sampling_,}distribution_shift_options "type". */
typedef enum { ARIA_SHIFT_NONE = 0, ARIA_SHIFT_LOGSNR, ARIA_SHIFT_FULL, ARIA_SHIFT_FLUX } aria_shift_type;
typedef struct {
    int   type;                                           /* aria_shift_type */
    float anchor_length, anchor_logsnr, rate, logsnr_end; /* LOGSNR: LogSNRShift */
    float base_shift, max_shift; int use_sine;            /* FULL:   DistributionShift */
    float alpha_min, alpha_max;                           /* FLUX:   FluxDistributionShift */
    float min_length, max_length;                         /* FULL + FLUX */
} aria_dist_shift;

/* The reference class defaults for `type` (note LogSNRShift's own rate default is 1). */
void aria_dist_shift_defaults(aria_dist_shift *s, int type);

/* build_schedule: linspace(sigma_max, 0, steps+1) warped by the shift at seq_len
 * (clamped >= 1), first value pinned to sigma_max. LOGSNR == aria_logsnr_schedule. */
void aria_build_schedule(float *out, int steps, float sigma_max,
                         const aria_dist_shift *s, int seq_len);

/* Denoiser callback: write velocity v_out[n] for input x[n] at timestep t. */
typedef void (*aria_denoiser_fn)(void *ctx, const float *x, float t, float *v_out, int n);

/* Pingpong loop over `steps` (sigmas has steps+1 entries). x[n] is updated in
 * place. injected_noise[steps*n] overrides the RNG when non-NULL (for parity). */
void aria_pingpong(float *x, int n, const float *sigmas, int steps,
                   aria_denoiser_fn denoise, void *ctx,
                   aria_rng *rng, const float *injected_noise);

/* Same, plus optional post-step hooks (both may be NULL):
 *  - progress(step+1, total): invoked after each completed step, for UI.
 *  - post_step(step, x, n): invoked after each step's pingpong update, with the
 *    updated latent x[n] and the 0-based step index -- the host-side hook the E12.2
 *    latent-steer arm uses to nudge x between denoise steps (works for CPU and CUDA
 *    since x lives on the host in this loop). Fires for the last step too, so the
 *    nudge survives to the decoder. */
void aria_pingpong_cb(float *x, int n, const float *sigmas, int steps,
                      aria_denoiser_fn denoise, void *ctx,
                      aria_rng *rng, const float *injected_noise,
                      void (*progress)(int step, int total, void *user), void *user,
                      void (*post_step)(int step, float *x, int n, void *puser), void *puser);

#endif /* ARIA_SAMPLER_H */
