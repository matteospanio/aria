/*
 * aria_sa3.h - Stable Audio 3 specific declarations (config struct, and later
 * the DiT / taae / conditioning forward passes).
 */

#ifndef ARIA_SA3_H
#define ARIA_SA3_H

#include "aria_sampler.h"

typedef struct {
    /* diffusion transformer (model.diffusion.config) */
    int   io_channels;          /* 256  (latent channels in/out) */
    int   embed_dim;            /* 1024 */
    int   depth;                /* 20   transformer blocks */
    int   num_heads;            /* 16 */
    int   head_dim;             /* embed_dim / num_heads = 64 */
    int   cond_token_dim;       /* 768  cross-attn cond input dim */
    int   global_cond_dim;      /* 768  adaLN global cond input dim */
    int   local_add_cond_dim;   /* 257  inpaint mask + masked input */
    int   num_memory_tokens;    /* 64 */
    float ff_mult;              /* 4.0 */
    int   qk_norm_rms;          /* 1 if attn qk_norm == "rms" */
    int   differential_attn;    /* 0 for small, 1 for medium */

    /* model.diffusion.sampling_distribution_shift_options: the inference schedule shift
     * (absent -> the stable_audio_3 wrapper default LogSNRShift(rate=0, -6.2, 2.0)), and
     * model.diffusion.distribution_shift_options: the training shift (`--schedule full`) */
    aria_dist_shift sample_shift, train_shift;

    /* autoencoder essentials (model.pretransform.config) */
    int   latent_dim;           /* 256 */
    int   downsampling_ratio;   /* 4096 */
    int   patch_size;           /* 256 */
} aria_sa3_config;

/* Parse model_config.json into cfg. Returns 0 on success, -1 on a missing
 * required field. */
int aria_sa3_parse_config(const char *json, aria_sa3_config *cfg);

#endif /* ARIA_SA3_H */
