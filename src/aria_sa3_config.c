/*
 * aria_sa3_config.c - parse Stable Audio 3 model_config.json.
 *
 * Nesting (small-music):
 *   model.diffusion.config.{io_channels, embed_dim, depth, num_heads, ...}
 *   model.diffusion.config.attn_kwargs.{qk_norm, differential}
 *   model.diffusion.config.ff_kwargs.mult
 *   model.pretransform.config.{latent_dim, downsampling_ratio}
 *   model.pretransform.config.pretransform.config.patch_size
 * Several keys (io_channels, config) repeat across sections, so navigation is
 * scoped to the right object span.
 */

#include "aria_sa3.h"
#include "aria_json.h"
#include <stddef.h>
#include <string.h>

/* one *distribution_shift_options object -> s (stable_audio_3 _create_dist_shift:
 * "type" defaults to "full"; omitted fields take the class defaults). -1 = unknown type. */
static int parse_shift(const char *o0, const char *o1, aria_dist_shift *s) {
    static const char *types[] = {"none", "logsnr", "full", "flux"};
    static const struct { const char *key; size_t off; } F[] = {
        {"anchor_length", offsetof(aria_dist_shift, anchor_length)},
        {"anchor_logsnr", offsetof(aria_dist_shift, anchor_logsnr)},
        {"rate",          offsetof(aria_dist_shift, rate)},
        {"logsnr_end",    offsetof(aria_dist_shift, logsnr_end)},
        {"base_shift",    offsetof(aria_dist_shift, base_shift)},
        {"max_shift",     offsetof(aria_dist_shift, max_shift)},
        {"alpha_min",     offsetof(aria_dist_shift, alpha_min)},
        {"alpha_max",     offsetof(aria_dist_shift, alpha_max)},
        {"min_length",    offsetof(aria_dist_shift, min_length)},
        {"max_length",    offsetof(aria_dist_shift, max_length)},
    };
    char ty[16] = "full";
    aria_json_get_string_in(o0, o1, "type", ty, sizeof(ty));
    int t = -1;
    for (int i = 0; i < 4; i++) if (strcmp(ty, types[i]) == 0) t = i;  /* == aria_shift_type */
    if (t < 0) return -1;
    aria_dist_shift_defaults(s, t);
    double v; int b;
    for (size_t i = 0; i < sizeof(F) / sizeof(*F); i++)
        if (aria_json_get_number_in(o0, o1, F[i].key, &v) == 0) *(float *)((char *)s + F[i].off) = (float)v;
    if (aria_json_get_bool_in(o0, o1, "use_sine", &b) == 0) s->use_sine = b;
    return 0;
}

int aria_sa3_parse_config(const char *json, aria_sa3_config *cfg) {
    memset(cfg, 0, sizeof(*cfg));

    const char *m0, *m1;        /* "model" object */
    if (aria_json_object(json, NULL, "model", &m0, &m1) != 0) return -1;

    /* ---- diffusion.config ---- */
    const char *d0, *d1;
    if (aria_json_object(m0, m1, "diffusion", &d0, &d1) != 0) return -1;
    const char *dc0, *dc1;
    if (aria_json_object(d0, d1, "config", &dc0, &dc1) != 0) return -1;

    double v;
    if (aria_json_get_number_in(dc0, dc1, "io_channels", &v) != 0) return -1;
    cfg->io_channels = (int)v;
    if (aria_json_get_number_in(dc0, dc1, "embed_dim", &v) != 0) return -1;
    cfg->embed_dim = (int)v;
    if (aria_json_get_number_in(dc0, dc1, "depth", &v) != 0) return -1;
    cfg->depth = (int)v;
    if (aria_json_get_number_in(dc0, dc1, "num_heads", &v) != 0) return -1;
    cfg->num_heads = (int)v;
    cfg->head_dim = cfg->num_heads ? cfg->embed_dim / cfg->num_heads : 0;

    if (aria_json_get_number_in(dc0, dc1, "cond_token_dim", &v) == 0) cfg->cond_token_dim = (int)v;
    if (aria_json_get_number_in(dc0, dc1, "global_cond_dim", &v) == 0) cfg->global_cond_dim = (int)v;
    if (aria_json_get_number_in(dc0, dc1, "local_add_cond_dim", &v) == 0) cfg->local_add_cond_dim = (int)v;
    if (aria_json_get_number_in(dc0, dc1, "num_memory_tokens", &v) == 0) cfg->num_memory_tokens = (int)v;

    /* ff_kwargs.mult */
    const char *ff0, *ff1;
    if (aria_json_object(dc0, dc1, "ff_kwargs", &ff0, &ff1) == 0 &&
        aria_json_get_number_in(ff0, ff1, "mult", &v) == 0) {
        cfg->ff_mult = (float)v;
    } else {
        cfg->ff_mult = 4.0f;
    }

    /* attn_kwargs.{qk_norm, differential} */
    const char *ak0, *ak1;
    if (aria_json_object(dc0, dc1, "attn_kwargs", &ak0, &ak1) == 0) {
        char buf[16];
        if (aria_json_get_string_in(ak0, ak1, "qk_norm", buf, sizeof(buf)) == 0)
            cfg->qk_norm_rms = (strcmp(buf, "rms") == 0);
        int db;
        if (aria_json_get_bool_in(ak0, ak1, "differential", &db) == 0)  /* JSON true/false */
            cfg->differential_attn = db;
    }

    /* schedule shifts (diffusion level, next to "config") */
    const char *s0, *s1;
    aria_dist_shift_defaults(&cfg->sample_shift, ARIA_SHIFT_LOGSNR);
    cfg->sample_shift.rate = 0.0f;   /* seq-len-invariant, matches legacy log_snr_sampling */
    if (aria_json_object(d0, d1, "sampling_distribution_shift_options", &s0, &s1) == 0 &&
        parse_shift(s0, s1, &cfg->sample_shift) != 0) return -1;
    aria_dist_shift_defaults(&cfg->train_shift, ARIA_SHIFT_FULL);
    if (aria_json_object(d0, d1, "distribution_shift_options", &s0, &s1) == 0 &&
        parse_shift(s0, s1, &cfg->train_shift) != 0) return -1;

    /* ---- pretransform.config ---- */
    const char *p0, *p1;
    if (aria_json_object(m0, m1, "pretransform", &p0, &p1) == 0) {
        const char *pc0, *pc1;
        if (aria_json_object(p0, p1, "config", &pc0, &pc1) == 0) {
            if (aria_json_get_number_in(pc0, pc1, "latent_dim", &v) == 0) cfg->latent_dim = (int)v;
            if (aria_json_get_number_in(pc0, pc1, "downsampling_ratio", &v) == 0) cfg->downsampling_ratio = (int)v;
            /* nested pretransform.config.pretransform.config.patch_size */
            const char *ip0, *ip1;
            if (aria_json_object(pc0, pc1, "pretransform", &ip0, &ip1) == 0) {
                const char *ipc0, *ipc1;
                if (aria_json_object(ip0, ip1, "config", &ipc0, &ipc1) == 0 &&
                    aria_json_get_number_in(ipc0, ipc1, "patch_size", &v) == 0) {
                    cfg->patch_size = (int)v;
                }
            }
        }
    }

    return 0;
}
