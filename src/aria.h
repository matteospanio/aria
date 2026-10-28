/*
 * aria.h - Public API for the aria audio-diffusion inference runtime.
 *
 *   aria_ctx *ctx = aria_load("path/to/model");      // dir with model_config.json
 *   aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
 *   p.prompt = "warm romantic piano"; p.seconds_total = 15.0f;
 *   aria_audio *audio = NULL;
 *   aria_generate(ctx, &p, &audio);
 *   aria_wav_write("out.wav", audio, 32);
 *   aria_audio_free(audio); aria_free(ctx);
 */

#ifndef ARIA_H
#define ARIA_H

#include <stdint.h>
#include "aria_wav.h"
#include "aria_quant.h"   /* aria_dtype (precision selection) */
#include "aria_steer.h"   /* aria_steer_set (E12 activation steering) */
#include "aria_lora.h"    /* aria_lora_adapter (E12.9 runtime LoRA arm) */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aria_ctx aria_ctx;

typedef enum { ARIA_DEVICE_CPU = 0, ARIA_DEVICE_CUDA = 1, ARIA_DEVICE_AUTO = 2 } aria_device;

/* Sampling schedule: the model's reference shift (what stable_audio_3 generate() uses),
 * aria's legacy length-dependent LogSNR (rate=1), or the training DistributionShift (what
 * the stable-audio-3 TensorRT runtime samples with). DEFAULT = the runtime's choice. */
typedef enum { ARIA_SCHED_DEFAULT = 0, ARIA_SCHED_REF, ARIA_SCHED_LEN, ARIA_SCHED_FULL } aria_schedule;

typedef struct {
    const char *prompt;             /* text prompt (Phase 1) */
    const char *prompt_embed_path;  /* optional precomputed cross-attn embedding, f32 [256,768] */
    float seconds_total;            /* requested duration in seconds */
    int   steps;                    /* denoising steps (0 = default: 6 on a GPU, 4 on the CPU) */
    float cfg_scale;                /* CFG scale (default 1.0 = off) */
    int64_t seed;                   /* RNG seed (-1 = random) */
    aria_device device;             /* DiT backend: auto (default) / cpu / cuda (E8.5) */
    aria_dtype precision;           /* DiT precision: ARIA_DTYPE_AUTO (default: the device's best) /
                                     * fp32 / fp16 / bf16 / q8 / q8a8 / q4 */
    const char *load_quant;         /* optional pre-quantized DiT (.aria) from aria-quantize (E9.2) */
    /* continue / inpaint (E7): regenerate part of an existing clip, keep the rest.
     * Set init_audio to a WAV path. inpaint_from_s..inpaint_to_s marks the region
     * to regenerate (seconds; to_s <= 0 means "to seconds_total"). inpaint_continue
     * = regenerate [clip_duration, seconds_total] (extend the clip). Runs on the
     * CPU and device DiT (per-block local-additive cond is uploaded to the GPU). */
    const char *init_audio;         /* WAV path, or NULL for plain text->audio */
    float inpaint_from_s;           /* start of the regenerated region (seconds) */
    float inpaint_to_s;             /* end of the regenerated region; <= 0 = seconds_total */
    int   inpaint_continue;         /* 1 = continue mode (regenerate the tail) */
    /* optional per-step progress callback, invoked after each denoise step with
     * (step, total, progress_user). NULL = no callback. */
    void (*progress)(int step, int total, void *user);
    void *progress_user;
    int   rng_torch;                /* 1 = PyTorch-matched noise (MT19937) for exact
                                     * torch.randn reproduction; 0 = xoshiro (default) */
    const aria_audio *init_audio_mem; /* in-memory continue/inpaint context (overrides
                                       * init_audio file); not freed by the runtime.
                                       * Used by the streaming loop to avoid temp WAVs. */
    const aria_steer_set *steer;      /* optional activation steering (E12); NULL = none */
    const aria_lora_adapter *lora;    /* optional runtime LoRA adapter (E12.9, CPU); NULL = none */
    /* B5 (E14.1) partial-emit decode: only [decode_from_s, decode_to_s) of the audio
     * is decoded (the rest of the returned buffer is silence). The full DiT still runs
     * (every latent frame is needed for coherence); only the AE decoder is restricted,
     * with a halo so the kept span is byte-identical to a full decode. Used by --stream
     * to decode ~2 s per chunk instead of the whole ~12.5 s window. 0/0 = full decode.
     * CPU decoders only; the device decoder ignores it (already fast, monolithic). */
    float decode_from_s, decode_to_s;
    aria_schedule schedule;         /* sampling schedule (default: ARIA_SCHED_DEFAULT) */
    float headroom_s;               /* stable_audio_3 duration_padding_sec: generate this many
                                     * extra seconds of latent, cropped after decode (text->audio
                                     * only; 0 = off; the reference uses 6) */
    int   oneshot;                  /* 1 = last request on this ctx: free each stage (text encoder,
                                     * DiT) once it is done, so peak memory is the largest stage
                                     * rather than their sum. Later requests still work (stages
                                     * reload). 0 = keep everything resident (batch/stream/server) */
} aria_gen_params;

#define ARIA_GEN_PARAMS_DEFAULT { NULL, NULL, 15.0f, 0, 1.0f, 0, ARIA_DEVICE_AUTO, ARIA_DTYPE_AUTO, NULL, NULL, 0.0f, 0.0f, 0, NULL, NULL, 0, NULL, NULL, NULL, 0.0f, 0.0f, ARIA_SCHED_DEFAULT, 0.0f, 0 }

/* Load a model from a directory containing model_config.json + model.safetensors.
 * Returns NULL on error (see aria_last_error()). */
aria_ctx *aria_load(const char *model_dir);
void aria_free(aria_ctx *ctx);

/* Introspection */
const char *aria_model_type(const aria_ctx *ctx);
int aria_sample_rate(const aria_ctx *ctx);
int aria_audio_channels(const aria_ctx *ctx);
int aria_num_tensors(const aria_ctx *ctx);
void aria_list_tensors(const aria_ctx *ctx, const char *prefix); /* prefix may be NULL */

/* Text-to-audio generation (Phase 1). Sets *out on success. Returns 0 / <0. */
int aria_generate(aria_ctx *ctx, const aria_gen_params *p, aria_audio **out);

const char *aria_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_H */
