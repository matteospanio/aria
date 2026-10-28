/*
 * aria_sa3_dit.h - Stable Audio 3 DiT: conditioning embedders + transformer block.
 *
 * Reference: stable_audio_tools models/dit.py (DiffusionTransformer) and
 * models/transformer.py (TransformerBlock, ContinuousTransformer).
 *
 * Block forward (adaLN), batch=1, operating on x[S, dim]:
 *   mod = to_scale_shift_gate[6*dim] + global_cond[6*dim]; chunk into
 *     (scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff) each [dim]
 *   self-attn: r=x; h=rmsnorm(x); h=h*(1+scale_self)+shift_self;
 *              h=selfattn(h, qk-rms, rope); h*=sigmoid(1-gate_self); x=r+h
 *   cross-attn: r=x; h=rmsnorm(x); h=crossattn(h, context, qk-rms, NO rope); x=r+h
 *   ffn:       r=x; h=rmsnorm(x); h=h*(1+scale_ff)+shift_ff;
 *              h=glu_ff(h); h*=sigmoid(1-gate_ff); x=r+h
 */

#ifndef ARIA_SA3_DIT_H
#define ARIA_SA3_DIT_H

#include "aria_safetensors.h"
#include "aria_sa3.h"
#include "aria_quant.h"
#include "aria_steer.h"   /* aria_steer_set (E12 residual-site steering) */
#include "aria_lora.h"    /* aria_lora_adapter (E12.9 runtime LoRA arm) */

#ifdef __cplusplus
extern "C" {
#endif

/* Two-layer MLP: Linear(in_dim->mid, W0,b0) -> SiLU -> Linear(mid->out_dim, W2,b2),
 * applied to N rows. b0/b2 may be NULL (no bias). out [N, out_dim]. */
void aria_sa3_mlp2(float *out, const float *in, int N, int in_dim, int mid, int out_dim,
                   const float *W0, const float *b0, const float *W2, const float *b2);

/* Timestep embedding: ExpoFourier(t, feat_dim, 0.5, 10000) -> mlp2(feat->ed->ed).
 * out [ed]. (t is the raw diffusion timestep.) */
void aria_sa3_timestep_embed(float *out, float t, int feat_dim, int ed,
                             const float *W0, const float *b0, const float *W2, const float *b2);

typedef struct {
    const float *pre_norm, *cross_norm, *ff_norm;   /* RMSNorm gammas [dim] */
    const float *sa_to_qkv, *sa_q_norm, *sa_k_norm, *sa_to_out;
    const float *ca_to_q, *ca_to_kv, *ca_q_norm, *ca_k_norm, *ca_to_out;
    const float *ff_in_w, *ff_in_b, *ff_out_w, *ff_out_b;
    const float *to_scale_shift_gate;               /* [6*dim] */
    /* local-additive (inpaint) cond MLP: Linear(257->dim) -> SiLU -> Linear(dim->dim) */
    const float *to_local0_w, *to_local0_b, *to_local2_w, *to_local2_b;
} aria_dit_block_w;

/* In-place adaLN block forward on x[S, dim].
 *   context [Sc, dim_ctx] (cross-attn kv), global_cond [6*dim] (post embedder),
 *   rope_cos/rope_sin [S, rot_dim/2] for self-attn (cross-attn has no rope).
 *   inner = ff inner dim (dim * ff_mult). */
void aria_dit_block_forward(float *x, int S, int dim, int num_heads, int head_dim, int inner,
                            const float *context, int Sc, int dim_ctx,
                            const float *global_cond,
                            const float *rope_cos, const float *rope_sin, int rot_dim,
                            const aria_dit_block_w *w, int differential);

/* ---- full DiT model ---- */

typedef struct aria_sa3_dit aria_sa3_dit;

/* Load all DiT weights (model.model.*) from sf using cfg. Returns NULL on a
 * missing tensor. Weights are copied to f32 buffers owned by the model. */
aria_sa3_dit *aria_sa3_dit_load(safetensors_file_t *sf, const aria_sa3_config *cfg);
void aria_sa3_dit_free(aria_sa3_dit *m);

/* Quantize the block GEMM weights to `dt` (Q8/Q4) as an additive overlay; the
 * denoise loop then dispatches those linears to dequant-on-use kernels. dt==F32
 * drops the overlay (zero-copy mmap path, bit-identical to no quantization).
 * Idempotent. The norms, biases, embedders, project_in/out stay f32. */
void aria_sa3_dit_quantize(aria_sa3_dit *m, aria_dtype dt);
/* total bytes of the block GEMM weights at the current precision (reporting). */
size_t aria_sa3_dit_weight_bytes(const aria_sa3_dit *m);

/* Offline packed quantized DiT (E9.2): save quantizes to `dt` and writes the
 * per-block GEMM overlay to `path`; load fills m->bq from such a file (the model
 * keeps its f32 weights for the non-block parts). 0 on success, <0 on error. */
int aria_sa3_dit_quant_save(aria_sa3_dit *m, aria_dtype dt, const char *path);
int aria_sa3_dit_quant_load(aria_sa3_dit *m, const char *path);

/* Per-request context: a reusable scratch arena plus the step-invariant state
 * cached once and reused across all denoising steps -- cross_ed (to_cond_embed),
 * RoPE tables, to_global_embed(seconds), and the per-block cross-attention K/V
 * (the prompt context is constant across steps, so it is projected just once).
 *
 * Thread-safety / batch foundation: the model (aria_sa3_dit) is immutable and
 * read-only during a step; each request owns its arena and caches with no shared
 * mutable state. Distinct requests may therefore run concurrently against one
 * shared model -- the basis for a request-parallel batch/server API.
 * (Each step uses OpenMP internally, so a server caps per-request
 * threads to avoid oversubscription.) */
typedef struct aria_sa3_dit_req aria_sa3_dit_req;

/* gpu=1 defers the per-block cross-K/V projection (cross_k/v/kd stay NULL, only
 * cross_ed is computed) so the CUDA backend can project on-device; gpu=0 projects
 * on the CPU as the f32 forward needs. */
aria_sa3_dit_req *aria_sa3_dit_req_begin(const aria_sa3_dit *m, int T,
                                         const float *cross_768, int n_cond,
                                         const float *global_768, int gpu);
/* Optional local-additive (inpaint) conditioning: raw [n_local, local_dim]
 * (=[T,257] of [inpaint_mask | inpaint_masked_input]); each block projects it via
 * its to_local_embed MLP and adds it to the residual stream (left-padded past the
 * memory tokens). Call after req_begin; omit for plain text->audio (no-op). */
void aria_sa3_dit_req_set_local(aria_sa3_dit_req *req, const float *local_raw,
                                int n_local, int local_dim);
/* E12: attach an activation-steering set (residual site applied in the block loop) + set the
 * current denoise step (for step-window gating). Both default to none/0; steer is borrowed. */
void aria_sa3_dit_req_set_steer(aria_sa3_dit_req *req, const aria_steer_set *steer);
void aria_sa3_dit_req_set_step(aria_sa3_dit_req *req, int step);
/* E12.9: attach a runtime LoRA adapter (CPU DiT). Applied at the 7 hooked block
 * GEMMs; the per-step 6 in the block loop and ca_to_kv re-folded into the cached
 * cross K/V here. Borrowed (not freed by the runtime). NULL = none. Call after
 * req_begin. No-op on the GPU path (host cross K/V deferred; see E12.9b). */
void aria_sa3_dit_req_set_lora(aria_sa3_dit_req *req, const aria_lora_adapter *lora);
/* one denoising step: out_CT = velocity(x_CT, t), reusing the request's caches. */
void aria_sa3_dit_step(const aria_sa3_dit *m, aria_sa3_dit_req *req,
                       float *out_CT, const float *x_CT, float t);
void aria_sa3_dit_req_end(aria_sa3_dit_req *req);

/* One-shot denoiser forward (matches DiffusionTransformer._forward, batch=1) =
 * begin + one step + end. x_CT/out_CT: latent [io_channels, T] (channel-major). */
void aria_sa3_dit_forward(const aria_sa3_dit *m, float *out_CT, const float *x_CT, int T,
                          float t, const float *cross_768, int n_cond, const float *global_768);

/* ---- read-only views (so the CUDA backend can upload weights/caches) ---- */
typedef struct {
    int depth, ed, num_heads, head_dim, inner, io_ch, n_mem, rot_dim;
    int differential;                 /* medium: differential self+cross attention */
    const float *preprocess, *postprocess, *project_in, *project_out, *memory_tokens;
    const aria_dit_block_w *blocks;   /* [depth] */
} aria_sa3_dit_view;
void aria_sa3_dit_get_view(const aria_sa3_dit *m, aria_sa3_dit_view *v);

typedef struct {
    int T, S, n_cond, depth;
    const float *global_seconds, *rope_cos, *rope_sin;
    float *const *cross_k, *const *cross_v;   /* [depth] each [H, n_cond, head_dim]; NULL if gpu-deferred */
    float *const *cross_kd;                   /* [depth] differential k_diff (post k_norm), or NULL */
    const float *cross_ed;                    /* [n_cond, ed] to_cond_embed(prompt); set when cross_k deferred */
    float *const *local_emb;                  /* [depth] each [S, ed]; inpaint local-additive cond (mem rows 0), or NULL */
    int has_local;                            /* continue/inpaint: add local_emb per block */
    const aria_steer_set *steer;              /* E12: residual-site steering (directions uploaded per block), or NULL */
} aria_sa3_dit_req_view;
void aria_sa3_dit_req_get_view(const aria_sa3_dit_req *r, aria_sa3_dit_req_view *v);

/* Helper: gcond[6*ed] = global_cond_embedder(global_seconds + timestep_embed(t)).
 * (Used by the CUDA path to compute the per-step modulation on the host.) */
void aria_sa3_dit_global_cond(const aria_sa3_dit *m, const float *global_seconds, float t, float *gcond);

/* ---- CUDA device-resident DiT (E8.5; implemented in aria_cuda.cu) ----
 * Weights are uploaded once as fp16 (fp32 compute); the denoise loop runs on the
 * device, only the per-step latent/velocity cross the bus. All host pointers. */
typedef struct aria_cuda_dit aria_cuda_dit;
/* precision: ARIA_F32 (fp16 storage), or ARIA_Q8/ARIA_Q4 (weights packed in VRAM,
 * dequant-on-use to fp16 -> medium fits low VRAM). NULL if no device / no fit. */
aria_cuda_dit *aria_cuda_dit_create(const aria_sa3_dit_view *v, aria_dtype precision);
void aria_cuda_dit_free(aria_cuda_dit *h);
void aria_cuda_dit_set_request(aria_cuda_dit *h, const aria_sa3_dit_req_view *rv);
/* E12: set the current denoise step for residual-steering step-window gating (call per step). */
void aria_cuda_dit_set_step(aria_cuda_dit *h, int step);
/* v_CT = velocity(x_CT, gcond); x_CT/v_CT are [io_channels, T], gcond is [6*ed]. */
void aria_cuda_dit_step(aria_cuda_dit *h, float *v_CT, const float *x_CT, const float *gcond);
/* E17.2: the whole pingpong loop on the device, bit-identical to aria_cuda_dit_step + the
 * host pingpong. gcond_all = [steps][6*ed] (per-step aria_sa3_dit_global_cond); noise(buf,
 * n, user) fills each step's N(0,1) draw in order. x_CT: in = initial noise, out = final
 * latent. Returns -1 (nothing run; use the host loop) if its staging can't be allocated. */
int aria_cuda_dit_sample(aria_cuda_dit *h, float *x_CT, const float *sched, int steps,
                         const float *gcond_all, void (*noise)(float *, int, void *), void *noise_user,
                         void (*progress)(int, int, void *), void *progress_user);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_SA3_DIT_H */
