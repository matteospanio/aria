/*
 * aria_sa3_enc.h - Stable Audio 3 taae_v2 autoencoder ENCODER (audio -> latent).
 *
 * Mirror of the decoder (aria_sa3_dec). Forward (audio[2, L] -> latent[256, T]):
 *   1. patchify:  patches[c*256+h, t] = audio[c, t*256+h]  (patch_size 256,
 *      zero-pad a short final patch)  ->  [512, T_patch], T_patch = ceil(L/256)
 *   2. SAME encoder: zero-pad T_patch to a multiple of 32; WNConv1d mapping
 *      512->768 (k1); group every 16 tokens + 1 learned new_token -> 17; two
 *      chunked (S=34) transformer halves (blocks 0-2 unshifted, 3-5 midpoint-
 *      shift halo); take the LAST token of each 17-group; Linear 768->256
 *      ->  enc[256, T], T = (T_patch padded)/16
 *   3. softnorm bottleneck: latent = (enc * scaling_factor + bias) / running_std
 * Per-block machinery (DyT norms, differential attn, SwiGLU) is shared with the
 * decoder via aria_taae.{c,h}.
 *
 * The medium model's SAME-L encoder (1536-d) differs in step 2 only: T_patch is padded
 * to a multiple of 16, the WNConv1d maps 512->1536, and 12 sliding-window blocks (band
 * +-17 tokens, absolute RoPE, as in the medium decoder) run over the whole sequence;
 * long inputs are encoded in windows with a 12-frame halo (byte-identical).
 */

#ifndef ARIA_SA3_ENC_H
#define ARIA_SA3_ENC_H

#include "aria_safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct aria_sa3_enc aria_sa3_enc;

aria_sa3_enc *aria_sa3_enc_load(safetensors_file_t *sf);
void aria_sa3_enc_free(aria_sa3_enc *m);

/* derived lengths: patch tokens for an audio of L samples, and latent tokens
 * for that many patch tokens (patches padded to a multiple of 32, then /16). */
int aria_sa3_patch_len(int L);
int aria_sa3_latent_len(int T_patch);                              /* SAME-S */
int aria_sa3_enc_latent_len(const aria_sa3_enc *m, int T_patch);   /* this encoder's */

/* full encode: audio[2, L] -> latent[256, T] (caller allocates >= 256*T_lat).
 * Returns the latent length T = aria_sa3_enc_latent_len(m, aria_sa3_patch_len(L)). */
int aria_sa3_enc_forward(const aria_sa3_enc *m, float *latent, const float *audio, int L);

/* ---- stage entry points (staged parity) ---- */
void aria_sa3_patchify(float *patches, const float *audio, int L, int T_patch);     /* patches[512,T_patch] */
void aria_sa3_same_encode(const aria_sa3_enc *m, float *enc, const float *patches,
                          int T_patch, int T_lat);                                   /* enc[256,T_lat] */
void aria_sa3_softnorm_encode(const aria_sa3_enc *m, float *latent, const float *enc, int T); /* latent[256,T] */
/* SAME-L with an explicit window (latent frames; tests check windowed == monolithic) */
void aria_sa3_same_encode_win(const aria_sa3_enc *m, float *enc, const float *patches,
                              int T_patch, int T_lat, int window_frames);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_SA3_ENC_H */
