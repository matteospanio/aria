/*
 * aria_wav.h - Minimal RIFF/WAVE reader & writer (no libsndfile)
 *
 * Audio is stored INTERLEAVED in float (natural for WAV files), nominal
 * range [-1, 1]. The SA3 autoencoder works in planar [channels, frames]
 * layout; conversion happens at the model boundary, not here.
 */

#ifndef ARIA_WAV_H
#define ARIA_WAV_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int sample_rate;
    int channels;
    int64_t num_frames;   /* samples per channel */
    float *data;          /* interleaved, channels * num_frames floats */
} aria_audio;

aria_audio *aria_audio_alloc(int sample_rate, int channels, int64_t num_frames);
void aria_audio_free(aria_audio *a);

/* bits: 16 -> PCM16, 32 -> IEEE float32. Returns 0 on success, -1 on error. */
int aria_wav_write(const char *path, const aria_audio *a, int bits);

/* Same encoding into a malloc'd buffer (*out, *out_len); caller frees. 0 ok / -1 error.
 * For in-memory consumers (e.g. the HTTP server response). */
int aria_wav_to_mem(const aria_audio *a, int bits, void **out, size_t *out_len);

/* The 44-byte header of a `bits` (16 PCM / 32 float) WAV holding `data_bytes` of samples:
 * for writers that stream audio to disk and patch the sizes as the file grows. */
void aria_wav_header(uint8_t h[44], int sample_rate, int ch, int bits, uint32_t data_bytes);

/* Reads PCM16 / PCM24 / PCM32 / float32 WAV into interleaved float. NULL on error. */
aria_audio *aria_wav_read(const char *path);

#endif /* ARIA_WAV_H */
