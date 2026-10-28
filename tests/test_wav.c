/* test_wav.c - WAV round-trip: float32 is exact, PCM16 within quantization. */
#include "../src/aria_wav.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "test_tmp.h"

int main(void) {
    int sr = 8000, ch = 2;
    int64_t nf = 256;
    aria_audio *a = aria_audio_alloc(sr, ch, nf);
    for (int64_t i = 0; i < nf; i++) {
        a->data[i*ch + 0] = 0.5f * sinf(2.0f * 3.14159265f * 220.0f * i / sr);
        a->data[i*ch + 1] = 0.25f * sinf(2.0f * 3.14159265f * 440.0f * i / sr);
    }

    char f32[512], p16[512];
    tmp_path(f32, sizeof f32, "aria_test_f32.wav");
    tmp_path(p16, sizeof p16, "aria_test_p16.wav");

    /* float32 round-trip: exact */
    if (aria_wav_write(f32, a, 32) != 0) { printf("FAIL write f32\n"); return 1; }
    aria_audio *b = aria_wav_read(f32);
    if (!b || b->num_frames != nf || b->channels != ch || b->sample_rate != sr) {
        printf("FAIL read f32 meta\n"); return 1;
    }
    int fails = 0;
    for (int64_t i = 0; i < nf * ch; i++)
        if (fabsf(a->data[i] - b->data[i]) > 1e-7f) fails++;
    if (fails) { printf("FAIL f32 roundtrip: %d mismatches\n", fails); return 1; }

    /* pcm16 round-trip: within 1/32768 */
    if (aria_wav_write(p16, a, 16) != 0) { printf("FAIL write p16\n"); return 1; }
    aria_audio *c = aria_wav_read(p16);
    if (!c) { printf("FAIL read p16\n"); return 1; }
    float maxerr = 0.0f;
    for (int64_t i = 0; i < nf * ch; i++) {
        float e = fabsf(a->data[i] - c->data[i]);
        if (e > maxerr) maxerr = e;
    }
    if (maxerr > 1.0f/32768.0f + 1e-6f) { printf("FAIL p16 maxerr=%.6f\n", maxerr); return 1; }

    aria_audio_free(a); aria_audio_free(b); aria_audio_free(c);
    printf("test_wav: OK (pcm16 maxerr=%.6f)\n", maxerr);
    return 0;
}
