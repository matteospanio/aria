/*
 * aria_wav.c - Minimal WAV reader/writer
 */

#include "aria_wav.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

aria_audio *aria_audio_alloc(int sample_rate, int channels, int64_t num_frames) {
    aria_audio *a = calloc(1, sizeof(aria_audio));
    if (!a) return NULL;
    a->sample_rate = sample_rate;
    a->channels = channels;
    a->num_frames = num_frames;
    a->data = calloc((size_t)num_frames * channels, sizeof(float));
    if (!a->data) { free(a); return NULL; }
    return a;
}

void aria_audio_free(aria_audio *a) {
    if (!a) return;
    free(a->data);
    free(a);
}

static void w_u32(FILE *f, uint32_t v) { fputc(v & 0xff, f); fputc((v>>8)&0xff, f); fputc((v>>16)&0xff, f); fputc((v>>24)&0xff, f); }
static void w_u16(FILE *f, uint16_t v) { fputc(v & 0xff, f); fputc((v>>8)&0xff, f); }

static float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

int aria_wav_write(const char *path, const aria_audio *a, int bits) {
    if (!a || (bits != 16 && bits != 32)) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { perror("aria_wav_write: fopen"); return -1; }

    int ch = a->channels;
    int64_t nf = a->num_frames;
    int bytes_per_sample = bits / 8;
    uint16_t fmt = (bits == 32) ? 3 : 1; /* 3=IEEE float, 1=PCM */
    uint32_t data_bytes = (uint32_t)(nf * ch * bytes_per_sample);
    uint32_t block_align = (uint32_t)(ch * bytes_per_sample);
    uint32_t byte_rate = (uint32_t)a->sample_rate * block_align;

    fwrite("RIFF", 1, 4, f);
    w_u32(f, 36 + data_bytes);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    w_u32(f, 16);
    w_u16(f, fmt);
    w_u16(f, (uint16_t)ch);
    w_u32(f, (uint32_t)a->sample_rate);
    w_u32(f, byte_rate);
    w_u16(f, (uint16_t)block_align);
    w_u16(f, (uint16_t)bits);
    fwrite("data", 1, 4, f);
    w_u32(f, data_bytes);

    int64_t n = nf * ch;
    if (bits == 32) {
        fwrite(a->data, sizeof(float), (size_t)n, f);
    } else {
        for (int64_t i = 0; i < n; i++) {
            float s = clampf(a->data[i], -1.0f, 1.0f);
            int32_t v = (int32_t)lrintf(s * 32767.0f);
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            w_u16(f, (uint16_t)(int16_t)v);
        }
    }
    fclose(f);
    return 0;
}

static uint8_t *m_u32(uint8_t *p, uint32_t v) { p[0]=v&0xff; p[1]=(v>>8)&0xff; p[2]=(v>>16)&0xff; p[3]=(v>>24)&0xff; return p+4; }
static uint8_t *m_u16(uint8_t *p, uint16_t v) { p[0]=v&0xff; p[1]=(v>>8)&0xff; return p+2; }

void aria_wav_header(uint8_t h[44], int sample_rate, int ch, int bits, uint32_t data_bytes) {
    uint32_t block_align = (uint32_t)(ch * (bits / 8));
    uint8_t *p = h;
    memcpy(p, "RIFF", 4); p += 4;
    p = m_u32(p, 36 + data_bytes);
    memcpy(p, "WAVE", 4); p += 4;
    memcpy(p, "fmt ", 4); p += 4;
    p = m_u32(p, 16);
    p = m_u16(p, (uint16_t)(bits == 32 ? 3 : 1));   /* 3=IEEE float, 1=PCM */
    p = m_u16(p, (uint16_t)ch);
    p = m_u32(p, (uint32_t)sample_rate);
    p = m_u32(p, (uint32_t)sample_rate * block_align);
    p = m_u16(p, (uint16_t)block_align);
    p = m_u16(p, (uint16_t)bits);
    memcpy(p, "data", 4); p += 4;
    m_u32(p, data_bytes);
}

int aria_wav_to_mem(const aria_audio *a, int bits, void **out, size_t *out_len) {
    if (!a || !out || !out_len || (bits != 16 && bits != 32)) return -1;
    int ch = a->channels;
    int64_t nf = a->num_frames, n = nf * ch;
    int bps = bits / 8;
    uint32_t data_bytes = (uint32_t)(n * bps);
    size_t total = 44 + data_bytes;
    uint8_t *buf = malloc(total);
    if (!buf) return -1;
    aria_wav_header(buf, a->sample_rate, ch, bits, data_bytes);
    uint8_t *p = buf + 44;
    if (bits == 32) {
        memcpy(p, a->data, (size_t)n * sizeof(float));
    } else {
        for (int64_t i = 0; i < n; i++) {
            float s = clampf(a->data[i], -1.0f, 1.0f);
            int32_t v = (int32_t)lrintf(s * 32767.0f);
            if (v > 32767) v = 32767;
            if (v < -32768) v = -32768;
            p = m_u16(p, (uint16_t)(int16_t)v);
        }
    }
    *out = buf; *out_len = total;
    return 0;
}

static uint32_t r_u32(const uint8_t *p) { return p[0] | (p[1]<<8) | (p[2]<<16) | ((uint32_t)p[3]<<24); }
static uint16_t r_u16(const uint8_t *p) { return (uint16_t)(p[0] | (p[1]<<8)); }

aria_audio *aria_wav_read(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("aria_wav_read: fopen"); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 44) { fclose(f); return NULL; }
    uint8_t *buf = malloc((size_t)sz);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    fclose(f);

    if (memcmp(buf, "RIFF", 4) != 0 || memcmp(buf + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "aria_wav_read: not a RIFF/WAVE file\n");
        free(buf);
        return NULL;
    }

    /* walk chunks to find fmt and data */
    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t sample_rate = 0;
    const uint8_t *data_ptr = NULL;
    uint32_t data_len = 0;
    size_t pos = 12;
    while (pos + 8 <= (size_t)sz) {
        const uint8_t *ch_id = buf + pos;
        uint32_t ch_len = r_u32(buf + pos + 4);
        const uint8_t *ch_data = buf + pos + 8;
        if (memcmp(ch_id, "fmt ", 4) == 0 && ch_len >= 16) {
            fmt = r_u16(ch_data);
            channels = r_u16(ch_data + 2);
            sample_rate = r_u32(ch_data + 4);
            bits = r_u16(ch_data + 14);
        } else if (memcmp(ch_id, "data", 4) == 0) {
            data_ptr = ch_data;
            data_len = ch_len;
            if (pos + 8 + ch_len > (size_t)sz) data_len = (uint32_t)((size_t)sz - (pos + 8));
        }
        pos += 8 + ch_len + (ch_len & 1); /* chunks are word-aligned */
    }
    if (!data_ptr || channels == 0 || bits == 0) {
        fprintf(stderr, "aria_wav_read: missing fmt/data chunk\n");
        free(buf);
        return NULL;
    }

    int bytes_per_sample = bits / 8;
    int64_t total_samples = data_len / bytes_per_sample;
    int64_t num_frames = total_samples / channels;
    aria_audio *a = aria_audio_alloc((int)sample_rate, (int)channels, num_frames);
    if (!a) { free(buf); return NULL; }

    int64_t n = num_frames * channels;
    if (fmt == 3 && bits == 32) {
        memcpy(a->data, data_ptr, (size_t)n * sizeof(float));
    } else if (fmt == 1 && bits == 16) {
        const uint8_t *p = data_ptr;
        for (int64_t i = 0; i < n; i++) {
            int16_t v = (int16_t)r_u16(p + i * 2);
            a->data[i] = (float)v / 32768.0f;
        }
    } else if (fmt == 1 && bits == 24) {
        const uint8_t *p = data_ptr;
        for (int64_t i = 0; i < n; i++) {
            int32_t v = p[i*3] | (p[i*3+1]<<8) | (p[i*3+2]<<16);
            if (v & 0x800000) v |= ~0xFFFFFF; /* sign extend */
            a->data[i] = (float)v / 8388608.0f;
        }
    } else if (fmt == 1 && bits == 32) {
        const uint8_t *p = data_ptr;
        for (int64_t i = 0; i < n; i++) {
            int32_t v = (int32_t)r_u32(p + i * 4);
            a->data[i] = (float)v / 2147483648.0f;
        }
    } else {
        fprintf(stderr, "aria_wav_read: unsupported format fmt=%u bits=%u\n", fmt, bits);
        aria_audio_free(a);
        free(buf);
        return NULL;
    }
    free(buf);
    return a;
}
