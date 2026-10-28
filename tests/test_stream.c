/* test_stream.c - hermetic: aria_stream (the --stream / live-session chunk loop) driven by
 * a fake generator (no model). Checks contiguity, prefix stability (a longer stream starts
 * with exactly the bytes of a shorter one), determinism, the evolve seed sequence, the
 * per-chunk params (window, continuation context, decode range) and the hold/anchor events. */
#include "../src/aria_stream.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SR 8000
#define CH 2

typedef struct { int calls; int64_t seeds[64]; float win[64], dfrom[64], dto[64]; int cont[64]; } fake_log;

/* deterministic "audio": a tone set by seed + prompt, phase-continued from the context's
 * last sample (so the context really conditions the output), plus seeded noise */
static int fake_generate(void *user, const aria_gen_params *p, aria_audio **out) {
    fake_log *lg = user;
    int k = lg->calls++;
    if (k < 64) {
        lg->seeds[k] = p->seed; lg->win[k] = p->seconds_total;
        lg->dfrom[k] = p->decode_from_s; lg->dto[k] = p->decode_to_s;
        lg->cont[k] = p->inpaint_continue && p->init_audio_mem;
    }
    int64_t n = (int64_t)(p->seconds_total * SR);
    aria_audio *a = aria_audio_alloc(SR, CH, n);
    if (!a) return -1;
    unsigned h = 2166136261u;
    for (const char *c = p->prompt ? p->prompt : ""; *c; c++) h = (h ^ (unsigned char)*c) * 16777619u;
    float f = 110.0f + (float)(h % 400) + 7.0f * (float)(p->seed % 13);
    float off = 0.0f;
    if (p->init_audio_mem && p->init_audio_mem->num_frames)
        off = p->init_audio_mem->data[(p->init_audio_mem->num_frames - 1) * CH];
    uint64_t r = (uint64_t)p->seed * 0x9E3779B97F4A7C15ULL + 1;
    for (int64_t i = 0; i < n; i++) {
        r ^= r << 13; r ^= r >> 7; r ^= r << 17;
        float v = 0.5f * sinf(2.0f * 3.14159265f * f * (float)i / SR) + 0.1f * off
                + 0.01f * ((float)(r >> 40) / (float)(1 << 24) - 0.5f);
        a->data[i * CH] = v; a->data[i * CH + 1] = -v;
    }
    *out = a;
    return 0;
}

typedef struct { float *buf; int64_t n; } acc;
static void push(acc *A, const float *x, int64_t n) {
    A->buf = realloc(A->buf, (size_t)(A->n + n) * CH * sizeof(float));
    memcpy(A->buf + A->n * CH, x, (size_t)n * CH * sizeof(float));
    A->n += n;
}

/* run `chunks` chunks (+ flush when `flush`), return the finalized stream */
static int run(int chunks, int flush, int evolve, int hold, int64_t seed, acc *A, fake_log *lg) {
    aria_gen_params base = ARIA_GEN_PARAMS_DEFAULT;
    base.seed = seed;
    aria_stream_cfg cfg = ARIA_STREAM_CFG_DEFAULT;
    cfg.evolve = evolve; cfg.hold = hold;
    cfg.generate = fake_generate; cfg.user = lg; cfg.sample_rate = SR; cfg.channels = CH;
    aria_stream *s = aria_stream_open(NULL, &base, &cfg);
    if (!s) return 1;
    int bad = 0; int64_t expect = 0;
    const char *pa = "warm piano", *pb = "ambient pads";
    char pa2[32]; strcpy(pa2, pa);   /* same text, another buffer: must not reset the loop */
    for (int i = 0; i < chunks; i++) {
        const float *x; aria_stream_stats st;
        const char *pr = (hold && i >= 3) ? pb : (hold && i == 2) ? pa2 : pa;   /* new text at chunk 3 */
        if (aria_stream_next(s, pr, NULL, &x, &st) != 0) { bad = 1; break; }
        if (st.chunk != i || st.start != expect || st.frames <= 0) bad = 1;
        if (hold && st.hold_set != (i == 0 || i == 3)) bad = 1;
        if (st.anchor_set != (i == 0)) bad = 1;
        expect += st.frames;
        push(A, x, st.frames);
    }
    if (flush) {
        const float *x;
        int64_t n = aria_stream_flush(s, &x);
        if (n != (int64_t)(ARIA_STREAM_XFADE_S * SR)) bad = 1;   /* the held-back crossfade */
        push(A, x, n);
    }
    aria_stream_close(s);
    return bad;
}

int main(void) {
    int fails = 0;
    acc a4 = {0}, a6 = {0}, b6 = {0}; fake_log l4 = {0}, l6 = {0}, m6 = {0};
    fails += run(4, 0, 0, 0, 7, &a4, &l4);
    fails += run(6, 1, 0, 0, 7, &a6, &l6);
    fails += run(6, 1, 0, 0, 7, &b6, &m6);
    if (a6.n <= a4.n || memcmp(a4.buf, a6.buf, (size_t)a4.n * CH * sizeof(float)))
        { printf("FAIL prefix stability\n"); fails++; }
    if (a6.n != b6.n || memcmp(a6.buf, b6.buf, (size_t)a6.n * CH * sizeof(float)))
        { printf("FAIL determinism\n"); fails++; }
    /* per-chunk params: chunk 0 = ctx+emit+tail plain, then ctx+skip+emit+tail continuations */
    if (fabsf(l6.win[0] - 11.0f) > 1e-6f || fabsf(l6.win[1] - 12.5f) > 1e-6f || l6.cont[0] || !l6.cont[1])
        { printf("FAIL window/continuation params\n"); fails++; }
    float d1 = (float)(int64_t)(7.5f * SR) / SR - 0.25f;   /* emit_start = (ctx+skip)*sr */
    if (l6.dfrom[0] != 0.0f || fabsf(l6.dfrom[1] - d1) > 1e-6f || !(l6.dto[1] > l6.dfrom[1] + 2.0f))
        { printf("FAIL decode range\n"); fails++; }

    acc e = {0}; fake_log le = {0};
    fails += run(5, 0, 2, 0, 5, &e, &le);
    int64_t want[5] = {5, 5, 6, 6, 7};
    for (int i = 0; i < 5; i++)
        if (le.seeds[i] != want[i]) { printf("FAIL evolve seed[%d]=%lld\n", i, (long long)le.seeds[i]); fails++; }

    acc r = {0}, r2 = {0}; fake_log lr = {0}, lr2 = {0};   /* seed -1 = a real random seed */
    fails += run(2, 0, 0, 0, -1, &r, &lr);
    fails += run(2, 0, 0, 0, -1, &r2, &lr2);
    if (lr.seeds[0] < 0 || lr.seeds[1] != lr.seeds[0] || lr2.seeds[0] < 0 || lr2.seeds[0] == lr.seeds[0])
        { printf("FAIL random seed (%lld, %lld)\n", (long long)lr.seeds[0], (long long)lr2.seeds[0]); fails++; }
    free(r.buf); free(r2.buf);

    {   /* set_chunk: longer chunks from the next one on, output stays contiguous */
        fake_log lc = {0};
        aria_gen_params base = ARIA_GEN_PARAMS_DEFAULT; base.seed = 9;
        aria_stream_cfg cfg = ARIA_STREAM_CFG_DEFAULT;
        cfg.generate = fake_generate; cfg.user = &lc; cfg.sample_rate = SR; cfg.channels = CH;
        aria_stream *st = aria_stream_open(NULL, &base, &cfg);
        const float *x; aria_stream_stats ss; int64_t expect = 0; int ok = st != NULL;
        for (int i = 0; ok && i < 5; i++) {
            if (i == 2 && aria_stream_set_chunk(st, 6.0f) != 0) ok = 0;
            if (aria_stream_next(st, "p", NULL, &x, &ss) != 0 || ss.start != expect) ok = 0;
            expect += ss.frames;
        }
        if (!ok || fabsf(lc.win[2] - 16.5f) > 1e-6f || fabsf(lc.win[1] - 12.5f) > 1e-6f) { printf("FAIL set_chunk\n"); fails++; }
        aria_stream_close(st);
    }

    acc h = {0}; fake_log lh = {0};
    fails += run(5, 1, 0, 1, 3, &h, &lh);   /* hold: loop set at chunk 0, reset at the new prompt */

    free(a4.buf); free(a6.buf); free(b6.buf); free(e.buf); free(h.buf);
    if (fails) { printf("test_stream: %d failures\n", fails); return 1; }
    printf("test_stream: OK\n");
    return 0;
}
