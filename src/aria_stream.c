/*
 * aria_stream.c - sliding-window continuation shared by the CLI and aria-server.
 * Moved from main.c's cmd_stream; every expression is kept as it was there, so the
 * CLI's --stream output is byte-identical. See aria_stream.h.
 */
#include "aria_stream.h"
#include "aria_model.h"   /* aria_set_error */
#include "aria_hpss.h"
#include "aria_wav.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct aria_stream {
    aria_ctx *ctx;
    aria_gen_params base;
    aria_stream_cfg cfg;
    int sr, ch, chunk;
    int64_t ctx_fr, xf_fr, maxlag, keep;
    aria_audio *outp;               /* the timeline's tail: local frame 0 = absolute `origin` */
    int64_t origin, written, cap;   /* `written` is local; absolute = origin + written */
    int64_t emitted;                /* absolute: finalized up to here */
    aria_audio *context, *anchor;
    int recapture;
    int64_t seed, base_seed; int evolve_k;
    float *held; int64_t held_len; const char *held_prompt;   /* --hold drum loop */
    float *out; int64_t out_cap;                              /* the finalized frames returned */
};

/* interleaved frames [start, start+len) of `a` -> new aria_audio */
static aria_audio *stream_slice(const aria_audio *a, int64_t start, int64_t len) {
    if (start < 0) start = 0;
    if (start + len > a->num_frames) len = a->num_frames - start;
    if (len < 0) len = 0;
    aria_audio *o = aria_audio_alloc(a->sample_rate, a->channels, len);
    if (o && len) memcpy(o->data, a->data + start * a->channels, (size_t)len * a->channels * sizeof(float));
    return o;
}

/* Append `src` onto `outp`, phase-aligning it to the existing tail before crossfading.
 * The continuation is only beat-plausible, not sample-accurate, so a plain crossfade
 * "flams" misaligned drum hits; here we cross-correlate `src`'s start against the output's
 * last ~60 ms over a 0..`maxlag` search, shift `src` by the best lag, then linearly
 * crossfade `xf` frames. maxlag=0 -> plain crossfade (used for the first segment). */
static void stream_append_aligned(aria_audio *outp, int64_t *written, int64_t cap,
                                  const aria_audio *src, int64_t xf, int64_t maxlag) {
    int ch = outp->channels; int64_t n = src->num_frames, lag = 0;
    const int64_t clen = 1764, cstride = 2;   /* ~40 ms window, every 2nd sample */
    int64_t W = *written;
    if (W >= clen && maxlag > 0 && n >= clen + maxlag) {
        const float *ref = outp->data + (W - clen) * ch;   /* mono via channel sum */
        double best = -1e30;
        for (int64_t L = 0; L <= maxlag; L++) {
            const float *cand = src->data + L * ch;
            double dot = 0, ea = 0, eb = 0;
            for (int64_t i = 0; i < clen; i += cstride) {
                float a = ref[i * ch] + (ch > 1 ? ref[i * ch + 1] : 0.0f);
                float b = cand[i * ch] + (ch > 1 ? cand[i * ch + 1] : 0.0f);
                dot += (double)a * b; ea += (double)a * a; eb += (double)b * b;
            }
            double corr = dot / (sqrt(ea * eb) + 1e-9);
            if (corr > best) { best = corr; lag = L; }
        }
    }
    int64_t xfe = xf;
    if (xfe > *written) xfe = *written;
    if (lag + xfe > n) xfe = n - lag;
    if (xfe < 0) xfe = 0;
    for (int64_t i = 0; i < xfe; i++) {
        float w = (float)(i + 1) / (float)(xfe + 1);
        int64_t o = (*written - xfe + i) * ch;
        for (int c = 0; c < ch; c++)
            outp->data[o + c] = (1.0f - w) * outp->data[o + c] + w * src->data[(lag + i) * ch + c];
    }
    int64_t rest = n - lag - xfe;
    if (*written + rest > cap) rest = cap - *written;
    if (rest > 0) memcpy(outp->data + (*written) * ch, src->data + (lag + xfe) * ch, (size_t)rest * ch * sizeof(float));
    *written += rest;
}

/* add the looped percussive `held` (held_len frames) into `dst`[n] starting at output
 * position `pos` (modulo tiling) — the held drum groove, steady across chunks. */
static void stream_add_tiled(float *dst, int64_t n, int ch, int64_t pos,
                             const float *held, int64_t held_len) {
    if (held_len <= 0) return;
    for (int64_t j = 0; j < n; j++) {
        int64_t k = (pos + j) % held_len;
        for (int c = 0; c < ch; c++) dst[j * ch + c] += held[k * ch + c];
    }
}

aria_stream *aria_stream_open(aria_ctx *ctx, const aria_gen_params *base, const aria_stream_cfg *cfg) {
    int sr = ctx ? aria_sample_rate(ctx) : cfg->sample_rate;
    int ch = ctx ? aria_audio_channels(ctx) : cfg->channels;
    if (!(cfg->chunk_s > 0.0f) || !(cfg->context_s > 0.0f) || sr <= 0 || ch <= 0) {
        aria_set_error("stream: chunk and context must be > 0");
        return NULL;
    }
    aria_stream *s = calloc(1, sizeof *s);
    if (!s) { aria_set_error("stream: OOM"); return NULL; }
    s->ctx = ctx; s->base = *base; s->cfg = *cfg; s->sr = sr; s->ch = ch;
    s->base.headroom_s = 0.0f;   /* text->audio only; chunks are continuations */
    if (s->cfg.anchor < 0.0f) s->cfg.anchor = 0.0f;
    if (s->cfg.anchor > 1.0f) s->cfg.anchor = 1.0f;
    s->ctx_fr = (int64_t)(cfg->context_s * sr);
    s->xf_fr = (int64_t)(ARIA_STREAM_XFADE_S * sr);
    s->maxlag = (int64_t)(ARIA_STREAM_MAXLAG_S * sr);
    /* the stitcher reads at most the last max(1764, xfade) frames and the next context is
     * the last ctx_fr, so a tail of `keep` frames behaves exactly like the whole timeline */
    s->keep = s->ctx_fr > s->xf_fr ? s->ctx_fr : s->xf_fr;
    if (s->keep < 1764) s->keep = 1764;
    s->cap = s->keep + (int64_t)((cfg->context_s + cfg->chunk_s) * sr) + s->maxlag + sr;
    s->outp = aria_audio_alloc(sr, ch, s->cap);
    if (!s->outp) { free(s); aria_set_error("stream: OOM"); return NULL; }
    s->seed = s->base_seed = base->seed;
    return s;
}

/* finalize [emitted, upto) (absolute) into s->out with the held loop mixed in */
static int64_t stream_take(aria_stream *s, int64_t upto, const float **audio) {
    int64_t n = upto > s->emitted ? upto - s->emitted : 0;
    if (n > s->out_cap) {
        float *o = realloc(s->out, (size_t)n * s->ch * sizeof(float));
        if (!o) { aria_set_error("stream: OOM"); return -1; }
        s->out = o; s->out_cap = n;
    }
    if (n) {
        memcpy(s->out, s->outp->data + (s->emitted - s->origin) * s->ch, (size_t)n * s->ch * sizeof(float));
        stream_add_tiled(s->out, n, s->ch, s->emitted, s->held, s->held_len);   /* no-op if held_len==0 */
    }
    *audio = s->out;
    s->emitted += n;
    return n;
}

int aria_stream_next(aria_stream *s, const char *prompt, const aria_steer_set *steer,
                     const float **audio, aria_stream_stats *st) {
    const int sr = s->sr, ch = s->ch, i = s->chunk;
    const float emit_s = s->cfg.chunk_s, context_s = s->cfg.context_s;
    const float skip_s = ARIA_STREAM_SKIP_S, tail_s = ARIA_STREAM_TAIL_S;
    aria_stream_stats z = {0}; *st = z;
    st->chunk = i;
    /* E14c: evolve -- fresh variation every N chunks (base seed stays the anchor of
     * the sequence so runs are reproducible); default (evolve==0) never re-seeds. */
    if (s->cfg.evolve > 0 && i > 0 && i % s->cfg.evolve == 0) {
        s->seed = (s->base_seed >= 0 ? s->base_seed : 0) + (++s->evolve_k);
        st->reseeded = 1;
    }
    /* i==0: plain text->audio, emit the strong front [0, context+emit] (drop its fade).
     * i>0 : continuation; emit the body [context+skip, context+skip+emit]. */
    float win_s = (i == 0) ? (context_s + emit_s + tail_s)
                           : (context_s + skip_s + emit_s + tail_s);
    int64_t emit_start = (i == 0) ? 0 : (int64_t)((context_s + skip_s) * sr);
    int64_t emit_len   = (i == 0) ? (int64_t)((context_s + emit_s) * sr)
                                  : (int64_t)(emit_s * sr);
    aria_gen_params p = s->base;
    p.prompt = prompt;
    p.steer = steer;
    p.seed = s->seed;
    p.seconds_total = win_s;
    p.init_audio_mem = (i == 0) ? NULL : s->context;
    p.inpaint_continue = (i == 0) ? 0 : 1;
    /* B5 (E14.1): decode only the region we actually emit (+ the maxlag search room
     * + a small halo margin), not the whole win_s window. CPU decoders only; the
     * GPU decode is already fast + monolithic and ignores this. */
    p.decode_from_s = (float)emit_start / sr - 0.25f;
    p.decode_to_s   = (float)(emit_start + emit_len + s->maxlag) / sr + 0.25f;
    if (p.decode_from_s < 0.0f) p.decode_from_s = 0.0f;

    aria_audio *win = NULL;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    int rc = s->cfg.generate ? s->cfg.generate(s->cfg.user, &p, &win) : aria_generate(s->ctx, &p, &win);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (rc != 0 || !win) { aria_audio_free(win); return -1; }
    st->gen_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    st->win_s = win_s; st->emit_s = (float)(emit_len / (double)sr); st->seed = s->seed;

    /* extra `maxlag` frames so the phase-align search has room to shift into */
    int64_t slice_len = emit_len + (i == 0 ? 0 : s->maxlag);
    if (emit_start + slice_len > win->num_frames) slice_len = win->num_frames - emit_start;
    aria_audio *emit = stream_slice(win, emit_start, slice_len);
    aria_audio_free(win);
    if (!emit) { aria_set_error("stream: OOM"); return -1; }
    if (s->cfg.hold) {
        /* split the emit; the timeline accumulates only the HARMONIC, the drum groove is
         * a held loop tiled on output -> drums stay perfectly steady across chunks. */
        int64_t en = emit->num_frames;
        float *H = malloc((size_t)en * ch * sizeof(float)), *P = malloc((size_t)en * ch * sizeof(float));
        if (!H || !P) { free(H); free(P); aria_audio_free(emit); aria_set_error("stream: OOM"); return -1; }
        aria_hpss_separate(emit->data, en, ch, H, P);
        if (!s->held || prompt != s->held_prompt) {   /* establish/reset the loop on chunk 0 or prompt change */
            free(s->held); s->held_len = en; s->held = malloc((size_t)s->held_len * ch * sizeof(float));
            memcpy(s->held, P, (size_t)s->held_len * ch * sizeof(float));
            s->held_prompt = prompt;
            st->hold_set = 1;
        }
        memcpy(emit->data, H, (size_t)en * ch * sizeof(float));   /* emit -> harmonic only */
        free(H); free(P);
    }
    if (s->written > s->keep) {   /* slide: drop what neither the stitcher nor the context reads */
        int64_t drop = s->written - s->keep;
        memmove(s->outp->data, s->outp->data + drop * ch, (size_t)s->keep * ch * sizeof(float));
        s->origin += drop; s->written = s->keep;
    }
    stream_append_aligned(s->outp, &s->written, s->cap, emit, (i == 0) ? 0 : s->xf_fr, (i == 0) ? 0 : s->maxlag);
    aria_audio_free(emit);

    aria_audio_free(s->context);
    s->context = stream_slice(s->outp, s->written - s->ctx_fr, s->ctx_fr);   /* just-emitted tail */
    if (!s->context) { aria_set_error("stream: OOM"); return -1; }
    const float beta = s->cfg.anchor;
    if (i == 0 || s->recapture) {
        /* the reference the stream keeps being pulled back toward (captured pre-hold-mix:
         * the harmonic timeline; drums are re-added below) */
        aria_audio_free(s->anchor);
        s->anchor = stream_slice(s->context, 0, s->context->num_frames);
        s->recapture = 0;
        st->anchor_set = 1;
    } else if (beta > 0.0f && s->anchor && s->anchor->num_frames > 0) {
        /* E14b: convex blend toward the reference, then pull the context's ENERGY back to
         * the reference level (boost-only, capped at 4x): a linear blend alone cannot
         * out-weigh a tail that already faded 15+ dB, so the model would still hear
         * "quiet outro" and keep fading. Peak guard keeps the taae encoder in range.
         * Conditioning only: the blend never reaches the output. */
        aria_audio *context = s->context, *anchor = s->anchor;
        int64_t bn = context->num_frames < anchor->num_frames
                   ? context->num_frames : anchor->num_frames;
        double e_ctx = 0.0, e_anc = 0.0;
        for (int64_t j = 0; j < bn * ch; j++) {
            float v = (1.0f - beta) * context->data[j] + beta * anchor->data[j];
            context->data[j] = v;
            e_ctx += (double)v * v;
            e_anc += (double)anchor->data[j] * anchor->data[j];
        }
        float g = 1.0f;
        if (e_ctx > 0.0 && e_anc > e_ctx)
            g = (float)sqrt(e_anc / e_ctx);
        if (g > 4.0f) g = 4.0f;
        float pk = 0.0f;
        for (int64_t j = 0; j < bn * ch; j++) {
            float v = context->data[j] * g;
            context->data[j] = v;
            float av = fabsf(v); if (av > pk) pk = av;
        }
        if (pk > 0.97f) {
            float sc = 0.97f / pk;
            for (int64_t j = 0; j < bn * ch; j++) context->data[j] *= sc;
        }
    }
    if (s->cfg.hold)
        stream_add_tiled(s->context->data, s->ctx_fr, ch, s->origin + s->written - s->ctx_fr, s->held, s->held_len);
    s->chunk++;

    /* settle everything but the crossfade region, which the next chunk still blends into */
    st->start = s->emitted;
    st->frames = stream_take(s, s->origin + s->written - s->xf_fr, audio);
    return st->frames < 0 ? -1 : 0;
}

int64_t aria_stream_flush(aria_stream *s, const float **audio) {
    return stream_take(s, s->origin + s->written, audio);
}

int64_t aria_stream_seed(const aria_stream *s) { return s->seed; }

void aria_stream_set_seed(aria_stream *s, int64_t seed) {
    s->seed = s->base_seed = seed; s->evolve_k = 0;
}
void aria_stream_set_evolve(aria_stream *s, int n) { s->cfg.evolve = n > 0 ? n : 0; }
void aria_stream_set_anchor(aria_stream *s, float beta) {
    s->cfg.anchor = beta < 0.0f ? 0.0f : beta > 1.0f ? 1.0f : beta;
    s->recapture = 1;
}

void aria_stream_close(aria_stream *s) {
    if (!s) return;
    aria_audio_free(s->outp); aria_audio_free(s->context); aria_audio_free(s->anchor);
    free(s->held); free(s->out); free(s);
}
