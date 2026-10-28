/*
 * aria_stream.h - endless sliding-window continuation ("streaming"), shared by the
 * CLI (--stream) and aria-server live sessions.
 *
 * Each chunk regenerates context + skip + emit + tail seconds conditioned on the last
 * `context_s` of output (continue mode), emits only the strong body (the ~1.5 s seam
 * after the context and the ~3 s outro fade are discarded) and stitches it on with a
 * phase-aligned crossfade. The session keeps only the tail it still needs, so a stream
 * can run for hours in constant memory. No threads, no I/O: the caller decides when to
 * generate, and what to do with the finalized audio.
 */
#ifndef ARIA_STREAM_H
#define ARIA_STREAM_H

#include <stdint.h>
#include "aria.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARIA_STREAM_SKIP_S   1.5f    /* post-context seam, regenerated but never emitted */
#define ARIA_STREAM_TAIL_S   3.0f    /* SA3's outro fade, regenerated but never emitted */
#define ARIA_STREAM_XFADE_S  0.25f   /* crossfade onto the previous chunk */
#define ARIA_STREAM_MAXLAG_S 0.18f   /* phase-align search range (< one beat) */

typedef struct {
    float chunk_s, context_s;   /* seconds emitted per chunk / rolling context */
    float anchor;               /* E14b re-anchoring strength 0..1 (0 = off) */
    int   evolve;               /* E14c: fresh seed every n chunks (0 = fixed seed) */
    int   hold;                 /* hold the chunk-0 HPSS drum loop under the harmonics */
    /* test seam: NULL = aria_generate(ctx, ...). With ctx NULL, sample_rate/channels
     * describe the audio the seam produces. */
    int (*generate)(void *user, const aria_gen_params *p, aria_audio **out);
    void *user;
    int sample_rate, channels;
} aria_stream_cfg;
#define ARIA_STREAM_CFG_DEFAULT { 2.0f, 6.0f, 0.0f, 0, 0, NULL, NULL, 0, 0 }

typedef struct {
    int     chunk;              /* 0-based index of the chunk just generated */
    int64_t start, frames;      /* the finalized audio returned: timeline [start, start+frames) */
    float   emit_s, win_s;      /* seconds emitted / generated window */
    double  gen_s;              /* generate wall time */
    int64_t seed;               /* seed this chunk used */
    int     reseeded, hold_set, anchor_set;   /* events, for the caller's log (the lib is quiet) */
} aria_stream_stats;

typedef struct aria_stream aria_stream;

/* `base` is copied (prompt/steer/seed come from it unless overridden per chunk); a
 * negative base->seed draws a random one (see aria_stream_seed). NULL +
 * aria_last_error() on bad config or OOM. */
aria_stream *aria_stream_open(aria_ctx *ctx, const aria_gen_params *base, const aria_stream_cfg *cfg);
/* Generate the next chunk with `prompt` (NULL = uncond) and `steer` (borrowed for the
 * call, NULL = none). *audio points at the newly finalized interleaved frames (owned by
 * the session, valid until the next call); the last crossfade region is held back until
 * the next chunk or aria_stream_flush. 0 / <0 (aria_last_error). */
int aria_stream_next(aria_stream *s, const char *prompt, const aria_steer_set *steer,
                     const float **audio, aria_stream_stats *st);
/* Finalize the held-back tail (end of stream); returns its frame count. */
int64_t aria_stream_flush(aria_stream *s, const float **audio);
int64_t aria_stream_seed(const aria_stream *s);
void aria_stream_set_seed(aria_stream *s, int64_t seed);   /* next chunk; restarts --evolve */
void aria_stream_set_evolve(aria_stream *s, int n);
void aria_stream_set_anchor(aria_stream *s, float beta);   /* re-captures the reference next chunk */
/* emit `chunk_s` seconds per chunk from the next chunk on (a slow machine streams in real
 * time with longer chunks: the context/seam/tail overhead per chunk is fixed). <0 = OOM. */
int aria_stream_set_chunk(aria_stream *s, float chunk_s);
void aria_stream_close(aria_stream *s);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_STREAM_H */
