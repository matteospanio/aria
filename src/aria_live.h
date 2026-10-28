/*
 * aria_live.h - wire protocol of aria-server live sessions (POST /stream), shared by the
 * server and aria-repl. Header-only, no I/O.
 *
 * The /stream response body is a sequence of frames, until the server closes:
 *   tag[4] | u32 little-endian payload length | payload
 *   HELO  JSON  {"session","sample_rate","channels","seed","chunk","context","ahead",...}
 *   AUDI  u64le start_frame | f32le interleaved audio (contiguous across AUDI frames)
 *   EVNT  JSON  one per chunk: {"chunk","at","end","gen_ms","rtf","ahead_ms","seed",
 *               "ctl","prompt_id","steer"} -- `at` is the first frame the chunk's controls
 *               (every control up to `ctl`) can be heard
 *   DONE  JSON  {"reason","frames"}       ERRO  JSON  {"error"}
 * Unknown tags are skipped. Audio and lengths are little-endian (every supported target).
 *
 * Pacing is by credit, not TCP backpressure: the client reports the frames it has played
 * (POST /stream/control {"session","played"}) and the server keeps at most `ahead` frames
 * generated past that.
 */
#ifndef ARIA_LIVE_H
#define ARIA_LIVE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ARIA_LIVE_MAX_FRAME (16u << 20)

static inline void live_hdr(unsigned char h[8], const char *tag, uint32_t len) {
    memcpy(h, tag, 4);
    for (int i = 0; i < 4; i++) h[4 + i] = (unsigned char)(len >> (8 * i));
}

/* Parse one frame from b[0..n). Returns the bytes it spans (header + payload), 0 when
 * more bytes are needed, -1 when the length is over `max` (a corrupt stream). */
static inline long live_parse(const unsigned char *b, size_t n, size_t max,
                              char tag[5], const unsigned char **payload, uint32_t *len) {
    if (n < 8) return 0;
    uint32_t l = (uint32_t)b[4] | (uint32_t)b[5] << 8 | (uint32_t)b[6] << 16 | (uint32_t)b[7] << 24;
    if (l > max) return -1;
    if (n < 8 + (size_t)l) return 0;
    memcpy(tag, b, 4); tag[4] = 0;
    *payload = b + 8; *len = l;
    return (long)(8 + l);
}

/* does a session need its next chunk? sent/played in frames; ahead 0 = free-run, then
 * only the outbox bound (queued vs qmax bytes) holds it back */
static inline int live_gate(int64_t sent, int64_t played, int64_t ahead, size_t queued, size_t qmax) {
    return queued < qmax && (ahead <= 0 || sent - played < ahead);
}

/* the first frame a control sent now can affect: the region after what is already
 * finalized, or one chunk later when the next chunk is already being generated */
static inline int64_t live_lands(int64_t sent, int busy, int64_t emit_fr) {
    return busy ? sent + emit_fr : sent;
}

#endif /* ARIA_LIVE_H */
