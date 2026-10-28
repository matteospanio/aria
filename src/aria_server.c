/*
 * aria_server.c - Minimal resident HTTP server for the aria runtime (E13.2).
 *
 * The ds4-server pattern at aria scale: each client connection is handled by a
 * small blocking thread that parses one HTTP request, then queues a job to the
 * SINGLE worker thread that owns the aria_ctx (model weights, GPU state, the
 * per-process caches: text-embed, device DiT/decoder, CUDA graph). That keeps
 * all model state in one place, serializes generation, and amortizes model
 * load + weight upload + per-prompt T5 encoding across every request -- the
 * same wins as --batch, but long-lived and remote-driveable.
 *
 *   aria-server -m <model_dir> [--port 8321] [--bind 127.0.0.1] [--log file]
 *               [--device cpu|cuda|auto] [-s steps] [-d seconds]
 *
 * API (JSON in, WAV out; flat keys -- aria_json is a key extractor):
 *   GET  /health            -> {"ok":true}
 *   GET  /info              -> model type / sample rate / channels
 *   GET  /steers            -> the named steering directions loaded with --steer-dir
 *   POST /stream            -> a live session: framed audio + events (see aria_live.h)
 *   POST /stream/control    -> change a live session: prompt, steer, seed, ahead, ...
 *   POST /generate          -> audio/wav (float32)
 *     {"prompt":"...", "seconds":10, "steps":8, "seed":11, "steer":"sweet:0.3,sour:0.1"}
 *     named steering is optional; the older single-steer form still works:
 *     {"steer_dir":"/path/sweet.atns", "steer_layer":16, "steer_scale":0.3,
 *      "steer_site":"residual", "steer_window":"0-7", "steer_op":"add"}
 */

#include "aria.h"
#include "aria_json.h"
#include "aria_live.h"
#include "aria_net.h"
#include "aria_parity.h"
#include "aria_stream.h"

#include <dirent.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_BODY (1u << 20)          /* 1 MB request cap */
#define IO_TIMEOUT_MS 30000

static volatile sig_atomic_t g_stop = 0;
static volatile aria_sock g_listen_fd = ARIA_SOCK_BAD;

static void on_sigint(int sig) {
    (void)sig;
    if (g_stop) _exit(130);
    g_stop = 1;
    if (g_listen_fd != ARIA_SOCK_BAD) aria_sock_wake(g_listen_fd);   /* unblocks accept() in main */
}

/* ---------------- log: one timestamped line per event, tagged with a request id -------- */

static FILE *g_log;
static pthread_mutex_t g_idmu = PTHREAD_MUTEX_INITIALIZER;
static unsigned g_nreq;

static void logline(const char *id, const char *fmt, ...) {
    char line[1024], ts[32];
    struct timespec t; clock_gettime(CLOCK_REALTIME, &t);
    time_t sec = t.tv_sec; struct tm tm;
    localtime_r(&sec, &tm);
    strftime(ts, sizeof ts, "%Y-%m-%d %H:%M:%S", &tm);
    int n = snprintf(line, sizeof line, "%s.%03ld %s%s%s", ts, t.tv_nsec / 1000000L,
                     id ? "[" : "", id ? id : "", id ? "] " : "");
    va_list ap; va_start(ap, fmt);
    if (n > 0 && n < (int)sizeof line) n += vsnprintf(line + n, sizeof line - (size_t)n, fmt, ap);
    va_end(ap);
    if (n >= (int)sizeof line - 1) n = (int)sizeof line - 2;
    line[n] = '\n'; line[n + 1] = 0;
    fputs(line, g_log);   /* one call per line: concurrent lines never interleave */
}

/* ---------------- job queue: connection threads enqueue, one worker owns ctx ------------ */

typedef struct job {
    /* request (owned) */
    char id[16];
    char *prompt;
    float seconds;
    int steps;
    long long seed;
    aria_steer steer;               /* legacy single steer: steer.dir != NULL iff requested */
    aria_steer named[32]; int n_named;   /* named steers (dirs borrowed from g_steers) */
    /* response (owned by the job until the connection thread takes it) */
    void *wav; size_t wav_len;
    char err[256];
    int done;
    double t_queued;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct job *next;
} job;

/* q.mu guards the job queue AND all live-session state below: one lock, one condvar the
 * worker sleeps on (jobs, controls, credits and disconnects all signal it) */
static struct {
    job *head, *tail; int n;
    pthread_mutex_t mu;
    pthread_cond_t cv;
} q = { NULL, NULL, 0, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER };

static int g_max_queue = 32, g_max_sessions = 1;
static double g_max_wait = 10.0;      /* a /generate job waits at most this long for live sessions */
static double g_job_rate = 1.0;       /* EMA: generate seconds per second of audio (clips and chunks) */
static int g_rate_n;
static void rate_update(double r) { g_job_rate = g_rate_n++ ? 0.7 * g_job_rate + 0.3 * r : r; }   /* q.mu held */

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

typedef struct {
    aria_ctx *ctx;
    aria_device device;
    int default_steps;
    float default_seconds;
} worker_cfg;

static void run_job(const worker_cfg *w, job *j) {
    {
        aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
        p.prompt = j->prompt;
        p.seconds_total = j->seconds > 0 ? j->seconds : w->default_seconds;
        p.steps = j->steps > 0 ? j->steps : w->default_steps;
        p.seed = j->seed;
        p.device = w->device;
        aria_steer_set set = { &j->steer, 1 }, nset = { j->named, j->n_named };
        if (j->steer.dir) p.steer = &set;
        else if (j->n_named) p.steer = &nset;

        aria_audio *audio = NULL;
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int rc = aria_generate(w->ctx, &p, &audio);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        char pe[128]; aria_json_escape(j->prompt, pe, sizeof pe);
        if (rc != 0 || !audio) {
            snprintf(j->err, sizeof j->err, "%s", aria_last_error());
            logline(j->id, "generate \"%s\" failed after %.2fs: %s", pe, dt, j->err);
        } else {
            if (aria_wav_to_mem(audio, 32, &j->wav, &j->wav_len) != 0)
                snprintf(j->err, sizeof j->err, "wav encode failed");
            aria_audio_free(audio);
            char stp[16] = "default";
            if (p.steps > 0) snprintf(stp, sizeof stp, "%d", p.steps);
            logline(j->id, "generate \"%s\" %.1fs steps=%s seed=%lld%s -> %.2fs", pe,
                    p.seconds_total, stp, (long long)p.seed, p.steer ? " steered" : "", dt);
        }
        if (rc == 0 && p.seconds_total > 0) { pthread_mutex_lock(&q.mu); rate_update(dt / p.seconds_total); pthread_mutex_unlock(&q.mu); }
        pthread_mutex_lock(&j->mu);
        j->done = 1;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
}

/* ---------------- tiny HTTP ------------------------------------------------------------ */

static void respond(aria_sock fd, int code, const char *status, const char *ctype,
                    const void *body, size_t blen) {
    char hdr[256];
    int n = snprintf(hdr, sizeof hdr,
                     "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                     "Connection: close\r\n\r\n", code, status, ctype, blen);
    if (aria_send_all(fd, hdr, (size_t)n) == 0 && blen) aria_send_all(fd, body, blen);
}

static void respond_json(aria_sock fd, int code, const char *status, const char *json) {
    respond(fd, code, status, "application/json", json, strlen(json));
}

/* {"error":"<msg>"} with the message escaped, logged under the request id */
static void respond_err(aria_sock fd, const char *id, int code, const char *status, const char *msg) {
    char e[300], body[340];
    aria_json_escape(msg, e, sizeof e);
    snprintf(body, sizeof body, "{\"error\":\"%s\"}", e);
    logline(id, "%d %s", code, e);
    respond_json(fd, code, status, body);
}

/* read the full request (headers + Content-Length body) into a NUL-terminated buffer;
 * *body points into it. Returns malloc'd buffer or NULL; *nread = bytes received
 * (0 = the peer connected and closed, e.g. a readiness probe). */
static char *read_request(aria_sock fd, char **body, size_t *nread) {
    size_t cap = 8192, len = 0;
    *nread = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    char *hdr_end = NULL;
    while (!hdr_end) {
        if (len + 1 >= cap) { cap *= 2; if (cap > MAX_BODY + 8192) { free(buf); return NULL; }
                              char *nb = realloc(buf, cap); if (!nb) { free(buf); return NULL; } buf = nb; }
        long k = aria_recv(fd, buf + len, cap - len - 1);
        if (k <= 0) { free(buf); return NULL; }
        len += (size_t)k; buf[len] = 0; *nread = len;
        hdr_end = strstr(buf, "\r\n\r\n");
    }
    size_t body_off = (size_t)(hdr_end - buf) + 4;
    size_t clen = 0;
    const char *cl = aria_strcasestr(buf, "\nContent-Length:");   /* line-anchored: match the header, not a substring */
    if (cl) clen = (size_t)strtoul(cl + 16, NULL, 10);
    if (clen > MAX_BODY) { free(buf); return NULL; }
    while (len - body_off < clen) {
        if (len + 1 >= cap) { cap = body_off + clen + 1;
                              char *nb = realloc(buf, cap); if (!nb) { free(buf); return NULL; } buf = nb; }
        long k = aria_recv(fd, buf + len, cap - len - 1);
        if (k <= 0) { free(buf); return NULL; }
        len += (size_t)k; buf[len] = 0;
    }
    *body = buf + body_off;
    return buf;
}

/* ---------------- request handling ------------------------------------------------------ */

static aria_ctx *g_ctx;   /* introspection only (worker owns generation) */

/* ---------------- named steering directions (--steer-dir) -------------------------------
 * Loaded once at startup into a fixed table that is read-only afterwards (no locks):
 * clients steer by name ("sweet:0.3") and never send file paths. */

#define MAX_STEERS 32
typedef struct { char name[48]; aria_steer st; } named_steer;   /* st.scale unused (per request) */
static named_steer g_steers[MAX_STEERS];
static int g_nsteers;

static int steer_cmp(const void *a, const void *b) {
    return strcmp(((const named_steer *)a)->name, ((const named_steer *)b)->name);
}

/* every <name>.atns in `dir`; the site/layer come from dirs_meta.json (the Hugging Face
 * matteospanio/aria layout): an entry with "layer" is a residual steer at that layer,
 * otherwise a _latent / _text suffix selects the latent / text-conditioning site. */
static void load_steer_dir(const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/dirs_meta.json", dir);
    char *meta = aria_read_file(path, NULL);
    DIR *d = opendir(dir);
    if (!d) { logline(NULL, "steer-dir %s: cannot open", dir); free(meta); return; }
    struct dirent *e;
    while ((e = readdir(d))) {
        size_t n = strlen(e->d_name);
        if (n < 6 || n - 5 >= sizeof g_steers[0].name || strcmp(e->d_name + n - 5, ".atns")) continue;
        char name[48]; memcpy(name, e->d_name, n - 5); name[n - 5] = 0;
        if (strspn(name, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != n - 5) {
            logline(NULL, "steer-dir %s: skip %s (name must be [A-Za-z0-9_-])", dir, e->d_name); continue;
        }
        int dup = 0;
        for (int i = 0; i < g_nsteers; i++) dup |= !strcmp(g_steers[i].name, name);
        if (dup || g_nsteers == MAX_STEERS) {
            logline(NULL, "steer-dir %s: skip %s (%s)", dir, name, dup ? "duplicate name" : "table full"); continue;
        }
        const char *o, *oe; double layer = -1;
        int has_meta = meta && aria_json_object(meta, NULL, name, &o, &oe) == 0;
        if (has_meta) aria_json_get_number_in(o, oe, "layer", &layer);
        aria_steer_site site;
        if (layer >= 0) site = ARIA_STEER_RESIDUAL;
        else if (n - 5 > 7 && !strcmp(name + n - 5 - 7, "_latent")) site = ARIA_STEER_LATENT;
        else if (n - 5 > 5 && !strcmp(name + n - 5 - 5, "_text")) site = ARIA_STEER_COND;
        else { logline(NULL, "steer-dir %s: skip %s (no layer in dirs_meta.json, no _latent/_text suffix)", dir, name); continue; }
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        aria_parity_tensor t;
        if (aria_parity_load(path, &t) != 0) { logline(NULL, "steer-dir %s: cannot load %s", dir, e->d_name); continue; }
        float *v = malloc((size_t)t.numel * sizeof(float));
        if (!v) { aria_parity_free(&t); continue; }
        memcpy(v, t.data, (size_t)t.numel * sizeof(float));
        double n2 = 0;
        for (int64_t c = 0; c < t.numel; c++) n2 += (double)v[c] * v[c];
        named_steer *ns = &g_steers[g_nsteers++];
        snprintf(ns->name, sizeof ns->name, "%s", name);
        ns->st.site = site; ns->st.op = ARIA_STEER_ADD; ns->st.layer = layer >= 0 ? (int)layer : 0;
        ns->st.dir = v; ns->st.dim = (int)t.numel; ns->st.dir_norm2 = (float)n2;
        ns->st.scale = 0.0f; ns->st.step_lo = 0; ns->st.step_hi = 1 << 30;
        aria_parity_free(&t);
        static const char *sn[] = {"residual", "latent", "text"};
        char at[16] = "";
        if (site == ARIA_STEER_RESIDUAL) snprintf(at, sizeof at, " L%d", (int)layer);
        logline(NULL, "steer %s: %s%s dim=%d (%s)", name, sn[site], at, ns->st.dim, dir);
    }
    closedir(d);
    free(meta);
    qsort(g_steers, (size_t)g_nsteers, sizeof g_steers[0], steer_cmp);
}

static int steer_find(const char *name, size_t len) {
    for (int i = 0; i < g_nsteers; i++)
        if (strlen(g_steers[i].name) == len && !strncmp(g_steers[i].name, name, len)) return i;
    return -1;
}

/* "name:scale[,name:scale...]" or "off" -> per-direction scales (merged into `scale`
 * when `merge`, else replacing it). Every name is checked before anything changes, so a
 * bad entry leaves `scale` untouched. Scale 0 removes a direction. */
static int parse_steer_spec(const char *spec, float *scale, int merge, char *err, size_t errlen) {
    float next[MAX_STEERS];
    int off = !strcmp(spec, "off");
    for (int i = 0; i < MAX_STEERS; i++) next[i] = merge && !off ? scale[i] : 0.0f;
    if (!off) {
        for (const char *p = spec; *p; ) {
            while (*p == ' ' || *p == ',') p++;
            if (!*p) break;
            size_t nl = strcspn(p, ":,");
            int k = steer_find(p, nl);
            if (k < 0) { snprintf(err, errlen, "unknown steer '%.*s' (GET /steers lists them)", (int)(nl < 40 ? nl : 40), p); return -1; }
            if (p[nl] != ':') { snprintf(err, errlen, "steer '%.47s' needs a scale (name:scale)", g_steers[k].name); return -1; }
            char *end; double v = strtod(p + nl + 1, &end);
            if (end == p + nl + 1) { snprintf(err, errlen, "bad scale for steer '%.47s'", g_steers[k].name); return -1; }
            next[k] = (float)v;
            p = end;
        }
    }
    memcpy(scale, next, sizeof next);
    return 0;
}

/* the steer items for `scale` (nonzero entries only) -> out[], returns the count */
static int steer_items(const float *scale, aria_steer *out) {
    int n = 0;
    for (int i = 0; i < g_nsteers; i++)
        if (scale[i] != 0.0f) { out[n] = g_steers[i].st; out[n].scale = scale[i]; n++; }
    return n;
}

static void steer_desc(const float *scale, char *out, size_t n) {
    size_t k = 0; out[0] = 0;
    for (int i = 0; i < g_nsteers && k + 64 < n; i++)
        if (scale[i] != 0.0f) k += (size_t)snprintf(out + k, n - k, "%s%s:%.2f", k ? "," : "", g_steers[i].name, scale[i]);
    if (!k) snprintf(out, n, "-");
}

/* ---------------- live sessions (POST /stream) ---------------------------------------------
 * The worker generates each session's chunks with aria_stream and queues the framed
 * output on the session's outbox; the session's connection thread writes it out. All
 * session fields are guarded by q.mu except `s` (worker only). Controls are latest-wins
 * and land on the next chunk. Two references: the worker's list and the connection. */

typedef struct outf { struct outf *next; size_t len; unsigned char b[]; } outf;

#define LIVE_SEED 1
#define LIVE_EVOLVE 2
#define LIVE_ANCHOR 4
#define OB_MAX ((size_t)2 << 20)   /* outbox bytes (~1.5 s of stereo f32) before the worker waits */

typedef struct live {
    struct live *next;
    char id[16];
    int refs;
    aria_gen_params base; aria_stream_cfg cfg;
    int sr, ch;
    int64_t ahead_fr, emit_fr;
    aria_stream *s;                       /* worker only */
    char prompt[1024]; int prompt_id;     /* controls (latest wins) */
    float scale[MAX_STEERS];
    int64_t seed; int evolve; float anchor; int dirty;
    int ctl, stop, closing;
    int busy, chunks, ended, finished;    /* progress; finished = DONE is queued */
    int64_t sent, played;
    double est_gen, t0;
    outf *oh, *ot; size_t ob_bytes;
    pthread_cond_t ob_cv;
    char reason[16];
} live;

static live *g_live;
static int g_nlive, g_nstreams;
static unsigned g_nsess;

static outf *frame_new(const char *tag, const void *pre, size_t plen, const void *pay, size_t len) {
    outf *f = malloc(sizeof *f + 8 + plen + len);
    if (!f) return NULL;
    f->next = NULL; f->len = 8 + plen + len;
    live_hdr(f->b, tag, (uint32_t)(plen + len));
    if (plen) memcpy(f->b + 8, pre, plen);
    if (len) memcpy(f->b + 8 + plen, pay, len);
    return f;
}
static outf *frame_json(const char *tag, const char *json) { return frame_new(tag, NULL, 0, json, strlen(json)); }

/* audio [start, start+n) as AUDI frames of <= 1 s, chained */
static outf *frames_audio(const float *a, int64_t start, int64_t n, int sr, int ch, outf **last) {
    outf *head = NULL, *tail = NULL;
    for (int64_t off = 0; off < n; off += sr) {
        int64_t k = n - off < sr ? n - off : sr;
        uint64_t at = (uint64_t)(start + off);
        outf *f = frame_new("AUDI", &at, 8, a + off * ch, (size_t)k * ch * sizeof(float));
        if (!f) break;
        if (tail) tail->next = f; else head = f;
        tail = f;
    }
    *last = tail;
    return head;
}

static void ob_append(live *L, outf *head, outf *tail) {   /* q.mu held */
    if (!head) return;
    for (outf *f = head; f; f = f->next) L->ob_bytes += f->len;
    if (L->ot) L->ot->next = head; else L->oh = head;
    L->ot = tail;
    pthread_cond_signal(&L->ob_cv);
}

static void live_unref(live *L) {   /* q.mu held */
    if (--L->refs) return;
    while (L->oh) { outf *f = L->oh; L->oh = f->next; free(f); }
    pthread_cond_destroy(&L->ob_cv);
    free(L);
}

static void fmt_t(char *b, size_t n, int64_t frames, int sr) {
    double t = (double)frames / sr;
    snprintf(b, n, "%02d:%05.2f", (int)(t / 60), t - 60 * (int)(t / 60));
}

/* worker: create the session's stream and announce it */
static void live_open(const worker_cfg *w, live *L) {   /* q.mu held (released inside) */
    pthread_mutex_unlock(&q.mu);
    aria_stream *s = aria_stream_open(w->ctx, &L->base, &L->cfg);
    char json[512], err[300];
    if (!s) aria_json_escape(aria_last_error(), err, sizeof err);
    pthread_mutex_lock(&q.mu);
    if (!s) {
        snprintf(json, sizeof json, "{\"error\":\"%s\"}", err);
        outf *f = frame_json("ERRO", json); ob_append(L, f, f);
        snprintf(L->reason, sizeof L->reason, "error");
        L->stop = 1;
        logline(L->id, "open failed: %s", err);
        return;
    }
    L->s = s;
    L->t0 = now_s();
    char sd[256]; steer_desc(L->scale, sd, sizeof sd);
    snprintf(json, sizeof json, "{\"session\":\"%s\",\"sample_rate\":%d,\"channels\":%d,\"seed\":%lld,"
             "\"chunk\":%.2f,\"context\":%.2f,\"ahead\":%.2f,\"xfade\":%.2f,\"prompt_id\":%d,\"steer\":\"%s\"}",
             L->id, L->sr, L->ch, (long long)aria_stream_seed(s), L->cfg.chunk_s, L->cfg.context_s,
             (double)L->ahead_fr / L->sr, ARIA_STREAM_XFADE_S, L->prompt_id, sd);
    outf *f = frame_json("HELO", json); ob_append(L, f, f);
}

/* worker: generate the session's next chunk with its current controls */
static void live_chunk(live *L) {   /* q.mu held (released while generating) */
    char prompt[1024]; float sc[MAX_STEERS];
    memcpy(prompt, L->prompt, sizeof prompt); memcpy(sc, L->scale, sizeof sc);
    int dirty = L->dirty, ctl = L->ctl, pid = L->prompt_id;
    int64_t seed = L->seed; int evolve = L->evolve; float anchor = L->anchor;
    L->dirty = 0; L->busy = 1;
    pthread_mutex_unlock(&q.mu);

    if (dirty & LIVE_SEED) aria_stream_set_seed(L->s, seed);
    if (dirty & LIVE_EVOLVE) aria_stream_set_evolve(L->s, evolve);
    if (dirty & LIVE_ANCHOR) aria_stream_set_anchor(L->s, anchor);
    aria_steer items[MAX_STEERS]; int n = steer_items(sc, items);
    aria_steer_set set = { items, n };
    const float *audio = NULL; aria_stream_stats st;
    int rc = aria_stream_next(L->s, prompt, n ? &set : NULL, &audio, &st);
    outf *head = NULL, *tail = NULL;
    char err[300] = "";
    if (rc == 0) head = frames_audio(audio, st.start, st.frames, L->sr, L->ch, &tail);
    else aria_json_escape(aria_last_error(), err, sizeof err);

    pthread_mutex_lock(&q.mu);
    L->busy = 0;
    if (rc != 0) {
        char json[400]; snprintf(json, sizeof json, "{\"error\":\"%s\"}", err);
        outf *f = frame_json("ERRO", json); ob_append(L, f, f);
        snprintf(L->reason, sizeof L->reason, "error");
        L->stop = 1;
        logline(L->id, "chunk %d failed: %s", L->chunks, err);
        return;
    }
    ob_append(L, head, tail);
    L->sent = st.start + st.frames;
    L->chunks++;
    L->est_gen = L->chunks == 1 ? st.gen_s : 0.7 * L->est_gen + 0.3 * st.gen_s;
    if (st.win_s > 0) rate_update(st.gen_s / st.win_s);
    double ahead = (double)(L->sent - L->played) / L->sr, rtf = st.gen_s > 0 ? st.emit_s / st.gen_s : 0.0;
    char sd[256], json[640], at[16];
    steer_desc(sc, sd, sizeof sd);
    snprintf(json, sizeof json, "{\"chunk\":%d,\"at\":%lld,\"end\":%lld,\"gen_ms\":%d,\"rtf\":%.2f,\"ahead_ms\":%d,"
             "\"seed\":%lld,\"ctl\":%d,\"prompt_id\":%d,\"steer\":\"%s\"}",
             st.chunk, (long long)st.start, (long long)L->sent, (int)(st.gen_s * 1000), rtf, (int)(ahead * 1000),
             (long long)st.seed, ctl, pid, sd);
    outf *f = frame_json("EVNT", json); ob_append(L, f, f);
    fmt_t(at, sizeof at, st.start, L->sr);
    logline(L->id, "chunk %d at=%s +%.2fs gen=%dms rtf=%.1fx ahead=%.2fs prompt#%d seed=%lld steer=%s",
            st.chunk, at, (double)st.frames / L->sr, (int)(st.gen_s * 1000), rtf, ahead, pid, (long long)st.seed, sd);
}

/* worker: finish a session (flush the held-back tail unless the client is gone) */
static void live_end(live *L) {   /* q.mu held (released while flushing) */
    L->ended = 1;
    if (!L->reason[0]) snprintf(L->reason, sizeof L->reason, "%s", L->closing ? "client" : L->stop ? "stop" : "shutdown");
    int flush = L->s && !L->closing;
    pthread_mutex_unlock(&q.mu);
    outf *head = NULL, *tail = NULL; int64_t start = 0, n = 0;
    if (flush) {
        const float *audio;
        n = aria_stream_flush(L->s, &audio);
        if (n > 0) head = frames_audio(audio, start = L->sent, n, L->sr, L->ch, &tail);
    }
    aria_stream_close(L->s);
    pthread_mutex_lock(&q.mu);
    L->s = NULL;
    if (n > 0) { ob_append(L, head, tail); L->sent = start + n; }
    char json[96]; snprintf(json, sizeof json, "{\"reason\":\"%s\",\"frames\":%lld}", L->reason, (long long)L->sent);
    outf *f = frame_json("DONE", json); ob_append(L, f, f);
    L->finished = 1;
    for (live **pp = &g_live; *pp; pp = &(*pp)->next) if (*pp == L) { *pp = L->next; break; }
    g_nlive--;
    char dur[16]; fmt_t(dur, sizeof dur, L->sent, L->sr);
    logline(L->id, "close reason=%s audio=%s chunks=%d", L->reason, dur, L->chunks);
    pthread_cond_broadcast(&L->ob_cv);
    live_unref(L);
}

/* the single worker: owns the model; a deadline-first scheduler over live sessions, with
 * /generate jobs in the gaps (or after --max-wait seconds regardless) */
static void *worker_main(void *arg) {
    const worker_cfg *w = (const worker_cfg *)arg;
    pthread_mutex_lock(&q.mu);
    for (;;) {
        live *L, *pick = NULL;
        for (L = g_live; L; L = L->next) if (L->stop || L->closing || g_stop) break;
        if (L) { live_end(L); continue; }
        for (L = g_live; L; L = L->next) if (!L->s) break;
        if (L) { live_open(w, L); continue; }
        double best = 1e30, min_slack = 1e30;
        for (L = g_live; L; L = L->next) {
            double slack = (double)(L->sent - L->played) / L->sr - L->est_gen;
            if (slack < min_slack) min_slack = slack;
            if (live_gate(L->sent, L->played, L->ahead_fr, L->ob_bytes, OB_MAX) && slack < best) { best = slack; pick = L; }
        }
        job *j = q.head;
        if (j && (now_s() - j->t_queued > g_max_wait ||
                  (!pick && (!g_live || min_slack > g_job_rate * (j->seconds > 0 ? j->seconds : w->default_seconds))))) {
            q.head = j->next; if (!q.head) q.tail = NULL; q.n--;
            pthread_mutex_unlock(&q.mu);
            run_job(w, j);
            pthread_mutex_lock(&q.mu);
            continue;
        }
        if (pick) { live_chunk(pick); continue; }
        if (g_stop && !g_live && !q.head) break;
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_nsec += 250000000L; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
        pthread_cond_timedwait(&q.cv, &q.mu, &ts);
    }
    pthread_mutex_unlock(&q.mu);
    return NULL;
}

static int parse_steer_json(const char *body, aria_steer *st, char *err, size_t errlen) {
    char path[1024];
    if (aria_json_get_string(body, "steer_dir", path, sizeof path) != 0) return 0; /* no steer */
    double layer = -1, scale = 0;
    if (aria_json_get_number(body, "steer_layer", &layer) != 0 ||
        aria_json_get_number(body, "steer_scale", &scale) != 0) {
        snprintf(err, errlen, "steer_dir needs steer_layer and steer_scale");
        return -1;
    }
    char site[32] = "residual", op[32] = "add", win[64] = "";
    aria_json_get_string(body, "steer_site", site, sizeof site);
    aria_json_get_string(body, "steer_op", op, sizeof op);
    aria_json_get_string(body, "steer_window", win, sizeof win);
    aria_steer_site s;
    if      (!strcmp(site, "residual")) s = ARIA_STEER_RESIDUAL;
    else if (!strcmp(site, "latent"))   s = ARIA_STEER_LATENT;   /* E12.2 host-side (CPU+GPU) */
    else if (!strcmp(site, "cond"))     s = ARIA_STEER_COND;     /* E12.4 host-side (CPU+GPU) */
    else { snprintf(err, errlen, "unknown steer_site '%s'", site); return -1; }
    aria_steer_op o = ARIA_STEER_ADD;
    if (!strcmp(op, "project") || !strcmp(op, "ablate")) o = ARIA_STEER_PROJECT;
    else if (strcmp(op, "add")) { snprintf(err, errlen, "unknown steer_op '%s'", op); return -1; }
    aria_parity_tensor t;
    if (aria_parity_load(path, &t) != 0) {
        snprintf(err, errlen, "cannot load steer direction %.200s", path);
        return -1;
    }
    float *d = malloc((size_t)t.numel * sizeof(float));
    if (!d) { aria_parity_free(&t); snprintf(err, errlen, "oom"); return -1; }
    memcpy(d, t.data, (size_t)t.numel * sizeof(float));
    int dim = (int)t.numel;
    aria_parity_free(&t);
    double n2 = 0;
    for (int c = 0; c < dim; c++) n2 += (double)d[c] * d[c];
    int lo = 0, hi = 1 << 30;
    if (win[0]) sscanf(win, "%d-%d", &lo, &hi);
    st->site = s; st->op = o; st->layer = (int)layer; st->dir = d; st->dim = dim;
    st->dir_norm2 = (float)n2; st->scale = (float)scale; st->step_lo = lo; st->step_hi = hi;
    return 1;
}

static void handle_generate(aria_sock fd, const char *id, const char *body) {
    char prompt[4096];
    if (aria_json_get_string(body, "prompt", prompt, sizeof prompt) != 0) {
        respond_err(fd, id, 400, "Bad Request", "missing 'prompt'");
        return;
    }
    job *j = calloc(1, sizeof(job));
    if (!j) { respond_err(fd, id, 500, "Internal Server Error", "oom"); return; }
    snprintf(j->id, sizeof j->id, "%s", id);
    pthread_mutex_init(&j->mu, NULL);
    pthread_cond_init(&j->cv, NULL);
    j->prompt = strdup(prompt);
    if (!j->prompt) {
        respond_err(fd, id, 500, "Internal Server Error", "oom");
        pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv); free(j);
        return;
    }
    double v = 0;
    j->seconds = aria_json_get_number(body, "seconds", &v) == 0 ? (float)v : 0.0f;
    j->steps = aria_json_get_number(body, "steps", &v) == 0 ? (int)v : 0;
    j->seed = aria_json_get_number(body, "seed", &v) == 0 ? (long long)v : -1;
    char err[256] = "", spec[512];
    if (aria_json_get_string(body, "steer", spec, sizeof spec) == 0) {
        float sc[MAX_STEERS] = {0};
        if (parse_steer_spec(spec, sc, 0, err, sizeof err) < 0) {
            respond_err(fd, id, 400, "Bad Request", err);
            pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv);
            free(j->prompt); free(j);
            return;
        }
        j->n_named = steer_items(sc, j->named);
    } else if (parse_steer_json(body, &j->steer, err, sizeof err) < 0) {
        respond_err(fd, id, 400, "Bad Request", err);
        pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv);
        free(j->prompt); free(j);
        return;
    }

    pthread_mutex_lock(&q.mu);
    if (g_stop || q.n >= g_max_queue) {
        int stopping = g_stop;
        pthread_mutex_unlock(&q.mu);
        respond_err(fd, id, 503, "Service Unavailable", stopping ? "shutting down" : "generate queue full");
        free(j->prompt); free((void *)j->steer.dir);
        pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv); free(j);
        return;
    }
    j->t_queued = now_s();
    if (q.tail) q.tail->next = j; else q.head = j;
    q.tail = j; q.n++;
    pthread_cond_signal(&q.cv);
    pthread_mutex_unlock(&q.mu);
    pthread_mutex_lock(&j->mu);
    while (!j->done) pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);

    if (j->wav) respond(fd, 200, "OK", "audio/x-wav", j->wav, j->wav_len);
    else respond_err(fd, id, 500, "Internal Server Error", j->err[0] ? j->err : "generate failed");
    free(j->wav); free(j->prompt); free((void *)j->steer.dir);
    pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv);
    free(j);
}

static const worker_cfg *g_wcfg;   /* defaults for live sessions (device, steps) */
static float g_chunk = 2.0f;       /* default session chunk (seconds), tuned by the probe */
static double g_probe_rtf;         /* the probe's predicted real-time factor at g_chunk (0 = no probe) */

/* Capacity probe (before listening): one warm-up chunk and one timed continuation chunk.
 * The cost of a chunk scales with its window (context + skip + chunk + tail), so one
 * measurement predicts the real-time factor of every chunk size: the default is the
 * smallest in the ladder reaching 1.25x real time, and --max-sessions defaults to how
 * many such streams fit. */
static void probe_capacity(aria_ctx *ctx, const worker_cfg *w, int sessions_set) {
    aria_gen_params base = ARIA_GEN_PARAMS_DEFAULT;
    base.seed = 1; base.steps = w->default_steps; base.device = w->device;
    aria_stream_cfg cfg = ARIA_STREAM_CFG_DEFAULT;
    aria_stream *s = aria_stream_open(ctx, &base, &cfg);
    const float *a; aria_stream_stats st;
    logline(NULL, "probing capacity (two %.1f s chunks)...", cfg.chunk_s);
    if (!s || aria_stream_next(s, NULL, NULL, &a, &st) || aria_stream_next(s, NULL, NULL, &a, &st)) {
        logline(NULL, "probe failed: %s (defaults: chunk %.1f s, %d session)", aria_last_error(), g_chunk, g_max_sessions);
        aria_stream_close(s);
        return;
    }
    aria_stream_close(s);
    double r = st.gen_s / st.win_s;   /* compute seconds per second of window */
    static const float ladder[] = { 2.0f, 3.0f, 4.0f, 6.0f };
    for (int i = 0; i < 4; i++) {
        float c = ladder[i];
        g_chunk = c;
        g_probe_rtf = c / (r * (cfg.context_s + ARIA_STREAM_SKIP_S + c + ARIA_STREAM_TAIL_S));
        if (g_probe_rtf >= 1.25) break;
    }
    if (!sessions_set) g_max_sessions = g_probe_rtf >= 2.5 ? (int)(g_probe_rtf / 1.25) : 1;
    if (g_probe_rtf >= 1.25)
        logline(NULL, "probe: %.1f s window in %.2f s -> chunk %.0f s at %.2fx real time, %d live session%s",
                st.win_s, st.gen_s, g_chunk, g_probe_rtf, g_max_sessions, g_max_sessions == 1 ? "" : "s");
    else if (g_probe_rtf >= 1.0)
        logline(NULL, "probe: %.1f s window in %.2f s -> real time with little headroom: %.2fx at chunk %.0f s "
                "(expect occasional buffering)", st.win_s, st.gen_s, g_probe_rtf, g_chunk);
    else
        logline(NULL, "probe: %.1f s window in %.2f s -> not real time on this device: best %.2fx at chunk %.0f s "
                "(clients will buffer; a GPU server streams in real time)", st.win_s, st.gen_s, g_probe_rtf, g_chunk);
}

/* POST /stream: open a live session, then write its frames until it ends */
static void handle_stream(aria_sock fd, const char *id, const char *body) {
    char err[256] = "", spec[512];
    double v;
    live *L = calloc(1, sizeof *L);
    if (!L) { respond_err(fd, id, 500, "Internal Server Error", "oom"); return; }
    if (aria_json_get_string(body, "prompt", L->prompt, sizeof L->prompt) != 0) {
        free(L); respond_err(fd, id, 400, "Bad Request", "missing 'prompt'"); return;
    }
    float chunk = aria_json_get_number(body, "chunk", &v) == 0 ? (float)v : g_chunk;
    float context = aria_json_get_number(body, "context", &v) == 0 ? (float)v : 6.0f;
    float ahead = aria_json_get_number(body, "ahead", &v) == 0 ? (float)v : chunk;
    if (!(chunk >= 0.5f && chunk <= 30.0f) || !(context >= 1.0f && context <= 30.0f) || !(ahead >= 0.0f && ahead <= 60.0f)) {
        free(L); respond_err(fd, id, 400, "Bad Request", "chunk must be 0.5-30 s, context 1-30 s, ahead 0-60 s"); return;
    }
    if (aria_json_get_string(body, "steer", spec, sizeof spec) == 0 &&
        parse_steer_spec(spec, L->scale, 0, err, sizeof err) < 0) {
        free(L); respond_err(fd, id, 400, "Bad Request", err); return;
    }
    aria_gen_params base = ARIA_GEN_PARAMS_DEFAULT;
    base.seed = aria_json_get_number(body, "seed", &v) == 0 ? (int64_t)v : -1;
    base.steps = aria_json_get_number(body, "steps", &v) == 0 && v > 0 ? (int)v : g_wcfg->default_steps;
    base.device = g_wcfg->device;
    aria_stream_cfg cfg = ARIA_STREAM_CFG_DEFAULT;
    cfg.chunk_s = chunk; cfg.context_s = context;
    cfg.anchor = aria_json_get_number(body, "anchor", &v) == 0 ? (float)v : 0.0f;
    cfg.evolve = aria_json_get_number(body, "evolve", &v) == 0 ? (int)v : 0;
    L->base = base; L->cfg = cfg;
    L->sr = aria_sample_rate(g_ctx); L->ch = aria_audio_channels(g_ctx);
    L->ahead_fr = (int64_t)(ahead * L->sr); L->emit_fr = (int64_t)(chunk * L->sr);
    L->prompt_id = 1; L->refs = 2;
    pthread_cond_init(&L->ob_cv, NULL);

    pthread_mutex_lock(&q.mu);
    if (g_stop || g_nlive >= g_max_sessions) {
        int stopping = g_stop;
        pthread_mutex_unlock(&q.mu);
        pthread_cond_destroy(&L->ob_cv); free(L);
        if (stopping) { respond_err(fd, id, 503, "Service Unavailable", "shutting down"); return; }
        char b[128]; snprintf(b, sizeof b, "{\"error\":\"at capacity (%d live session%s)\",\"retry_after\":5}",
                              g_max_sessions, g_max_sessions == 1 ? "" : "s");
        logline(id, "503 at capacity (%d live)", g_max_sessions);
        respond_json(fd, 503, "Service Unavailable", b);
        return;
    }
    snprintf(L->id, sizeof L->id, "s%u", ++g_nsess);
    live **pp = &g_live; while (*pp) pp = &(*pp)->next; *pp = L;
    g_nlive++; g_nstreams++;
    pthread_cond_signal(&q.cv);
    pthread_mutex_unlock(&q.mu);
    char pe[128], sd[256]; aria_json_escape(L->prompt, pe, sizeof pe); steer_desc(L->scale, sd, sizeof sd);
    logline(L->id, "open (%s) \"%s\" chunk=%.1f context=%.1f ahead=%.1f seed=%lld steer=%s",
            id, pe, chunk, context, ahead, (long long)base.seed, sd);

    static const char hdr[] = "HTTP/1.1 200 OK\r\nContent-Type: application/x-aria-stream\r\n"
                              "Cache-Control: no-store\r\nConnection: close\r\n\r\n";
    int ok = aria_send_all(fd, hdr, sizeof hdr - 1) == 0;
    unsigned char up[512]; size_t upn = 0;   /* upstream frames from the client (CRED) */
    pthread_mutex_lock(&q.mu);
    while (ok) {
        while (!L->oh && !L->finished) {
            struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 20000000L; if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            if (pthread_cond_timedwait(&L->ob_cv, &q.mu, &ts) == 0) continue;
            /* credits arrive on this connection (one-way, no new TCP connection per credit):
             * CRED {"played":frames[,"ahead":seconds]} */
            long k = aria_recv_nb(fd, up + upn, sizeof up - upn);
            if (k == 0) { ok = 0; break; }   /* client gone */
            if (k > 0) upn += (size_t)k;
            for (;;) {
                char tag[5]; const unsigned char *pl; uint32_t pn;
                long used = live_parse(up, upn, sizeof up - 8, tag, &pl, &pn);
                if (used < 0) { ok = 0; break; }
                if (!used) break;
                if (!strcmp(tag, "CRED")) {
                    char js[256]; double v;
                    size_t l = pn < sizeof js - 1 ? pn : sizeof js - 1;
                    memcpy(js, pl, l); js[l] = 0;
                    if (aria_json_get_number(js, "played", &v) == 0 && (int64_t)v > L->played) L->played = (int64_t)v;
                    if (aria_json_get_number(js, "ahead", &v) == 0 && v >= 0 && v <= 60 && (int64_t)(v * L->sr) != L->ahead_fr) {
                        L->ahead_fr = (int64_t)(v * L->sr);
                        logline(L->id, "client buffer -> %.2f s", v);
                    }
                    pthread_cond_signal(&q.cv);
                }
                memmove(up, up + used, upn - (size_t)used); upn -= (size_t)used;
            }
            if (!ok) break;
        }
        if (!ok || !L->oh) break;   /* client gone, or finished and drained */
        outf *f = L->oh;
        L->oh = f->next; if (!L->oh) L->ot = NULL;
        L->ob_bytes -= f->len;
        pthread_cond_signal(&q.cv);   /* outbox space: the worker may generate again */
        pthread_mutex_unlock(&q.mu);
        ok = aria_send_all(fd, f->b, f->len) == 0;
        free(f);
        pthread_mutex_lock(&q.mu);
    }
    if (!L->ended) { L->closing = 1; pthread_cond_signal(&q.cv); }
    g_nstreams--;
    live_unref(L);
    pthread_mutex_unlock(&q.mu);
}

/* POST /stream/control: change a live session (applies to its next chunk) and/or credit
 * the frames the client has played */
static void handle_control(aria_sock fd, const char *id, const char *body) {
    char sid[16], err[256] = "", spec[512], prompt[1024];
    double v;
    if (aria_json_get_string(body, "session", sid, sizeof sid) != 0) { respond_err(fd, id, 400, "Bad Request", "missing 'session'"); return; }
    pthread_mutex_lock(&q.mu);
    live *L = g_live;
    while (L && strcmp(L->id, sid)) L = L->next;
    if (!L || L->ended) {
        pthread_mutex_unlock(&q.mu);
        respond_err(fd, id, 404, "Not Found", "no such session");
        return;
    }
    float sc[MAX_STEERS]; memcpy(sc, L->scale, sizeof sc);
    int has_steer = aria_json_get_string(body, "steer", spec, sizeof spec) == 0;
    if (has_steer && parse_steer_spec(spec, sc, 1, err, sizeof err) < 0) {   /* validate before changing anything */
        pthread_mutex_unlock(&q.mu);
        respond_err(fd, id, 400, "Bad Request", err);
        return;
    }
    int changed = 0;
    char what[1400] = "";
    size_t wl = 0;
    if (aria_json_get_string(body, "prompt", prompt, sizeof prompt) == 0 && prompt[0]) {
        memcpy(L->prompt, prompt, sizeof prompt); L->prompt_id++; changed = 1;
        char pe[128]; aria_json_escape(prompt, pe, sizeof pe);
        wl += (size_t)snprintf(what + wl, sizeof what - wl, " prompt#%d=\"%s\"", L->prompt_id, pe);
    }
    if (has_steer) {
        memcpy(L->scale, sc, sizeof sc); changed = 1;
        char sd[256]; steer_desc(sc, sd, sizeof sd);
        wl += (size_t)snprintf(what + wl, sizeof what - wl, " steer=%s", sd);
    }
    if (aria_json_get_number(body, "seed", &v) == 0) {
        L->seed = (int64_t)v; L->dirty |= LIVE_SEED; changed = 1;
        wl += (size_t)snprintf(what + wl, sizeof what - wl, " seed=%lld", (long long)L->seed);
    }
    if (aria_json_get_number(body, "evolve", &v) == 0) {
        L->evolve = (int)v; L->dirty |= LIVE_EVOLVE; changed = 1;
        wl += (size_t)snprintf(what + wl, sizeof what - wl, " evolve=%d", L->evolve);
    }
    if (aria_json_get_number(body, "anchor", &v) == 0) {
        L->anchor = (float)v; L->dirty |= LIVE_ANCHOR; changed = 1;
        wl += (size_t)snprintf(what + wl, sizeof what - wl, " anchor=%.2f", L->anchor);
    }
    int stop = 0;
    if (aria_json_get_bool_in(body, NULL, "stop", &stop) == 0 && stop) { L->stop = 1; changed = 1; wl += (size_t)snprintf(what + wl, sizeof what - wl, " stop"); }
    if (aria_json_get_number(body, "played", &v) == 0 && (int64_t)v > L->played) L->played = (int64_t)v;
    if (changed) L->ctl++;
    int64_t lands = live_lands(L->sent, L->busy, L->emit_fr);
    char resp[160];
    snprintf(resp, sizeof resp, "{\"ok\":true,\"ctl\":%d,\"lands\":%lld,\"sent\":%lld,\"played\":%lld}",
             L->ctl, (long long)lands, (long long)L->sent, (long long)L->played);
    if (changed) { char at[16]; fmt_t(at, sizeof at, lands, L->sr); logline(L->id, "ctl#%d%s lands@%s", L->ctl, what, at); }
    pthread_cond_signal(&q.cv);
    pthread_mutex_unlock(&q.mu);
    respond_json(fd, 200, "OK", resp);
}

static void *conn_main(void *arg) {
    aria_sock fd = (aria_sock)(intptr_t)arg;
    char id[16];
    pthread_mutex_lock(&g_idmu); snprintf(id, sizeof id, "r%u", ++g_nreq); pthread_mutex_unlock(&g_idmu);
    aria_sock_timeouts(fd, IO_TIMEOUT_MS);
    char *body = NULL; size_t nread;
    char *req = read_request(fd, &body, &nread);
    if (!req) {
        if (nread) logline(id, "bad or incomplete request (closed)");
        aria_sock_close(fd); return NULL;
    }

    if (!strncmp(req, "GET /health", 11)) {
        respond_json(fd, 200, "OK", "{\"ok\":true}");
    } else if (!strncmp(req, "GET /info", 9)) {
        char info[256];
        snprintf(info, sizeof info,
                 "{\"model_type\":\"%s\",\"sample_rate\":%d,\"channels\":%d,\"tensors\":%d,"
                 "\"chunk\":%.1f,\"rtf\":%.2f,\"max_sessions\":%d}",
                 aria_model_type(g_ctx), aria_sample_rate(g_ctx),
                 aria_audio_channels(g_ctx), aria_num_tensors(g_ctx), g_chunk, g_probe_rtf, g_max_sessions);
        respond_json(fd, 200, "OK", info);
    } else if (!strncmp(req, "GET /steers", 11)) {
        static const char *sn[] = {"residual", "latent", "text"};
        char out[4096]; size_t n = (size_t)snprintf(out, sizeof out, "{\"steers\":[");
        for (int i = 0; i < g_nsteers && n < sizeof out - 128; i++)
            n += (size_t)snprintf(out + n, sizeof out - n, "%s{\"name\":\"%s\",\"site\":\"%s\",\"layer\":%d,\"dim\":%d}",
                                  i ? "," : "", g_steers[i].name, sn[g_steers[i].st.site], g_steers[i].st.layer, g_steers[i].st.dim);
        snprintf(out + n, sizeof out - n, "]}");
        respond_json(fd, 200, "OK", out);
    } else if (!strncmp(req, "POST /generate", 14)) {
        handle_generate(fd, id, body);
    } else if (!strncmp(req, "POST /stream/control", 20)) {
        handle_control(fd, id, body);
    } else if (!strncmp(req, "POST /stream", 12)) {
        handle_stream(fd, id, body);   /* lasts as long as the client listens */
    } else {
        char line[64]; size_t k = strcspn(req, "\r\n"); if (k > 48) k = 48;
        memcpy(line, req, k); line[k] = 0;
        char msg[192]; snprintf(msg, sizeof msg, "no route for '%s' (GET /health, GET /info, GET /steers, POST /generate, POST /stream, POST /stream/control)", line);
        respond_err(fd, id, 404, "Not Found", msg);
    }
    free(req);
    aria_sock_close(fd);
    return NULL;
}

/* ---------------- main ------------------------------------------------------------------ */

static void usage(FILE *f, const char *prog) {
    fprintf(f,
            "aria-server - resident HTTP server for the aria runtime\n\n"
            "Usage: %s -m <model_dir> [--port 8321] [--bind 127.0.0.1] [--log <file>] [--steer-dir <dir>]...\n"
            "          [--device cpu|cuda|auto] [-s <steps (default: the model's, 6)>] [-d <default_seconds=10>]\n\n"
            "  --bind   address to listen on (default 127.0.0.1; 0.0.0.0 = every interface, no auth)\n"
            "  --log    append the activity log to <file> (default stderr)\n"
            "  --steer-dir <dir>  load named steering directions (*.atns + dirs_meta.json, the\n"
            "           Hugging Face matteospanio/aria layout, e.g. directions/small); repeatable\n"
            "  --max-sessions <n>  concurrent live sessions (default: from the capacity probe)\n"
            "  --no-probe          skip the startup capacity probe (chunk 2 s, 1 session)\n"
            "  --queue <n>         queued /generate jobs before 503 (default 32)\n"
            "  --max-wait <s>      longest a /generate job yields to live sessions (default 10)\n", prog);
}

#ifndef ARIA_SERVER_NO_MAIN   /* tests/test_server.c includes this file for its helpers */
int main(int argc, char **argv) {
    const char *model_dir = NULL, *bind_ip = "127.0.0.1", *log_path = NULL;
    const char *steer_dirs[8]; int n_steer_dirs = 0, probe = 1, sessions_set = 0;
    int port = 8321;
    worker_cfg w = { NULL, ARIA_DEVICE_AUTO, 0, 10.0f };   /* steps 0 = the model's default */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bind") && i + 1 < argc) bind_ip = argv[++i];
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) log_path = argv[++i];
        else if (!strcmp(argv[i], "--steer-dir") && i + 1 < argc && n_steer_dirs < 8) steer_dirs[n_steer_dirs++] = argv[++i];
        else if (!strcmp(argv[i], "--max-sessions") && i + 1 < argc) { g_max_sessions = atoi(argv[++i]); sessions_set = 1; }
        else if (!strcmp(argv[i], "--no-probe")) probe = 0;
        else if (!strcmp(argv[i], "--queue") && i + 1 < argc) g_max_queue = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-wait") && i + 1 < argc) g_max_wait = atof(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(stdout, argv[0]); return 0; }
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) w.default_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) w.default_seconds = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            const char *d = argv[++i];
            w.device = !strcmp(d, "cpu") ? ARIA_DEVICE_CPU
                     : (!strcmp(d, "cuda") || !strcmp(d, "gpu")) ? ARIA_DEVICE_CUDA : ARIA_DEVICE_AUTO;
        } else { usage(stderr, argv[0]); return 1; }
    }
    if (!model_dir) { usage(stderr, argv[0]); return 1; }
    if (isatty(2)) fprintf(stderr, "\n%s  live music generation server\n\n", aria_banner());
    g_log = stderr;
    if (log_path && !(g_log = fopen(log_path, "a"))) { fprintf(stderr, "aria-server: cannot open log %s\n", log_path); return 1; }
    setvbuf(g_log, NULL, _IOLBF, 0);
    if (aria_net_init() != 0) { logline(NULL, "network init failed"); return 1; }

    logline(NULL, "loading %s", model_dir);
    g_ctx = aria_load(model_dir);
    if (!g_ctx) { logline(NULL, "load error: %s", aria_last_error()); return 1; }
    w.ctx = g_ctx;
    g_wcfg = &w;
    logline(NULL, "model type=%s sr=%d ch=%d", aria_model_type(g_ctx), aria_sample_rate(g_ctx), aria_audio_channels(g_ctx));
    for (int i = 0; i < n_steer_dirs; i++) load_steer_dir(steer_dirs[i]);
    if (probe) probe_capacity(g_ctx, &w, sessions_set);

    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    signal(SIGPIPE, SIG_IGN);

    aria_sock lfd = aria_listen(bind_ip, port);
    if (lfd == ARIA_SOCK_BAD) { logline(NULL, "cannot listen on %s:%d", bind_ip, port); return 1; }
    g_listen_fd = lfd;
    logline(NULL, "listening on %s:%d", bind_ip, port);

    pthread_t worker;
    pthread_create(&worker, NULL, worker_main, &w);

    while (!g_stop) {
        aria_sock cfd = aria_accept(lfd);
        if (cfd == ARIA_SOCK_BAD) { if (g_stop) break; continue; }
        pthread_t th;
        if (pthread_create(&th, NULL, conn_main, (void *)(intptr_t)cfd) == 0) pthread_detach(th);
        else aria_sock_close(cfd);
    }
    g_stop = 1;
    aria_sock_close(lfd);
    pthread_mutex_lock(&q.mu); pthread_cond_broadcast(&q.cv); pthread_mutex_unlock(&q.mu);
    pthread_join(worker, NULL);
    for (int i = 0; i < 20; i++) {   /* let stream connections send their DONE */
        pthread_mutex_lock(&q.mu); int n = g_nstreams; pthread_mutex_unlock(&q.mu);
        if (!n) break;
        struct timespec ts = { 0, 50000000L }; nanosleep(&ts, NULL);
    }
    aria_free(g_ctx);
    logline(NULL, "bye");
    return 0;
}
#endif
