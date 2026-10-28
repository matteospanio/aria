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
 *   POST /generate          -> audio/wav (float32)
 *     {"prompt":"...", "seconds":10, "steps":8, "seed":11, "steer":"sweet:0.3,sour:0.1"}
 *     named steering is optional; the older single-steer form still works:
 *     {"steer_dir":"/path/sweet.atns", "steer_layer":16, "steer_scale":0.3,
 *      "steer_site":"residual", "steer_window":"0-7", "steer_op":"add"}
 */

#include "aria.h"
#include "aria_json.h"
#include "aria_net.h"
#include "aria_parity.h"

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
    if (g_listen_fd != ARIA_SOCK_BAD) { aria_sock fd = g_listen_fd; g_listen_fd = ARIA_SOCK_BAD; aria_sock_close(fd); }
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
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct job *next;
} job;

static struct {
    job *head, *tail;
    pthread_mutex_t mu;
    pthread_cond_t cv;
} q = { NULL, NULL, PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER };

static void q_push(job *j) {
    pthread_mutex_lock(&q.mu);
    if (q.tail) q.tail->next = j; else q.head = j;
    q.tail = j;
    pthread_cond_signal(&q.cv);
    pthread_mutex_unlock(&q.mu);
}

static job *q_pop_wait(void) {
    pthread_mutex_lock(&q.mu);
    while (!q.head && !g_stop) {
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += 1;
        pthread_cond_timedwait(&q.cv, &q.mu, &ts);
    }
    job *j = q.head;
    if (j) { q.head = j->next; if (!q.head) q.tail = NULL; }
    pthread_mutex_unlock(&q.mu);
    return j;
}

typedef struct {
    aria_ctx *ctx;
    aria_device device;
    int default_steps;
    float default_seconds;
} worker_cfg;

static void *worker_main(void *arg) {
    worker_cfg *w = (worker_cfg *)arg;
    for (;;) {
        job *j = q_pop_wait();
        if (!j) { if (g_stop) return NULL; continue; }
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

    q_push(j);
    pthread_mutex_lock(&j->mu);
    while (!j->done) pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);

    if (j->wav) respond(fd, 200, "OK", "audio/x-wav", j->wav, j->wav_len);
    else respond_err(fd, id, 500, "Internal Server Error", j->err[0] ? j->err : "generate failed");
    free(j->wav); free(j->prompt); free((void *)j->steer.dir);
    pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv);
    free(j);
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
                 "{\"model_type\":\"%s\",\"sample_rate\":%d,\"channels\":%d,\"tensors\":%d}",
                 aria_model_type(g_ctx), aria_sample_rate(g_ctx),
                 aria_audio_channels(g_ctx), aria_num_tensors(g_ctx));
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
    } else {
        char line[64]; size_t k = strcspn(req, "\r\n"); if (k > 48) k = 48;
        memcpy(line, req, k); line[k] = 0;
        char msg[192]; snprintf(msg, sizeof msg, "no route for '%s' (GET /health, GET /info, GET /steers, POST /generate)", line);
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
            "           Hugging Face matteospanio/aria layout, e.g. directions/small); repeatable\n", prog);
}

#ifndef ARIA_SERVER_NO_MAIN   /* tests/test_server.c includes this file for its helpers */
int main(int argc, char **argv) {
    const char *model_dir = NULL, *bind_ip = "127.0.0.1", *log_path = NULL;
    const char *steer_dirs[8]; int n_steer_dirs = 0;
    int port = 8321;
    worker_cfg w = { NULL, ARIA_DEVICE_AUTO, 0, 10.0f };   /* steps 0 = the model's default */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bind") && i + 1 < argc) bind_ip = argv[++i];
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) log_path = argv[++i];
        else if (!strcmp(argv[i], "--steer-dir") && i + 1 < argc && n_steer_dirs < 8) steer_dirs[n_steer_dirs++] = argv[++i];
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
    g_log = stderr;
    if (log_path && !(g_log = fopen(log_path, "a"))) { fprintf(stderr, "aria-server: cannot open log %s\n", log_path); return 1; }
    setvbuf(g_log, NULL, _IOLBF, 0);
    if (aria_net_init() != 0) { logline(NULL, "network init failed"); return 1; }

    logline(NULL, "loading %s", model_dir);
    g_ctx = aria_load(model_dir);
    if (!g_ctx) { logline(NULL, "load error: %s", aria_last_error()); return 1; }
    w.ctx = g_ctx;
    logline(NULL, "model type=%s sr=%d ch=%d", aria_model_type(g_ctx), aria_sample_rate(g_ctx), aria_audio_channels(g_ctx));
    for (int i = 0; i < n_steer_dirs; i++) load_steer_dir(steer_dirs[i]);

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
        if (cfd == ARIA_SOCK_BAD) { if (g_stop || g_listen_fd == ARIA_SOCK_BAD) break; continue; }
        pthread_t th;
        if (pthread_create(&th, NULL, conn_main, (void *)(intptr_t)cfd) == 0) pthread_detach(th);
        else aria_sock_close(cfd);
    }
    g_stop = 1;
    pthread_cond_broadcast(&q.cv);
    pthread_join(worker, NULL);
    aria_free(g_ctx);
    logline(NULL, "bye");
    return 0;
}
#endif
