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
 *   POST /generate          -> audio/wav (float32)
 *     {"prompt":"...", "seconds":10, "steps":8, "seed":11,
 *      "steer_dir":"/path/sweet.atns", "steer_layer":16, "steer_scale":0.3,
 *      "steer_site":"residual", "steer_window":"0-7", "steer_op":"add"}
 *     steering is optional: present iff steer_dir is set (single steer/request).
 */

#include "aria.h"
#include "aria_json.h"
#include "aria_net.h"
#include "aria_parity.h"

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
    aria_steer steer;               /* steer.dir != NULL iff steering requested */
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
        aria_steer_set set = { &j->steer, 1 };
        if (j->steer.dir) p.steer = &set;

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
            logline(j->id, "generate \"%s\" %.1fs steps=%d seed=%lld%s -> %.2fs", pe,
                    p.seconds_total, p.steps, (long long)p.seed, j->steer.dir ? " steered" : "", dt);
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
    char err[256] = "";
    if (parse_steer_json(body, &j->steer, err, sizeof err) < 0) {
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
    } else if (!strncmp(req, "POST /generate", 14)) {
        handle_generate(fd, id, body);
    } else {
        char line[64]; size_t k = strcspn(req, "\r\n"); if (k > 48) k = 48;
        memcpy(line, req, k); line[k] = 0;
        char msg[128]; snprintf(msg, sizeof msg, "no route for '%s' (GET /health, GET /info, POST /generate)", line);
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
            "Usage: %s -m <model_dir> [--port 8321] [--bind 127.0.0.1] [--log <file>]\n"
            "          [--device cpu|cuda|auto] [-s <steps (default: the model's, 6)>] [-d <default_seconds=10>]\n\n"
            "  --bind   address to listen on (default 127.0.0.1; 0.0.0.0 = every interface, no auth)\n"
            "  --log    append the activity log to <file> (default stderr)\n", prog);
}

int main(int argc, char **argv) {
    const char *model_dir = NULL, *bind_ip = "127.0.0.1", *log_path = NULL;
    int port = 8321;
    worker_cfg w = { NULL, ARIA_DEVICE_AUTO, 0, 10.0f };   /* steps 0 = the model's default */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bind") && i + 1 < argc) bind_ip = argv[++i];
        else if (!strcmp(argv[i], "--log") && i + 1 < argc) log_path = argv[++i];
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
