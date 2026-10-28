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
 *   aria-server -m <model_dir> [--port 8321] [--device cpu|cuda|auto] [-s steps]
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
#include "aria_parity.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define MAX_BODY (1u << 20)          /* 1 MB request cap */
#define IO_TIMEOUT_SEC 30

static volatile sig_atomic_t g_stop = 0;
static volatile int g_listen_fd = -1;

static void on_sigint(int sig) {
    (void)sig;
    if (g_stop) _exit(130);
    g_stop = 1;
    if (g_listen_fd >= 0) { int fd = g_listen_fd; g_listen_fd = -1; close(fd); }
}

/* ---------------- job queue: connection threads enqueue, one worker owns ctx ------------ */

typedef struct job {
    /* request (owned) */
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
        if (rc != 0 || !audio) {
            snprintf(j->err, sizeof j->err, "%s", aria_last_error());
        } else {
            if (aria_wav_to_mem(audio, 32, &j->wav, &j->wav_len) != 0)
                snprintf(j->err, sizeof j->err, "wav encode failed");
            aria_audio_free(audio);
            fprintf(stderr, "[serve] \"%.48s\" %.1fs/%d steps seed=%lld%s -> %.2fs\n",
                    j->prompt, p.seconds_total, p.steps, (long long)p.seed,
                    j->steer.dir ? " steered" : "",
                    (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
        }
        pthread_mutex_lock(&j->mu);
        j->done = 1;
        pthread_cond_signal(&j->cv);
        pthread_mutex_unlock(&j->mu);
    }
}

/* ---------------- tiny HTTP ------------------------------------------------------------ */

static int send_all(int fd, const void *buf, size_t n) {
    const char *p = buf;
    while (n) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}

static void respond(int fd, int code, const char *status, const char *ctype,
                    const void *body, size_t blen) {
    char hdr[256];
    int n = snprintf(hdr, sizeof hdr,
                     "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                     "Connection: close\r\n\r\n", code, status, ctype, blen);
    if (send_all(fd, hdr, (size_t)n) == 0 && blen) send_all(fd, body, blen);
}

static void respond_json(int fd, int code, const char *status, const char *json) {
    respond(fd, code, status, "application/json", json, strlen(json));
}

/* read the full request (headers + Content-Length body) into a NUL-terminated buffer;
 * *body points into it. Returns malloc'd buffer or NULL. */
static char *read_request(int fd, char **body) {
    size_t cap = 8192, len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    char *hdr_end = NULL;
    while (!hdr_end) {
        if (len + 1 >= cap) { cap *= 2; if (cap > MAX_BODY + 8192) { free(buf); return NULL; }
                              char *nb = realloc(buf, cap); if (!nb) { free(buf); return NULL; } buf = nb; }
        ssize_t k = recv(fd, buf + len, cap - len - 1, 0);
        if (k <= 0) { free(buf); return NULL; }
        len += (size_t)k; buf[len] = 0;
        hdr_end = strstr(buf, "\r\n\r\n");
    }
    size_t body_off = (size_t)(hdr_end - buf) + 4;
    size_t clen = 0;
    const char *cl = strcasestr(buf, "\nContent-Length:");   /* line-anchored: match the header, not a substring */
    if (cl) clen = (size_t)strtoul(cl + 16, NULL, 10);
    if (clen > MAX_BODY) { free(buf); return NULL; }
    while (len - body_off < clen) {
        if (len + 1 >= cap) { cap = body_off + clen + 1;
                              char *nb = realloc(buf, cap); if (!nb) { free(buf); return NULL; } buf = nb; }
        ssize_t k = recv(fd, buf + len, cap - len - 1, 0);
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

static void handle_generate(int fd, const char *body) {
    char prompt[4096];
    if (aria_json_get_string(body, "prompt", prompt, sizeof prompt) != 0) {
        respond_json(fd, 400, "Bad Request", "{\"error\":\"missing 'prompt'\"}");
        return;
    }
    job *j = calloc(1, sizeof(job));
    if (!j) { respond_json(fd, 500, "Internal Server Error", "{\"error\":\"oom\"}"); return; }
    pthread_mutex_init(&j->mu, NULL);
    pthread_cond_init(&j->cv, NULL);
    j->prompt = strdup(prompt);
    if (!j->prompt) {
        respond_json(fd, 500, "Internal Server Error", "{\"error\":\"oom\"}");
        pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv); free(j);
        return;
    }
    double v = 0;
    j->seconds = aria_json_get_number(body, "seconds", &v) == 0 ? (float)v : 0.0f;
    j->steps = aria_json_get_number(body, "steps", &v) == 0 ? (int)v : 0;
    j->seed = aria_json_get_number(body, "seed", &v) == 0 ? (long long)v : -1;
    char err[256] = "";
    if (parse_steer_json(body, &j->steer, err, sizeof err) < 0) {
        char resp[320];
        snprintf(resp, sizeof resp, "{\"error\":\"%.256s\"}", err);
        respond_json(fd, 400, "Bad Request", resp);
        pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv);
        free(j->prompt); free(j);
        return;
    }

    q_push(j);
    pthread_mutex_lock(&j->mu);
    while (!j->done) pthread_cond_wait(&j->cv, &j->mu);
    pthread_mutex_unlock(&j->mu);

    if (j->wav) respond(fd, 200, "OK", "audio/x-wav", j->wav, j->wav_len);
    else {
        char resp[320];
        snprintf(resp, sizeof resp, "{\"error\":\"%.256s\"}", j->err[0] ? j->err : "generate failed");
        respond_json(fd, 500, "Internal Server Error", resp);
    }
    free(j->wav); free(j->prompt); free((void *)j->steer.dir);
    pthread_mutex_destroy(&j->mu); pthread_cond_destroy(&j->cv);
    free(j);
}

static void *conn_main(void *arg) {
    int fd = (int)(intptr_t)arg;
    struct timeval tv = { IO_TIMEOUT_SEC, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    char *body = NULL;
    char *req = read_request(fd, &body);
    if (!req) { close(fd); return NULL; }

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
        handle_generate(fd, body);
    } else {
        respond_json(fd, 404, "Not Found", "{\"error\":\"routes: GET /health, GET /info, POST /generate\"}");
    }
    free(req);
    close(fd);
    return NULL;
}

/* ---------------- main ------------------------------------------------------------------ */

static void usage(const char *prog) {
    fprintf(stderr,
            "aria-server - resident HTTP server for the aria runtime\n\n"
            "Usage: %s -m <model_dir> [--port 8321] [--device cpu|cuda|auto]\n"
            "          [-s <steps (default: the model's, 6)>] [-d <default_seconds=10>]\n", prog);
}

int main(int argc, char **argv) {
    const char *model_dir = NULL;
    int port = 8321;
    worker_cfg w = { NULL, ARIA_DEVICE_AUTO, 0, 10.0f };   /* steps 0 = the model's default */
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc) model_dir = argv[++i];
        else if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) w.default_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-d") && i + 1 < argc) w.default_seconds = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) {
            const char *d = argv[++i];
            w.device = !strcmp(d, "cpu") ? ARIA_DEVICE_CPU
                     : (!strcmp(d, "cuda") || !strcmp(d, "gpu")) ? ARIA_DEVICE_CUDA : ARIA_DEVICE_AUTO;
        } else { usage(argv[0]); return 1; }
    }
    if (!model_dir) { usage(argv[0]); return 1; }

    g_ctx = aria_load(model_dir);
    if (!g_ctx) { fprintf(stderr, "aria-server: load error: %s\n", aria_last_error()); return 1; }
    w.ctx = g_ctx;
    fprintf(stderr, "aria-server: %s | type=%s sr=%d ch=%d\n", model_dir,
            aria_model_type(g_ctx), aria_sample_rate(g_ctx), aria_audio_channels(g_ctx));

    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);
    signal(SIGPIPE, SIG_IGN);

    int lfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof addr) != 0 || listen(lfd, 16) != 0) {
        fprintf(stderr, "aria-server: cannot listen on port %d: %s\n", port, strerror(errno));
        return 1;
    }
    g_listen_fd = lfd;
    fprintf(stderr, "aria-server: listening on :%d (POST /generate)\n", port);

    pthread_t worker;
    pthread_create(&worker, NULL, worker_main, &w);

    while (!g_stop) {
        int cfd = accept(lfd, NULL, NULL);
        if (cfd < 0) { if (g_stop || errno == EBADF) break; continue; }
        pthread_t th;
        if (pthread_create(&th, NULL, conn_main, (void *)(intptr_t)cfd) == 0) pthread_detach(th);
        else close(cfd);
    }
    g_stop = 1;
    pthread_cond_broadcast(&q.cv);
    pthread_join(worker, NULL);
    aria_free(g_ctx);
    fprintf(stderr, "aria-server: bye\n");
    return 0;
}
