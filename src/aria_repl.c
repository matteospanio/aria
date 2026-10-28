/*
 * aria_repl.c - aria-repl: an interactive client for aria-server live sessions.
 *
 *   aria-repl [--host 127.0.0.1:8321] [--record take.wav] [--history take.jsonl] ...
 *
 * Type a prompt to start streaming; type another at any time to change it; /steer,
 * /seed, /pause, /stop, /help for the rest. Three threads: the UI (this one: line
 * editor, commands, status line), the network reader (POST /stream frames -> ring
 * buffer, recording, events) and playback (ring -> an external player's stdin; its
 * blocking writes pace real time). The server keeps about one chunk ahead of what has
 * been played (credits the UI thread sends on the stream connection itself), so a change
 * is heard ~one buffer later; the buffer starts small and grows after each underrun.
 * Playback uses an existing player (pw-play, paplay, aplay, sox play, ffplay) through
 * a pipe -- no audio library.
 */
#include "aria_json.h"
#include "aria_live.h"
#include "aria_net.h"
#include "aria_wav.h"

#include <errno.h>
#include <fcntl.h>
#include <langinfo.h>
#include <locale.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

#define RING_SEC 30          /* client buffer (the server only sends ~ahead past `played`) */
#define BLOCK 1024           /* playback write size, frames */
#define MAX_STEERS 32

enum { IDLE, OPENING, LIVE, DRAINING };

typedef struct { int ctl; int64_t at; double t_sent; char what[300], val[1100]; char type[8]; int resolved, shown; } pending;
typedef struct { int64_t at; char steer[256]; } audible;

static struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;                   /* ring space / data, state changes */
    /* config */
    char host[256]; int port;
    double chunk, context, ahead, latency, duration;
    long long seed; int seed_set;
    const char *rec_path, *hist_path, *player;
    int no_play, to_stdout, tty, utf8;
    /* session */
    int state, quit, sigq, stdin_eof;
    char sid[16]; int sr, ch;
    aria_sock sock;                      /* the /stream connection (credits go upstream on it) */
    double buffer, buffer_sent;          /* client buffer target = the server's lookahead, s */
    double chunk_s;                      /* the server's current chunk length */
    char open_body[2600];
    float *ring; int64_t ring_cap;       /* frames */
    int64_t recv_end, played, credited;  /* session timeline, frames */
    int done_recv, paused, buffering, stop_sent;
    double rtf, t_live;
    int chunks, underruns; int64_t silence;
    double ahead_min, ahead_max, ahead_sum; int ahead_n;
    pending pend[64]; int npend;
    audible aud[16]; int naud;
    char req_steer[MAX_STEERS][48]; float req_scale[MAX_STEERS]; int nreq;   /* what we asked for */
    char pre_steer[512];                 /* /steer before the stream: sent with the open */
    double lag_sum, lag_max; int lag_n;
    /* files */
    FILE *rec; uint32_t rec_bytes; double rec_offset;
    FILE *hist;
    /* UI events (net/playback threads -> UI thread) */
    int wake[2];
    char evq[32][400]; int evn;
} R = { .mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER, .port = 8321,
        .latency = 0.25, .seed = -1, .sock = ARIA_SOCK_BAD };

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static const char *G(const char *u, const char *a) { return R.utf8 ? u : a; }

static void fmt_t(char *b, size_t n, double t) {
    if (t < 0) t = 0;
    snprintf(b, n, "%02d:%02d", (int)(t / 60), (int)t % 60);
}

/* ---------------- events: any thread queues a line, the UI thread prints it ---------- */

static void post_event(const char *fmt, ...) {   /* R.mu held */
    if (R.evn == 32) return;
    va_list ap; va_start(ap, fmt);
    vsnprintf(R.evq[R.evn++], sizeof R.evq[0], fmt, ap);
    va_end(ap);
    char c = 1; if (write(R.wake[1], &c, 1)) {}
}

static void hist_line(const char *type, const char *fmt, ...) {   /* R.mu held; fmt = extra JSON fields */
    if (!R.hist) return;
    char ts[32]; time_t t = time(NULL); struct tm tm; localtime_r(&t, &tm);
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &tm);
    fprintf(R.hist, "{\"t_wall\":\"%s\",\"session\":\"%s\",\"type\":\"%s\"", ts, R.sid, type);
    va_list ap; va_start(ap, fmt);
    vfprintf(R.hist, fmt, ap);
    va_end(ap);
    fputs("}\n", R.hist);
    fflush(R.hist);
}

/* ---------------- HTTP (short requests: controls, /steers) ---------------------------- */

static int http(const char *method, const char *path, const char *body, char *resp, size_t cap) {
    aria_sock s = aria_connect(R.host, R.port);
    if (s == ARIA_SOCK_BAD) { snprintf(resp, cap, "cannot connect to %s:%d", R.host, R.port); return -1; }
    aria_sock_timeouts(s, 5000);
    char hdr[256];
    size_t bl = body ? strlen(body) : 0;
    int n = snprintf(hdr, sizeof hdr, "%s %s HTTP/1.1\r\nHost: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
                     method, path, R.host, bl);
    if (aria_send_all(s, hdr, (size_t)n) || (bl && aria_send_all(s, body, bl))) {
        aria_sock_close(s); snprintf(resp, cap, "send failed"); return -1;
    }
    char buf[16384]; size_t len = 0; long k;
    while (len < sizeof buf - 1 && (k = aria_recv(s, buf + len, sizeof buf - 1 - len)) > 0) len += (size_t)k;
    aria_sock_close(s);
    buf[len] = 0;
    int code = 0;
    sscanf(buf, "HTTP/1.%*d %d", &code);
    const char *b = strstr(buf, "\r\n\r\n");
    snprintf(resp, cap, "%s", b ? b + 4 : "");
    return code ? code : -1;
}

/* ---------------- player ------------------------------------------------------------- */

static pid_t player_pid = -1;
static int player_fd = -1;

static int spawn(const char *cmd) {   /* /bin/sh -c cmd, stdin = a new pipe; -1 if it exits at once */
    int p[2];
    if (pipe2(p, O_CLOEXEC)) return -1;
    posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, p[0], 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    char *argv[] = { "sh", "-c", (char *)cmd, NULL };
    pid_t pid;
    int rc = posix_spawn(&pid, "/bin/sh", &fa, NULL, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(p[0]);
    if (rc) { close(p[1]); return -1; }
    struct timespec ts = { 0, 300000000L }; nanosleep(&ts, NULL);   /* bad flags / no device: exits at once */
    if (waitpid(pid, NULL, WNOHANG) == pid) { close(p[1]); return -1; }
    player_pid = pid; player_fd = p[1];
    return 0;
}

static int in_path(const char *exe) {
    const char *path = getenv("PATH");
    char dir[1024];
    for (const char *p = path ? path : "/usr/bin:/bin"; *p; ) {
        size_t n = strcspn(p, ":");
        snprintf(dir, sizeof dir, "%.*s/%s", (int)n, p, exe);
        if (access(dir, X_OK) == 0) return 1;
        p += n; if (*p) p++;
    }
    return 0;
}

static void subst(char *out, size_t n, const char *tpl) {   /* {sr} {ch} */
    size_t k = 0;
    for (const char *t = tpl; *t && k + 16 < n; t++) {
        if (!strncmp(t, "{sr}", 4)) { k += (size_t)snprintf(out + k, n - k, "%d", R.sr); t += 3; }
        else if (!strncmp(t, "{ch}", 4)) { k += (size_t)snprintf(out + k, n - k, "%d", R.ch); t += 3; }
        else out[k++] = *t;
    }
    out[k] = 0;
}

static const char *player_open(void) {   /* returns the player's name or NULL */
    static const char *cand[][2] = {
        { "pw-play", "pw-play --raw --format f32 --rate {sr} --channels {ch} -" },
        { "paplay",  "paplay --raw --format=float32le --rate={sr} --channels={ch}" },
        { "aplay",   "aplay -q -t raw -f FLOAT_LE -r {sr} -c {ch} -" },
        { "play",    "play -q -t raw -r {sr} -e float -b 32 -c {ch} -" },
        { "ffplay",  "ffplay -nodisp -autoexit -loglevel quiet -f f32le -ar {sr} -ac {ch} -i -" },
    };
    char cmd[512];
    if (R.player) { subst(cmd, sizeof cmd, R.player); return spawn(cmd) ? NULL : R.player; }
    for (int i = 0; i < 5; i++) {
        if (!in_path(cand[i][0])) continue;
        subst(cmd, sizeof cmd, cand[i][1]);
        if (!spawn(cmd)) return cand[i][0];
    }
    return NULL;
}

static void player_close(void) {
    if (player_fd >= 0) { close(player_fd); player_fd = -1; }
    if (player_pid > 0) { waitpid(player_pid, NULL, 0); player_pid = -1; }
}

/* ---------------- recording ------------------------------------------------------------ */

static void rec_patch(void) {   /* R.mu held */
    if (!R.rec) return;
    uint8_t h[44];
    aria_wav_header(h, R.sr, R.ch, 32, R.rec_bytes);
    long pos = ftell(R.rec);
    fseek(R.rec, 0, SEEK_SET); fwrite(h, 1, 44, R.rec); fseek(R.rec, pos, SEEK_SET);
    fflush(R.rec);
}

/* ---------------- network thread: one /stream at a time ------------------------------- */

static void steer_pretty(const char *spec, char *out, size_t n) {   /* "sweet:0.30,sour:0.10" -> "sweet +0.30 · sour +0.10" */
    size_t k = 0; out[0] = 0;
    if (!spec[0] || !strcmp(spec, "-")) return;
    for (const char *p = spec; *p && k + 40 < n; ) {
        size_t nl = strcspn(p, ":,");
        double v = p[nl] == ':' ? atof(p + nl + 1) : 0.0;
        k += (size_t)snprintf(out + k, n - k, "%s%.*s %+.2f", k ? G(" · ", ", ") : "", (int)nl, p, v);
        p += nl; p += strcspn(p, ","); if (*p) p++;
    }
}

static void on_evnt(const char *j) {   /* R.mu held */
    double v; char steer[256] = "-";
    int ctl = aria_json_get_number(j, "ctl", &v) == 0 ? (int)v : 0;
    int64_t at = aria_json_get_number(j, "at", &v) == 0 ? (int64_t)v : 0;
    if (aria_json_get_number(j, "rtf", &v) == 0) R.rtf = v;
    if (aria_json_get_number(j, "ahead_ms", &v) == 0 && R.chunks > 0) {   /* chunk 0 starts the buffer */
        double a = v / 1000.0;
        if (!R.ahead_n || a < R.ahead_min) R.ahead_min = a;
        if (a > R.ahead_max) R.ahead_max = a;
        R.ahead_sum += a; R.ahead_n++;
    }
    aria_json_get_string(j, "steer", steer, sizeof steer);
    if (aria_json_get_number(j, "chunk_s", &v) == 0 && R.chunk_s > 0 && v != R.chunk_s)
        post_event("  %s server chunk %.0f %s %.0f s (%s)", G("~", "~"), R.chunk_s, G("→", "->"), v,
                   v > R.chunk_s ? "below real time: changes land later" : "headroom: changes land sooner");
    if (aria_json_get_number(j, "chunk_s", &v) == 0) R.chunk_s = v;
    R.chunks++;
    if (R.naud == 16) { memmove(R.aud, R.aud + 1, 15 * sizeof R.aud[0]); R.naud = 15; }
    R.aud[R.naud].at = at; snprintf(R.aud[R.naud].steer, sizeof R.aud[0].steer, "%s", steer); R.naud++;
    for (int i = 0; i < R.npend; i++) {
        pending *p = &R.pend[i];
        if (p->resolved || p->ctl > ctl) continue;
        p->resolved = 1; p->at = at;
        double t_audio = (double)at / R.sr, lag = t_audio - p->t_sent;
        if (lag < 0) lag = 0;
        R.lag_sum += lag; R.lag_n++; if (lag > R.lag_max) R.lag_max = lag;
        hist_line(p->type, ",\"value\":\"%s\",\"ctl\":%d,\"t_sent\":%.2f,\"t_audio\":%.2f,\"rec_at\":%.2f",
                  p->val, p->ctl, p->t_sent, t_audio, R.rec_offset + t_audio);
    }
    rec_patch();
}

static void *net_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&R.mu);
    for (;;) {
        while (R.state != OPENING && !R.quit) pthread_cond_wait(&R.cv, &R.mu);
        if (R.quit) break;
        char body[sizeof R.open_body]; memcpy(body, R.open_body, sizeof body);
        pthread_mutex_unlock(&R.mu);

        aria_sock s = aria_connect(R.host, R.port);
        char hdr[256]; int ok = s != ARIA_SOCK_BAD;
        if (ok) {
            int n = snprintf(hdr, sizeof hdr, "POST /stream HTTP/1.1\r\nHost: %s\r\nContent-Length: %zu\r\n\r\n", R.host, strlen(body));
            ok = !aria_send_all(s, hdr, (size_t)n) && !aria_send_all(s, body, strlen(body));
        }
        size_t cap = 1 << 22, len = 0; unsigned char *buf = malloc(cap);
        long k; int code = 0, inhdr = 1;
        pthread_mutex_lock(&R.mu);
        R.sock = ok ? s : ARIA_SOCK_BAD;
        if (!ok) post_event("  %s cannot reach the server at %s:%d", G("✗", "x"), R.host, R.port);
        while (ok && !R.quit) {
            pthread_mutex_unlock(&R.mu);
            k = len < cap ? aria_recv(s, buf + len, cap - len) : 0;
            pthread_mutex_lock(&R.mu);
            if (k <= 0) break;
            len += (size_t)k;
            if (inhdr) {
                unsigned char *e = NULL;
                for (size_t i = 0; i + 3 < len; i++) if (!memcmp(buf + i, "\r\n\r\n", 4)) { e = buf + i; break; }
                if (!e) continue;
                sscanf((char *)buf, "HTTP/1.%*d %d", &code);
                size_t h = (size_t)(e + 4 - buf);
                memmove(buf, buf + h, len - h); len -= h; inhdr = 0;
                if (code != 200) {   /* the error body follows; read it all */
                    while (len < cap - 1) {
                        pthread_mutex_unlock(&R.mu);
                        k = aria_recv(s, buf + len, cap - 1 - len);
                        pthread_mutex_lock(&R.mu);
                        if (k <= 0) break;
                        len += (size_t)k;
                    }
                    buf[len] = 0;
                    char err[300] = "";
                    aria_json_get_string((char *)buf, "error", err, sizeof err);
                    post_event("  %s server: %s", G("✗", "x"), err[0] ? err : "no stream");
                    break;
                }
            }
            for (;;) {
                char tag[5]; const unsigned char *pl; uint32_t pn;
                long used = live_parse(buf, len, cap - 8, tag, &pl, &pn);
                if (used < 0) { post_event("  %s corrupt stream", G("✗", "x")); ok = 0; break; }
                if (!used) break;
                char js[1024];
                if (!strcmp(tag, "AUDI") && pn >= 8 && R.sr) {
                    uint64_t start; memcpy(&start, pl, 8);
                    int64_t n = (int64_t)(pn - 8) / (4 * R.ch);
                    const float *a = (const float *)(pl + 8);
                    if ((int64_t)start != R.recv_end) post_event("  %s gap in stream at frame %lld", G("✗", "x"), (long long)start);
                    for (int64_t off = 0; off < n && !R.quit; ) {   /* into the ring, waiting for space */
                        int64_t room = R.ring_cap - (R.recv_end - R.played);
                        if (room <= 0) { pthread_cond_wait(&R.cv, &R.mu); continue; }
                        int64_t m = n - off < room ? n - off : room;
                        for (int64_t f = 0; f < m; f++) {
                            int64_t pos = (R.recv_end + f) % R.ring_cap;
                            memcpy(R.ring + pos * R.ch, a + (off + f) * R.ch, (size_t)R.ch * sizeof(float));
                        }
                        if (R.rec) { fwrite(a + off * R.ch, sizeof(float), (size_t)(m * R.ch), R.rec); R.rec_bytes += (uint32_t)(m * R.ch * 4); }
                        R.recv_end += m; off += m;
                        pthread_cond_broadcast(&R.cv);
                    }
                } else if (!strcmp(tag, "HELO") || !strcmp(tag, "EVNT") || !strcmp(tag, "DONE") || !strcmp(tag, "ERRO")) {
                    size_t l = pn < sizeof js - 1 ? pn : sizeof js - 1;
                    memcpy(js, pl, l); js[l] = 0;
                    double v;
                    if (tag[0] == 'H') {
                        aria_json_get_string(js, "session", R.sid, sizeof R.sid);
                        R.sr = aria_json_get_number(js, "sample_rate", &v) == 0 ? (int)v : 44100;
                        R.ch = aria_json_get_number(js, "channels", &v) == 0 ? (int)v : 2;
                        if (!R.ring) { R.ring_cap = (int64_t)RING_SEC * R.sr; R.ring = calloc((size_t)(R.ring_cap * R.ch), sizeof(float)); }
                        R.recv_end = R.played = R.credited = 0;
                        R.done_recv = R.stop_sent = 0; R.chunks = 0; R.naud = 0; R.t_live = now_s();
                        R.rec_offset = R.rec ? R.rec_bytes / (4.0 * R.ch * R.sr) : 0;
                        if (R.rec && !R.rec_bytes) { uint8_t h[44] = {0}; fwrite(h, 1, 44, R.rec); }
                        long long seed = aria_json_get_number(js, "seed", &v) == 0 ? (long long)v : -1;
                        double ch_s = aria_json_get_number(js, "chunk", &v) == 0 ? v : 0;
                        double ah = aria_json_get_number(js, "ahead", &v) == 0 ? v : 0;
                        R.buffer = R.buffer_sent = ah;
                        R.chunk_s = ch_s;
                        char pr[512] = "", pe[1100]; aria_json_get_string(R.open_body, "prompt", pr, sizeof pr);
                        aria_json_escape(pr, pe, sizeof pe);
                        hist_line("open", ",\"prompt\":\"%s\",\"seed\":%lld,\"chunk\":%.2f,\"ahead\":%.2f,\"rec_offset\":%.2f",
                                  pe, seed, ch_s, ah, R.rec_offset);
                        R.state = LIVE;
                        post_event("  %s 00:00 playing %s seed %lld %s chunk %.0f s", G("▶", ">"), G("·", "-"), seed, G("·", "-"), ch_s);
                        pthread_cond_broadcast(&R.cv);
                    } else if (tag[0] == 'E' && tag[1] == 'V') {
                        on_evnt(js);
                    } else if (tag[0] == 'D') {
                        char why[32] = ""; aria_json_get_string(js, "reason", why, sizeof why);
                        hist_line("close", ",\"reason\":\"%s\",\"seconds\":%.2f", why, (double)R.recv_end / R.sr);
                        R.done_recv = 1;
                    } else {
                        char err[300] = ""; aria_json_get_string(js, "error", err, sizeof err);
                        post_event("  %s server: %s", G("✗", "x"), err);
                    }
                }
                memmove(buf, buf + used, len - (size_t)used); len -= (size_t)used;
            }
            if (!ok) break;
        }
        R.sock = ARIA_SOCK_BAD;
        if (s != ARIA_SOCK_BAD) aria_sock_close(s);
        free(buf);
        R.done_recv = 1;
        rec_patch();
        if (R.state == OPENING) R.state = IDLE;          /* never went live */
        else if (R.state == LIVE) R.state = DRAINING;    /* playback finishes what was received */
        pthread_cond_broadcast(&R.cv);
        char c = 1; if (write(R.wake[1], &c, 1)) {}
    }
    pthread_mutex_unlock(&R.mu);
    return NULL;
}

/* ---------------- playback thread ------------------------------------------------------ */

static void *play_main(void *arg) {
    (void)arg;
    float *blk = malloc((size_t)BLOCK * 8 * sizeof(float));
    pthread_mutex_lock(&R.mu);
    for (;;) {
        while (R.state != LIVE && R.state != DRAINING && !R.quit) pthread_cond_wait(&R.cv, &R.mu);
        if (R.quit) break;
        int sr = R.sr, ch = R.ch;
        int fd = -1; const char *name = NULL;
        if (R.to_stdout) fd = 1;
        else if (!R.no_play) {
            pthread_mutex_unlock(&R.mu);
            name = player_open();
            pthread_mutex_lock(&R.mu);
            if (name) fd = player_fd;
            else post_event("  %s no audio player found (pw-play, paplay, aplay, play, ffplay); continuing silently (use --player)", G("✗", "x"));
        }
        double t_next = now_s();
        for (;;) {
            int64_t avail = R.recv_end - R.played;
            int drained = R.state == DRAINING && avail == 0;
            if (R.quit || drained) break;
            int silent = R.paused || avail == 0 || (R.buffering && avail < sr && !R.done_recv);
            if (!silent && R.buffering) { R.buffering = 0; post_event("  %s playing again", G("▶", ">")); }
            if (avail == 0 && !R.paused && !R.buffering && R.state == LIVE && R.played > 0) {
                R.buffering = 1; R.underruns++;
                if (R.buffer < 6.0) R.buffer = R.buffer + 0.5 < 6.0 ? R.buffer + 0.5 : 6.0;   /* adaptive */
                post_event("  %s buffering %s buffer %.1f s%s", G("…", "..."), G("·", "-"), R.buffer,
                           R.underruns == 1 && R.rtf > 0 && R.rtf < 1.1 ? " (the server is near or below real time; a larger --chunk helps)" : "");
            }
            int64_t n = silent ? BLOCK : (avail < BLOCK ? avail : BLOCK);
            if (silent) { memset(blk, 0, (size_t)n * ch * sizeof(float)); if (R.buffering) R.silence += n; }
            else for (int64_t f = 0; f < n; f++)
                memcpy(blk + f * ch, R.ring + ((R.played + f) % R.ring_cap) * ch, (size_t)ch * sizeof(float));
            pthread_mutex_unlock(&R.mu);
            int werr = 0;
            if (fd >= 0) {
                const char *p = (const char *)blk; size_t left = (size_t)n * ch * sizeof(float);
                while (left) { ssize_t w = write(fd, p, left); if (w < 0 && errno == EINTR) continue; if (w <= 0) { werr = 1; break; } p += w; left -= (size_t)w; }
            } else {   /* no player: pace by the clock */
                t_next += (double)n / sr;
                double d = t_next - now_s();
                if (d > 0) { struct timespec ts = { (time_t)d, (long)((d - (time_t)d) * 1e9) }; nanosleep(&ts, NULL); }
            }
            pthread_mutex_lock(&R.mu);
            if (werr && fd >= 0) {
                post_event("  %s audio output closed; continuing silently", G("✗", "x"));
                if (fd != 1) player_close();
                fd = -1; t_next = now_s();
            }
            if (!silent) { R.played += n; pthread_cond_broadcast(&R.cv); }
        }
        if (fd >= 0 && fd != 1) { pthread_mutex_unlock(&R.mu); player_close(); pthread_mutex_lock(&R.mu); }
        (void)name;
        if (R.state == DRAINING) {
            R.state = IDLE;
            post_event("  %s stream ended (%.1f s)", G("■", "#"), (double)R.played / sr);
        }
    }
    pthread_mutex_unlock(&R.mu);
    free(blk);
    return NULL;
}

/* ---------------- line editor (raw terminal) ------------------------------------------- */

typedef struct {
    char buf[1024]; int len, pos;
    char hist[64][1024]; int nhist, hidx;
    int esc; char seq[8]; int nseq;
} le_state;
enum { LE_NONE, LE_REDRAW, LE_SUBMIT, LE_EOF, LE_INTR };

static int le_key(le_state *e, unsigned char c) {
    if (e->esc) {   /* ESC [ X / ESC [ n ~ / ESC O X */
        if (e->nseq < 7) e->seq[e->nseq++] = (char)c;
        if (e->nseq == 1 && c != '[' && c != 'O') { e->esc = 0; return LE_NONE; }
        if (e->nseq >= 2 && (c >= 'A' && c <= 'Z' ? 1 : c == '~')) {
            e->esc = 0;
            char k = e->seq[e->nseq - 1], d = e->nseq >= 3 ? e->seq[1] : 0;
            if (k == 'A' || k == 'B') {   /* history */
                int h = e->hidx + (k == 'A' ? -1 : 1);
                if (h < 0 || h > e->nhist) return LE_NONE;
                e->hidx = h;
                const char *src = h == e->nhist ? "" : e->hist[h];
                size_t l = strlen(src);
                memcpy(e->buf, src, l + 1);
                e->len = e->pos = (int)l;
            } else if (k == 'C') { if (e->pos < e->len) do e->pos++; while (e->pos < e->len && (e->buf[e->pos] & 0xC0) == 0x80); }
            else if (k == 'D') { if (e->pos > 0) do e->pos--; while (e->pos > 0 && (e->buf[e->pos] & 0xC0) == 0x80); }
            else if (k == 'H' || (k == '~' && (d == '1' || d == '7'))) e->pos = 0;
            else if (k == 'F' || (k == '~' && (d == '4' || d == '8'))) e->pos = e->len;
            else if (k == '~' && d == '3' && e->pos < e->len) {   /* delete */
                int n = 1; while (e->pos + n < e->len && (e->buf[e->pos + n] & 0xC0) == 0x80) n++;
                memmove(e->buf + e->pos, e->buf + e->pos + n, (size_t)(e->len - e->pos - n + 1)); e->len -= n;
            }
            return LE_REDRAW;
        }
        return LE_NONE;
    }
    switch (c) {
    case 27: e->esc = 1; e->nseq = 0; return LE_NONE;
    case '\r': case '\n':
        if (e->len && (!e->nhist || strcmp(e->hist[e->nhist - 1], e->buf))) {
            if (e->nhist == 64) { memmove(e->hist, e->hist + 1, sizeof e->hist[0] * 63); e->nhist = 63; }
            memcpy(e->hist[e->nhist++], e->buf, (size_t)e->len + 1);
        }
        e->hidx = e->nhist;
        return LE_SUBMIT;
    case 3: if (e->len) { e->len = e->pos = 0; e->buf[0] = 0; return LE_REDRAW; } return LE_INTR;   /* ^C */
    case 4: return e->len ? LE_NONE : LE_EOF;                                                       /* ^D */
    case 1: e->pos = 0; return LE_REDRAW;                                                           /* ^A */
    case 5: e->pos = e->len; return LE_REDRAW;                                                      /* ^E */
    case 11: e->len = e->pos; e->buf[e->len] = 0; return LE_REDRAW;                                 /* ^K */
    case 12: return LE_REDRAW;                                                                      /* ^L */
    case 21: memmove(e->buf, e->buf + e->pos, (size_t)(e->len - e->pos + 1)); e->len -= e->pos; e->pos = 0; return LE_REDRAW;   /* ^U */
    case 23: {                                                                                      /* ^W */
        int p = e->pos;
        while (p > 0 && e->buf[p - 1] == ' ') p--;
        while (p > 0 && e->buf[p - 1] != ' ') p--;
        memmove(e->buf + p, e->buf + e->pos, (size_t)(e->len - e->pos + 1)); e->len -= e->pos - p; e->pos = p;
        return LE_REDRAW;
    }
    case 8: case 127:                                                                               /* backspace */
        if (e->pos > 0) {
            int n = 1; while (e->pos - n > 0 && (e->buf[e->pos - n] & 0xC0) == 0x80) n++;
            memmove(e->buf + e->pos - n, e->buf + e->pos, (size_t)(e->len - e->pos + 1)); e->len -= n; e->pos -= n;
        }
        return LE_REDRAW;
    default:
        if (c < 32 || e->len + 1 >= (int)sizeof e->buf) return LE_NONE;
        memmove(e->buf + e->pos + 1, e->buf + e->pos, (size_t)(e->len - e->pos + 1));
        e->buf[e->pos++] = (char)c; e->len++;
        return LE_REDRAW;
    }
}

static int u8len(const char *s, int n) { int w = 0; for (int i = 0; i < n; i++) w += (s[i] & 0xC0) != 0x80; return w; }

/* ---------------- UI ------------------------------------------------------------------- */

static le_state ED;
static struct termios tty_saved; static int tty_raw;

static void tty_restore(void) {
    if (tty_raw) { fputs("\r\x1b[J", stderr); fflush(stderr); tcsetattr(0, TCSAFLUSH, &tty_saved); tty_raw = 0; }
}

static int term_cols(void) { struct winsize w; return ioctl(2, TIOCGWINSZ, &w) == 0 && w.ws_col > 20 ? w.ws_col : 80; }

static void status_line(char *out, size_t n) {   /* R.mu held */
    char t[16], st[300] = "";
    double pos = R.sr ? (double)R.played / R.sr - R.latency : 0;
    fmt_t(t, sizeof t, pos);
    const char *steer = "-";
    for (int i = R.naud - 1; i >= 0; i--) if ((double)R.aud[i].at / (R.sr ? R.sr : 1) <= pos + R.latency) { steer = R.aud[i].steer; break; }
    steer_pretty(steer, st, sizeof st);
    const char *dot = G(" · ", " | ");
    if (R.state == IDLE || !R.sr)
        snprintf(out, n, "[%s idle%s type a prompt%s /help]", G("■", "#"), dot, dot);
    else
        snprintf(out, n, "[%s %s buf %.1f s%s%.1f%s%s%s%s%s]", t,
                 R.paused ? G("‖", "||") : R.buffering ? G("…", "...") : G("▶", ">"),
                 (double)(R.recv_end - R.played) / R.sr, dot, R.rtf, G("×", "x"),
                 st[0] ? dot : "", st, R.rec ? dot : "", R.rec ? G("rec ●", "rec *") : "");
}

static void ui_draw(void) {   /* R.mu held; TTY: input line + status line below it */
    if (!R.tty) return;
    int cols = term_cols();
    const char *pr = "aria> ";
    int avail = cols - 7, cw = u8len(ED.buf, ED.pos), start = 0;
    while (cw - u8len(ED.buf, start) > avail - 1) { do start++; while ((ED.buf[start] & 0xC0) == 0x80); }
    char st[400]; status_line(st, sizeof st);
    int sw = 0, cut = 0;   /* truncate the status to the width */
    for (cut = 0; st[cut]; cut++) { if ((st[cut] & 0xC0) != 0x80 && ++sw > cols - 1) break; }
    st[cut] = 0;
    int vis = 0, end = start;
    while (end < ED.len && vis < avail) { end++; while (end < ED.len && (ED.buf[end] & 0xC0) == 0x80) end++; vis++; }
    fprintf(stderr, "\r\x1b[J%s%.*s\r\n\x1b[2m%s\x1b[0m\x1b[A\r\x1b[%dC", pr, end - start, ED.buf + start, st,
            6 + u8len(ED.buf + start, ED.pos - start));
    fflush(stderr);
}

static void ui_flush_events(void) {   /* R.mu held */
    if (!R.evn) return;
    if (R.tty) fputs("\r\x1b[J", stderr);
    for (int i = 0; i < R.evn; i++) fprintf(stderr, "%s%s", R.evq[i], R.tty ? "\r\n" : "\n");
    R.evn = 0;
    ui_draw();
}

/* the "now" notices: resolved changes that playback has reached */
static void ui_audible(void) {   /* R.mu held */
    if (!R.sr) return;
    double pos = (double)R.played / R.sr - R.latency;
    for (int i = 0; i < R.npend; i++) {
        pending *p = &R.pend[i];
        if (!p->resolved || p->shown || (double)p->at / R.sr > pos) continue;
        char t[16]; fmt_t(t, sizeof t, (double)p->at / R.sr);
        post_event("  %s %s now: %s", G("▶", ">"), t, p->what);
        p->shown = 1;
    }
    int k = 0;   /* drop shown entries */
    for (int i = 0; i < R.npend; i++) if (!R.pend[i].shown) R.pend[k++] = R.pend[i];
    R.npend = k;
}

static void json_str(char *out, size_t n, const char *s) { aria_json_escape(s, out, n); }

/* send a control; on success queue "lands in ~x s" and a pending entry resolved by EVNT */
static void control(const char *type, const char *what, const char *val, const char *fields) {
    char body[1600], resp[512];
    pthread_mutex_lock(&R.mu);
    snprintf(body, sizeof body, "{\"session\":\"%s\",%s}", R.sid, fields);
    int64_t played = R.played; int sr = R.sr;
    pthread_mutex_unlock(&R.mu);
    int code = http("POST", "/stream/control", body, resp, sizeof resp);
    pthread_mutex_lock(&R.mu);
    char err[300] = "";
    if (code != 200) {
        aria_json_get_string(resp, "error", err, sizeof err);
        post_event("  %s %s", G("✗", "x"), err[0] ? err : resp);
    } else if (what) {
        double v, lands = aria_json_get_number(resp, "lands", &v) == 0 ? v : 0;
        int ctl = aria_json_get_number(resp, "ctl", &v) == 0 ? (int)v : 0;
        double in = (lands - (double)played) / sr + R.latency;
        char t[16]; fmt_t(t, sizeof t, (double)played / sr);
        post_event("  %s %s %s  (lands in ~%.1f s)", G("↳", "->"), t, what, in < 0 ? 0 : in);
        if (R.npend < 64) {
            pending *p = &R.pend[R.npend++];
            memset(p, 0, sizeof *p);
            p->ctl = ctl; p->t_sent = (double)played / sr;
            snprintf(p->what, sizeof p->what, "%s", what); snprintf(p->type, sizeof p->type, "%s", type);
            snprintf(p->val, sizeof p->val, "%s", val ? val : "");
        }
    }
    pthread_mutex_unlock(&R.mu);
}

static void open_stream(const char *prompt) {
    char pe[1100], extra[600] = "";
    json_str(pe, sizeof pe, prompt);
    size_t k = 0;
    if (R.seed_set) k += (size_t)snprintf(extra + k, sizeof extra - k, ",\"seed\":%lld", R.seed);
    if (R.chunk > 0) k += (size_t)snprintf(extra + k, sizeof extra - k, ",\"chunk\":%.2f", R.chunk);
    if (R.context > 0) k += (size_t)snprintf(extra + k, sizeof extra - k, ",\"context\":%.2f", R.context);
    if (R.ahead >= 0) k += (size_t)snprintf(extra + k, sizeof extra - k, ",\"ahead\":%.2f", R.ahead);
    R.buffer = 0;   /* HELO reports the server's choice */
    if (R.pre_steer[0]) k += (size_t)snprintf(extra + k, sizeof extra - k, ",\"steer\":\"%s\"", R.pre_steer);
    pthread_mutex_lock(&R.mu);
    snprintf(R.open_body, sizeof R.open_body, "{\"prompt\":\"%s\"%s}", pe, extra);
    R.state = OPENING;
    pthread_cond_broadcast(&R.cv);
    pthread_mutex_unlock(&R.mu);
}

static int steer_index(const char *name) {
    for (int i = 0; i < R.nreq; i++) if (!strcmp(R.req_steer[i], name)) return i;
    if (R.nreq == MAX_STEERS) return -1;
    snprintf(R.req_steer[R.nreq], sizeof R.req_steer[0], "%s", name);
    R.req_scale[R.nreq] = 0;
    return R.nreq++;
}

static void help(void) {
    static const char *h[] = {
        "  <text>              a prompt: starts the stream, or changes it (lands on the next chunk)",
        "  /steer <name> <x>   steer toward a taste (e.g. /steer sweet 0.3; 0 removes it)",
        "  /steer off          remove all steering        /steers   list the server's directions",
        "  /seed <n>  /evolve <n>  /anchor <0..1>          new seed / reseed every n chunks / re-anchor",
        "  /pause  /resume     pause playback (the server waits)",
        "  /stop               end the stream (what is buffered still plays)",
        "  /status  /help  /quit (or Ctrl-D / Ctrl-C on an empty line)",
    };
    pthread_mutex_lock(&R.mu);
    for (size_t i = 0; i < sizeof h / sizeof *h; i++) post_event("%s", h[i]);
    pthread_mutex_unlock(&R.mu);
}

static void handle_line(char *line) {
    while (*line == ' ' || *line == '\t') line++;   /* prompts are trimmed here, not by the engine */
    size_t n = strlen(line);
    while (n && (line[n - 1] == ' ' || line[n - 1] == '\t' || line[n - 1] == '\r')) line[--n] = 0;
    if (!n) return;
    pthread_mutex_lock(&R.mu);
    int state = R.state;
    pthread_mutex_unlock(&R.mu);
    char f[1200], w[400];
    if (line[0] != '/') {
        if (state == IDLE) { open_stream(line); return; }
        if (state != LIVE) { pthread_mutex_lock(&R.mu); post_event("  %s the stream is %s; try again in a moment", G("…", "..."), state == OPENING ? "starting" : "ending"); pthread_mutex_unlock(&R.mu); return; }
        char pe[1100]; json_str(pe, sizeof pe, line);
        snprintf(f, sizeof f, "\"prompt\":\"%s\"", pe);
        snprintf(w, sizeof w, "prompt \"%.300s\"", line);
        control("prompt", w, pe, f);
        return;
    }
    char cmd[32] = "", a1[128] = "", a2[64] = "";
    sscanf(line, "/%31s %127s %63s", cmd, a1, a2);
    if (!strcmp(cmd, "quit") || !strcmp(cmd, "q") || !strcmp(cmd, "exit")) { pthread_mutex_lock(&R.mu); R.quit = 1; pthread_cond_broadcast(&R.cv); pthread_mutex_unlock(&R.mu); return; }
    if (!strcmp(cmd, "help") || !strcmp(cmd, "h")) { help(); return; }
    if (!strcmp(cmd, "steers")) {
        char resp[8192]; int code = http("GET", "/steers", NULL, resp, sizeof resp);
        pthread_mutex_lock(&R.mu);
        if (code != 200) post_event("  %s %s", G("✗", "x"), resp);
        else {
            char out[400] = ""; size_t k = 0;
            for (const char *p = resp; (p = strstr(p, "\"name\":\"")); ) {
                p += 8; size_t l = strcspn(p, "\"");
                const char *site = strstr(p, "\"site\":\""); char sn[16] = "";
                if (site) sscanf(site + 8, "%15[^\"]", sn);
                k += (size_t)snprintf(out + k, sizeof out - k, "%s%.*s (%s)", k ? ", " : "", (int)l, p, sn);
                if (k > sizeof out - 64) break;
            }
            post_event("  steers: %s", k ? out : "none (start the server with --steer-dir)");
        }
        pthread_mutex_unlock(&R.mu);
        return;
    }
    if (!strcmp(cmd, "pause") || !strcmp(cmd, "resume")) {
        pthread_mutex_lock(&R.mu); R.paused = cmd[0] == 'p'; post_event("  %s %s", R.paused ? G("‖", "||") : G("▶", ">"), R.paused ? "paused" : "resumed"); pthread_mutex_unlock(&R.mu);
        return;
    }
    if (!strcmp(cmd, "status")) {
        pthread_mutex_lock(&R.mu);
        char st[400]; status_line(st, sizeof st);
        post_event("  %s  session %s, %d chunks, underruns %d (%.1f s), lookahead %.1f-%.1f s",
                   st, R.sid[0] ? R.sid : "-", R.chunks, R.underruns, R.sr ? (double)R.silence / R.sr : 0.0, R.ahead_min, R.ahead_max);
        pthread_mutex_unlock(&R.mu);
        return;
    }
    if (!strcmp(cmd, "steer")) {
        char spec[200];
        if (!strcmp(a1, "off")) { snprintf(spec, sizeof spec, "off"); snprintf(w, sizeof w, "steer off"); R.nreq = 0; }
        else if (a1[0] && a2[0]) {
            int i = steer_index(a1);
            float from = i >= 0 ? R.req_scale[i] : 0, to = (float)atof(a2);
            if (i >= 0) R.req_scale[i] = to;
            snprintf(spec, sizeof spec, "%s:%g", a1, to);
            snprintf(w, sizeof w, "%s %.2f %s %.2f", a1, from, G("→", "->"), to);
        } else { pthread_mutex_lock(&R.mu); post_event("  usage: /steer <name> <scale> | /steer off"); pthread_mutex_unlock(&R.mu); return; }
        if (state != LIVE) {   /* before a stream: sent with the open */
            if (!strcmp(spec, "off")) R.pre_steer[0] = 0;
            else { size_t k = strlen(R.pre_steer); snprintf(R.pre_steer + k, sizeof R.pre_steer - k, "%s%s", k ? "," : "", spec); }
            pthread_mutex_lock(&R.mu); post_event("  %s %s (applies when the stream starts)", G("↳", "->"), w); pthread_mutex_unlock(&R.mu);
            return;
        }
        snprintf(f, sizeof f, "\"steer\":\"%s\"", spec);
        control("steer", w, spec, f);
        return;
    }
    if (!strcmp(cmd, "seed") || !strcmp(cmd, "evolve") || !strcmp(cmd, "anchor")) {
        if (!a1[0]) { pthread_mutex_lock(&R.mu); post_event("  usage: /%s <value>", cmd); pthread_mutex_unlock(&R.mu); return; }
        if (state != LIVE && !strcmp(cmd, "seed")) { R.seed = atoll(a1); R.seed_set = 1; pthread_mutex_lock(&R.mu); post_event("  %s seed %s (applies when the stream starts)", G("↳", "->"), a1); pthread_mutex_unlock(&R.mu); return; }
        if (state != LIVE) { pthread_mutex_lock(&R.mu); post_event("  /%s needs a live stream", cmd); pthread_mutex_unlock(&R.mu); return; }
        snprintf(f, sizeof f, "\"%s\":%s", cmd, a1);
        snprintf(w, sizeof w, "%s %s", cmd, a1);
        control(cmd, w, a1, f);
        return;
    }
    if (!strcmp(cmd, "stop")) {
        if (state != LIVE) return;
        pthread_mutex_lock(&R.mu); R.stop_sent = 1; pthread_mutex_unlock(&R.mu);
        control("stop", NULL, NULL, "\"stop\":true");
        return;
    }
    pthread_mutex_lock(&R.mu); post_event("  unknown command /%s (try /help)", cmd); pthread_mutex_unlock(&R.mu);
}

/* ---------------- main ------------------------------------------------------------------ */
#ifndef ARIA_REPL_NO_MAIN   /* tests/test_repl.c includes this file for the editor and helpers */

static volatile sig_atomic_t g_sig;
static void on_sig(int s) { g_sig = s; char c = 1; if (write(R.wake[1], &c, 1)) {} }

static void usage(FILE *f) {
    fputs("aria-repl - interactive client for aria-server live sessions\n\n"
          "Usage: aria-repl [--host 127.0.0.1:8321] [--record take.wav] [--history take.jsonl]\n"
          "                 [--player \"cmd\" | --no-play | --out -] [--chunk s] [--context s]\n"
          "                 [--buffer s] [--seed n] [--latency s] [--duration s]\n\n"
          "  --record f.wav     save everything received (float WAV, valid even if interrupted)\n"
          "  --history f.jsonl  log every prompt / steer / seed change with when it became audible\n"
          "  --player \"cmd\"     play through cmd ({sr} {ch} are substituted; raw f32 on its stdin);\n"
          "                     default: the first of pw-play, paplay, aplay, play, ffplay found\n"
          "  --no-play          no audio output (recording still works)\n"
          "  --out -            raw f32 to stdout instead of a player (pipe it anywhere)\n"
          "  --buffer s         audio kept ahead of playback (default: one chunk); lower = changes\n"
          "                     land sooner; it grows by 0.5 s after each underrun\n"
          "  --chunk/--context  session shape (default: the server's tuned values)\n"
          "  --latency s        player latency estimate for \"lands in\" (default 0.25)\n"
          "  --duration s       stop after s seconds of audio (scripts)\n\n"
          "Then type a prompt. /help lists the commands.\n", f);
}

int main(int argc, char **argv) {
    snprintf(R.host, sizeof R.host, "127.0.0.1");
    R.ahead = -1;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(stdout); return 0; }
        else if (!strcmp(a, "--no-play")) R.no_play = 1;
        else if (!v) { usage(stderr); return 1; }
        else if (!strcmp(a, "--host")) {
            const char *c = strrchr(v, ':');
            if (c) { snprintf(R.host, sizeof R.host, "%.*s", (int)(c - v), v); R.port = atoi(c + 1); }
            else snprintf(R.host, sizeof R.host, "%s", v);
            i++;
        }
        else if (!strcmp(a, "--record")) { R.rec_path = v; i++; }
        else if (!strcmp(a, "--history")) { R.hist_path = v; i++; }
        else if (!strcmp(a, "--player")) { R.player = v; i++; }
        else if (!strcmp(a, "--out")) { if (strcmp(v, "-")) { usage(stderr); return 1; } R.to_stdout = 1; i++; }
        else if (!strcmp(a, "--chunk")) { R.chunk = atof(v); i++; }
        else if (!strcmp(a, "--context")) { R.context = atof(v); i++; }
        else if (!strcmp(a, "--ahead") || !strcmp(a, "--buffer")) { R.ahead = atof(v); i++; }
        else if (!strcmp(a, "--seed")) { R.seed = atoll(v); R.seed_set = 1; i++; }
        else if (!strcmp(a, "--latency")) { R.latency = atof(v); i++; }
        else if (!strcmp(a, "--duration")) { R.duration = atof(v); i++; }
        else { usage(stderr); return 1; }
    }
    if (R.to_stdout && isatty(1)) { fprintf(stderr, "aria-repl: --out - writes raw audio; pipe it to a player\n"); return 1; }
    setlocale(LC_CTYPE, "");
    R.utf8 = !strcmp(nl_langinfo(CODESET), "UTF-8");
    R.tty = isatty(0) && isatty(2);
    if (R.rec_path && !(R.rec = fopen(R.rec_path, "wb"))) { fprintf(stderr, "aria-repl: cannot write %s\n", R.rec_path); return 1; }
    if (R.hist_path && !(R.hist = fopen(R.hist_path, "a"))) { fprintf(stderr, "aria-repl: cannot write %s\n", R.hist_path); return 1; }
    if (pipe2(R.wake, O_CLOEXEC | O_NONBLOCK)) return 1;
    signal(SIGPIPE, SIG_IGN);
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_handler = on_sig;
    sigaction(SIGINT, &sa, NULL); sigaction(SIGTERM, &sa, NULL); sigaction(SIGHUP, &sa, NULL); sigaction(SIGWINCH, &sa, NULL);

    char info[1024];
    int code = http("GET", "/info", NULL, info, sizeof info);
    if (code != 200) { fprintf(stderr, "aria-repl: no aria-server at %s:%d (%s)\n", R.host, R.port, info); return 1; }
    char mt[64] = "?"; double v, sr = 0, chs = 0, rtf = 0;
    aria_json_get_string(info, "model_type", mt, sizeof mt);
    aria_json_get_number(info, "sample_rate", &sr); aria_json_get_number(info, "chunk", &chs); aria_json_get_number(info, "rtf", &rtf);
    (void)v;
    if (isatty(2)) fprintf(stderr, "\n%s\n", aria_banner());
    char speed[96];
    if (rtf > 0) snprintf(speed, sizeof speed, "server %.1f%s real time at %.0f s chunks", rtf, G("×", "x"), chs);
    else snprintf(speed, sizeof speed, "%.0f s chunks", chs);
    fprintf(stderr, "aria live %s %s @ %s:%d %s %.1f kHz %s %s\n",
            G("·", "-"), mt, R.host, R.port, G("·", "-"), sr / 1000, G("·", "-"), speed);
    fprintf(stderr, "type a prompt to start %s /help for commands\n\n", G("·", "-"));

    if (R.tty) {
        tcgetattr(0, &tty_saved);
        struct termios t = tty_saved;
        t.c_lflag &= (tcflag_t)~(ECHO | ICANON | ISIG | IEXTEN);
        t.c_iflag &= (tcflag_t)~(IXON | ICRNL);
        t.c_cc[VMIN] = 1; t.c_cc[VTIME] = 0;
        tcsetattr(0, TCSAFLUSH, &t);
        tty_raw = 1;
        atexit(tty_restore);
    }
    pthread_t net, play;
    pthread_create(&net, NULL, net_main, NULL);
    pthread_create(&play, NULL, play_main, NULL);

    char lb[4096]; size_t ll = 0;
    double t_credit = 0;
    pthread_mutex_lock(&R.mu); ui_draw(); pthread_mutex_unlock(&R.mu);
    for (;;) {
        pthread_mutex_lock(&R.mu);
        int quit = R.quit;
        pthread_mutex_unlock(&R.mu);
        if (quit) break;
        fd_set fds; FD_ZERO(&fds); FD_SET(R.wake[0], &fds);
        if (!R.stdin_eof) FD_SET(0, &fds);
        struct timeval tv = { 0, 200000 };
        int r = select(R.wake[0] + 1, &fds, NULL, NULL, &tv);
        if (g_sig && g_sig != SIGWINCH) {   /* first signal: stop gracefully; second: now */
            pthread_mutex_lock(&R.mu);
            if (R.sigq++) { tty_restore(); _exit(130); }
            R.quit = 1; pthread_cond_broadcast(&R.cv);
            pthread_mutex_unlock(&R.mu);
            g_sig = 0;
            break;
        }
        g_sig = 0;
        if (r > 0 && FD_ISSET(R.wake[0], &fds)) { char d[64]; while (read(R.wake[0], d, sizeof d) > 0) {} }
        if (r > 0 && !R.stdin_eof && FD_ISSET(0, &fds)) {
            unsigned char in[512];
            ssize_t k = read(0, in, sizeof in);
            if (k <= 0) R.stdin_eof = 1;
            for (ssize_t i = 0; i < k; i++) {
                if (R.tty) {
                    int e = le_key(&ED, in[i]);
                    if (e == LE_SUBMIT) {
                        char line[1024]; snprintf(line, sizeof line, "%s", ED.buf);
                        ED.len = ED.pos = 0; ED.buf[0] = 0;
                        pthread_mutex_lock(&R.mu);
                        fprintf(stderr, "\r\x1b[J%saria> %s\x1b[0m\r\n", "\x1b[1m", line);   /* echo it into the log */
                        ui_draw();
                        pthread_mutex_unlock(&R.mu);
                        handle_line(line);
                    } else if (e == LE_EOF || e == LE_INTR) { pthread_mutex_lock(&R.mu); R.quit = 1; pthread_cond_broadcast(&R.cv); pthread_mutex_unlock(&R.mu); break; }
                    else if (e == LE_REDRAW) { pthread_mutex_lock(&R.mu); ui_draw(); pthread_mutex_unlock(&R.mu); }
                } else if (in[i] == '\n') {
                    lb[ll] = 0; ll = 0;
                    fprintf(stderr, "aria> %s\n", lb);
                    handle_line(lb);
                } else if (ll < sizeof lb - 1) lb[ll++] = (char)in[i];
            }
        }
        /* credits, auto-stop, notices, status */
        pthread_mutex_lock(&R.mu);
        double t = now_s();
        int live = R.state == LIVE;
        int64_t played = R.played, credited = R.credited;
        char sid[16]; snprintf(sid, sizeof sid, "%s", R.sid);
        int want_stop = live && R.duration > 0 && !R.stop_sent && R.sr && (double)R.played / R.sr >= R.duration;
        if (want_stop) R.stop_sent = 1;
        if (R.duration > 0 && R.stdin_eof && R.state == IDLE && R.chunks > 0) R.quit = 1;   /* scripted run finished */
        if (!R.tty && R.stdin_eof && R.duration <= 0 && R.state == IDLE) R.quit = 1;
        ui_audible();
        ui_flush_events();
        ui_draw();
        pthread_mutex_unlock(&R.mu);
        pthread_mutex_lock(&R.mu);
        aria_sock sock = R.sock; double buffer = R.buffer; int resend = buffer != R.buffer_sent;
        pthread_mutex_unlock(&R.mu);
        if (live && sock != ARIA_SOCK_BAD && (played != credited || resend) && t - t_credit >= 0.1) {
            unsigned char fr[96]; char js[80];   /* CRED on the stream connection: no new TCP connection */
            int n = snprintf(js, sizeof js, "{\"played\":%lld,\"ahead\":%.2f}", (long long)played, buffer);
            live_hdr(fr, "CRED", (uint32_t)n); memcpy(fr + 8, js, (size_t)n);
            if (aria_send_all(sock, fr, 8 + (size_t)n) == 0) {
                pthread_mutex_lock(&R.mu); R.credited = played; R.buffer_sent = buffer; pthread_mutex_unlock(&R.mu);
            }
            t_credit = t;
        }
        if (want_stop) control("stop", NULL, NULL, "\"stop\":true");
    }

    /* shutdown: stop the stream (what is buffered is dropped), finalize files */
    pthread_mutex_lock(&R.mu);
    int live = R.state == LIVE || R.state == OPENING;
    char sid[16]; snprintf(sid, sizeof sid, "%s", R.sid);
    R.quit = 1; pthread_cond_broadcast(&R.cv);
    pthread_mutex_unlock(&R.mu);
    if (live && sid[0]) { char b[96], resp[256]; snprintf(b, sizeof b, "{\"session\":\"%s\",\"stop\":true}", sid); http("POST", "/stream/control", b, resp, sizeof resp); }
    tty_restore();
    pthread_join(play, NULL);
    pthread_join(net, NULL);
    pthread_mutex_lock(&R.mu);
    ui_flush_events();
    rec_patch();
    if (R.rec) fclose(R.rec);
    if (R.hist) fclose(R.hist);
    if (R.played) fprintf(stderr, "[aria-repl] played %.1f s %s underruns %d (%.1f s) %s lookahead %.2f/%.2f/%.2f s %s %.1f%s %s change lag avg %.1f / max %.1f s\n",
            R.sr ? (double)R.played / R.sr : 0.0, G("·", "-"), R.underruns, R.sr ? (double)R.silence / R.sr : 0.0, G("·", "-"),
            R.ahead_min, R.ahead_n ? R.ahead_sum / R.ahead_n : 0.0, R.ahead_max, G("·", "-"), R.rtf, G("×", "x"), G("·", "-"),
            R.lag_n ? R.lag_sum / R.lag_n : 0.0, R.lag_max);
    pthread_mutex_unlock(&R.mu);
    return 0;
}
#endif
