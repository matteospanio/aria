/*
 * main.c - aria CLI
 *
 * Phase 0 subcommands:
 *   aria -m <dir> --info
 *   aria -m <dir> --list-tensors [prefix]
 *   aria --wav-roundtrip <in.wav> <out.wav>
 *
 * Phase 1 (text-to-audio):
 *   aria -m <dir> -p "prompt" -d 30 -s 8 -o out.wav
 */

#include "aria.h"
#include "aria_wav.h"
#include "aria_hpss.h"
#include "aria_parity.h"   /* .atns reader for --steer direction loading (E12.7) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>        /* sqrt for the phase-aligned crossfade */
#ifdef _WIN32
#include "aria_pthread_compat.h"
#else
#include <pthread.h>     /* writer thread so generation overlaps playback (--stream -o -) */
#endif
#ifndef _WIN32
#include <unistd.h>      /* isatty */
#include <sys/select.h>  /* non-blocking stdin for live --stream prompting */
#else
#include <io.h>
#include <fcntl.h>       /* _O_BINARY for raw f32 on stdout (--stream -o -) */
#define isatty _isatty
/* _putenv_s always overwrites; honour the overwrite=0 flag explicitly so a
 * user-provided OMP_WAIT_POLICY still wins (see main()). */
#define setenv(k, v, o) (((o) || !getenv(k)) ? _putenv_s((k), (v)) : 0)
#endif

/* per-step progress, drawn on one line; only attached when stderr is a TTY */
static void cli_progress(int step, int total, void *user) {
    (void)user;
    fprintf(stderr, "\r  denoising [%d/%d]%s", step, total, step == total ? "\n" : "");
    fflush(stderr);
}

/* ---- unified table-driven CLI (argp-like, no external deps) ---------------------
 * OPTS is the single source of truth: it drives both parsing and --help, so they
 * cannot drift. cli_parse fills a cli_config; main() dispatches off its fields. */
typedef struct {
    const char *model_dir, *prompt, *prompt_embed, *out_path, *list_prefix;
    const char *init_audio, *load_quant, *util_a, *util_b;
    float seconds, inpaint_from, inpaint_to, stream_chunk, stream_context;
    float stream_anchor;                    /* E14b: 0 = off; else blend context toward chunk-0 */
    int steps, bench, stream_chunks;
    int stream_evolve;                      /* E14c: 0 = fixed seed (continuity); else re-seed every N chunks */
    long long seed;
    aria_device device;
    aria_dtype precision;
    int do_info, do_list, do_generate, do_stream, inpaint_continue, stream_hold, rng_torch;
    int steps_set, fast;                    /* A1: --fast = 6-step preset unless -s is explicit */
    int util;                               /* 0 none / 1 wav-roundtrip / 2 hpss-test */
    const char *steer_specs[16]; int n_steer;   /* E12: repeatable --steer site:layer:dir.atns:scale:lo-hi */
    const char *batch_file;                 /* E13: --batch jobs TSV (resident multi-gen) */
    const char *lora_spec;                  /* E12.9: --lora file.safetensors[:alpha] */
    const char *steer_ramp;                 /* E14: --steer-ramp lo:hi[:tri] (stream chunks) */
} cli_config;

static cli_config cli_defaults(void) {
    cli_config c = {0};
    c.out_path = "out.wav"; c.seconds = 15.0f; c.steps = 8; c.bench = 1;
    c.seed = -1; c.device = ARIA_DEVICE_AUTO; c.precision = ARIA_F32;
    c.stream_chunk = 2.0f; c.stream_context = 6.0f; c.stream_chunks = 8;
    return c;
}

enum {   /* keys: short flags use their ASCII char; long-only options use ids past 256 */
    K_MODEL='m', K_PROMPT='p', K_DUR='d', K_STEPS='s', K_OUT='o', K_HELP='h',
    K_SEED=256, K_DEVICE, K_BENCH, K_UNCOND, K_PEMB, K_INFO, K_LIST,
    K_INPAINT, K_CONTINUE, K_FROM, K_TO, K_PREC, K_LOADQ, K_RNG,
    K_STREAM, K_CHUNK, K_CTX, K_CHUNKS, K_HOLD, K_WAVRT, K_HPSS, K_STEER,
    K_BATCH, K_LORA, K_STEERRAMP, K_ANCHOR, K_EVOLVE, K_FAST,
};
typedef enum { A_NONE, A_ONE, A_TWO, A_OPT } argkind;  /* flag / 1 arg / 2 args / optional 1 arg */
typedef struct {
    int key; const char *lng; char shrt; argkind arg;
    const char *meta, *group, *help;
} opt_spec;

static const opt_spec OPTS[] = {
    {K_MODEL,    "model",        'm', A_ONE,  "<dir>",      "core",     "model directory (required, except for utilities)"},
    {K_OUT,      "out",          'o', A_ONE,  "<file>",     "core",     "output WAV ('-' = raw f32 to stdout, for --stream)"},
    {K_HELP,     "help",         'h', A_NONE, NULL,         "core",     "show this help and exit"},

    {K_PROMPT,   "prompt",       'p', A_ONE,  "<text>",     "generate", "text prompt -> audio"},
    {K_PEMB,     "prompt-embed",  0,  A_ONE,  "<file>",     "generate", "precomputed prompt embedding (.atns)"},
    {K_UNCOND,   "uncond",        0,  A_NONE, NULL,         "generate", "unconditional generation"},
    {K_DUR,      "",             'd', A_ONE,  "<sec>",      "generate", "duration in seconds (default 15)"},
    {K_STEPS,    "",             's', A_ONE,  "<n>",        "generate", "denoise steps (default 8)"},
    {K_FAST,     "fast",          0,  A_NONE, NULL,         "generate", "fast preset: 6 steps (calibrated: taste-drift < seed noise; overridden by -s)"},
    {K_SEED,     "seed",          0,  A_ONE,  "<n>",        "generate", "RNG seed (default random)"},
    {K_DEVICE,   "device",        0,  A_ONE,  "<dev>",      "generate", "cpu | cuda | auto (default auto)"},
    {K_PREC,     "precision",     0,  A_ONE,  "<p>",        "generate", "fp32 | fp16 | bf16 | q8 | q4 (q8/q4 force CPU)"},
    {K_LOADQ,    "load-quant",    0,  A_ONE,  "<file>",     "generate", "load a pre-quantized .aria DiT overlay"},
    {K_RNG,      "rng",           0,  A_ONE,  "<mode>",     "generate", "xoshiro (default) | torch (parity)"},
    {K_STEER,    "steer",         0,  A_ONE,  "<spec>",     "generate", "steer site:layer:dir.atns:scale:lo-hi[:add|project] (repeatable; CPU+GPU)"},
    {K_LORA,     "lora",          0,  A_ONE,  "<file[:a]>", "generate", "apply a LoRA adapter safetensors (optional :alpha; forces CPU DiT)"},
    {K_BENCH,    "bench",         0,  A_ONE,  "<n>",        "generate", "generate N times resident, report warm-min"},
    {K_BATCH,    "batch",         0,  A_ONE,  "<jobs.tsv>", "generate", "run jobs (out<TAB>seed<TAB>steer|-<TAB>prompt) against one resident model"},

    {K_CONTINUE, "continue",      0,  A_ONE,  "<in.wav>",   "edit",     "extend a clip (GPU or CPU)"},
    {K_INPAINT,  "inpaint",       0,  A_ONE,  "<in.wav>",   "edit",     "regenerate a region of a clip"},
    {K_FROM,     "from",          0,  A_ONE,  "<sec>",      "edit",     "inpaint region start"},
    {K_TO,       "to",            0,  A_ONE,  "<sec>",      "edit",     "inpaint region end"},

    {K_STREAM,   "stream",        0,  A_NONE, NULL,         "stream",   "continuous sliding-window generation"},
    {K_CHUNK,    "chunk",         0,  A_ONE,  "<sec>",      "stream",   "seconds emitted per chunk (default 2)"},
    {K_CTX,      "context",       0,  A_ONE,  "<sec>",      "stream",   "rolling context seconds (default 6)"},
    {K_CHUNKS,   "chunks",        0,  A_ONE,  "<n>",        "stream",   "number of chunks (default 8)"},
    {K_HOLD,     "hold",          0,  A_NONE, NULL,         "stream",   "hold a steady HPSS drum loop"},
    {K_STEERRAMP,"steer-ramp",    0,  A_ONE,  "<lo:hi[:tri]>","stream",  "ramp the first --steer scale across chunks (tri = up-then-down)"},
    {K_ANCHOR,   "anchor",        0,  A_ONE,  "<beta>",     "stream",   "re-anchor each context toward the chunk-0 reference (0..1; fights energy decay)"},
    {K_EVOLVE,   "evolve",        0,  A_ONE,  "<n>",        "stream",   "fresh seed every n chunks (default 0 = fixed seed, the voice carries forward)"},

    {K_INFO,     "info",          0,  A_NONE, NULL,         "inspect",  "print model info and exit"},
    {K_LIST,     "list-tensors",  0,  A_OPT,  "[prefix]",   "inspect",  "list tensors (optional name prefix)"},

    {K_WAVRT,    "wav-roundtrip", 0,  A_TWO,  "<in> <out>", "util",     "decode + re-encode a WAV (no model)"},
    {K_HPSS,     "hpss-test",     0,  A_ONE,  "<in.wav>",   "util",     "write harmonic.wav + percussive.wav (no model)"},
    {0}
};

/* apply one matched option to the config; returns 0 ok, 1 error (message printed). */
static int cli_apply(int key, char **a, cli_config *c) {
    switch (key) {
    case K_MODEL:  c->model_dir = a[0]; break;
    case K_OUT:    c->out_path = a[0]; break;
    case K_PROMPT: c->prompt = a[0]; c->do_generate = 1; break;
    case K_PEMB:   c->prompt_embed = a[0]; c->do_generate = 1; break;
    case K_UNCOND: c->do_generate = 1; break;
    case K_DUR:    c->seconds = (float)atof(a[0]); break;
    case K_STEPS:  c->steps = atoi(a[0]); c->steps_set = 1; break;
    case K_FAST:   c->fast = 1; break;
    case K_SEED:   c->seed = atoll(a[0]); break;
    case K_BENCH:  c->bench = atoi(a[0]); if (c->bench < 1) c->bench = 1; break;
    case K_DEVICE:
        if (!strcmp(a[0], "cpu")) c->device = ARIA_DEVICE_CPU;
        else if (!strcmp(a[0], "cuda") || !strcmp(a[0], "gpu")) c->device = ARIA_DEVICE_CUDA;
        else c->device = ARIA_DEVICE_AUTO;
        break;
    case K_PREC:
        if (aria_dtype_parse(a[0], &c->precision) != 0) {
            fprintf(stderr, "unknown --precision %s (use fp32|fp16|bf16|q8|q4)\n", a[0]); return 1;
        }
        break;
    case K_LOADQ:  c->load_quant = a[0]; break;
    case K_RNG:
        if (!strcmp(a[0], "torch")) c->rng_torch = 1;
        else if (!strcmp(a[0], "xoshiro")) c->rng_torch = 0;
        else { fprintf(stderr, "unknown --rng %s (use xoshiro|torch)\n", a[0]); return 1; }
        break;
    case K_CONTINUE: c->init_audio = a[0]; c->inpaint_continue = 1; c->do_generate = 1; break;
    case K_INPAINT:  c->init_audio = a[0]; c->do_generate = 1; break;
    case K_FROM:   c->inpaint_from = (float)atof(a[0]); break;
    case K_TO:     c->inpaint_to = (float)atof(a[0]); break;
    case K_STREAM: c->do_stream = 1; break;
    case K_CHUNK:  c->stream_chunk = (float)atof(a[0]); break;
    case K_CTX:    c->stream_context = (float)atof(a[0]); break;
    case K_CHUNKS: c->stream_chunks = atoi(a[0]); break;
    case K_HOLD:   c->stream_hold = 1; break;
    case K_ANCHOR: c->stream_anchor = (float)atof(a[0]); break;
    case K_EVOLVE: c->stream_evolve = atoi(a[0]); break;
    case K_STEER:
        if (c->n_steer >= 16) { fprintf(stderr, "too many --steer (max 16)\n"); return 1; }
        c->steer_specs[c->n_steer++] = a[0]; break;
    case K_BATCH:  c->batch_file = a[0]; break;
    case K_LORA:   c->lora_spec = a[0]; c->do_generate = 1; break;
    case K_STEERRAMP: c->steer_ramp = a[0]; break;
    case K_INFO:   c->do_info = 1; break;
    case K_LIST:   c->do_list = 1; if (a[0]) c->list_prefix = a[0]; break;
    case K_WAVRT:  c->util = 1; c->util_a = a[0]; c->util_b = a[1]; break;
    case K_HPSS:   c->util = 2; c->util_a = a[0]; break;
    default: return 1;
    }
    return 0;
}

static void cli_help(const char *prog) {
    fprintf(stderr, "aria - audio diffusion inference runtime\n\nUsage: %s [options]\n", prog);
    static const char *groups[] = {"core","generate","edit","stream","inspect","util"};
    static const char *titles[] = {"Core","Generation","Continue / inpaint","Streaming","Inspect","Utilities (no model)"};
    for (size_t g = 0; g < sizeof(groups)/sizeof(*groups); g++) {
        fprintf(stderr, "\n%s:\n", titles[g]);
        for (const opt_spec *o = OPTS; o->key; o++) {
            if (strcmp(o->group, groups[g])) continue;
            char left[56]; int n = 0;
            if (o->shrt) n += snprintf(left+n, sizeof left-n, "-%c", o->shrt);
            if (o->shrt && o->lng && o->lng[0]) n += snprintf(left+n, sizeof left-n, ", ");
            if (o->lng && o->lng[0]) n += snprintf(left+n, sizeof left-n, "--%s", o->lng);
            if (o->meta) snprintf(left+n, sizeof left-n, " %s", o->meta);
            fprintf(stderr, "  %-26s %s\n", left, o->help);
        }
    }
    fprintf(stderr, "\nLive play + re-steer (type a new prompt + Enter any time):\n"
                    "  %s -m <dir> --stream -o - -p \"...\" | play -t raw -r <sr> -e float -b 32 -c <ch> -\n", prog);
}

/* parse argv into cfg; returns 0 ok, 1 error, or -1 when --help was shown. */
static int cli_parse(int argc, char **argv, cli_config *c) {
    *c = cli_defaults();
    for (int i = 1; i < argc; i++) {
        const char *tok = argv[i];
        const opt_spec *o = NULL;
        if (tok[0] == '-' && tok[1] == '-') {
            for (const opt_spec *s = OPTS; s->key; s++)
                if (s->lng && s->lng[0] && !strcmp(tok + 2, s->lng)) { o = s; break; }
        } else if (tok[0] == '-' && tok[1] && !tok[2]) {
            for (const opt_spec *s = OPTS; s->key; s++)
                if (s->shrt == tok[1]) { o = s; break; }
        }
        if (!o) { fprintf(stderr, "unknown option: %s\n", tok); cli_help(argv[0]); return 1; }
        if (o->key == K_HELP) { cli_help(argv[0]); return -1; }
        char *args[2] = {NULL, NULL};
        int need = (o->arg == A_ONE) ? 1 : (o->arg == A_TWO) ? 2 : 0;
        for (int k = 0; k < need; k++) {
            if (i + 1 >= argc) { fprintf(stderr, "option %s needs %d argument(s)\n", tok, need); return 1; }
            args[k] = argv[++i];
        }
        if (o->arg == A_OPT && i + 1 < argc && argv[i + 1][0] != '-') args[0] = argv[++i];
        if (cli_apply(o->key, args, c)) return 1;
    }
    if (c->fast && !c->steps_set) c->steps = 6;   /* A1: preset unless -s was explicit (order-independent) */
    return 0;
}

/* E12: parse ONE steer spec (site:layer:dir.atns:scale:lo-hi[:add|project]) into *it,
 * loading the .atns direction (caller frees it->dir, e.g. via free_steer_set). 0 ok / 1 error. */
static int parse_steer_spec(const char *spec, aria_steer *it) {
    char buf[1024];
    snprintf(buf, sizeof buf, "%s", spec);
    char *site = strtok(buf, ":"), *layer = strtok(NULL, ":"), *path = strtok(NULL, ":");
    char *scale = strtok(NULL, ":"), *win = strtok(NULL, ":"), *op = strtok(NULL, ":");
    if (!site || !layer || !path || !scale) {
        fprintf(stderr, "bad --steer '%s' (want site:layer:dir.atns:scale:lo-hi[:add|project])\n", spec);
        return 1;
    }
    aria_steer_site s;
    if      (!strcmp(site, "residual")) s = ARIA_STEER_RESIDUAL;
    else if (!strcmp(site, "latent"))   s = ARIA_STEER_LATENT;   /* E12.2 host-side (CPU+GPU) */
    else if (!strcmp(site, "cond"))     s = ARIA_STEER_COND;     /* E12.4 host-side (CPU+GPU) */
    else { fprintf(stderr, "unknown --steer site '%s' (residual|latent|cond)\n", site); return 1; }
    aria_steer_op o = ARIA_STEER_ADD;
    if (op) {
        if      (!strcmp(op, "project") || !strcmp(op, "ablate")) o = ARIA_STEER_PROJECT;
        else if (strcmp(op, "add")) { fprintf(stderr, "unknown --steer op '%s' (add|project)\n", op); return 1; }
    }
    aria_parity_tensor t;
    if (aria_parity_load(path, &t) != 0) { fprintf(stderr, "cannot load steer direction %s\n", path); return 1; }
    float *d = malloc((size_t)t.numel * sizeof(float));
    if (!d) { aria_parity_free(&t); fprintf(stderr, "steer: out of memory\n"); return 1; }
    memcpy(d, t.data, (size_t)t.numel * sizeof(float));
    aria_parity_free(&t);
    int lo = 0, hi = 1 << 30;
    if (win) sscanf(win, "%d-%d", &lo, &hi);
    double n2 = 0; for (int c = 0; c < (int)t.numel; c++) n2 += (double)d[c] * d[c];
    it->site = s; it->op = o; it->layer = atoi(layer); it->dir = d; it->dim = (int)t.numel;
    it->dir_norm2 = (float)n2; it->scale = (float)atof(scale); it->step_lo = lo; it->step_hi = hi;
    fprintf(stderr, "[steer] %s/%s layer=%d dim=%d scale=%.4g steps[%d,%d] <- %s\n",
            site, o == ARIA_STEER_PROJECT ? "project" : "add", it->layer, it->dim, it->scale, lo, hi, path);
    return 0;
}

/* E12: parse the repeatable --steer specs into a steer set. `items` is caller storage
 * [>=n_steer]; on success fills `set` (free with free_steer_set after generation). */
static int build_steer_set(const cli_config *cfg, aria_steer *items, aria_steer_set *set) {
    set->items = items; set->n = 0;
    for (int i = 0; i < cfg->n_steer; i++) {
        if (parse_steer_spec(cfg->steer_specs[i], &items[set->n]) != 0) return 1;
        set->n++;
    }
    return 0;
}
static void free_steer_set(aria_steer_set *set) {
    for (int i = 0; i < set->n; i++) free((void *)set->items[i].dir);
}

/* E12.9: load a '--lora <file.safetensors>[:alpha]' adapter. A trailing ':<number>'
 * overrides alpha (scale = alpha/rank); anything else is treated as part of the path.
 * Returns NULL on error (message printed). Caller frees with aria_lora_free. */
static aria_lora_adapter *load_lora_arg(const char *spec) {
    char buf[1024];
    snprintf(buf, sizeof buf, "%s", spec);
    float alpha = 0.0f;
    char *colon = strrchr(buf, ':');
    if (colon) {
        char *end = NULL;
        float v = strtof(colon + 1, &end);
        if (end != colon + 1 && *end == '\0') { alpha = v; *colon = '\0'; }
    }
    aria_lora_adapter *a = aria_lora_load(buf, alpha);
    if (!a) fprintf(stderr, "cannot load LoRA adapter %s\n", buf);
    return a;
}

/* E13: batch mode -- run many generations against ONE resident ctx, amortizing model
 * open/parse, GPU weight upload/quantize and per-prompt text encoding (the caches in
 * aria_model_sa3 -- cached_emb, cdit, cdec -- only live per process; one-shot CLI sweeps
 * re-pay them every invocation). Jobs file: one line per job,
 *   out.wav<TAB>seed<TAB>steer-spec|-<TAB>prompt
 * '#'/empty lines skipped. Duration/steps/device/precision are shared from the CLI. */
static int cmd_batch(aria_ctx *ctx, const cli_config *cfg) {
    FILE *fh = fopen(cfg->batch_file, "r");
    if (!fh) { fprintf(stderr, "cannot open batch file %s\n", cfg->batch_file); return 1; }
    char line[4096];
    int ok = 0, failed = 0;
    struct timespec tb0, tb1;
    clock_gettime(CLOCK_MONOTONIC, &tb0);
    while (fgets(line, sizeof line, fh)) {
        char *nl = strchr(line, '\n'); if (nl) *nl = 0;
        if (!line[0] || line[0] == '#') continue;
        char *save = NULL;
        char *out    = strtok_r(line, "\t", &save);
        char *seed   = strtok_r(NULL, "\t", &save);
        char *spec   = strtok_r(NULL, "\t", &save);
        char *prompt = save;   /* rest of the line (prompts may contain ':' etc., not tabs) */
        if (!out || !seed || !spec || !prompt || !prompt[0]) {
            fprintf(stderr, "[batch] bad line (want out<TAB>seed<TAB>steer|-<TAB>prompt)\n");
            failed++; continue;
        }
        aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
        p.prompt = prompt; p.seconds_total = cfg->seconds; p.steps = cfg->steps;
        p.seed = atoll(seed); p.device = cfg->device; p.precision = cfg->precision;
        p.rng_torch = cfg->rng_torch;
        aria_steer it; aria_steer_set set = { &it, 0 };
        if (strcmp(spec, "-") != 0) {
            if (parse_steer_spec(spec, &it) != 0) { failed++; continue; }
            set.n = 1; p.steer = &set;
        }
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        aria_audio *audio = NULL;
        int rc = aria_generate(ctx, &p, &audio);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        free_steer_set(&set);
        if (rc != 0 || !audio) {
            fprintf(stderr, "[batch] %s: generate error: %s\n", out, aria_last_error());
            failed++; continue;
        }
        if (aria_wav_write(out, audio, 32) != 0) { fprintf(stderr, "[batch] cannot write %s\n", out); failed++; }
        else {
            ok++;
            fprintf(stderr, "[batch %d] %s (%.2fs)\n", ok,
                    out, (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
        }
        aria_audio_free(audio);
    }
    fclose(fh);
    clock_gettime(CLOCK_MONOTONIC, &tb1);
    double tot = (tb1.tv_sec - tb0.tv_sec) + (tb1.tv_nsec - tb0.tv_nsec) / 1e9;
    printf("batch: %d ok, %d failed, %.1fs total (%.2fs/job)\n", ok, failed, tot, ok ? tot / ok : 0.0);
    return failed ? 1 : 0;
}

static int cmd_wav_roundtrip(const char *in, const char *out) {
    aria_audio *a = aria_wav_read(in);
    if (!a) { fprintf(stderr, "failed to read %s\n", in); return 1; }
    printf("read: sr=%d ch=%d frames=%lld (%.2fs)\n",
           a->sample_rate, a->channels, (long long)a->num_frames,
           (double)a->num_frames / a->sample_rate);
    int rc = aria_wav_write(out, a, 32);
    if (rc != 0) { fprintf(stderr, "failed to write %s\n", out); aria_audio_free(a); return 1; }
    printf("wrote: %s (float32)\n", out);
    aria_audio_free(a);
    return 0;
}

/* --hpss-test <in.wav>: separate -> harmonic.wav + percussive.wav, report split. */
static int cmd_hpss_test(const char *in) {
    aria_audio *a = aria_wav_read(in);
    if (!a) { fprintf(stderr, "hpss: cannot read %s\n", in); return 1; }
    int64_t nf = a->num_frames; int ch = a->channels;
    float *h = malloc((size_t)nf * ch * sizeof(float)), *p = malloc((size_t)nf * ch * sizeof(float));
    aria_hpss_separate(a->data, nf, ch, h, p);
    double res = 0, eh = 0, ep = 0, ei = 0; int64_t M = nf * ch;
    for (int64_t i = 0; i < M; i++) {
        double r = a->data[i] - (h[i] + p[i]);
        res += r * r; eh += (double)h[i] * h[i]; ep += (double)p[i] * p[i]; ei += (double)a->data[i] * a->data[i];
    }
    fprintf(stderr, "hpss: residual-rms=%.2e  harmonic=%.0f%%  percussive=%.0f%% (of input energy)\n",
            sqrt(res / M), 100 * eh / (ei + 1e-9), 100 * ep / (ei + 1e-9));
    aria_audio ho = { a->sample_rate, ch, nf, h }, po = { a->sample_rate, ch, nf, p };
    aria_wav_write("harmonic.wav", &ho, 32); aria_wav_write("percussive.wav", &po, 32);
    fprintf(stderr, "hpss: wrote harmonic.wav + percussive.wav\n");
    free(h); free(p); aria_audio_free(a);
    return 0;
}

/* read a new prompt from stdin if a line is waiting (non-blocking); 1 if updated.
 * POSIX only: select() on fd 0. Windows has no equivalent for a console/pipe
 * handle, so the shim is a no-op there and cmd_stream() does not advertise live
 * re-steering on that platform. */
static int stream_poll_prompt(char *buf, size_t cap) {
#ifndef _WIN32
    fd_set fds; FD_ZERO(&fds); FD_SET(0, &fds);
    struct timeval tv = {0, 0};
    if (select(1, &fds, NULL, NULL, &tv) > 0 && FD_ISSET(0, &fds) && fgets(buf, (int)cap, stdin)) {
        size_t n = strlen(buf);
        while (n && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) buf[--n] = 0;
        return n > 0;
    }
#else
    (void)buf;
    (void)cap;
#endif
    return 0;
}

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

/* Bounded FIFO of settled audio chunks. The generator (main thread) pushes; a writer
 * thread drains to stdout. This decouples generation from the blocking pipe write so the
 * next chunk is generated WHILE the player drains the current one, instead of alternating
 * gen/play. The bound throttles a faster-than-realtime producer (e.g. a future GPU path). */
typedef struct schunk { float *data; int64_t n; struct schunk *next; } schunk;
typedef struct {
    schunk *head, *tail; int count, done;
    pthread_mutex_t m; pthread_cond_t ne, nf;
} squeue;
#define SQ_MAX 32   /* up to ~32 settled regions buffered ahead */

static void sq_push(squeue *q, float *data, int64_t n) {
    schunk *c = malloc(sizeof *c); c->data = data; c->n = n; c->next = NULL;
    pthread_mutex_lock(&q->m);
    while (q->count >= SQ_MAX) pthread_cond_wait(&q->nf, &q->m);
    if (q->tail) q->tail->next = c; else q->head = c;
    q->tail = c; q->count++;
    pthread_cond_signal(&q->ne);
    pthread_mutex_unlock(&q->m);
}
static void *sq_writer(void *arg) {   /* drains the queue to stdout at the player's pace */
    squeue *q = arg;
    for (;;) {
        pthread_mutex_lock(&q->m);
        while (!q->head && !q->done) pthread_cond_wait(&q->ne, &q->m);
        schunk *c = q->head;
        if (!c) { pthread_mutex_unlock(&q->m); break; }   /* done and drained */
        q->head = c->next; if (!q->head) q->tail = NULL; q->count--;
        pthread_cond_signal(&q->nf);
        pthread_mutex_unlock(&q->m);
        fwrite(c->data, sizeof(float), (size_t)c->n, stdout);
        fflush(stdout);
        free(c->data); free(c);
    }
    return NULL;
}

/* settle the frames [*emitted, upto) of `outp` (the harmonic timeline) as raw interleaved
 * float32, mixing in the held drum loop, and hand them to the writer thread. Lets
 * `aria --stream -o - | play -t raw -r <sr> -e float -b 32 -c <ch> -` play live. */
static void stream_flush(const aria_audio *outp, const float *held, int64_t held_len,
                         int ch, int64_t *emitted, int64_t upto, squeue *q) {
    if (upto <= *emitted) return;
    int64_t n = upto - *emitted;
    float *tmp = malloc((size_t)n * ch * sizeof(float));
    memcpy(tmp, outp->data + (*emitted) * ch, (size_t)n * ch * sizeof(float));
    stream_add_tiled(tmp, n, ch, *emitted, held, held_len);   /* no-op if held_len==0 */
    sq_push(q, tmp, n * ch);   /* the writer thread frees tmp after writing it out */
    *emitted = upto;
}

/* Streaming / interactive generation: keep the model resident and emit `emit_s`-second
 * segments by sliding-window continuation over a `context_s` rolling context. The naive
 * tail-inpaint fades (SA3 makes the regenerated end an outro), so each continuation
 * regenerates context + skip + emit + tail and emits only the STRONG BODY — skipping the
 * ~1.5 s post-context seam and discarding the ~3 s fade-out (profile measured on a long
 * continuation) — crossfaded onto the output. Each step re-reads the prompt (live
 * re-steering on a TTY). `emit_s` is the --chunk value. */
static int cmd_stream(aria_ctx *ctx, aria_gen_params *p, float emit_s, float context_s,
                      int n_chunks, int hold, const char *out_path,
                      aria_steer *ramp_target, float ramp_lo, float ramp_hi, int ramp_tri,
                      float anchor_beta, int evolve_n) {
    int sr = aria_sample_rate(ctx), ch = aria_audio_channels(ctx);
    const float skip_s = 1.5f, tail_s = 3.0f, xfade_s = 0.25f;   /* seam / fade / crossfade */
    int64_t ctx_fr = (int64_t)(context_s * sr), xf_fr = (int64_t)(xfade_s * sr);
    int64_t maxlag = (int64_t)(0.18f * sr);   /* phase-align search range (~< one beat) */
#ifdef _WIN32
    int interactive = 0;                    /* stream_poll_prompt is a no-op on Windows */
#else
    int interactive = isatty(fileno(stdin));
#endif
    char promptbuf[1024];
    float *held = NULL; int64_t held_len = 0; const char *held_prompt = NULL;  /* --hold drum loop */
    /* E14b re-anchoring: keep the chunk-0 context tail as an energy/character reference and
     * blend every later context toward it -- chained continuations can no longer inherit an
     * outro fade and decay to silence (the documented streaming ceiling). Conditioning-only:
     * the blend never reaches the output (only the regenerated body is emitted). */
    if (anchor_beta < 0.0f) anchor_beta = 0.0f;
    if (anchor_beta > 1.0f) anchor_beta = 1.0f;
    aria_audio *anchor = NULL;
    /* E14c seed policy (SA3-Realtime semantics): default = FIXED seed every chunk, so the
     * per-chunk noise is identical and the voice carries forward ("continue"); --evolve N
     * re-seeds every N chunks for fresh variation. */
    int64_t base_seed = p->seed; int evolve_k = 0;
    int to_stdout = (strcmp(out_path, "-") == 0);   /* -o - : stream raw f32 to stdout for a player */
#ifdef _WIN32
    if (to_stdout) _setmode(_fileno(stdout), _O_BINARY);   /* text mode would turn every 0x0A into CRLF */
#endif
    int64_t emitted = 0;
    squeue q = { .m = PTHREAD_MUTEX_INITIALIZER, .ne = PTHREAD_COND_INITIALIZER, .nf = PTHREAD_COND_INITIALIZER };
    pthread_t writer; int have_writer = 0;
    if (to_stdout) { pthread_create(&writer, NULL, sq_writer, &q); have_writer = 1; }

    int64_t cap = (int64_t)((context_s + emit_s) * sr) + (int64_t)n_chunks * ((int64_t)(emit_s * sr) + maxlag) + sr;
    aria_audio *outp = aria_audio_alloc(sr, ch, cap);
    if (!outp) { fprintf(stderr, "stream: OOM\n"); return 1; }
    int64_t written = 0;
    aria_audio *context = NULL;

    const char *devname = p->device == ARIA_DEVICE_CPU ? "CPU" : p->device == ARIA_DEVICE_CUDA ? "CUDA" : "auto";
    fprintf(stderr, "[stream] emit=%.1fs context=%.1fs (skip %.1f / tail %.1f / xfade %.2f) steps=%d (%s)%s%s%s\n",
            emit_s, context_s, skip_s, tail_s, xfade_s, p->steps, devname,
            anchor_beta > 0.0f ? " +anchor" : "", evolve_n > 0 ? " +evolve" : "",
            interactive ? " — type a prompt + Enter to re-steer" : "");

    for (int i = 0; i < n_chunks; i++) {
        /* poll stdin (TTY or pipe) for a new prompt -> live re-steering between chunks */
        if (stream_poll_prompt(promptbuf, sizeof promptbuf)) {
            p->prompt = promptbuf;
            fprintf(stderr, "[stream] prompt -> \"%s\"\n", promptbuf);
        }
        /* E14c: evolve -- fresh variation every N chunks (base seed stays the anchor of
         * the sequence so runs are reproducible); default (evolve_n==0) never re-seeds. */
        if (evolve_n > 0 && i > 0 && i % evolve_n == 0) {
            p->seed = (base_seed >= 0 ? base_seed : 0) + (++evolve_k);
            fprintf(stderr, "[stream] evolve: seed -> %lld\n", (long long)p->seed);
        }
        /* i==0: plain text->audio, emit the strong front [0, context+emit] (drop its fade).
         * i>0 : continuation; emit the body [context+skip, context+skip+emit]. */
        float win_s = (i == 0) ? (context_s + emit_s + tail_s)
                               : (context_s + skip_s + emit_s + tail_s);
        int64_t emit_start = (i == 0) ? 0 : (int64_t)((context_s + skip_s) * sr);
        int64_t emit_len   = (i == 0) ? (int64_t)((context_s + emit_s) * sr)
                                      : (int64_t)(emit_s * sr);
        p->seconds_total = win_s;
        p->init_audio_mem = (i == 0) ? NULL : context;
        p->inpaint_continue = (i == 0) ? 0 : 1;
        /* B5 (E14.1): decode only the region we actually emit (+ the maxlag search room
         * + a small halo margin), not the whole win_s window. CPU decoders only; the
         * GPU decode is already fast + monolithic and ignores this. */
        p->decode_from_s = (float)emit_start / sr - 0.25f;
        p->decode_to_s   = (float)(emit_start + emit_len + maxlag) / sr + 0.25f;
        if (p->decode_from_s < 0.0f) p->decode_from_s = 0.0f;
        if (ramp_target && n_chunks > 1) {
            /* E14: live steering ramp -- linear lo->hi over the chunks, or triangular
             * (lo->hi->lo) with 'tri'. The steer set points at this item, so the new
             * scale flows into the next generate (device path re-uploads per request). */
            float t = (float)i / (float)(n_chunks - 1);
            float u = ramp_tri ? (t <= 0.5f ? 2.0f * t : 2.0f * (1.0f - t)) : t;
            ramp_target->scale = ramp_lo + (ramp_hi - ramp_lo) * u;
            fprintf(stderr, "[stream] chunk %d steer scale %.3f\n", i + 1, ramp_target->scale);
        }

        aria_audio *win = NULL;
        struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
        int rc = aria_generate(ctx, p, &win);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (rc != 0 || !win) {
            fprintf(stderr, "stream: generate failed: %s\n", aria_last_error());
            if (have_writer) { pthread_mutex_lock(&q.m); q.done = 1; pthread_cond_signal(&q.ne); pthread_mutex_unlock(&q.m); pthread_join(writer, NULL); }
            aria_audio_free(outp); aria_audio_free(context); return 1;
        }
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

        /* extra `maxlag` frames so the phase-align search has room to shift into */
        int64_t slice_len = emit_len + (i == 0 ? 0 : maxlag);
        if (emit_start + slice_len > win->num_frames) slice_len = win->num_frames - emit_start;
        aria_audio *emit = stream_slice(win, emit_start, slice_len);
        if (hold) {
            /* split the emit; `outp` accumulates only the HARMONIC, the drum groove is a
             * held loop tiled later -> drums stay perfectly steady across chunks. */
            int64_t en = emit->num_frames;
            float *H = malloc((size_t)en * ch * sizeof(float)), *P = malloc((size_t)en * ch * sizeof(float));
            aria_hpss_separate(emit->data, en, ch, H, P);
            if (!held || p->prompt != held_prompt) {   /* establish/reset the loop on chunk 0 or prompt change */
                free(held); held_len = en; held = malloc((size_t)held_len * ch * sizeof(float));
                memcpy(held, P, (size_t)held_len * ch * sizeof(float));
                held_prompt = p->prompt;
                fprintf(stderr, "[stream] drum loop set (%.1fs)\n", held_len / (double)sr);
            }
            memcpy(emit->data, H, (size_t)en * ch * sizeof(float));   /* emit -> harmonic only */
            free(H); free(P);
        }
        stream_append_aligned(outp, &written, cap, emit, (i == 0) ? 0 : xf_fr, (i == 0) ? 0 : maxlag);
        aria_audio_free(emit);

        double es = emit_len / (double)sr;
        fprintf(stderr, "[stream] chunk %d/%d: emit %.1fs (win %.1fs) in %.2fs (RTF %.1fx)  \"%s\"\n",
                i + 1, n_chunks, es, win_s, dt, dt > 0 ? es / dt : 0.0, p->prompt ? p->prompt : "");

        aria_audio_free(context);
        context = stream_slice(outp, written - ctx_fr, ctx_fr);   /* just-emitted tail */
        if (anchor_beta > 0.0f) {
            if (i == 0) {
                /* chunk-0 tail = the reference the stream keeps being pulled back toward
                 * (captured pre-hold-mix: the harmonic timeline; drums are re-added below) */
                anchor = stream_slice(context, 0, context->num_frames);
                fprintf(stderr, "[stream] anchor set (%.1fs, beta %.2f)\n",
                        anchor ? anchor->num_frames / (double)sr : 0.0, anchor_beta);
            } else if (anchor && anchor->num_frames > 0) {
                /* convex blend toward the reference, then pull the context's ENERGY back to
                 * the reference level (boost-only, capped at 4x): a linear blend alone cannot
                 * out-weigh a tail that already faded 15+ dB, so the model would still hear
                 * "quiet outro" and keep fading. Peak guard keeps the taae encoder in range. */
                int64_t bn = context->num_frames < anchor->num_frames
                           ? context->num_frames : anchor->num_frames;
                double e_ctx = 0.0, e_anc = 0.0;
                for (int64_t j = 0; j < bn * ch; j++) {
                    float v = (1.0f - anchor_beta) * context->data[j] + anchor_beta * anchor->data[j];
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
        }
        if (hold) stream_add_tiled(context->data, ctx_fr, ch, written - ctx_fr, held, held_len);
        aria_audio_free(win);

        /* live playback: hand settled audio to the writer thread, holding back the crossfade
         * region (xf_fr) so the next chunk can still crossfade into it. The next chunk
         * generates while the writer feeds the player. */
        if (to_stdout) stream_flush(outp, held, held_len, ch, &emitted, written - xf_fr, &q);
    }
    int wrc = 0;
    if (to_stdout) {
        stream_flush(outp, held, held_len, ch, &emitted, written, &q);   /* final tail */
        pthread_mutex_lock(&q.m); q.done = 1; pthread_cond_signal(&q.ne); pthread_mutex_unlock(&q.m);
        pthread_join(writer, NULL); have_writer = 0;
        fprintf(stderr, "[stream] streamed %.1fs to stdout\n", written / (double)sr);
    } else {
        if (hold) stream_add_tiled(outp->data, written, ch, 0, held, held_len);   /* mix the steady drums in */
        outp->num_frames = written;
        wrc = aria_wav_write(out_path, outp, 32);
        fprintf(stderr, "[stream] wrote %s (%.1fs total)\n", out_path, written / (double)sr);
    }
    free(held);
    aria_audio_free(outp); aria_audio_free(context); aria_audio_free(anchor);
    return wrc;
}

int main(int argc, char **argv) {
    if (argc < 2) { cli_help(argv[0]); return 1; }
#ifdef _OPENMP
    /* A4: a generation is thousands of short parallel regions (per-op fork/join);
     * active waiting keeps the worker pool spinning between them instead of paying a
     * futex sleep/wake per region. Set before the first region (libgomp reads it at
     * pool init); a user-provided value always wins (setenv overwrite=0). */
    setenv("OMP_WAIT_POLICY", "active", 0);
#endif

    cli_config cfg;
    int pr = cli_parse(argc, argv, &cfg);
    if (pr < 0) return 0;   /* --help shown */
    if (pr > 0) return 1;   /* parse error (message already printed) */

    /* standalone utilities run without loading a model */
    if (cfg.util == 1) return cmd_wav_roundtrip(cfg.util_a, cfg.util_b);
    if (cfg.util == 2) return cmd_hpss_test(cfg.util_a);

    if (!cfg.model_dir) { cli_help(argv[0]); return 1; }

    aria_ctx *ctx = aria_load(cfg.model_dir);
    if (!ctx) { fprintf(stderr, "load error: %s\n", aria_last_error()); return 1; }

    fprintf(stderr, "model: %s | type=%s | sr=%d ch=%d | tensors=%d\n",
            cfg.model_dir, aria_model_type(ctx), aria_sample_rate(ctx),
            aria_audio_channels(ctx), aria_num_tensors(ctx));

    int rc = 0;
    if (cfg.do_stream) {
        aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
        p.prompt = cfg.prompt; p.steps = cfg.steps; p.seed = cfg.seed;
        p.precision = cfg.precision; p.device = cfg.device;  /* GPU continuation supported (sm_70+) */
        p.rng_torch = cfg.rng_torch;
        aria_steer steer_items[16]; aria_steer_set steer_set = {0};
        aria_steer *ramp_target = NULL; float ramp_lo = 0, ramp_hi = 0; int ramp_tri = 0;
        if (cfg.n_steer > 0) {
            if (build_steer_set(&cfg, steer_items, &steer_set) != 0) { aria_free(ctx); return 1; }
            p.steer = &steer_set;
            if (cfg.steer_ramp) {
                char rb[64]; snprintf(rb, sizeof rb, "%s", cfg.steer_ramp);
                char *lo = strtok(rb, ":"), *hi = strtok(NULL, ":"), *m = strtok(NULL, ":");
                if (!lo || !hi) { fprintf(stderr, "bad --steer-ramp (want lo:hi[:tri])\n"); aria_free(ctx); return 1; }
                ramp_lo = (float)atof(lo); ramp_hi = (float)atof(hi);
                ramp_tri = m && !strcmp(m, "tri");
                ramp_target = &steer_items[0];
            }
        } else if (cfg.steer_ramp) {
            fprintf(stderr, "--steer-ramp needs a --steer\n"); aria_free(ctx); return 1;
        }
        rc = cmd_stream(ctx, &p, cfg.stream_chunk, cfg.stream_context, cfg.stream_chunks, cfg.stream_hold, cfg.out_path,
                        ramp_target, ramp_lo, ramp_hi, ramp_tri, cfg.stream_anchor, cfg.stream_evolve);
        free_steer_set(&steer_set);
    } else if (cfg.do_list) {
        aria_list_tensors(ctx, cfg.list_prefix);
    } else if (cfg.batch_file) {
        rc = cmd_batch(ctx, &cfg);
    } else if (cfg.do_generate) {
        aria_gen_params p = ARIA_GEN_PARAMS_DEFAULT;
        p.prompt = cfg.prompt;
        p.prompt_embed_path = cfg.prompt_embed;
        p.seconds_total = cfg.seconds;
        p.steps = cfg.steps;
        p.seed = cfg.seed;
        p.device = cfg.device;
        p.precision = cfg.precision;
        p.load_quant = cfg.load_quant;
        p.init_audio = cfg.init_audio;
        p.inpaint_from_s = cfg.inpaint_from;
        p.inpaint_to_s = cfg.inpaint_to;
        p.inpaint_continue = cfg.inpaint_continue;
        p.rng_torch = cfg.rng_torch;
        aria_steer steer_items[16]; aria_steer_set steer_set = {0};   /* E12 activation steering */
        if (cfg.n_steer > 0) {
            if (build_steer_set(&cfg, steer_items, &steer_set) != 0) { aria_free(ctx); return 1; }
            p.steer = &steer_set;
        }
        aria_lora_adapter *lora = NULL;   /* E12.9 runtime LoRA adapter (CPU DiT) */
        if (cfg.lora_spec) {
            lora = load_lora_arg(cfg.lora_spec);
            if (!lora) { free_steer_set(&steer_set); aria_free(ctx); return 1; }
            p.lora = lora;
            if (p.device != ARIA_DEVICE_CPU) {   /* GPU LoRA is E12.9b (not implemented) */
                p.device = ARIA_DEVICE_CPU;
                fprintf(stderr, "[lora] GPU LoRA is out of scope (E12.9b); running the DiT on the CPU\n");
            }
        }
        if (isatty(fileno(stderr))) p.progress = cli_progress;  /* live progress on a terminal */
        double best = 1e9;
        for (int b = 0; b < cfg.bench && rc == 0; b++) {
            aria_audio *audio = NULL;
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            rc = aria_generate(ctx, &p, &audio);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double gen_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            if (rc != 0 || !audio) { fprintf(stderr, "generate error: %s\n", aria_last_error()); rc = 1; break; }
            if (gen_s < best) best = gen_s;
            if (cfg.bench > 1) fprintf(stderr, "  [bench %d/%d] %.2fs\n", b + 1, cfg.bench, gen_s);
            if (b == cfg.bench - 1) {
                aria_wav_write(cfg.out_path, audio, 32);
                printf("wrote %s (%.2fs audio, generated in %.2fs%s)\n",
                       cfg.out_path, (double)audio->num_frames / audio->sample_rate, best,
                       cfg.bench > 1 ? " warm-min" : "");
            }
            aria_audio_free(audio);
        }
        free_steer_set(&steer_set);
        if (lora) aria_lora_free(lora);
    } else if (cfg.do_info) {
        /* header already printed */
    }

    aria_free(ctx);
    return rc;
}
