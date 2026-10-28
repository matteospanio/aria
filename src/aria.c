/*
 * aria.c - Orchestrator: model load, registry, generation entry points.
 */

#include "aria.h"
#include "aria_model.h"
#include "aria_json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#ifndef _WIN32
#include <unistd.h>   /* sysconf (aria_win_compat.h shims it on Windows) */
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* ---- process-wide last error ---- */
static char g_error[512] = "";
const char *aria_last_error(void) { return g_error; }
void aria_set_error(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    vsnprintf(g_error, sizeof(g_error), fmt, ap);
    va_end(ap);
}

/* ---- model registry: append future modules here ---- */
static const aria_model_module *g_registry[] = {
    &aria_module_sa3,
    /* &aria_module_acestep,  (Phase 6) */
};
static const int g_registry_n = (int)(sizeof(g_registry) / sizeof(g_registry[0]));

const aria_model_module *aria_registry_detect(const char *model_type) {
    for (int i = 0; i < g_registry_n; i++)
        if (g_registry[i]->detect && g_registry[i]->detect(model_type))
            return g_registry[i];
    return NULL;
}

static char *join_path(const char *dir, const char *file) {
    size_t n = strlen(dir) + 1 + strlen(file) + 1;
    char *p = malloc(n);
    if (!p) return NULL;
    snprintf(p, n, "%s/%s", dir, file);
    return p;
}

aria_ctx *aria_load(const char *model_dir) {
    g_error[0] = '\0';
    aria_ctx *ctx = calloc(1, sizeof(aria_ctx));
    if (!ctx) { aria_set_error("oom"); return NULL; }
    ctx->model_dir = strdup(model_dir);

    /* model_config.json */
    char *cfg_path = join_path(model_dir, "model_config.json");
    ctx->config_json = aria_read_file(cfg_path, NULL);
    free(cfg_path);
    if (!ctx->config_json) {
        aria_set_error("cannot read %s/model_config.json", model_dir);
        aria_free(ctx);
        return NULL;
    }

    if (aria_json_get_string(ctx->config_json, "model_type", ctx->model_type, sizeof(ctx->model_type)) != 0) {
        aria_set_error("model_config.json: missing model_type");
        aria_free(ctx);
        return NULL;
    }
    double d;
    ctx->sample_rate = (aria_json_get_number(ctx->config_json, "sample_rate", &d) == 0) ? (int)d : 44100;
    ctx->channels    = (aria_json_get_number(ctx->config_json, "audio_channels", &d) == 0) ? (int)d : 2;

    /* dispatch to a module */
    ctx->module = aria_registry_detect(ctx->model_type);
    if (!ctx->module) {
        aria_set_error("no module handles model_type '%s'", ctx->model_type);
        aria_free(ctx);
        return NULL;
    }

    /* weights */
    char *st_path = join_path(model_dir, "model.safetensors");
    ctx->sf = safetensors_open(st_path);
    free(st_path);
    if (!ctx->sf) {
        aria_set_error("cannot open %s/model.safetensors", model_dir);
        aria_free(ctx);
        return NULL;
    }

    ctx->state = ctx->module->load(ctx, ctx->sf, ctx->config_json);
    if (!ctx->state) {
        /* load() sets a specific error */
        aria_free(ctx);
        return NULL;
    }
    return ctx;
}

void aria_free(aria_ctx *ctx) {
    if (!ctx) return;
    if (ctx->module && ctx->module->unload && ctx->state) ctx->module->unload(ctx->state);
    if (ctx->sf) safetensors_close(ctx->sf);
    free(ctx->config_json);
    free(ctx->model_dir);
    free(ctx);
}

const char *aria_model_type(const aria_ctx *ctx) { return ctx ? ctx->model_type : ""; }
int aria_sample_rate(const aria_ctx *ctx) { return ctx ? ctx->sample_rate : 0; }
int aria_audio_channels(const aria_ctx *ctx) { return ctx ? ctx->channels : 0; }
int aria_num_tensors(const aria_ctx *ctx) { return (ctx && ctx->sf) ? ctx->sf->num_tensors : 0; }

void aria_list_tensors(const aria_ctx *ctx, const char *prefix) {
    if (!ctx || !ctx->sf) return;
    size_t plen = prefix ? strlen(prefix) : 0;
    int shown = 0;
    for (int i = 0; i < ctx->sf->num_tensors; i++) {
        const safetensor_t *t = &ctx->sf->tensors[i];
        if (prefix && strncmp(t->name, prefix, plen) != 0) continue;
        safetensor_print(t);
        shown++;
    }
    printf("(%d tensors%s%s)\n", shown, prefix ? " under prefix " : "", prefix ? prefix : "");
}

#ifdef _OPENMP
/* physical cores = distinct SMT sibling sets (0 when the topology is unreadable) */
static int physical_cores(void) {
    static int n = -1;
    if (n >= 0) return n;
    static char seen[1024][64];
    int k = 0;
    long ncpu = sysconf(_SC_NPROCESSORS_CONF);
    for (long c = 0; c < ncpu && k < 1024; c++) {
        char path[96], buf[64] = "";
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%ld/topology/thread_siblings_list", c);
        FILE *f = fopen(path, "r");
        if (!f) continue;   /* offline / absent cpu */
        if (!fgets(buf, sizeof buf, f)) buf[0] = 0;
        fclose(f);
        int dup = 0;
        for (int i = 0; i < k && !dup; i++) dup = !strcmp(seen[i], buf);
        if (!dup) strcpy(seen[k++], buf);
    }
    n = k;
    return n;
}
#endif

int aria_generate(aria_ctx *ctx, const aria_gen_params *p, aria_audio **out) {
    if (!ctx || !ctx->module || !ctx->module->generate) {
        aria_set_error("generate: no module loaded");
        return -1;
    }
#ifdef _OPENMP
    /* default team: one thread per physical core -- the GEMMs saturate the FMA ports, so
     * SMT siblings only contend (measured faster on the i9). Exact: no op reduces across
     * threads. Set per call: OpenMP team sizes are per calling thread (aria-server
     * generates on a worker). OMP_NUM_THREADS still wins. */
    if (!getenv("OMP_NUM_THREADS")) {
        int pc = physical_cores();
        if (pc > 0 && pc < omp_get_num_procs()) omp_set_num_threads(pc);
    }
#endif
    return ctx->module->generate(ctx, ctx->state, p, out);
}
