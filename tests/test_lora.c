/* test_lora.c - hermetic tests for the LoRA runtime adapter (E12.9), no model.
 *
 * Covers: (a) rank-0 / scale-0 is a bit-identical no-op over the base GEMM,
 * (b) the low-rank residual matches a hand-computed x@W^T + scale*(x@downT)@upT,
 * (c) name-mapping picks the right block/projection, and the safetensors loader
 * (shape->rank/in/out, PEFT names, alpha/rank scale). */
#include "../src/aria_lora.h"
#include "../src/aria_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "test_tmp.h"
#ifdef _WIN32
#include <io.h>      /* _mktemp_s */
#endif

static int fails = 0;
static void ok(const char *name, int cond) {
    if (!cond) { printf("FAIL %s\n", name); fails++; }
}
static void close_to(const char *name, float got, float want) {
    if (fabsf(got - want) > 1e-5f) { printf("FAIL %s: got %.7f want %.7f\n", name, got, want); fails++; }
}

/* deterministic pseudo-random fill in [-1,1] */
static void fill(float *a, int n, unsigned seed) {
    for (int i = 0; i < n; i++) { seed = seed * 1664525u + 1013904223u; a[i] = (seed >> 8) / (float)(1u << 23) - 1.0f; }
}

/* ---- (b) reference: y = x@W^T + scale*(x@downT)@upT ---- */
static void ref_lora(float *y, const float *x, const float *W, int M, int in, int out,
                     int rank, float scale, const float *down, const float *up) {
    for (int m = 0; m < M; m++)
        for (int o = 0; o < out; o++) {
            float base = 0.0f;
            for (int k = 0; k < in; k++) base += x[m * in + k] * W[o * in + k];
            float low = 0.0f;
            for (int r = 0; r < rank; r++) {
                float t = 0.0f;
                for (int k = 0; k < in; k++) t += x[m * in + k] * down[r * in + k];
                low += t * up[o * rank + r];
            }
            y[m * out + o] = base + scale * low;
        }
}

static void test_op(void) {
    enum { M = 3, in = 6, out = 5, rank = 4 };   /* constants, not VLAs (MSVC has none) */
    float x[M * in], W[out * in], down[rank * in], up[out * rank];
    fill(x, M * in, 1); fill(W, out * in, 2); fill(down, rank * in, 3); fill(up, out * rank, 4);

    float y0[M * out];
    aria_linear(y0, x, W, NULL, M, in, out);   /* base GEMM */

    /* (a) scale==0 and rank==0 are exact no-ops over the base output */
    float y[M * out], scratch[M * rank];
    memcpy(y, y0, sizeof y0);
    aria_lora_linear(y, x, M, in, out, rank, 0.0f, down, up, scratch);
    ok("scale0 is bit-identical", memcmp(y, y0, sizeof y0) == 0);
    memcpy(y, y0, sizeof y0);
    aria_lora_linear(y, x, M, in, out, 0, 1.5f, down, up, scratch);
    ok("rank0 is bit-identical", memcmp(y, y0, sizeof y0) == 0);

    /* (b) matches the hand-computed low-rank forward (scratch + malloc paths) */
    const float scale = 1.75f;
    float want[M * out];
    ref_lora(want, x, W, M, in, out, rank, scale, down, up);
    memcpy(y, y0, sizeof y0);
    aria_lora_linear(y, x, M, in, out, rank, scale, down, up, scratch);
    for (int i = 0; i < M * out; i++) close_to("lora fwd (scratch)", y[i], want[i]);
    memcpy(y, y0, sizeof y0);
    aria_lora_linear(y, x, M, in, out, rank, scale, down, up, NULL);   /* internal malloc */
    for (int i = 0; i < M * out; i++) close_to("lora fwd (malloc)", y[i], want[i]);
}

/* ---- (c) name mapping ---- */
static void expect_map(const char *nm, int want_layer, aria_lora_proj want_proj) {
    int layer = -1; aria_lora_proj proj = ARIA_LORA_NPROJ;
    int m = aria_lora_match_name(nm, &layer, &proj);
    ok(nm, m == 1 && layer == want_layer && proj == want_proj);
}
static void test_name_map(void) {
    const char *P = "model.transformer.layers.";
    char nm[256];
    snprintf(nm, sizeof nm, "%s3.self_attn.to_qkv.parametrizations.weight.0.lora_A", P);
    expect_map(nm, 3, ARIA_LORA_SA_TO_QKV);
    snprintf(nm, sizeof nm, "%s3.self_attn.to_out.parametrizations.weight.0.lora_B", P);
    expect_map(nm, 3, ARIA_LORA_SA_TO_OUT);
    snprintf(nm, sizeof nm, "%s11.cross_attn.to_q.lora_A.weight", P);   /* plain PEFT name */
    expect_map(nm, 11, ARIA_LORA_CA_TO_Q);
    snprintf(nm, sizeof nm, "%s0.cross_attn.to_kv.parametrizations.weight.0.lora_A", P);
    expect_map(nm, 0, ARIA_LORA_CA_TO_KV);
    snprintf(nm, sizeof nm, "%s7.cross_attn.to_out.parametrizations.weight.0.lora_B", P);
    expect_map(nm, 7, ARIA_LORA_CA_TO_OUT);
    snprintf(nm, sizeof nm, "%s5.ff.ff.0.proj.parametrizations.weight.0.lora_A", P);
    expect_map(nm, 5, ARIA_LORA_FF_IN);
    snprintf(nm, sizeof nm, "%s5.ff.ff.2.parametrizations.weight.0.lora_B", P);
    expect_map(nm, 5, ARIA_LORA_FF_OUT);

    /* negatives: adapted-but-unhooked layers have no block-GEMM substring */
    int layer, m;
    m = aria_lora_match_name("model.transformer.project_in.parametrizations.weight.0.lora_A", &layer, NULL);
    ok("project_in not mapped", m == 0);
    m = aria_lora_match_name("model.transformer.layers.5.to_scale_shift_gate.lora_A.weight", &layer, NULL);
    ok("scale_shift_gate not mapped", m == 0);

    /* find on a hand-built adapter */
    aria_lora_item it[2] = {
        { .layer = 2, .proj = ARIA_LORA_FF_IN, .rank = 1 },
        { .layer = 4, .proj = ARIA_LORA_CA_TO_KV, .rank = 1 },
    };
    aria_lora_adapter a = { .n = 2, .items = it };
    ok("find hit",  aria_lora_find(&a, 4, ARIA_LORA_CA_TO_KV) == &it[1]);
    ok("find miss", aria_lora_find(&a, 2, ARIA_LORA_SA_TO_QKV) == NULL);
}

/* ---- loader: write a minimal safetensors adapter and load it ---- */
static void test_loader(void) {
    enum { rank = 4, in = 6, out = 9 };
    float down[rank * in], up[out * rank];
    fill(down, rank * in, 7); fill(up, out * rank, 8);
    size_t dsz = sizeof down, usz = sizeof up;   /* 96, 144 bytes */

    /* header: __metadata__ carries the (escaped) lora_config with alpha=8; two tensors
     * under one block GEMM. data_offsets are byte ranges into the tensor-data region. */
    char header[1024];
    int hn = snprintf(header, sizeof header,
        "{\"__metadata__\":{\"lora_config\":\"{\\\"rank\\\": 4, \\\"alpha\\\": 8}\"},"
        "\"model.transformer.layers.2.self_attn.to_qkv.parametrizations.weight.0.lora_A\":"
        "{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[0,%zu]},"
        "\"model.transformer.layers.2.self_attn.to_qkv.parametrizations.weight.0.lora_B\":"
        "{\"dtype\":\"F32\",\"shape\":[%d,%d],\"data_offsets\":[%zu,%zu]}}",
        rank, in, dsz, out, rank, dsz, dsz + usz);

    char path[512];
    tmp_path(path, sizeof path, "aria_lora_testXXXXXX");
#ifdef _WIN32
    FILE *f = _mktemp_s(path, strlen(path) + 1) == 0 ? fopen(path, "wb") : NULL;
#else
    int fd = mkstemp(path);
    FILE *f = fd >= 0 ? fdopen(fd, "wb") : NULL;
#endif
    ok("temp file", f != NULL);
    if (!f) return;
    uint64_t hlen = (uint64_t)hn;   /* x86 host: little-endian */
    fwrite(&hlen, 8, 1, f);
    fwrite(header, 1, hn, f);
    fwrite(down, 1, dsz, f);
    fwrite(up, 1, usz, f);
    fclose(f);

    /* alpha from metadata (=8), rank from shape (=4) -> scale = 2.0 */
    aria_lora_adapter *a = aria_lora_load(path, 0.0f);
    ok("load ok", a != NULL);
    if (a) {
        ok("one item", a->n == 1);
        const aria_lora_item *it = &a->items[0];
        ok("layer", it->layer == 2);
        ok("proj", it->proj == ARIA_LORA_SA_TO_QKV);
        ok("rank", it->rank == rank);
        ok("in_dim", it->in_dim == in);
        ok("out_dim", it->out_dim == out);
        close_to("scale (alpha/rank)", it->scale, 2.0f);
        close_to("down[0]", it->down[0], down[0]);
        close_to("down[last]", it->down[rank * in - 1], down[rank * in - 1]);
        close_to("up[0]", it->up[0], up[0]);
        close_to("up[last]", it->up[out * rank - 1], up[out * rank - 1]);
        aria_lora_free(a);
    }

    /* alpha override wins: scale = 4/4 = 1.0 */
    aria_lora_adapter *b = aria_lora_load(path, 4.0f);
    ok("load override ok", b != NULL);
    if (b) { close_to("scale (override/rank)", b->items[0].scale, 1.0f); aria_lora_free(b); }

    /* missing file errors cleanly (NULL, no crash) */
    ok("missing file -> NULL", aria_lora_load("/nonexistent/aria_lora_missing.safetensors", 0.0f) == NULL);

    remove(path);
}

int main(void) {
    test_op();
    test_name_map();
    test_loader();
    if (fails) { printf("test_lora: %d failures\n", fails); return 1; }
    printf("test_lora: OK\n");
    return 0;
}
