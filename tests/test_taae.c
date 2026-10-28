/* test_taae.c - hermetic: the batched chunk block (taae_block_forward over nch chunks)
 * is byte-identical to running each 34-token chunk on its own, on the fp32 path and the
 * fp16 overlay path, and independent of the thread count. Random weights, real dims. */
#include "../src/aria_taae.h"
#include "../src/aria_ops.h"
#include "../src/aria_arena.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rs = 0x9E3779B97F4A7C15ULL;
static float urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)(rs >> 40) / (float)(1 << 24) - 0.5f; }
static float *rnd(size_t n, float sc) { float *p = malloc(n * sizeof(float)); for (size_t i = 0; i < n; i++) p[i] = sc * urand(); return p; }

static void make_block(taae_block_w *w) {
    memset(w, 0, sizeof(*w));
    const int D = TAAE_D, hd = TAAE_HD;
    w->pre_alpha = 0.7f; w->qn_alpha = 0.9f; w->kn_alpha = 1.1f; w->ff_alpha = 0.8f;
    w->pre_gamma = rnd(D, 2); w->pre_beta = rnd(D, 0.2f);
    w->qn_gamma = rnd(hd, 2); w->qn_beta = rnd(hd, 0.2f);
    w->kn_gamma = rnd(hd, 2); w->kn_beta = rnd(hd, 0.2f);
    w->ff_gamma = rnd(D, 2); w->ff_beta = rnd(D, 0.2f);
    w->to_qkv = rnd((size_t)TAAE_QKV * D, 0.08f);
    w->to_out = rnd((size_t)D * D, 0.08f);
    w->ff_in_w = rnd((size_t)2 * TAAE_INNER * D, 0.08f); w->ff_in_b = rnd(2 * TAAE_INNER, 0.1f);
    w->ff_out_w = rnd((size_t)D * TAAE_INNER, 0.05f); w->ff_out_b = rnd(D, 0.1f);
}

static int check(const char *name, const taae_block_w *w, const float *x0, int nch, int nth) {
    const int S = TAAE_S, D = TAAE_D;
    size_t n = (size_t)nch * S * D;
    float rc[TAAE_S * (TAAE_ROT / 2)], rsn[TAAE_S * (TAAE_ROT / 2)];
    aria_rope_freqs(rc, rsn, S, TAAE_ROT, 10000.0f);
    float *a = malloc(n * sizeof(float)), *b = malloc(n * sizeof(float));
    memcpy(a, x0, n * sizeof(float)); memcpy(b, x0, n * sizeof(float));
#ifdef _OPENMP
    int prev = omp_get_max_threads();
    omp_set_num_threads(nth);
#endif
    aria_arena ar; aria_arena_init(&ar, taae_block_arena_bytes(nch * S));
    taae_block_forward(a, nch, S, w, rc, rsn, &ar);                                   /* batched */
    for (int c = 0; c < nch; c++) taae_block_forward(b + (size_t)c * S * D, 1, S, w, rc, rsn, &ar);  /* per chunk */
    aria_arena_free(&ar);
#ifdef _OPENMP
    omp_set_num_threads(prev);
#endif
    int ok = memcmp(a, b, n * sizeof(float)) == 0;
    printf("%s %s (nch=%d threads=%d)\n", ok ? "ok  " : "FAIL", name, nch, nth);
    free(a); free(b);
    return ok ? 0 : 1;
}

int main(void) {
    const int nch = 7;
    float *x0 = rnd((size_t)nch * TAAE_S * TAAE_D, 2.0f);
    taae_block_w w; make_block(&w);
    int fails = 0, maxth = 1;
#ifdef _OPENMP
    maxth = omp_get_max_threads();
#endif
    fails += check("fp32 batched == per-chunk", &w, x0, nch, maxth);
    fails += check("fp32 batched == per-chunk", &w, x0, nch, 1);
    /* fp16 overlay (B3 path: unfused FF through aria_linear_qw); quantize a fresh copy
     * because the overlay releases the fp32 source pages */
    taae_block_w wq; make_block(&wq);
    taae_block_quantize(&wq, TAAE_D, TAAE_INNER, ARIA_F16);
    fails += check("fp16 overlay batched == per-chunk", &wq, x0, nch, maxth);
    if (fails) { printf("test_taae: %d failures\n", fails); return 1; }
    printf("test_taae: OK\n");
    return 0;
}
