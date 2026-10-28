/* test_gemm.c - hermetic: the packed GEMM is bit-exact against a scalar fmaf chain
 * (acc = +0, k ascending, then + bias) for edge M/N tiles, the K-blocked path and the
 * fp16/bf16 weight sources -- the per-element invariant that keeps every model output
 * byte-identical across GEMM driver changes -- and does not depend on the thread count. */
#include "../src/aria_ops.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rs = 88172645463325252ULL;
static float urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)(rs >> 40) / (float)(1 << 24) - 0.5f; }

/* wt: 0 fp32, 1 fp16, 2 bf16 (weights already rounded to that storage) */
static void ref(float *y, const float *x, const float *Wf, const float *b, int M, int K, int N) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc = fmaf(x[(size_t)m * K + k], Wf[(size_t)n * K + k], acc);
            y[(size_t)m * N + n] = b ? acc + b[n] : acc;
        }
}

static int run(int M, int K, int N, int wt, int bias, int nth) {
    size_t nx = (size_t)M * K, nw = (size_t)N * K, ny = (size_t)M * N;
    float *x = malloc(nx * 4), *Wf = malloc(nw * 4), *b = malloc((size_t)N * 4);
    uint16_t *Wh = malloc(nw * 2);
    float *y = malloc(ny * 4), *yr = malloc(ny * 4);
    for (size_t i = 0; i < nx; i++) x[i] = urand();
    for (size_t i = 0; i < nw; i++) {
        float w = urand();
        if (wt == 1) { Wh[i] = aria_float_to_half(w); w = aria_half_to_float(Wh[i]); }
        if (wt == 2) { Wh[i] = aria_float_to_bf16(w); w = aria_bf16_to_float(Wh[i]); }
        Wf[i] = w;
    }
    for (int n = 0; n < N; n++) b[n] = urand();
#ifdef _OPENMP
    int prev = omp_get_max_threads();
    omp_set_num_threads(nth);
#endif
    if (wt == 0) aria_linear_packed_only(y, x, Wf, bias ? b : NULL, M, K, N);
    else         aria_linear_hw(y, x, Wh, wt == 2, bias ? b : NULL, M, K, N);
#ifdef _OPENMP
    omp_set_num_threads(prev);
#endif
    ref(yr, x, Wf, bias ? b : NULL, M, K, N);
    /* K>2048 GEMMs hand edge-N columns (N % 16) to aria_dot: check the kernel-owned ones */
    int Nk = K > 2048 ? N / 16 * 16 : N, bad = 0;
    for (int m = 0; m < M && !bad; m++)
        if (memcmp(y + (size_t)m * N, yr + (size_t)m * N, (size_t)Nk * 4)) bad = 1;
    printf("%s gemm M=%d K=%d N=%d %s%s threads=%d\n", bad ? "FAIL" : "ok  ", M, K, N,
           wt == 0 ? "fp32" : wt == 1 ? "fp16" : "bf16", bias ? "+bias" : "", nth);
    free(x); free(Wf); free(b); free(Wh); free(y); free(yr);
    return bad;
}

int main(void) {
    static const int S[][3] = {   /* M, K, N: edge tiles, K blocks, attention/decoder shapes */
        {1, 64, 16}, {5, 100, 17}, {6, 1024, 256}, {7, 2048, 1000}, {34, 768, 3840},
        {64, 64, 257}, {64, 64, 710}, {172, 1024, 1024}, {13, 2049, 1024}, {40, 2304, 768},
        {9, 4096, 1024}, {8, 2304, 100}};
    int maxth = 1, fails = 0;
#ifdef _OPENMP
    maxth = omp_get_max_threads();
#endif
    for (size_t i = 0; i < sizeof(S) / sizeof(*S); i++)
        for (int bias = 0; bias < 2; bias++) fails += run(S[i][0], S[i][1], S[i][2], 0, bias, maxth);
    fails += run(172, 1024, 1024, 0, 1, 1);                 /* thread-count independence */
    fails += run(34, 2304, 768, 0, 1, 1);
    for (int wt = 1; wt <= 2; wt++) {                        /* half sources (packed above 2^18) */
        fails += run(6, 1024, 256, wt, 1, maxth);
        fails += run(64, 768, 1000, wt, 0, maxth);
        fails += run(9, 4096, 1024, wt, 1, maxth);
    }
    if (fails) { printf("test_gemm: %d failures\n", fails); return 1; }
    printf("test_gemm: OK\n");
    return 0;
}
