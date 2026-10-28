/* bench_gemm.c - microbenchmark for aria_linear on the real SA3 shapes. Weights rotate
 * through more than an L3's worth of distinct copies, so they stream from DRAM as they
 * do in a denoise step (a single hot W flatters every driver). */
#include "../src/aria_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

static void bench(int M, int K, int N, const char *name) {
    size_t wbytes = (size_t)N * K * sizeof(float);
    int nW = (int)(256u * 1024 * 1024 / wbytes) + 1;     /* >= 256 MB of weights in rotation */
    if (nW > 64) nW = 64;
    float *x = malloc((size_t)M * K * sizeof(float));
    float *y = malloc((size_t)M * N * sizeof(float));
    float **W = malloc((size_t)nW * sizeof(float *));
    for (size_t i = 0; i < (size_t)M * K; i++) x[i] = (float)((int)(i % 17) - 8) / 8.0f;
    for (int w = 0; w < nW; w++) {
        W[w] = malloc(wbytes);
        for (size_t i = 0; i < (size_t)N * K; i++) W[w][i] = (float)((int)((i + w) % 13) - 6) / 6.0f;
    }
    aria_linear(y, x, W[0], NULL, M, K, N);  /* warmup */
    int reps = nW > 12 ? nW : 12;
    double t0 = now();
    for (int r = 0; r < reps; r++) aria_linear(y, x, W[r % nW], NULL, M, K, N);
    double dt = (now() - t0) / reps;
    printf("%-14s M=%5d K=%5d N=%5d  %7.2f ms  %6.1f GFLOP/s\n",
           name, M, K, N, dt * 1e3, 2.0 * M * N * K / dt / 1e9);
    for (int w = 0; w < nW; w++) free(W[w]);
    free(W); free(x); free(y);
}

int main(void) {
    /* small-music DiT at 10 s / 60 s (S = 64 memory tokens + T) */
    static const int Ms[] = {172, 710};
    for (int i = 0; i < 2; i++) {
        bench(Ms[i], 1024, 3072, "dit_s_qkv");
        bench(Ms[i], 1024, 8192, "dit_s_ff_in");
        bench(Ms[i], 4096, 1024, "dit_s_ff_out");
        bench(Ms[i], 1024, 1024, "dit_s_out");
        bench(Ms[i], 1536, 7680, "dit_m_qkv");
        bench(Ms[i], 6144, 1536, "dit_m_ff_out");
    }
    bench(1088,  768, 3840, "dec_s_qkv");     /* small decoder, one 32-chunk group */
    bench(1088, 2304,  768, "dec_s_ff_out");
    bench(1836, 1536, 7680, "dec_m_qkv");     /* medium decoder, 10 s */
    bench(1836, 1536, 9216, "dec_m_ff_in");
    bench(1836, 4608, 1536, "dec_m_ff_out");
    bench(  24,  768, 2048, "t5_mlp_in");     /* T5 on a ~24-token prompt */
    return 0;
}
