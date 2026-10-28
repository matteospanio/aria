/* test_quant.c - hermetic tests for Q8/Q4 weight quantization + dequant GEMM.
 * Checks: dtype parse, round-trip max error within the format's resolution, and
 * quantized-GEMM vs fp32 reference within the expected quantization tolerance. */
#include "../src/aria_quant.h"
#include "../src/aria_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int fails = 0;
static unsigned rng = 12345u;
static float frand(void) { rng = rng * 1664525u + 1013904223u; return (float)(rng >> 8) / (float)(1u << 24); }
static float frandn(void) { /* ~N(0,1) via two uniforms (Box-Muller) */
    float u1 = frand() + 1e-7f, u2 = frand();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* dequant a Q8 weight back to f32 for round-trip checks */
static void q8_dequant(float *W, const int8_t *q, const float *scale, int N, int K) {
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++) W[(size_t)n * K + k] = (float)q[(size_t)n * K + k] * scale[n];
}

static void check_le(const char *what, float got, float thr) {
    if (got > thr) { printf("FAIL %s: %.3e > %.3e\n", what, got, thr); fails++; }
    else printf("ok   %s: %.3e (<= %.3e)\n", what, got, thr);
}

/* max relative GEMM error: max|y_q - y_ref| / max|y_ref| */
static float gemm_relerr(const float *yq, const float *yref, int n) {
    float md = 0, mr = 0;
    for (int i = 0; i < n; i++) { float d = fabsf(yq[i] - yref[i]); if (d > md) md = d;
                                  float a = fabsf(yref[i]); if (a > mr) mr = a; }
    return mr > 0 ? md / mr : md;
}

int main(void) {
    /* dtype parse round-trip */
    {
        const char *names[] = {"fp32", "fp16", "bf16", "q8", "q4"};
        aria_dtype dts[] = {ARIA_F32, ARIA_F16, ARIA_BF16, ARIA_Q8, ARIA_Q4};
        for (int i = 0; i < 5; i++) {
            aria_dtype d;
            if (aria_dtype_parse(names[i], &d) != 0 || d != dts[i] ||
                strcmp(aria_dtype_name(dts[i]), names[i]) != 0) { printf("FAIL dtype %s\n", names[i]); fails++; }
        }
        aria_dtype d;
        if (aria_dtype_parse("nope", &d) == 0) { printf("FAIL dtype: accepted junk\n"); fails++; }
        printf("ok   dtype parse/name\n");
    }

    /* shapes mirroring SA3 linears: K (contracted) x N (out), plus an odd K */
    int shapes[][2] = { {768, 1024}, {1024, 3072}, {64, 64}, {1024, 257}, {130, 96} };
    int M = 5;
    for (int si = 0; si < (int)(sizeof(shapes) / sizeof(shapes[0])); si++) {
        int K = shapes[si][0], N = shapes[si][1];
        float *W = malloc((size_t)N * K * sizeof(float));
        float *x = malloc((size_t)M * K * sizeof(float));
        for (int i = 0; i < N * K; i++) W[i] = 0.1f * frandn();    /* weight-like scale */
        for (int i = 0; i < M * K; i++) x[i] = frandn();
        float *yref = malloc((size_t)M * N * sizeof(float));
        aria_linear(yref, x, W, NULL, M, K, N);

        char tag[64];

        /* ---- Q8 ---- */
        {
            int8_t *q = malloc(aria_q8_qbytes(N, K));
            float *scale = malloc(aria_q8_nscale(N, K) * sizeof(float));
            aria_q8_quant(q, scale, W, N, K);
            /* round-trip: per-row error <= scale (half-step is scale/2; allow scale) */
            float *Wd = malloc((size_t)N * K * sizeof(float));
            q8_dequant(Wd, q, scale, N, K);
            float rmax = 0;
            for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) {
                float e = fabsf(Wd[(size_t)n * K + k] - W[(size_t)n * K + k]);
                float s = scale[n] * 0.5f + 1e-9f;            /* worst-case round error */
                if (e / s > rmax) rmax = e / s;
            }
            snprintf(tag, sizeof(tag), "q8 roundtrip[%dx%d] (err/halfstep)", N, K);
            check_le(tag, rmax, 1.001f);
            float *yq = malloc((size_t)M * N * sizeof(float));
            aria_linear_q8(yq, x, q, scale, NULL, M, K, N);
            snprintf(tag, sizeof(tag), "q8 gemm relerr[%dx%d]", N, K);
            check_le(tag, gemm_relerr(yq, yref, M * N), 0.03f);
            /* W8A8: int8 activations on the same weights (adds per-row activation
             * quantization error on top of q8's -- looser threshold) */
            aria_linear_q8a8(yq, x, q, scale, NULL, M, K, N);
            snprintf(tag, sizeof(tag), "w8a8 gemm relerr[%dx%d]", N, K);
            check_le(tag, gemm_relerr(yq, yref, M * N), 0.06f);
            free(q); free(scale); free(Wd); free(yq);
        }
        /* ---- Q4 ---- */
        {
            uint8_t *q = malloc(aria_q4_qbytes(N, K));
            float *scale = malloc(aria_q4_nscale(N, K) * sizeof(float));
            aria_q4_quant(q, scale, W, N, K);
            float *yq = malloc((size_t)M * N * sizeof(float));
            aria_linear_q4(yq, x, q, scale, NULL, M, K, N);
            snprintf(tag, sizeof(tag), "q4 gemm relerr[%dx%d]", N, K);
            check_le(tag, gemm_relerr(yq, yref, M * N), 0.12f);
            free(q); free(scale); free(yq);
        }
        /* ---- B2: fp16 / bf16 storage GEMM (fp32 compute, so error is the weight-store
         * rounding only: fp16 ~2^-11 rel, bf16 ~2^-8 rel per element) ---- */
        for (int bf16 = 0; bf16 < 2; bf16++) {
            aria_qweight hw;
            aria_qweight_set(&hw, W, N, K, bf16 ? ARIA_BF16 : ARIA_F16);
            float *yq = malloc((size_t)M * N * sizeof(float));
            aria_linear_hw(yq, x, (const uint16_t *)hw.q, bf16, NULL, M, K, N);
            snprintf(tag, sizeof(tag), "%s gemm relerr[%dx%d]", bf16 ? "bf16" : "fp16", N, K);
            check_le(tag, gemm_relerr(yq, yref, M * N), bf16 ? 0.02f : 0.004f);
            /* qw dispatch must route dt==F16/BF16 through the same kernel (bit-identical) */
            float *yd = malloc((size_t)M * N * sizeof(float));
            aria_linear_qw(yd, x, &hw, NULL, M);
            int same = 1;
            for (int i = 0; i < M * N; i++) if (yd[i] != yq[i]) { same = 0; break; }
            if (!same) { printf("FAIL %s: linear_qw != linear_hw\n", bf16 ? "bf16" : "fp16"); fails++; }
            aria_qweight_free(&hw); free(yq); free(yd);
        }
        free(W); free(x); free(yref);
    }

    /* bias path sanity (Q8) */
    {
        int K = 64, N = 8, Mb = 2;
        float W[64 * 8], x[2 * 64], b[8], yref[16], yq[16];
        for (int i = 0; i < N * K; i++) W[i] = 0.1f * frandn();
        for (int i = 0; i < Mb * K; i++) x[i] = frandn();
        for (int i = 0; i < N; i++) b[i] = frandn();
        aria_linear(yref, x, W, b, Mb, K, N);
        int8_t q[64 * 8]; float scale[8];
        aria_q8_quant(q, scale, W, N, K);
        aria_linear_q8(yq, x, q, scale, b, Mb, K, N);
        check_le("q8 gemm+bias relerr", gemm_relerr(yq, yref, Mb * N), 0.03f);
        aria_linear_q8a8(yq, x, q, scale, b, Mb, K, N);
        check_le("w8a8 gemm+bias relerr", gemm_relerr(yq, yref, Mb * N), 0.06f);
    }

    /* W8A8 must be exact int32 math on every path (scalar, AVX2 maddubs, AVX-VNNI,
     * NEON sdot): bit-identical to a scalar reference with the same quantization.
     * The shapes cover full and edge register tiles and a K tail. */
    {
        static const int shp[][3] = { {1, 32, 1}, {3, 64, 2}, {4, 96, 3}, {5, 100, 4}, {7, 1030, 7}, {9, 1024, 13}, {12, 256, 6} };
        int bad = 0;
        for (size_t si = 0; si < sizeof shp / sizeof *shp; si++) {
            int M = shp[si][0], K = shp[si][1], N = shp[si][2];
            float *W = malloc((size_t)N * K * sizeof(float)), *x = malloc((size_t)M * K * sizeof(float));
            float *b = malloc((size_t)N * sizeof(float)), *y = malloc((size_t)M * N * sizeof(float));
            int8_t *q = malloc((size_t)N * K), *xq = malloc((size_t)M * K); float *scale = malloc((size_t)N * sizeof(float));
            for (int i = 0; i < N * K; i++) W[i] = 0.1f * frandn();
            for (int i = 0; i < M * K; i++) x[i] = frandn();
            for (int i = 0; i < N; i++) b[i] = frandn();
            aria_q8_quant(q, scale, W, N, K);
            aria_linear_q8a8(y, x, q, scale, b, M, K, N);
            for (int m = 0; m < M && !bad; m++) {
                float amax = 0; for (int k = 0; k < K; k++) if (fabsf(x[m * K + k]) > amax) amax = fabsf(x[m * K + k]);
                float sx = amax / 127.0f, inv = 127.0f / amax;
                for (int k = 0; k < K; k++) { int v = (int)lrintf(x[m * K + k] * inv); xq[m * K + k] = (int8_t)(v < -127 ? -127 : v > 127 ? 127 : v); }
                for (int n = 0; n < N; n++) {
                    int32_t acc = 0;
                    for (int k = 0; k < K; k++) acc += (int32_t)xq[m * K + k] * q[(size_t)n * K + k];
                    float r = (float)acc * sx * scale[n] + b[n];
                    if (memcmp(&r, &y[m * N + n], sizeof r)) { printf("FAIL w8a8 exact M=%d K=%d N=%d at (%d,%d)\n", M, K, N, m, n); bad = 1; break; }
                }
            }
            free(W); free(x); free(b); free(y); free(q); free(xq); free(scale);
        }
        if (bad) fails++; else printf("ok   w8a8 exact int32 (bit-identical to the scalar reference)\n");
    }

    /* E9.2: aria_qweight serialization round-trip must be bit-identical */
    for (int qi = 0; qi < 2; qi++) {
        aria_dtype dt = qi ? ARIA_Q4 : ARIA_Q8;
        int K = 130, N = 48, Mb = 3;   /* odd K to exercise the tail */
        float *W = malloc((size_t)N * K * sizeof(float)), *x = malloc((size_t)Mb * K * sizeof(float));
        float *y1 = malloc((size_t)Mb * N * sizeof(float)), *y2 = malloc((size_t)Mb * N * sizeof(float));
        for (int i = 0; i < N * K; i++) W[i] = 0.1f * frandn();
        for (int i = 0; i < Mb * K; i++) x[i] = frandn();
        aria_qweight a, c;
        aria_qweight_set(&a, W, N, K, dt);
        aria_linear_qw(y1, x, &a, NULL, Mb);
        FILE *f = tmpfile();
        int io = aria_qweight_write(f, &a); rewind(f);
        io |= aria_qweight_read(f, &c); fclose(f);
        aria_linear_qw(y2, x, &c, NULL, Mb);
        float md = 0; for (int i = 0; i < Mb * N; i++) { float d = fabsf(y1[i] - y2[i]); if (d > md) md = d; }
        char tag[48]; snprintf(tag, sizeof(tag), "%s serialize roundtrip (exact)", aria_dtype_name(dt));
        if (io != 0) { printf("FAIL %s: io error\n", tag); fails++; }
        else check_le(tag, md, 0.0f);
        aria_qweight_free(&a); aria_qweight_free(&c);
        free(W); free(x); free(y1); free(y2);
    }

    if (fails) { printf("test_quant: %d failures\n", fails); return 1; }
    printf("test_quant: OK\n");
    return 0;
}
