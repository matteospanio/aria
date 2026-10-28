/* test_ops.c - known-answer tests for the CPU op kernels. */
#include "../src/aria_ops.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void check(const char *name, float got, float want) {
    float tol = 1e-4f;
    if (fabsf(got - want) > tol) {
        printf("FAIL %s: got %.6f want %.6f\n", name, got, want);
        fails++;
    }
}

/* the pre-vectorization banded attention, kept verbatim as the exactness reference */
static void band_ref(float *out, const float *q, const float *k, const float *v, int H, int N, int D, int W) {
    float scale = 1.0f / sqrtf((float)D);
    float *scores = malloc((size_t)(2 * W + 1) * sizeof(float));
    for (int h = 0; h < H; h++) {
        const float *qh = q + (size_t)h * N * D, *kh = k + (size_t)h * N * D, *vh = v + (size_t)h * N * D;
        float *oh = out + (size_t)h * N * D;
        for (int i = 0; i < N; i++) {
            int lo = i - W < 0 ? 0 : i - W, hi = i + W >= N ? N - 1 : i + W;
            const float *qi = qh + (size_t)i * D;
            float maxv = -INFINITY;
            for (int j = lo; j <= hi; j++) {
                const float *kj = kh + (size_t)j * D;
                float s = 0.0f;
                for (int d = 0; d < D; d++) s += qi[d] * kj[d];
                s *= scale;
                scores[j - lo] = s;
                if (s > maxv) maxv = s;
            }
            float sum = 0.0f;
            for (int j = lo; j <= hi; j++) { float e = expf(scores[j - lo] - maxv); scores[j - lo] = e; sum += e; }
            float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
            float *oi = oh + (size_t)i * D;
            for (int d = 0; d < D; d++) oi[d] = 0.0f;
            for (int j = lo; j <= hi; j++) {
                float wgt = scores[j - lo] * inv;
                const float *vj = vh + (size_t)j * D;
                for (int d = 0; d < D; d++) oi[d] += wgt * vj[d];
            }
        }
    }
    free(scores);
}

int main(void) {
    /* aria_linear: y = x @ W^T + b */
    {
        float x[2] = {1.0f, 2.0f};            /* M=1, K=2 */
        float W[6] = {1,0, 0,1, 1,1};         /* N=3, K=2 */
        float b[3] = {0,0,0};
        float y[3];
        aria_linear(y, x, W, b, 1, 2, 3);
        check("linear[0]", y[0], 1.0f);
        check("linear[1]", y[1], 2.0f);
        check("linear[2]", y[2], 3.0f);
    }
    /* aria_matmul: C = A@B */
    {
        float A[6] = {1,2,3, 4,5,6};   /* 2x3 */
        float B[3] = {1,1,1};          /* 3x1 */
        float C[2];
        aria_matmul(C, A, B, 2, 3, 1);
        check("matmul[0]", C[0], 6.0f);
        check("matmul[1]", C[1], 15.0f);
    }
    /* aria_rmsnorm: x=[3,4], dim=2 */
    {
        float x[2] = {3.0f, 4.0f};
        float y[2];
        aria_rmsnorm(y, x, NULL, 1, 2, 0.0f);
        check("rmsnorm[0]", y[0], 0.848528f);
        check("rmsnorm[1]", y[1], 1.131371f);
    }
    /* aria_silu */
    {
        float v[2] = {0.0f, 1.0f};
        aria_silu(v, 2);
        check("silu(0)", v[0], 0.0f);
        check("silu(1)", v[1], 0.731059f);
    }
    /* aria_gelu_tanh(1.0) ~= 0.841192 */
    {
        float v[1] = {1.0f};
        aria_gelu_tanh(v, 1);
        check("gelu_tanh(1)", v[0], 0.841192f);
    }
    /* aria_softmax row [1,2,3] */
    {
        float x[3] = {1.0f, 2.0f, 3.0f};
        aria_softmax_inplace(x, 1, 3, NULL);
        check("softmax[0]", x[0], 0.090031f);
        check("softmax[1]", x[1], 0.244728f);
        check("softmax[2]", x[2], 0.665241f);
    }
    /* aria_silu_gate */
    {
        float gate[1] = {1.0f}, up[1] = {2.0f}, out[1];
        aria_silu_gate(out, gate, up, 1);
        check("silu_gate", out[0], 0.731059f * 2.0f);
    }

    /* aria_attention_band: byte-identical to the scalar reference (incl. N < 2W+1, N = 1) */
    {
        const int cases[][3] = {{3, 200, 17}, {2, 20, 17}, {1, 1, 17}, {4, 130, 5}};   /* H, N, W */
        unsigned long long r = 0x2545F4914F6CDD1DULL;
        for (int c = 0; c < 4; c++) {
            int H = cases[c][0], N = cases[c][1], W = cases[c][2], D = 64;
            size_t n = (size_t)H * N * D;
            float *qq = malloc(n * sizeof(float)), *kk = malloc(n * sizeof(float)), *vv = malloc(n * sizeof(float));
            float *o1 = malloc(n * sizeof(float)), *o2 = malloc(n * sizeof(float));
            for (size_t i = 0; i < n; i++) {
                r ^= r << 13; r ^= r >> 7; r ^= r << 17; qq[i] = (float)(r >> 40) / (1 << 22) - 2.0f;
                r ^= r << 13; r ^= r >> 7; r ^= r << 17; kk[i] = (float)(r >> 40) / (1 << 22) - 2.0f;
                r ^= r << 13; r ^= r >> 7; r ^= r << 17; vv[i] = (float)(r >> 40) / (1 << 22) - 2.0f;
            }
            band_ref(o1, qq, kk, vv, H, N, D, W);
            aria_attention_band(o2, qq, kk, vv, H, N, D, W);
            if (memcmp(o1, o2, n * sizeof(float))) { printf("FAIL attention_band not bit-identical (H=%d N=%d W=%d)\n", H, N, W); fails++; }
            free(qq); free(kk); free(vv); free(o1); free(o2);
        }
    }

    /* aria_f32_to_f16_n (F16C bulk) == aria_float_to_half bit for bit: fp16 subnormals,
     * ties, the 65504..65520 rounding edge and overflow included */
    {
        enum { NF = 70001 };
        static float src[NF]; static uint16_t a[NF];
        unsigned long long r = 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < NF; i++) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            int e = (int)(r % 48) - 30;                 /* 2^-30 .. 2^17 */
            float f = ldexpf((float)((r >> 20) & 0xFFFFF) / (float)(1 << 20) + 1.0f, e);
            src[i] = (r >> 63) ? -f : f;
        }
        src[0] = 0.0f; src[1] = -0.0f; src[2] = 65504.0f; src[3] = 65519.996f; src[4] = 65520.0f;
        src[5] = 5.9604645e-8f; src[6] = 2.9802322e-8f; src[7] = 1e30f;
        aria_f32_to_f16_n(a, src, NF);
        int bad = 0;
        for (int i = 0; i < NF; i++) if (a[i] != aria_float_to_half(src[i])) bad++;
        if (bad) { printf("FAIL f32_to_f16_n: %d mismatches\n", bad); fails++; }
    }

    if (fails) { printf("test_ops: %d failures\n", fails); return 1; }
    printf("test_ops: OK\n");
    return 0;
}
