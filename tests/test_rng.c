/*
 * test_rng.c - hermetic RNG tests.
 *  (1) xoshiro determinism: same seed -> identical sequence.
 *  (2) torch-matched mode: aria_rng_randn(TORCH) matches golden torch.randn(seed)
 *      values (captured from PyTorch) to within libm last-ULP rounding (~2e-6).
 */
#include "aria_sampler.h"
#include <stdio.h>
#include <math.h>

/* torch.manual_seed(s); torch.randn(16).tolist() */
static const float TORCH_SEED0[16] = {
    -1.12583983f, -1.15236020f, -0.25057858f, -0.43387881f, 0.84871036f, 0.69200915f,
    -0.31601277f, -2.11521935f, 0.32227492f, -1.26333475f, 0.34998319f, 0.30813393f,
    0.11984151f, 1.23765790f, 1.11677718f, -0.24727815f };
static const float TORCH_SEED42[16] = {
    1.92691529f, 1.48728406f, 0.90071720f, -2.10552096f, 0.67841846f, -1.23454487f,
    -0.04306748f, -1.60466695f, -0.75213528f, 1.64872301f, -0.39247864f, -1.40360713f,
    -0.72788131f, -0.55943018f, -0.76883888f, 0.76244539f };

static int check_torch(const char *name, uint64_t seed, const float *golden) {
    aria_rng r; aria_rng_init(&r, seed, ARIA_RNG_TORCH);
    float out[16]; aria_rng_randn(&r, out, 16);
    float maxd = 0.0f;
    for (int i = 0; i < 16; i++) { float d = fabsf(out[i] - golden[i]); if (d > maxd) maxd = d; }
    int ok = maxd < 2e-5f;   /* generous of the ~2e-6 libm residual */
    printf("%-4s torch-rng seed=%llu: max|aria-torch.randn|=%.2e  %s\n",
           ok ? "ok" : "FAIL", (unsigned long long)seed, maxd, ok ? "" : "<-- mismatch");
    return ok;
}

int main(void) {
    int ok = 1;

    /* (1) xoshiro determinism */
    aria_rng a, b;
    aria_rng_seed(&a, 12345); aria_rng_seed(&b, 12345);
    float xa[32], xb[32];
    aria_rng_randn(&a, xa, 32); aria_rng_randn(&b, xb, 32);
    int same = 1; for (int i = 0; i < 32; i++) if (xa[i] != xb[i]) same = 0;
    printf("%-4s xoshiro determinism (same seed -> same sequence)\n", same ? "ok" : "FAIL");
    ok &= same;
    /* the parallel vector fill == n sequential aria_rng_gaussian draws, bit for bit,
     * including the state it leaves behind */
    {
        enum { NV = 4099 };
        static float va[NV], vb[NV];
        aria_rng c, d;
        aria_rng_seed(&c, 777); aria_rng_seed(&d, 777);
        aria_rng_randn(&c, va, NV);
        for (int i = 0; i < NV; i++) vb[i] = aria_rng_gaussian(&d);
        int eq = 1; for (int i = 0; i < NV; i++) if (va[i] != vb[i]) eq = 0;
        eq &= aria_rng_gaussian(&c) == aria_rng_gaussian(&d);
        printf("%-4s xoshiro vector fill == sequential draws\n", eq ? "ok" : "FAIL");
        ok &= eq;
    }

    /* (2) torch-matched mode vs golden torch.randn */
    ok &= check_torch("", 0, TORCH_SEED0);
    ok &= check_torch("", 42, TORCH_SEED42);

    /* (3) torch mode with n<16 must be deterministic + finite (no uninitialized
     * xoshiro read), even though it is not bit-exact to torch's small-n path. */
    aria_rng t1, t2; aria_rng_init(&t1, 5, ARIA_RNG_TORCH); aria_rng_init(&t2, 5, ARIA_RNG_TORCH);
    float s1[7], s2[7]; aria_rng_randn(&t1, s1, 7); aria_rng_randn(&t2, s2, 7);
    int small_ok = 1;
    for (int i = 0; i < 7; i++) if (s1[i] != s2[i] || !isfinite(s1[i])) small_ok = 0;
    printf("%-4s torch-rng n<16 deterministic + finite\n", small_ok ? "ok" : "FAIL");
    ok &= small_ok;

    if (!ok) { printf("test_rng FAILED\n"); return 1; }
    printf("test_rng passed\n");
    return 0;
}
