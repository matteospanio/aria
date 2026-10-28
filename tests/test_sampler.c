/* test_sampler.c - hermetic: pingpong update, schedule shape, RNG stats. */
#include "../src/aria_sampler.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

static int fails = 0;
static void check(const char *n, float got, float want, float tol) {
    if (fabsf(got - want) > tol) { printf("FAIL %s: got %.6f want %.6f\n", n, got, want); fails++; }
}

/* denoiser: v = 0.5 (constant), independent of x */
static void const_denoiser(void *ctx, const float *x, float t, float *v, int n) {
    (void)ctx; (void)x; (void)t;
    for (int i = 0; i < n; i++) v[i] = 0.5f;
}

int main(void) {
    /* pingpong with injected zero noise: hand-computed trajectory */
    {
        float x[3] = {1.0f, 1.0f, 1.0f};
        float sigmas[3] = {1.0f, 0.5f, 0.0f};
        float noise[6] = {0,0,0, 0,0,0};
        aria_pingpong(x, 3, sigmas, 2, const_denoiser, NULL, NULL, noise);
        /* step0: denoised=1-1*0.5=0.5; x=(1-0.5)*0.5+0.5*0=0.25
           step1: denoised=0.25-0.5*0.5=0.0; x=(1-0)*0+0*z=0.0 */
        for (int i = 0; i < 3; i++) check("pingpong_zero", x[i], 0.0f, 1e-6f);
    }
    /* pingpong with nonzero injected noise on a single step */
    {
        float x[1] = {2.0f};
        float sigmas[2] = {1.0f, 0.5f};   /* 1 step */
        float noise[1] = {1.0f};
        aria_pingpong(x, 1, sigmas, 1, const_denoiser, NULL, NULL, noise);
        /* denoised=2-1*0.5=1.5; x=(1-0.5)*1.5+0.5*1=0.75+0.5=1.25 */
        check("pingpong_noise", x[0], 1.25f, 1e-6f);
    }
    /* schedule: endpoints + monotonic descending */
    {
        float sched[9];
        aria_logsnr_schedule(sched, 8, 1.0f, -6.2f, 2000.0f, 1.0f, 2.0f, 323.0f);
        check("sched_first", sched[0], 1.0f, 1e-6f);
        check("sched_last", sched[8], 0.0f, 1e-6f);
        int mono = 1;
        for (int i = 1; i < 9; i++) if (sched[i] > sched[i-1] + 1e-7f) mono = 0;
        if (!mono) { printf("FAIL schedule not descending\n"); fails++; }
    }
    /* P0.1 shifts via aria_build_schedule */
    {
        float a[9], b[9];
        /* LOGSNR == the legacy call, byte-for-byte (the `len` schedule must not move) */
        aria_dist_shift len; aria_dist_shift_defaults(&len, ARIA_SHIFT_LOGSNR);
        for (int T = 108; T <= 646; T += 538) {
            aria_logsnr_schedule(a, 8, 1.0f, -6.2f, 2000.0f, 1.0f, 2.0f, (float)T);
            aria_build_schedule(b, 8, 1.0f, &len, T);
            if (memcmp(a, b, sizeof a)) { printf("FAIL len schedule moved at T=%d\n", T); fails++; }
        }
        /* reference sampling default LogSNRShift(rate=0): stable_audio_3 8-step values,
         * identical at every length */
        static const float ref8[9] = {1.0f, 0.9944f, 0.9845f, 0.9579f, 0.8909f, 0.7455f, 0.5125f, 0.2739f, 0.0f};
        aria_dist_shift ref = len; ref.rate = 0.0f;
        aria_build_schedule(a, 8, 1.0f, &ref, 108);
        aria_build_schedule(b, 8, 1.0f, &ref, 646);
        for (int i = 0; i < 9; i++) check("ref8", a[i], ref8[i], 1e-4f);
        if (memcmp(a, b, sizeof a)) { printf("FAIL ref schedule depends on length\n"); fails++; }
        /* DistributionShift (full): a*t/(1+(a-1)t), a = e^0.5 at L<=256, e^1.15 at 4096 */
        aria_dist_shift full; aria_dist_shift_defaults(&full, ARIA_SHIFT_FULL);
        const int Ls[2] = {100, 4096}; const double as[2] = {exp(0.5), exp(1.15)};
        for (int k = 0; k < 2; k++) {
            aria_build_schedule(a, 8, 1.0f, &full, Ls[k]);
            for (int i = 0; i < 9; i++) {
                double t = 1.0 - i / 8.0, w = as[k] * t / (1.0 + (as[k] - 1.0) * t);
                check("full", a[i], (float)w, 1e-6f);
            }
        }
        /* flux with alpha 1 is the identity */
        aria_dist_shift fl; aria_dist_shift_defaults(&fl, ARIA_SHIFT_FLUX);
        aria_build_schedule(a, 8, 1.0f, &fl, 500);
        for (int i = 0; i < 9; i++) check("flux_id", a[i], 1.0f - i / 8.0f, 1e-6f);
    }
    /* RNG: gaussian mean ~0, var ~1 */
    {
        aria_rng r; aria_rng_seed(&r, 12345);
        int N = 200000; double m = 0, v = 0;
        for (int i = 0; i < N; i++) { float z = aria_rng_gaussian(&r); m += z; v += (double)z*z; }
        m /= N; v = v / N - m*m;
        if (fabs(m) > 0.02) { printf("FAIL rng mean=%.4f\n", m); fails++; }
        if (fabs(v - 1.0) > 0.03) { printf("FAIL rng var=%.4f\n", v); fails++; }
    }

    if (fails) { printf("test_sampler: %d failures\n", fails); return 1; }
    printf("test_sampler: OK\n");
    return 0;
}
