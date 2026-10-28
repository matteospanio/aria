/* test_schedule.c - parity: aria_build_schedule vs the reference build_schedule for the
 * shifts the reference wrapper builds from the model's config (ref = what generate()
 * samples with, full = the training shift, len = aria's legacy LogSNR rate=1).
 * Needs ARIA_DUMPS + ARIA_MODEL (config parse + dispatch are part of what's tested). */
#include "../src/aria_sampler.h"
#include "../src/aria_sa3.h"
#include "../src/aria_json.h"
#include "../src/aria_parity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static const char *DUMPS;
static int fails = 0;

static void run(const char *name, const aria_dist_shift *s, int steps, int seq_len) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/sampler/sched_%s_s%d_L%d.atns", DUMPS, name, steps, seq_len);
    aria_parity_tensor ref;
    if (aria_parity_load(path, &ref) != 0) { printf("FAIL: load %s\n", path); fails++; return; }
    float *got = malloc((size_t)(steps + 1) * sizeof(float));
    aria_build_schedule(got, steps, 1.0f, s, seq_len);
    float md = aria_parity_maxabsdiff(got, ref.data, steps + 1);
    if (md > 1e-5f) { printf("FAIL sched %s s%d L%d: maxdiff=%.3e\n", name, steps, seq_len, md); fails++; }
    else printf("ok   sched %s s%d L%d: maxdiff=%.3e\n", name, steps, seq_len, md);
    free(got);
    aria_parity_free(&ref);
}

int main(void) {
    DUMPS = getenv("ARIA_DUMPS");
    const char *model = getenv("ARIA_MODEL");
    if (!DUMPS || !model) { printf("test_schedule: SKIP (set ARIA_DUMPS and ARIA_MODEL)\n"); return 0; }
    char cfgp[1024];
    snprintf(cfgp, sizeof(cfgp), "%s/model_config.json", model);
    char *json = aria_read_file(cfgp, NULL);
    aria_sa3_config cfg;
    if (!json || aria_sa3_parse_config(json, &cfg) != 0) { printf("FAIL: config %s\n", cfgp); return 1; }
    free(json);
    aria_dist_shift len; aria_dist_shift_defaults(&len, ARIA_SHIFT_LOGSNR);
    const struct { const char *name; const aria_dist_shift *s; } S[] = {
        {"ref", &cfg.sample_shift}, {"full", &cfg.train_shift}, {"len", &len}};
    const int cases[][2] = {{8, 108}, {6, 108}, {8, 323}, {8, 1293}, {16, 512}};
    for (size_t i = 0; i < sizeof(S) / sizeof(*S); i++)
        for (size_t c = 0; c < sizeof(cases) / sizeof(*cases); c++)
            run(S[i].name, S[i].s, cases[c][0], cases[c][1]);
    if (fails) { printf("test_schedule: FAILED\n"); return 1; }
    printf("test_schedule: OK\n");
    return 0;
}
