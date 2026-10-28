/* test_server.c - hermetic (POSIX): aria-server's named-steer table. Loads synthetic
 * --steer-dir folders in the Hugging Face layout (layer from dirs_meta.json, or the
 * site from a _latent/_text suffix; anything else is skipped) and checks the
 * "name:scale,..." parser: merge vs replace, "off", and no partial apply on errors. */
#define ARIA_SERVER_NO_MAIN
#pragma GCC diagnostic ignored "-Wunused-function"   /* the server's main-only helpers */
#include "../src/aria_server.c"

#include <sys/stat.h>

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("FAIL %s\n", what); fails++; } }

static void write_atns(const char *path, int n) {
    FILE *f = fopen(path, "wb");
    unsigned hdr[3] = {1, 1, 0}; int64_t shape = n;
    fwrite("ATNS", 1, 4, f); fwrite(hdr, 4, 3, f); fwrite(&shape, 8, 1, f);
    for (int i = 0; i < n; i++) { float v = (float)(i % 7) - 3.0f; fwrite(&v, 4, 1, f); }
    fclose(f);
}

int main(void) {
    g_log = stderr;
    char a[] = "/tmp/aria_steersXXXXXX", p[256];
    if (!mkdtemp(a)) return 1;
    snprintf(p, sizeof p, "%s/small", a); mkdir(p, 0700);
    snprintf(p, sizeof p, "%s/sites", a); mkdir(p, 0700);
    snprintf(p, sizeof p, "%s/small/dirs_meta.json", a);
    FILE *f = fopen(p, "w"); fputs("{\"sweet\":{\"layer\":16,\"dim\":1024},\"sour\":{\"layer\":19}}", f); fclose(f);
    snprintf(p, sizeof p, "%s/small/sweet.atns", a); write_atns(p, 1024);
    snprintf(p, sizeof p, "%s/small/sour.atns", a); write_atns(p, 1024);
    snprintf(p, sizeof p, "%s/small/umami.atns", a); write_atns(p, 1024);       /* no meta, no suffix: skipped */
    snprintf(p, sizeof p, "%s/sites/sweet_latent.atns", a); write_atns(p, 256);  /* no meta needed */
    snprintf(p, sizeof p, "%s/sites/sour_text.atns", a); write_atns(p, 768);
    snprintf(p, sizeof p, "%s/small", a); load_steer_dir(p);
    snprintf(p, sizeof p, "%s/sites", a); load_steer_dir(p);
    snprintf(p, sizeof p, "%s/small", a); load_steer_dir(p);                      /* again: duplicates skipped */

    check(g_nsteers == 4, "4 directions loaded (umami and duplicates skipped)");
    int sw = steer_find("sweet", 5), so = steer_find("sour", 4), sl = steer_find("sweet_latent", 12), st = steer_find("sour_text", 9);
    check(sw >= 0 && g_steers[sw].st.site == ARIA_STEER_RESIDUAL && g_steers[sw].st.layer == 16 && g_steers[sw].st.dim == 1024, "sweet: residual L16");
    check(so >= 0 && g_steers[so].st.layer == 19, "sour: residual L19");
    check(sl >= 0 && g_steers[sl].st.site == ARIA_STEER_LATENT && g_steers[sl].st.dim == 256, "sweet_latent: latent");
    check(st >= 0 && g_steers[st].st.site == ARIA_STEER_COND && g_steers[st].st.dim == 768, "sour_text: text");
    check(steer_find("umami", 5) < 0, "umami skipped");

    float sc[MAX_STEERS] = {0}; char err[256]; aria_steer items[MAX_STEERS];
    check(parse_steer_spec("sweet:0.3, sour:0.1", sc, 0, err, sizeof err) == 0 && sc[sw] == 0.3f && sc[so] == 0.1f, "parse two");
    check(steer_items(sc, items) == 2, "two items");
    check(parse_steer_spec("sour:0", sc, 1, err, sizeof err) == 0 && sc[sw] == 0.3f && sc[so] == 0.0f, "merge: scale 0 removes one");
    check(parse_steer_spec("sweet:1,umami:1", sc, 1, err, sizeof err) < 0 && sc[sw] == 0.3f, "unknown name: nothing applied");
    check(parse_steer_spec("sweet", sc, 1, err, sizeof err) < 0 && strstr(err, "needs a scale"), "missing scale");
    check(parse_steer_spec("sour_text:-0.5", sc, 0, err, sizeof err) == 0 && sc[st] == -0.5f && sc[sw] == 0.0f, "replace + negative scale");
    check(parse_steer_spec("off", sc, 1, err, sizeof err) == 0 && steer_items(sc, items) == 0, "off clears");

    char cmd[300]; snprintf(cmd, sizeof cmd, "rm -rf %s", a); if (system(cmd)) {}
    if (fails) { printf("test_server: %d failures\n", fails); return 1; }
    printf("test_server: OK\n");
    return 0;
}
