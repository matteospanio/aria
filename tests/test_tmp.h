/* test_tmp.h - temp file paths for the hermetic tests: %TEMP% on Windows, else $TMPDIR or /tmp. */
#include <stdio.h>
#include <stdlib.h>

static void tmp_path(char *buf, size_t n, const char *name) {
#ifdef _WIN32
    const char *dir = getenv("TEMP");
#else
    const char *dir = getenv("TMPDIR");
#endif
    snprintf(buf, n, "%s/%s", dir ? dir : "/tmp", name);
}
