/* test_live.c - hermetic: the pieces of the live client/server protocol that need no
 * model or network. aria_json_escape must produce a string body the aria_json getters
 * read back unchanged (quotes, backslashes), with control characters as spaces and
 * truncation that never splits an escape or a UTF-8 character. */
#include "../src/aria_json.h"
#include <stdio.h>
#include <string.h>

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("FAIL %s\n", what); fails++; } }

static void roundtrip(const char *in, const char *expect) {
    char esc[256], json[300], back[256];
    aria_json_escape(in, esc, sizeof esc);
    snprintf(json, sizeof json, "{\"k\":\"%s\"}", esc);
    check(aria_json_get_string(json, "k", back, sizeof back) == 0 && strcmp(back, expect) == 0, in);
}

int main(void) {
    roundtrip("plain prompt", "plain prompt");
    roundtrip("say \"hi\"", "say \"hi\"");
    roundtrip("back\\slash", "back\\slash");
    roundtrip("tab\tnew\nline", "tab new line");
    roundtrip("caffè lungo — ok", "caffè lungo — ok");
    char out[8];
    aria_json_escape("ab\"cdef", out, sizeof out);          /* room for 7 bytes */
    check(strcmp(out, "ab\\\"cde") == 0, "truncation keeps escapes whole");
    aria_json_escape("abcd\"x", out, 6);                     /* the escape would not fit */
    check(strcmp(out, "abcd") == 0, "no half escape");
    aria_json_escape("abcdè", out, 6);                       /* è = 2 bytes, 1 would fit */
    check(strcmp(out, "abcd") == 0, "no half UTF-8 character");
    if (fails) { printf("test_live: %d failures\n", fails); return 1; }
    printf("test_live: OK\n");
    return 0;
}
