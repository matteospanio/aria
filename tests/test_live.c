/* test_live.c - hermetic: the pieces of the live client/server protocol that need no
 * model or network. The frame codec (aria_live.h) must round-trip, wait on partial input
 * and reject corrupt lengths; the credit gate and "lands" prediction follow their rules;
 * aria_json_escape must produce a string body the aria_json getters read back unchanged
 * (quotes, backslashes), with control characters as spaces and truncation that never
 * splits an escape or a UTF-8 character. */
#include "../src/aria_json.h"
#include "../src/aria_live.h"
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

static void frames(void) {
    unsigned char b[64]; char tag[5]; const unsigned char *pl; uint32_t len;
    live_hdr(b, "HELO", 5); memcpy(b + 8, "hello", 5);
    live_hdr(b + 13, "AUDI", 3); memcpy(b + 21, "abc", 3);
    long k = live_parse(b, 24, 100, tag, &pl, &len);
    check(k == 13 && !strcmp(tag, "HELO") && len == 5 && !memcmp(pl, "hello", 5), "parse first frame");
    k = live_parse(b + 13, 11, 100, tag, &pl, &len);
    check(k == 11 && !strcmp(tag, "AUDI") && len == 3 && !memcmp(pl, "abc", 3), "parse second frame");
    check(live_parse(b, 7, 100, tag, &pl, &len) == 0, "partial header: need more");
    check(live_parse(b, 12, 100, tag, &pl, &len) == 0, "partial payload: need more");
    check(live_parse(b, 24, 4, tag, &pl, &len) == -1, "oversize length: corrupt");
    live_hdr(b, "EVNT", 0x01020304u);
    check(b[4] == 4 && b[5] == 3 && b[6] == 2 && b[7] == 1, "length is little-endian");
    /* credit gate: generate while less than `ahead` is buffered and the outbox has room */
    check(live_gate(100, 0, 200, 0, 10) && !live_gate(200, 0, 200, 0, 10) && !live_gate(100, 0, 200, 10, 10), "gate");
    check(live_gate(1000000, 0, 0, 0, 10), "ahead 0 = free-run");
    check(live_lands(500, 0, 80) == 500 && live_lands(500, 1, 80) == 580, "lands");
}

int main(void) {
    frames();
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
