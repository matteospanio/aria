/* test_repl.c - hermetic (POSIX): aria-repl's line editor as a key-byte state machine
 * (insert, cursor keys, UTF-8 aware backspace/delete/movement, ^W ^U ^C ^D, history)
 * and the steer formatting the status line shows. */
#define ARIA_REPL_NO_MAIN
#pragma GCC diagnostic ignored "-Wunused-function"   /* the client's main-only helpers */
#include "../src/aria_repl.c"

static int fails;
static void check(int ok, const char *what) { if (!ok) { printf("FAIL %s\n", what); fails++; } }

static int keys(le_state *e, const char *s) { int r = LE_NONE; for (; *s; s++) r = le_key(e, (unsigned char)*s); return r; }

int main(void) {
    le_state e; memset(&e, 0, sizeof e);
    keys(&e, "abc");
    check(!strcmp(e.buf, "abc") && e.pos == 3, "type");
    keys(&e, "\x1b[D" "X");
    check(!strcmp(e.buf, "abXc") && e.pos == 3, "left + insert");
    keys(&e, "\x1b[H" "\x1b[3~");
    check(!strcmp(e.buf, "bXc") && e.pos == 0, "home + delete");
    keys(&e, "\x1b[F" "\x7f");
    check(!strcmp(e.buf, "bX") && e.pos == 2, "end + backspace");
    check(le_key(&e, 3) == LE_REDRAW && e.len == 0, "^C clears a line");
    check(le_key(&e, 3) == LE_INTR, "^C on an empty line");
    check(le_key(&e, 4) == LE_EOF, "^D on an empty line");

    keys(&e, "caffè");   /* è = 2 bytes */
    check(e.len == 6, "utf-8 bytes");
    keys(&e, "\x1b[D");
    check(e.pos == 4, "left skips a whole character");
    keys(&e, "\x1b[C" "\x7f");
    check(!strcmp(e.buf, "caff") && e.pos == 4, "backspace removes a whole character");
    keys(&e, " lungo\x17");
    check(!strcmp(e.buf, "caff "), "^W deletes a word");
    keys(&e, "\x15");
    check(e.len == 0, "^U clears to the start");

    check(keys(&e, "first\r") == LE_SUBMIT, "submit");
    e.len = e.pos = 0; e.buf[0] = 0;
    keys(&e, "second\r"); e.len = e.pos = 0; e.buf[0] = 0;
    keys(&e, "\x1b[A");
    check(!strcmp(e.buf, "second"), "up: last entry");
    keys(&e, "\x1b[A");
    check(!strcmp(e.buf, "first"), "up again");
    keys(&e, "\x1b[B" "\x1b[B");
    check(e.len == 0, "down past the end: empty line");

    char out[128];
    R.utf8 = 0;
    steer_pretty("sweet:0.30,sour:-0.10", out, sizeof out);
    check(!strcmp(out, "sweet +0.30, sour -0.10"), "steer display");
    steer_pretty("-", out, sizeof out);
    check(out[0] == 0, "no steer");

    if (fails) { printf("test_repl: %d failures\n", fails); return 1; }
    printf("test_repl: OK\n");
    return 0;
}
