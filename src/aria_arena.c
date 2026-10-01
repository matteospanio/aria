/*
 * aria_arena.c - bump-pointer scratch arena.
 */

#include "aria_arena.h"
#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
/* MSVC has no aligned_alloc; plain malloc is only 16-aligned (test_arena checks 64) */
#define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#endif

#define ARIA_ARENA_ALIGN 64

int aria_arena_init(aria_arena *a, size_t cap) {
    a->base = aligned_alloc(ARIA_ARENA_ALIGN, (cap + ARIA_ARENA_ALIGN - 1) / ARIA_ARENA_ALIGN * ARIA_ARENA_ALIGN);
    if (!a->base) return -1;
    a->cap = cap;
    a->used = 0;
    a->peak = 0;
    return 0;
}

void aria_arena_free(aria_arena *a) {
    if (!a) return;
#ifdef _WIN32
    _aligned_free(a->base);   /* _aligned_malloc memory must not go through free() */
#else
    free(a->base);
#endif
    a->base = NULL;
    a->cap = a->used = a->peak = 0;
}

void *aria_arena_alloc(aria_arena *a, size_t bytes) {
    size_t off = (a->used + ARIA_ARENA_ALIGN - 1) & ~((size_t)ARIA_ARENA_ALIGN - 1);
    if (off + bytes > a->cap) return NULL;
    a->used = off + bytes;
    if (a->used > a->peak) a->peak = a->used;
    return a->base + off;
}

float *aria_arena_floats(aria_arena *a, size_t n) {
    return (float *)aria_arena_alloc(a, n * sizeof(float));
}

size_t aria_arena_save(const aria_arena *a) { return a->used; }
void   aria_arena_restore(aria_arena *a, size_t mark) { a->used = mark; }
