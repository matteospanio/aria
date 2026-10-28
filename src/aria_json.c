/*
 * aria_json.c - Tiny read-only JSON value extractor.
 */

#include "aria_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *aria_read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    __int64 sz = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
#else
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
#endif
    if (sz < 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
    buf[sz] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

/* Locate the value position just after the colon following "key", searching
 * only within [start, end) (end NULL = to the NUL terminator). Naive first-match
 * scan that respects the bound -- adequate for config fields. */
static const char *find_key_b(const char *start, const char *end, const char *key) {
    size_t klen = strlen(key);
    const char *e = end ? end : (start + strlen(start));
    const char *p = start;
    while (p < e) {
        if (*p != '"') { p++; continue; }
        const char *ks = p + 1;
        if (ks + klen < e && strncmp(ks, key, klen) == 0 && ks[klen] == '"') {
            const char *q = ks + klen + 1;
            while (q < e && (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r')) q++;
            if (q < e && *q == ':') {
                q++;
                while (q < e && (*q == ' ' || *q == '\t' || *q == '\n' || *q == '\r')) q++;
                return q;
            }
        }
        p = ks;
    }
    return NULL;
}

static const char *find_key(const char *json, const char *key) {
    return find_key_b(json, NULL, key);
}

int aria_json_object(const char *start, const char *end, const char *key,
                     const char **obj_start, const char **obj_end) {
    const char *v = find_key_b(start, end, key);
    if (!v || *v != '{') return -1;
    const char *p = v;
    int depth = 0;
    const char *e = end ? end : (v + strlen(v));
    while (p < e) {
        if (*p == '{') depth++;
        else if (*p == '}') { depth--; if (depth == 0) { *obj_start = v; *obj_end = p + 1; return 0; } }
        p++;
    }
    return -1;
}

int aria_json_get_number_in(const char *start, const char *end, const char *key, double *out) {
    const char *v = find_key_b(start, end, key);
    if (!v) return -1;
    char *e = NULL;
    double d = strtod(v, &e);
    if (e == v) return -1;
    *out = d;
    return 0;
}

int aria_json_get_bool_in(const char *start, const char *end, const char *key, int *out) {
    const char *v = find_key_b(start, end, key);
    if (!v) return -1;
    if (*v == 't' || *v == '1') { *out = 1; return 0; }
    if (*v == 'f' || *v == '0') { *out = 0; return 0; }
    return -1;
}

int aria_json_get_string_in(const char *start, const char *end, const char *key,
                            char *out, size_t outlen) {
    const char *v = find_key_b(start, end, key);
    if (!v || *v != '"') return -1;
    v++;
    size_t i = 0;
    while (*v && *v != '"' && i < outlen - 1) {
        if (*v == '\\' && v[1]) v++;
        out[i++] = *v++;
    }
    out[i] = '\0';
    return 0;
}

int aria_json_get_string(const char *json, const char *key, char *out, size_t outlen) {
    const char *v = find_key(json, key);
    if (!v || *v != '"') return -1;
    v++;
    size_t i = 0;
    while (*v && *v != '"' && i < outlen - 1) {
        if (*v == '\\' && v[1]) v++;
        out[i++] = *v++;
    }
    out[i] = '\0';
    return 0;
}

int aria_json_get_number(const char *json, const char *key, double *out) {
    const char *v = find_key(json, key);
    if (!v) return -1;
    char *end = NULL;
    double d = strtod(v, &end);
    if (end == v) return -1;
    *out = d;
    return 0;
}
