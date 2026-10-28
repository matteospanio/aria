/*
 * aria_json.h - Tiny read-only JSON value extractor for model configs.
 *
 * Not a full parser: enough to pull scalar fields out of model_config.json /
 * config.json by key name. Phase 1 extends this for nested DiT/AE configs.
 */

#ifndef ARIA_JSON_H
#define ARIA_JSON_H

#include <stddef.h>

/* Read an entire file into a NUL-terminated heap buffer (caller frees). */
char *aria_read_file(const char *path, size_t *out_len);

/* Find the first "key": "string" and copy the value. Returns 0 on success. */
int aria_json_get_string(const char *json, const char *key, char *out, size_t outlen);

/* Find the first "key": <number> and store as double. Returns 0 on success. */
int aria_json_get_number(const char *json, const char *key, double *out);

/* ---- scoped variants: search only within [start, end) (end NULL = to NUL) ----
 * Needed because nested configs reuse key names (e.g. "io_channels", "config").
 */

/* Locate "key": { ... } and return the brace-balanced object span:
 * *obj_start points at '{', *obj_end just past the matching '}'. Returns 0/-1. */
int aria_json_object(const char *start, const char *end, const char *key,
                     const char **obj_start, const char **obj_end);

int aria_json_get_number_in(const char *start, const char *end, const char *key, double *out);
int aria_json_get_string_in(const char *start, const char *end, const char *key,
                            char *out, size_t outlen);
/* Find "key": true/false (or 1/0) -> *out (0/1). Returns 0 on success. */
int aria_json_get_bool_in(const char *start, const char *end, const char *key, int *out);

/* Escape `s` as a JSON string body (no quotes) that the getters above read back:
 * '"' and '\\' get a backslash, control characters become spaces. Truncates to fit
 * `outlen`, never mid-escape. Returns `out`. */
char *aria_json_escape(const char *s, char *out, size_t outlen);

#endif /* ARIA_JSON_H */
