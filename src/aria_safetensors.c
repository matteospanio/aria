/*
 * aria_safetensors.c - Safetensors reader (ported from references/iris.c)
 */

#include "aria_safetensors.h"
#include "aria_win_compat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#ifndef _WIN32
#include <unistd.h>
#include <sys/mman.h>
#endif
#include <sys/stat.h>

/* ---- minimal JSON scanning for the safetensors header ---- */

static void skip_whitespace(const char **p) {
    while (**p == ' ' || **p == '\n' || **p == '\r' || **p == '\t') (*p)++;
}

static int parse_string(const char **p, char *out, size_t max_len) {
    skip_whitespace(p);
    if (**p != '"') return -1;
    (*p)++;
    size_t i = 0;
    while (**p && **p != '"' && i < max_len - 1) {
        if (**p == '\\') {
            (*p)++;
            if (**p == 'n') out[i++] = '\n';
            else if (**p == 't') out[i++] = '\t';
            else if (**p == 'r') out[i++] = '\r';
            else if (**p == '"') out[i++] = '"';
            else if (**p == '\\') out[i++] = '\\';
            else out[i++] = **p;
        } else {
            out[i++] = **p;
        }
        (*p)++;
    }
    out[i] = '\0';
    if (**p != '"') return -1;
    (*p)++;
    return 0;
}

static int64_t parse_int(const char **p) {
    skip_whitespace(p);
    int64_t val = 0;
    int neg = 0;
    if (**p == '-') { neg = 1; (*p)++; }
    while (**p >= '0' && **p <= '9') {
        val = val * 10 + (**p - '0');
        (*p)++;
    }
    return neg ? -val : val;
}

static safetensor_dtype_t parse_dtype(const char *s) {
    if (strcmp(s, "F32") == 0) return DTYPE_F32;
    if (strcmp(s, "F16") == 0) return DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return DTYPE_BF16;
    if (strcmp(s, "I32") == 0) return DTYPE_I32;
    if (strcmp(s, "I64") == 0) return DTYPE_I64;
    if (strcmp(s, "BOOL") == 0) return DTYPE_BOOL;
    return DTYPE_UNKNOWN;
}

static int parse_tensor_entry(const char **p, safetensor_t *t) {
    skip_whitespace(p);
    if (**p != '{') return -1;
    (*p)++;
    t->dtype = DTYPE_UNKNOWN;
    t->ndim = 0;
    t->data_offset = 0;
    t->data_size = 0;

    while (**p && **p != '}') {
        skip_whitespace(p);
        if (**p == ',') { (*p)++; continue; }
        char key[64];
        if (parse_string(p, key, sizeof(key)) != 0) return -1;
        skip_whitespace(p);
        if (**p != ':') return -1;
        (*p)++;
        skip_whitespace(p);

        if (strcmp(key, "dtype") == 0) {
            char dtype_str[32];
            if (parse_string(p, dtype_str, sizeof(dtype_str)) != 0) return -1;
            t->dtype = parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            t->ndim = 0;
            while (**p && **p != ']' && t->ndim < 8) {
                skip_whitespace(p);
                if (**p == ',') { (*p)++; continue; }
                t->shape[t->ndim++] = parse_int(p);
            }
            if (**p == ']') (*p)++;
        } else if (strcmp(key, "data_offsets") == 0) {
            if (**p != '[') return -1;
            (*p)++;
            skip_whitespace(p);
            size_t start = (size_t)parse_int(p);
            skip_whitespace(p);
            if (**p == ',') (*p)++;
            skip_whitespace(p);
            size_t end = (size_t)parse_int(p);
            t->data_offset = start;
            t->data_size = end - start;
            skip_whitespace(p);
            if (**p == ']') (*p)++;
        } else {
            /* skip unknown value (string/array/object/literal) */
            if (**p == '"') {
                (*p)++;
                while (**p && **p != '"') { if (**p == '\\') (*p)++; if (**p) (*p)++; }
                if (**p == '"') (*p)++;
            } else if (**p == '[') {
                int depth = 1; (*p)++;
                while (**p && depth > 0) { if (**p=='[') depth++; else if (**p==']') depth--; (*p)++; }
            } else if (**p == '{') {
                int depth = 1; (*p)++;
                while (**p && depth > 0) { if (**p=='{') depth++; else if (**p=='}') depth--; (*p)++; }
            } else {
                while (**p && **p != ',' && **p != '}') (*p)++;
            }
        }
    }
    if (**p == '}') (*p)++;
    return 0;
}

static int ensure_capacity(safetensors_file_t *sf) {
    if (sf->num_tensors < sf->tensors_capacity) return 0;
    int newcap = sf->tensors_capacity ? sf->tensors_capacity * 2 : 1024;
    if (newcap > SAFETENSORS_MAX_TENSORS) newcap = SAFETENSORS_MAX_TENSORS;
    if (sf->num_tensors >= newcap) return -1;
    safetensor_t *grown = realloc(sf->tensors, (size_t)newcap * sizeof(safetensor_t));
    if (!grown) return -1;
    sf->tensors = grown;
    sf->tensors_capacity = newcap;
    return 0;
}

static int parse_header(safetensors_file_t *sf) {
    const char *p = sf->header_json;
    skip_whitespace(&p);
    if (*p != '{') return -1;
    p++;
    sf->num_tensors = 0;

    while (*p && *p != '}') {
        skip_whitespace(&p);
        if (*p == ',') { p++; continue; }
        if (*p == '}') break;

        char name[256];
        if (parse_string(&p, name, sizeof(name)) != 0) return -1;
        skip_whitespace(&p);
        if (*p != ':') return -1;
        p++;

        if (strcmp(name, "__metadata__") == 0) {
            skip_whitespace(&p);
            if (*p == '{') {
                int depth = 1; p++;
                while (*p && depth > 0) { if (*p=='{') depth++; else if (*p=='}') depth--; p++; }
            }
            continue;
        }
        if (ensure_capacity(sf) != 0) return -1;
        safetensor_t *t = &sf->tensors[sf->num_tensors];
        snprintf(t->name, sizeof(t->name), "%s", name);
        if (parse_tensor_entry(&p, t) != 0) return -1;
        sf->num_tensors++;
    }
    return 0;
}

safetensors_file_t *safetensors_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("safetensors_open: open"); return NULL; }

#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) < 0) { perror("safetensors_open: fstat"); _close(fd); return NULL; }
#else
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("safetensors_open: fstat"); close(fd); return NULL; }
#endif
    size_t file_size = (size_t)st.st_size;
    if (file_size < 8) { fprintf(stderr, "safetensors_open: file too small\n"); close(fd); return NULL; }

    void *data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (data == MAP_FAILED) { perror("safetensors_open: mmap"); return NULL; }

    uint64_t header_size = 0;
    memcpy(&header_size, data, 8);
    if (header_size > file_size - 8) {
        fprintf(stderr, "safetensors_open: invalid header size\n");
        munmap(data, file_size);
        return NULL;
    }

    safetensors_file_t *sf = calloc(1, sizeof(safetensors_file_t));
    if (!sf) { munmap(data, file_size); return NULL; }
    sf->path = strdup(path);
    sf->data = data;
    sf->file_size = file_size;
    sf->header_size = (size_t)header_size;

    sf->header_json = malloc(header_size + 1);
    if (!sf->header_json) { safetensors_close(sf); return NULL; }
    memcpy(sf->header_json, (char *)data + 8, header_size);
    sf->header_json[header_size] = '\0';

    if (parse_header(sf) != 0) {
        fprintf(stderr, "safetensors_open: failed to parse header\n");
        safetensors_close(sf);
        return NULL;
    }

    /* Validate every tensor fits inside the data region (catches truncated files). */
    size_t data_region = file_size - 8 - (size_t)header_size;
    for (int i = 0; i < sf->num_tensors; i++) {
        safetensor_t *t = &sf->tensors[i];
        if (t->data_offset + t->data_size > data_region) {
            fprintf(stderr, "safetensors_open: %s: truncated (tensor '%s')\n", path, t->name);
            safetensors_close(sf);
            return NULL;
        }
    }
    return sf;
}

void safetensors_advise_dontneed(const safetensors_file_t *sf) {
    if (sf && sf->data) madvise(sf->data, sf->file_size, MADV_DONTNEED);
}

void safetensors_close(safetensors_file_t *sf) {
    if (!sf) return;
    if (sf->data) munmap(sf->data, sf->file_size);
    free(sf->path);
    free(sf->header_json);
    free(sf->tensors);
    free(sf);
}

const safetensor_t *safetensors_find(const safetensors_file_t *sf, const char *name) {
    for (int i = 0; i < sf->num_tensors; i++)
        if (strcmp(sf->tensors[i].name, name) == 0) return &sf->tensors[i];
    return NULL;
}

const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t) {
    size_t offset = 8 + sf->header_size + t->data_offset;
    return (const char *)sf->data + offset;
}

int64_t safetensor_numel(const safetensor_t *t) {
    int64_t n = 1;
    for (int i = 0; i < t->ndim; i++) n *= t->shape[i];
    return n;
}

static float bf16_to_f32(uint16_t bf16) {
    uint32_t f32 = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

static float f16_to_f32(uint16_t f16) {
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mant = f16 & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t) {
    int64_t n = safetensor_numel(t);
    float *out = malloc((size_t)n * sizeof(float));
    if (!out) return NULL;
    const void *data = safetensors_data(sf, t);
    switch (t->dtype) {
        case DTYPE_F32:
            memcpy(out, data, (size_t)n * sizeof(float));
            break;
        case DTYPE_F16: {
            const uint16_t *src = (const uint16_t *)data;
            for (int64_t i = 0; i < n; i++) out[i] = f16_to_f32(src[i]);
            break;
        }
        case DTYPE_BF16: {
            const uint16_t *src = (const uint16_t *)data;
            for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32(src[i]);
            break;
        }
        default:
            fprintf(stderr, "safetensors_get_f32: unsupported dtype %d for %s\n", t->dtype, t->name);
            free(out);
            return NULL;
    }
    return out;
}

const float *safetensors_f32_ptr(const safetensors_file_t *sf, const safetensor_t *t) {
    if (!sf || !t || t->dtype != DTYPE_F32) return NULL;
    return (const float *)safetensors_data(sf, t);
}

void safetensor_print(const safetensor_t *t) {
    const char *names[] = {"F32", "F16", "BF16", "I32", "I64", "BOOL"};
    const char *dn = (t->dtype >= 0 && t->dtype <= 5) ? names[t->dtype] : "UNKNOWN";
    printf("%s: dtype=%s, shape=[", t->name, dn);
    for (int i = 0; i < t->ndim; i++)
        printf("%ld%s", (long)t->shape[i], i < t->ndim - 1 ? ", " : "");
    printf("], offset=%zu, size=%zu\n", t->data_offset, t->data_size);
}

void safetensors_print_all(const safetensors_file_t *sf) {
    printf("Safetensors file: %s\n", sf->path);
    printf("File size: %zu bytes\n", sf->file_size);
    printf("Header size: %zu bytes\n", sf->header_size);
    printf("Number of tensors: %d\n\n", sf->num_tensors);
    for (int i = 0; i < sf->num_tensors; i++) safetensor_print(&sf->tensors[i]);
}

int safetensors_count_prefix(const safetensors_file_t *sf, const char *prefix) {
    size_t n = strlen(prefix);
    int count = 0;
    for (int i = 0; i < sf->num_tensors; i++)
        if (strncmp(sf->tensors[i].name, prefix, n) == 0) count++;
    return count;
}
