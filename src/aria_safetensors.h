/*
 * aria_safetensors.h - Safetensors file format reader
 *
 * Ported from references/iris.c (iris_safetensors.*). The format is:
 *   - 8 bytes: uint64 little-endian header size
 *   - N bytes: JSON header with tensor metadata
 *   - Remaining: raw tensor data (page-cache friendly via mmap)
 *
 * Stable Audio 3 ships a single model.safetensors with three top-level
 * prefixes: "conditioner.", "model." (the DiT) and "pretransform." (the
 * taae_v2 autoencoder). All weights are F32.
 */

#ifndef ARIA_SAFETENSORS_H
#define ARIA_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SA3 small-music has 685 tensors; medium is larger. Keep generous. */
#define SAFETENSORS_MAX_TENSORS 4096

typedef enum {
    DTYPE_F32 = 0,
    DTYPE_F16 = 1,
    DTYPE_BF16 = 2,
    DTYPE_I32 = 3,
    DTYPE_I64 = 4,
    DTYPE_BOOL = 5,
    DTYPE_UNKNOWN = -1
} safetensor_dtype_t;

typedef struct {
    char name[256];
    safetensor_dtype_t dtype;
    int ndim;
    int64_t shape[8];
    size_t data_offset;   /* offset within the tensor-data region */
    size_t data_size;     /* bytes */
} safetensor_t;

typedef struct {
    char *path;
    void *data;           /* mmap'd file data */
    size_t file_size;
    size_t header_size;
    char *header_json;
    int num_tensors;
    safetensor_t *tensors;     /* heap array, num_tensors entries */
    int tensors_capacity;
} safetensors_file_t;

/* Open a safetensors file (memory-mapped). Returns NULL on error. */
safetensors_file_t *safetensors_open(const char *path);

/* Close and free resources. */
void safetensors_close(safetensors_file_t *sf);

/* Hint the kernel to drop the mmap's resident pages (MADV_DONTNEED). Use after the
 * weights have been copied elsewhere (e.g. uploaded to the GPU) and won't be read on
 * the host again -- frees the host RSS; any re-access just re-faults from the file. */
void safetensors_advise_dontneed(const safetensors_file_t *sf);

/* Same, for one range: drops the whole pages inside [p, p+bytes) when the range lies in
 * an open safetensors mapping; a no-op for any other memory (heap buffers stay intact).
 * Called right after a weight is uploaded, so host RSS never holds the whole model. */
void safetensors_release(const void *p, size_t bytes);

/* Find a tensor by exact name, returns NULL if not found. */
const safetensor_t *safetensors_find(const safetensors_file_t *sf, const char *name);

/* Raw pointer to tensor data within the mmap'd region (no copy). */
const void *safetensors_data(const safetensors_file_t *sf, const safetensor_t *t);

/* Tensor data as a newly allocated f32 array (caller frees). Converts F16/BF16. */
float *safetensors_get_f32(const safetensors_file_t *sf, const safetensor_t *t);

/* Zero-copy pointer into the mmap'd F32 tensor data (no copy; valid while the
 * file is open). Returns NULL if the tensor is not F32. Tensor data is 4-byte
 * aligned, fine for (unaligned) SIMD loads. */
const float *safetensors_f32_ptr(const safetensors_file_t *sf, const safetensor_t *t);

int64_t safetensor_numel(const safetensor_t *t);

void safetensor_print(const safetensor_t *t);
void safetensors_print_all(const safetensors_file_t *sf);

/* Convenience: number of tensors whose name starts with prefix. */
int safetensors_count_prefix(const safetensors_file_t *sf, const char *prefix);

#ifdef __cplusplus
}
#endif

#endif /* ARIA_SAFETENSORS_H */
