#ifndef ARIA_WIN_COMPAT_H
#define ARIA_WIN_COMPAT_H
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <time.h>
#define __thread __declspec(thread)
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
static inline int aria_clock_gettime(int clock_id, struct timespec *t) {
    (void)clock_id; static LARGE_INTEGER f; static int init; LARGE_INTEGER q;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&q); t->tv_sec=(time_t)(q.QuadPart/f.QuadPart); t->tv_nsec=(long)((q.QuadPart%f.QuadPart)*1000000000LL/f.QuadPart); return 0;
}
#define clock_gettime aria_clock_gettime
#endif
#ifndef strtok_r
#define strtok_r strtok_s
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef PROT_READ
#define PROT_READ 1
#endif
#ifndef MADV_DONTNEED
#define MADV_DONTNEED 4
#endif
#ifndef MAP_PRIVATE
#define MAP_PRIVATE 2
#endif
#ifndef MAP_FAILED
#define MAP_FAILED ((void *)-1)
#endif
static inline void *aria_mmap(void *addr, size_t len, int prot, int flags, int fd, long long off) {
    (void)addr; (void)flags;
    HANDLE file = (HANDLE)_get_osfhandle(fd);
    if (file == INVALID_HANDLE_VALUE) return MAP_FAILED;
    DWORD protect = (prot & 2) ? PAGE_READWRITE : PAGE_READONLY;
    DWORD access = (prot & 2) ? FILE_MAP_WRITE : FILE_MAP_READ;
    LARGE_INTEGER size; size.QuadPart = (LONGLONG)len + off;
    HANDLE mapping = CreateFileMappingA(file, NULL, protect, size.HighPart, size.LowPart, NULL);
    if (!mapping) return MAP_FAILED;
    void *p = MapViewOfFile(mapping, access, (DWORD)((uint64_t)off >> 32), (DWORD)off, len);
    CloseHandle(mapping);
    return p ? p : MAP_FAILED;
}
static inline int aria_munmap(void *p, size_t len) { (void)len; return UnmapViewOfFile(p) ? 0 : -1; }
static inline int aria_madvise(void *p, size_t len, int advice) { (void)p; (void)len; (void)advice; return 0; }
#define mmap aria_mmap
#define munmap aria_munmap
#define madvise aria_madvise
#define sysconf(name) 4096L
#define _SC_PAGESIZE 1
#endif
#endif
