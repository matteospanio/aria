#ifndef ARIA_PTHREAD_COMPAT_H
#define ARIA_PTHREAD_COMPAT_H
/*
 * Minimal pthread shim for the MSVC build path (POSIX builds use the real
 * <pthread.h>). Only the primitives the Windows CLI needs are provided.
 *
 * SRWLOCK, not CRITICAL_SECTION: a static PTHREAD_MUTEX_INITIALIZER has to be a
 * genuine initializer, and SRWLOCK_INIT is exactly {0}. A CRITICAL_SECTION has
 * no static initializer, so the earlier version called InitializeCriticalSection
 * lazily on first lock -- guarded by a `static int` that is per-translation-unit
 * rather than per-mutex (a second mutex in the same TU would enter an
 * uninitialized object) and that races whenever another thread locks the mutex
 * before the initializing one. SRWLOCK needs no lazy init at all.
 *
 * It pairs with SleepConditionVariableSRW, which requires the lock to be held
 * exclusively -- exactly what pthread_cond_wait guarantees on the caller.
 */
#ifdef _WIN32
#include <windows.h>
#include <process.h>
typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef HANDLE pthread_t;
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define PTHREAD_COND_INITIALIZER CONDITION_VARIABLE_INIT
static inline int pthread_mutex_lock(pthread_mutex_t *m){AcquireSRWLockExclusive(m);return 0;}
static inline int pthread_mutex_unlock(pthread_mutex_t *m){ReleaseSRWLockExclusive(m);return 0;}
static inline int pthread_cond_wait(pthread_cond_t *c,pthread_mutex_t*m){SleepConditionVariableSRW(c,m,INFINITE,0);return 0;}
static inline int pthread_cond_signal(pthread_cond_t *c){WakeConditionVariable(c);return 0;}
typedef unsigned (__stdcall *aria_thread_fn)(void*);
static inline int pthread_create(pthread_t*t,void*a,void*(*fn)(void*),void*x){(void)a; *t=(HANDLE)_beginthreadex(NULL,0,(aria_thread_fn)fn,x,0,NULL);return *t?0:-1;}
static inline int pthread_join(pthread_t t,void**r){(void)r; WaitForSingleObject(t,INFINITE); CloseHandle(t);return 0;}
#endif
#endif
