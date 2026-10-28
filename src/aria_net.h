/*
 * aria_net.h - the few socket calls aria-server and aria-repl need, behind one small
 * header so both build on POSIX and (with the Winsock branch) on Windows.
 * Header-only; nothing here is part of libaria.
 */
#ifndef ARIA_NET_H
#define ARIA_NET_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#  error "aria_net.h: Winsock branch not implemented yet"
#else
#  include <arpa/inet.h>
#  include <errno.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <strings.h>
#  include <sys/socket.h>
#  include <sys/time.h>
#  include <unistd.h>
typedef int aria_sock;
#  define ARIA_SOCK_BAD (-1)

static inline int aria_net_init(void) { return 0; }
static inline void aria_sock_close(aria_sock s) { close(s); }
/* wake a thread blocked in accept() on `s` (async-signal-safe; close() alone does not) */
static inline void aria_sock_wake(aria_sock s) { shutdown(s, SHUT_RDWR); }

/* send/recv timeouts; 0 = none */
static inline void aria_sock_timeouts(aria_sock s, int ms) {
    struct timeval tv = { ms / 1000, (ms % 1000) * 1000 };
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
}

static inline int aria_send_all(aria_sock s, const void *buf, size_t n) {
    const char *p = (const char *)buf;
    while (n) {
        ssize_t k = send(s, p, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}

static inline long aria_recv(aria_sock s, void *buf, size_t n) {
    ssize_t k;
    do k = recv(s, buf, n, 0); while (k < 0 && errno == EINTR);
    return (long)k;
}

/* non-blocking read: >0 bytes, 0 = the peer closed, -1 = nothing to read now (or error) */
static inline long aria_recv_nb(aria_sock s, void *buf, size_t n) {
    ssize_t k = recv(s, buf, n, MSG_DONTWAIT);
    return k > 0 ? (long)k : k == 0 ? 0 : -1;
}

/* 1 if the peer has closed its end (orderly shutdown), without consuming data */
static inline int aria_peer_gone(aria_sock s) {
    char c;
    return recv(s, &c, 1, MSG_PEEK | MSG_DONTWAIT) == 0;
}

/* listening TCP socket on ip:port (ip NULL = 127.0.0.1); ARIA_SOCK_BAD on error */
static inline aria_sock aria_listen(const char *ip, int port) {
    aria_sock s = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return ARIA_SOCK_BAD;
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip ? ip : "127.0.0.1", &a.sin_addr) != 1 ||
        bind(s, (struct sockaddr *)&a, sizeof a) != 0 || listen(s, 64) != 0) {
        close(s); return ARIA_SOCK_BAD;
    }
    return s;
}

static inline aria_sock aria_accept(aria_sock l) {
    aria_sock s = accept4(l, NULL, NULL, SOCK_CLOEXEC);
    if (s >= 0) { int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one); }
    return s < 0 ? ARIA_SOCK_BAD : s;
}

/* connected TCP socket to host:port; ARIA_SOCK_BAD on error */
static inline aria_sock aria_connect(const char *host, int port) {
    char ps[16]; struct addrinfo hints, *res = NULL, *ai;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    snprintf(ps, sizeof ps, "%d", port);
    if (getaddrinfo(host, ps, &hints, &res) != 0) return ARIA_SOCK_BAD;
    aria_sock s = ARIA_SOCK_BAD;
    for (ai = res; ai; ai = ai->ai_next) {
        s = socket(ai->ai_family, ai->ai_socktype | SOCK_CLOEXEC, ai->ai_protocol);
        if (s < 0) { s = ARIA_SOCK_BAD; continue; }
        if (connect(s, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(s); s = ARIA_SOCK_BAD;
    }
    freeaddrinfo(res);
    if (s != ARIA_SOCK_BAD) { int one = 1; setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one); }
    return s;
}

#  define aria_strcasestr strcasestr
#endif

#endif /* ARIA_NET_H */
