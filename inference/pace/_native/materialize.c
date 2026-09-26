/* Batched pread-based line materialization for PACE.
 *
 * Python's buffered file object (what open(path, "rb") returns) does real
 * per-call work on every .seek()/.read(): it mutates shared cursor state,
 * goes through its own buffering layer, and pays two Python-level method
 * dispatches per line. For a candidate set of tens of thousands of lines
 * (the common case for a non-selective search term), that overhead was
 * measured to dominate materialize time -- see this project's perf
 * investigation history for the numbers.
 *
 * This single function replaces that loop with a tight C loop of raw
 * pread(2) calls (one syscall per span, no separate seek, no shared cursor)
 * into one caller-allocated output buffer. Called via ctypes from Python;
 * ctypes releases the GIL around the call by default, which also removes
 * the GIL-contention tax measured under this project's real 29-way threaded
 * eval workload -- a plain Python loop holds the GIL the whole time it runs,
 * this doesn't.
 *
 * Deliberately scoped to just this one operation: line-span -> byte-span
 * computation, candidate-set intersection, and everything else stay in
 * Python. This keeps the native surface small, auditable, and easy to keep
 * correct, while still capturing the actual measured bottleneck.
 */
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>

/* Read `n` spans out of the file at `path`: span i is `lengths[i]` bytes
 * starting at `offsets[i]`. Writes them back-to-back, in the given order,
 * into `out` (caller-allocated, must hold at least sum(lengths) bytes).
 *
 * Returns 0 on success. Returns -1 on error (open() failure, a read error,
 * or unexpected EOF mid-span) with errno set -- callers should fall back to
 * the pure-Python path on any nonzero return, never assume `out` is
 * meaningfully populated.
 */
int materialize_spans(const char *path, const int64_t *offsets, const int64_t *lengths,
                       int64_t n, unsigned char *out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    int64_t out_pos = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t remaining = lengths[i];
        int64_t off = offsets[i];
        if (remaining < 0) {
            close(fd);
            errno = EINVAL;
            return -1;
        }
        while (remaining > 0) {
            ssize_t r = pread(fd, out + out_pos, (size_t)remaining, (off_t)off);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                close(fd);
                return -1;
            }
            if (r == 0) {
                /* Unexpected EOF mid-span: the corpus file shrank or the
                 * offsets are stale relative to it. Treat as a hard error --
                 * never silently return a truncated/wrong buffer. */
                close(fd);
                errno = EIO;
                return -1;
            }
            out_pos += r;
            off += r;
            remaining -= r;
        }
    }

    if (close(fd) < 0) {
        return -1;
    }
    return 0;
}

#include <string.h>

/* Copy `n` spans out of an already-mapped file image `base` (the caller
 * mmaps the corpus once per process and keeps it): span i is `lengths[i]`
 * bytes at `offsets[i]`, written back-to-back into `out`. No syscalls per
 * span -- for dense candidate sets (a million short lines) the per-pread
 * syscall cost of materialize_spans() dominated; a memcpy loop over a
 * page-cache-resident mapping is bounded by memory bandwidth instead.
 * `base_len` guards against stale offsets. Returns 0 on success, -1 (errno
 * EINVAL) if any span falls outside the mapping.
 */
int copy_spans(const unsigned char *base, int64_t base_len, const int64_t *offsets,
               const int64_t *lengths, int64_t n, unsigned char *out) {
    int64_t out_pos = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t off = offsets[i];
        int64_t len = lengths[i];
        if (off < 0 || len < 0 || off + len > base_len) {
            errno = EINVAL;
            return -1;
        }
        memcpy(out + out_pos, base + off, (size_t)len);
        out_pos += len;
    }
    return 0;
}
