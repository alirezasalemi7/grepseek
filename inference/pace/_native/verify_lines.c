/* In-process literal verification of candidate lines (replaces spawning
 * `bash -c "rg -F A | rg -i -F B | head -n N"` for the dominant pipeline
 * shapes). The caller guarantees the shape is one whose output rg would
 * produce identically (see core.py: ASCII-only fixed-string patterns,
 * flags limited to -F/-i, no NUL bytes, and for -i no bytes of U+017F /
 * U+212A, the only non-ASCII characters whose simple case folding is an
 * ASCII letter). Lines are '\n'-terminated; a final unterminated line is
 * treated as a line, as rg does.
 *
 * verify_lines() scans `buf` line by line; a line is selected when every
 * pattern matches it (pattern i case-insensitively when ci[i] != 0). It
 * stops after `max_matches` selections (0 = unlimited) and records each
 * selected line's [start, end) byte span (end excludes the '\n').
 * Returns the number of selected lines.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>

static const uint8_t *find_ci(const uint8_t *hay, size_t hlen, const uint8_t *needle_lower, size_t nlen) {
    if (nlen == 0) return hay;
    if (hlen < nlen) return NULL;
    uint8_t first = needle_lower[0];
    for (size_t i = 0; i + nlen <= hlen; i++) {
        uint8_t c = hay[i];
        if (c >= 'A' && c <= 'Z') c = (uint8_t)(c + 32);
        if (c != first) continue;
        size_t j = 1;
        for (; j < nlen; j++) {
            uint8_t d = hay[i + j];
            if (d >= 'A' && d <= 'Z') d = (uint8_t)(d + 32);
            if (d != needle_lower[j]) break;
        }
        if (j == nlen) return hay + i;
    }
    return NULL;
}

int64_t verify_lines(const uint8_t *buf, int64_t len,
                     const uint8_t *const *patterns, const int64_t *pat_lens, const int32_t *ci, int32_t n_pat,
                     int64_t max_matches, int64_t *out_starts, int64_t *out_ends, int64_t out_cap) {
    int64_t n_out = 0;
    int64_t pos = 0;
    while (pos < len) {
        const uint8_t *nl = memchr(buf + pos, '\n', (size_t)(len - pos));
        int64_t end = nl ? (int64_t)(nl - buf) : len;
        int ok = 1;
        for (int32_t k = 0; k < n_pat && ok; k++) {
            const uint8_t *hit;
            if (ci[k]) {
                hit = find_ci(buf + pos, (size_t)(end - pos), patterns[k], (size_t)pat_lens[k]);
            } else {
                hit = memmem(buf + pos, (size_t)(end - pos), patterns[k], (size_t)pat_lens[k]);
            }
            if (!hit) ok = 0;
        }
        if (ok) {
            if (n_out < out_cap) {
                out_starts[n_out] = pos;
                out_ends[n_out] = end;
            }
            n_out++;
            if (max_matches > 0 && n_out >= max_matches) break;
        }
        pos = end + 1;
    }
    return n_out;
}

/* 1 if `buf` contains any byte sequence that makes rg's behaviour differ
 * from the plain byte comparison above: a NUL (binary detection) always,
 * and when `check_fold` != 0 the UTF-8 encodings of U+017F (C5 BF) and
 * U+212A (E2 84 AA). */
int verify_unsafe_bytes(const uint8_t *buf, int64_t len, int32_t check_fold) {
    if (memchr(buf, 0, (size_t)len)) return 1;
    if (check_fold) {
        static const uint8_t long_s[2] = {0xC5, 0xBF};
        static const uint8_t kelvin[3] = {0xE2, 0x84, 0xAA};
        if (memmem(buf, (size_t)len, long_s, 2)) return 1;
        if (memmem(buf, (size_t)len, kelvin, 3)) return 1;
    }
    return 0;
}
