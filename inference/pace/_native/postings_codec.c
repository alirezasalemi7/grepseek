/* Rank support for the separator bitvector used to map a vocabulary
 * suffix-array position back to its term id (see vocab_search.c and
 * core.py's PaceStructure.term_ids()).
 *
 * Plain loop meant to be called via ctypes with numpy-backed buffers; it
 * never allocates.
 */
#include <stdint.h>
#include <stddef.h>

/* Rank support for the separator bitvector: number of set bits in
 * `words[0..n_words)` restricted to bit positions < pos, given per-word
 * cumulative counts `cum` (cum[w] = set bits in words 0..w-1). Vectorized
 * over `n` query positions. */
int rank_positions(const uint64_t *words, const uint32_t *cum, const int64_t *positions, int64_t n, int64_t *out) {
    for (int64_t i = 0; i < n; i++) {
        int64_t p = positions[i];
        int64_t w = p >> 6;
        uint64_t mask = (p & 63) ? ((uint64_t)1 << (p & 63)) - 1 : 0;
        out[i] = (int64_t)cum[w] + __builtin_popcountll(words[w] & mask);
    }
    return 0;
}
