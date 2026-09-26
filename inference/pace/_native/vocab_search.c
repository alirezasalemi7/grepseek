/* Substring search over a generalized suffix array (GSA) built by libsais
 * over the corpus's distinct vocabulary terms, concatenated with 0x00
 * separators (see build_pace() in core.py).
 *
 * Why this exists: a word-level exact-token lookup structure (the original
 * design) cannot find a query substring that lands *inside* a larger token
 * -- "16" inside "2016", or inside a JSON \uXXXX escape sequence merged
 * with adjacent text. A GSA over the vocabulary lets us find every
 * vocabulary term *containing* a query substring, not just terms exactly
 * equal to it, closing that gap without needing any escape-sequence
 * special-casing at tokenization time.
 */
#include <stdint.h>
#include <string.h>

/* Compare the suffix starting at vocab[sa_pos] against `query` (length
 * query_len), for the purpose of a prefix-range binary search. `vocab` is
 * the same buffer used to build `SA`; it may contain 0x00 term separators,
 * which can never equal a query byte (queries are always [a-z0-9]+ text),
 * so a separator falling within the compared window always breaks the
 * match correctly without needing separate per-term bounds checking.
 *
 * Returns <0 if the suffix sorts before query, 0 if query is a prefix of
 * the suffix (a match), >0 if the suffix sorts after query.
 */
static int compare_prefix(const uint8_t *vocab, int64_t vocab_len, int32_t sa_pos,
                           const uint8_t *query, int64_t query_len) {
    int64_t avail = vocab_len - (int64_t)sa_pos;
    int64_t cmp_len = avail < query_len ? avail : query_len;
    int cmp = memcmp(vocab + sa_pos, query, (size_t)cmp_len);
    if (cmp != 0) {
        return cmp;
    }
    if (avail < query_len) {
        /* The suffix ran out of bytes before query did: it's a proper
         * prefix of query, so it sorts strictly before query itself. */
        return -1;
    }
    return 0;
}

/* Find the contiguous range [*out_lo, *out_hi) of positions into `SA` (of
 * length sa_len) such that for every i in that range, the suffix starting
 * at vocab[SA[i]] has `query` as a prefix -- i.e. every vocabulary term
 * containing `query` as a substring has at least one suffix-array entry in
 * this range (there may be more than one per term if `query` occurs
 * multiple times within the same term; callers dedupe by mapping back to
 * the term id).
 *
 * Returns 0 on success (an empty range, out_lo == out_hi, means no vocabulary
 * term contains `query`), -1 if query_len <= 0.
 */
int vocab_search_range(const uint8_t *vocab, int64_t vocab_len,
                        const int32_t *SA, int64_t sa_len,
                        const uint8_t *query, int64_t query_len,
                        int64_t *out_lo, int64_t *out_hi) {
    if (query_len <= 0) {
        *out_lo = 0;
        *out_hi = 0;
        return -1;
    }

    /* Lower bound: first position whose suffix is a match-or-greater. */
    int64_t lo = 0, hi = sa_len;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int c = compare_prefix(vocab, vocab_len, SA[mid], query, query_len);
        if (c < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int64_t range_lo = lo;

    /* Upper bound: first position whose suffix is strictly greater (i.e. no
     * longer a match). Matches (c == 0) always sort as one contiguous
     * block between range_lo and range_hi, since sharing a common prefix
     * is preserved under lexicographic order. */
    lo = range_lo;
    hi = sa_len;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int c = compare_prefix(vocab, vocab_len, SA[mid], query, query_len);
        if (c <= 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    int64_t range_hi = lo;

    *out_lo = range_lo;
    *out_hi = range_hi;
    return 0;
}
