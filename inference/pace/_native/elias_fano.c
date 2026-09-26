/* Elias-Fano codec for sorted uint32 posting lists (the auxiliary-structure
 * posting format).
 *
 * A list of n strictly increasing values below universe u is stored as
 *   header: 1 byte l (number of low bits), then
 *   low part: n * l bits, value_i & ((1<<l)-1), packed little-endian, then
 *   high part: unary-coded high halves: bit (value_i >> l) + i set, in a
 *   bitvector of n + (u >> l) + 1 bits (also little-endian bit order),
 * each part padded to a byte boundary. l = floor(log2(u / n)) gives
 * ~2 + log2(u/n) bits per value, the information-theoretic bound for a
 * sorted set up to a constant, whatever the gap distribution.
 *
 * Decoding is a sequential scan of the high bitvector (word at a time,
 * ctz to find set bits) combined with the packed low bits.
 */
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static inline int64_t ef_low_bits(int64_t n, int64_t u) {
    if (n <= 0) return 0;
    int64_t l = 0;
    while (((int64_t)1 << (l + 1)) <= u / n) l++;
    return l;
}

/* Bytes needed to encode a list of n values with universe u. */
int64_t ef_encoded_size(int64_t n, int64_t u) {
    int64_t l = ef_low_bits(n, u);
    int64_t low_bytes = (n * l + 7) / 8;
    int64_t high_bits = n + (u >> l) + 1;
    return 1 + low_bytes + (high_bits + 7) / 8;
}

static inline void set_bit(uint8_t *bits, int64_t pos) { bits[pos >> 3] |= (uint8_t)(1u << (pos & 7)); }

/* Encode; `out` must hold ef_encoded_size(n, u) bytes, zero-filled by the
 * callee. Returns bytes written. */
int64_t ef_encode_list(const uint32_t *vals, int64_t n, int64_t u, uint8_t *out) {
    int64_t l = ef_low_bits(n, u);
    int64_t low_bytes = (n * l + 7) / 8;
    int64_t high_bits = n + (u >> l) + 1;
    int64_t total = 1 + low_bytes + (high_bits + 7) / 8;
    memset(out, 0, (size_t)total);
    out[0] = (uint8_t)l;
    uint8_t *low = out + 1;
    uint8_t *high = out + 1 + low_bytes;
    uint64_t mask = l ? (((uint64_t)1 << l) - 1) : 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t v = vals[i];
        if (l) {
            uint64_t lo = v & mask;
            int64_t bitpos = i * l;
            /* write l bits little-endian starting at bitpos */
            for (int64_t b = 0; b < l; b++) {
                if ((lo >> b) & 1) set_bit(low, bitpos + b);
            }
        }
        set_bit(high, (int64_t)(v >> l) + i);
    }
    return total;
}

static inline uint64_t read_bits(const uint8_t *low, int64_t bitpos, int64_t l) {
    /* read up to 32 bits little-endian at arbitrary bit position */
    uint64_t w = 0;
    int64_t byte = bitpos >> 3;
    int shift = (int)(bitpos & 7);
    int64_t need = (shift + l + 7) / 8;  /* <= 5 bytes for l <= 32 */
    for (int64_t k = 0; k < need; k++) w |= (uint64_t)low[byte + k] << (8 * k);
    return (w >> shift) & (l ? (((uint64_t)1 << l) - 1) : 0);
}

/* Decode n values from `in` into `out`. Returns bytes consumed. */
int64_t ef_decode_list(const uint8_t *in, int64_t n, int64_t u, uint32_t *out) {
    int64_t l = in[0];
    int64_t low_bytes = (n * l + 7) / 8;
    int64_t high_bits = n + (u >> l) + 1;
    const uint8_t *low = in + 1;
    const uint8_t *high = in + 1 + low_bytes;
    int64_t high_bytes = (high_bits + 7) / 8;
    /* scan high bitvector: the k-th set bit at position p encodes high = p - k */
    int64_t i = 0;
    int64_t word_idx = 0;
    int64_t nwords = (high_bytes + 7) / 8;
    while (i < n && word_idx < nwords) {
        uint64_t w = 0;
        int64_t rem = high_bytes - word_idx * 8;
        memcpy(&w, high + word_idx * 8, (size_t)(rem >= 8 ? 8 : rem));
        while (w && i < n) {
            int tz = __builtin_ctzll(w);
            int64_t p = word_idx * 64 + tz;
            uint64_t hi = (uint64_t)(p - i);
            uint64_t lo = l ? read_bits(low, i * l, l) : 0;
            out[i] = (uint32_t)((hi << l) | lo);
            i++;
            w &= w - 1;
        }
        word_idx++;
    }
    return 1 + low_bytes + high_bytes;
}

/* Bulk: decode `n_lists` lists (byte offsets and counts selected via
 * list_idx) back to back into out; returns total values written. */
int64_t ef_decode_lists(const uint8_t *base, const uint64_t *byte_offsets, const uint32_t *counts,
                        const int64_t *list_idx, int64_t n_lists, int64_t u, uint32_t *out) {
    int64_t written = 0;
    for (int64_t k = 0; k < n_lists; k++) {
        int64_t t = list_idx[k];
        int64_t c = (int64_t)counts[t];
        ef_decode_list(base + byte_offsets[t], c, u, out + written);
        written += c;
    }
    return written;
}

/* Bulk encode: lists laid out back to back in vals (list k at el_offsets[k],
 * counts[k] values); writes encoded lists into out (sized by the caller via
 * ef_encoded_size sums) and byte offsets into out_byteoffsets. Returns bytes. */
int64_t ef_encode_lists(const uint32_t *vals, const int64_t *el_offsets, const uint32_t *counts,
                        int64_t n_lists, int64_t u, uint8_t *out, int64_t *out_byteoffsets) {
    int64_t pos = 0;
    for (int64_t k = 0; k < n_lists; k++) {
        out_byteoffsets[k] = pos;
        pos += ef_encode_list(vals + el_offsets[k], (int64_t)counts[k], u, out + pos);
    }
    return pos;
}
