"""PACE (Pruned Adaptive Command Execution): a one-time, offline auxiliary
structure over a flat JSONL corpus that narrows literal `rg`/`grep`
pipelines to candidate lines before the real pipeline runs, instead of a
full linear scan of every line on every call.

Correctness contract
--------------------
PACE only ever narrows the *input* of the real `rg`/`grep` process; it
never produces output itself. For that to be output-preserving the
candidate set must be a superset of every line the real command would
print. The guarantee rests on three facts:

1. Tokenization is "maximal Unicode-alphanumeric runs of casefold(line)".
   If a literal pattern P occurs in a line L (byte-wise, or
   case-insensitively under rg's simple case folding), then every
   alphanumeric run of casefold(P) is a substring of some alphanumeric run
   of casefold(L) -- runs in L can only be longer than the corresponding
   run in P, never shorter.
2. The vocabulary (every distinct token of length >= min_term_len) sits in
   a generalized suffix array (libsais), so a query word is resolved to
   *every* vocabulary term containing it as a substring ("16" -> "2016",
   "160", ...), not only to an exact token. Query words shorter than
   min_term_len are never used for narrowing.
3. Candidates are the intersection of those per-word posting unions, over
   the words of every positive per-line filter stage at the head of the
   pipeline (`rg -F A corpus | rg -i -F B | ...`). Dropping a word from the
   intersection can only widen the set, so words that are too expensive to
   resolve are simply skipped.

Everything the planner cannot prove safe (unknown flags, `-v` as the first
stage, regex metacharacters without `-F`, a second file argument, ...)
returns "not served" and the caller runs the original command unmodified.

Cost model
----------
Resolving a word costs O(#suffix-array matches) to map to vocabulary
entries, then O(sum of their posting counts) to gather. The posting counts
are stored per vocabulary term, so the gather cost of a word is known
*before* touching a single posting; the planner resolves words cheapest
first, stops once the candidate set is small, and never gathers a word
whose estimated cost exceeds the per-structure budget. See `PaceBudget`.

Materialization streams the candidate lines (coalesced into contiguous
byte runs) into the real command's stdin in file order, so a pipeline that
terminates early (`| head -n 3`) stops the feed early as well, exactly as a
full scan would stop on SIGPIPE.

Callers load one PACE structure per process via `load_pace_from_env()`
(None when `GREPSEEK_PACE_DIR` is unset -- the opt-in gate) and then call
`decide_for_pace()` per tool call, streaming the candidate buffer via
`PaceStructure.iter_materialized_chunks()`.

This build supports exactly one on-disk layout: singletons-first
vocabulary, Elias-Fano posting lists, a pruned vocabulary suffix array,
blocked line offsets, and a separator-rank bitvector, built over
Unicode-alphanumeric, casefolded tokens. Anything else fails to load (see
`PaceStructure.__init__`).
"""
from __future__ import annotations

import ctypes
import json
import logging
import mmap
import os
import re
import threading
import time
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native extension (optional for materialize, required for substring lookup)
# ---------------------------------------------------------------------------
# See _native/materialize.c, _native/vocab_search.c, _native/postings_codec.c,
# _native/verify_lines.c, _native/elias_fano.c and the vendored
# _native/libsais/. Built once per target environment via _native/build.sh.
# materialize falls back to a pure-Python pread loop if this is missing;
# substring lookup has no fallback: without it every decision is
# "not served" (safe: callers then run the unmodified command).
_NATIVE_MATERIALIZE = None
_NATIVE_COPY_SPANS = None
_RANK_POSITIONS = None
_VERIFY_LINES = None
_VERIFY_UNSAFE_BYTES = None
_EF_ENCODED_SIZE = None
_EF_ENCODE_LISTS = None
_EF_DECODE_LISTS = None
_LIBSAIS_GSA = None
_VOCAB_SEARCH_RANGE = None
try:
    _native_lib = ctypes.CDLL(str(Path(__file__).resolve().parent / "_native" / "materialize.so"))

    _native_lib.materialize_spans.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_char_p,
    ]
    _native_lib.materialize_spans.restype = ctypes.c_int
    _NATIVE_MATERIALIZE = _native_lib.materialize_spans

    _native_lib.copy_spans.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
        ctypes.c_char_p,
    ]
    _native_lib.copy_spans.restype = ctypes.c_int
    _NATIVE_COPY_SPANS = _native_lib.copy_spans

    _native_lib.rank_positions.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    _native_lib.rank_positions.restype = ctypes.c_int
    _RANK_POSITIONS = _native_lib.rank_positions

    _native_lib.verify_lines.argtypes = [
        ctypes.c_void_p, ctypes.c_int64, ctypes.POINTER(ctypes.c_char_p), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
    ]
    _native_lib.verify_lines.restype = ctypes.c_int64
    _VERIFY_LINES = _native_lib.verify_lines
    _native_lib.verify_unsafe_bytes.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int32]
    _native_lib.verify_unsafe_bytes.restype = ctypes.c_int
    _VERIFY_UNSAFE_BYTES = _native_lib.verify_unsafe_bytes

    _native_lib.ef_encoded_size.argtypes = [ctypes.c_int64, ctypes.c_int64]
    _native_lib.ef_encoded_size.restype = ctypes.c_int64
    _EF_ENCODED_SIZE = _native_lib.ef_encoded_size
    _native_lib.ef_encode_lists.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    _native_lib.ef_encode_lists.restype = ctypes.c_int64
    _EF_ENCODE_LISTS = _native_lib.ef_encode_lists
    _native_lib.ef_decode_lists.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    _native_lib.ef_decode_lists.restype = ctypes.c_int64
    _EF_DECODE_LISTS = _native_lib.ef_decode_lists

    _native_lib.libsais_gsa.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    _native_lib.libsais_gsa.restype = ctypes.c_int32
    _LIBSAIS_GSA = _native_lib.libsais_gsa

    _native_lib.vocab_search_range.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    _native_lib.vocab_search_range.restype = ctypes.c_int
    _VOCAB_SEARCH_RANGE = _native_lib.vocab_search_range
except OSError:
    logger.info(
        "native extension not available (build it with "
        "inference/pace/_native/build.sh); materialize will use its "
        "pure-Python fallback and every decision will be 'not served'"
    )

# Tokens are maximal runs of Unicode letters/digits, so "baş" or
# "trémoille" stay one token instead of splitting at the non-ASCII letter.
_WORD_RE = re.compile(r"[^\W_]+")
TOKEN_CLASS_UNICODE = "unicode_alnum"

VOCAB_FILENAME = "vocab.bin"
VOCAB_SA_FILENAME = "vocab_sa.bin"
VOCAB_SEP_WORDS_FILENAME = "vocab_sep_words.bin"
VOCAB_SEP_CUM_FILENAME = "vocab_sep_cum.bin"
META_FILENAME = "meta.json"
# singles.bin                 uint32 line number of each singleton term (ids 0..n_singletons-1)
# multi_count.bin             uint32 count of each multi-posting term (id - n_singletons)
# multi_ef_byteoffset.bin     uint64 byte offset of each multi-posting term's Elias-Fano list
# postings_ef.bin             Elias-Fano encoded lists (_native/elias_fano.c)
# offsets_base.bin/_delta.bin uint64 base per 1024 lines + uint32 delta per line
# Vocabulary order is singletons first, so singleton terms carry no metadata
# at all; the suffix array keeps only suffixes that can match a query of
# >= min_term_len characters (drops separator and last-character suffixes).
SINGLES_FILENAME = "singles.bin"
MULTI_COUNT_FILENAME = "multi_count.bin"
MULTI_EF_BYTEOFFSET_FILENAME = "multi_ef_byteoffset.bin"
POSTINGS_EF_FILENAME = "postings_ef.bin"
OFFSETS_BASE_FILENAME = "offsets_base.bin"
OFFSETS_DELTA_FILENAME = "offsets_delta.bin"
OFFSETS_BLOCK = 1024

FORMAT_VERSION = 4
TOKEN_FOLD_CASEFOLD = "casefold"


def _fold(text: str) -> str:
    """`casefold()` rather than `lower()` because rg's `-i` uses Unicode
    simple case folding: a line containing the long s "ſ" *is* matched by
    `rg -i -F "s..."`, and only casefold maps "ſ" -> "s" on both sides."""
    return text.casefold()


def _tokenize(line: str) -> set[str]:
    """Alphanumeric tokens (maximal runs of Unicode letters/digits) of the
    case-folded line. No JSON-escape handling on purpose: a `\\uXXXX` or
    `\\n` escape glued to real text yields a merged token
    ("annphotosynthesis"), and substring lookup still finds the query word
    inside it."""
    return set(_WORD_RE.findall(_fold(line)))


def _iter_lines_with_offsets(f: BinaryIO) -> Iterator[tuple[int, bytes]]:
    """Yield (byte_offset, raw_line_bytes) for each line of an open binary
    file. Offsets come from the raw bytes so they line up exactly with a
    later pread against the same file."""
    offset = 0
    for raw_line in f:
        yield offset, raw_line
        offset += len(raw_line)


@dataclass(frozen=True, slots=True)
class PaceMetadata:
    corpus_path: str
    corpus_size: int
    corpus_mtime: float
    num_lines: int
    num_terms: int
    min_term_len: int
    built_at: float
    token_fold: str = TOKEN_FOLD_CASEFOLD
    format_version: int = FORMAT_VERSION
    n_singletons: int = 0  # terms [0, n_singletons) have exactly one posting
    token_class: str = TOKEN_CLASS_UNICODE  # which alphanumeric-run definition built the vocabulary

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "PaceMetadata":
        return cls(**json.loads(text))


class StalePaceError(RuntimeError):
    """Raised when the on-disk PACE structure no longer matches the corpus
    it was built from."""


def _build_suffix_array(vocab_bytes: bytes) -> array:
    """Generalized suffix array over `vocab_bytes` (0x00-terminated terms)
    via libsais. No pure-Python fallback: an O(n log n) Python suffix sort
    over a ~300MB vocabulary would make building PACE data impractical."""
    if _LIBSAIS_GSA is None:
        raise RuntimeError(
            "libsais native extension is required to build PACE data. "
            "Run inference/pace/_native/build.sh first."
        )
    n = len(vocab_bytes)
    if n == 0:
        return array("i")
    if n >= 2**31:
        raise RuntimeError(f"vocabulary too large for libsais_gsa's int32 API: {n} bytes")
    t_buf = (ctypes.c_uint8 * n).from_buffer_copy(vocab_bytes)
    sa_buf = (ctypes.c_int32 * n)()
    rc = _LIBSAIS_GSA(t_buf, sa_buf, n, 0, None)
    if rc != 0:
        raise RuntimeError(f"libsais_gsa failed with code {rc}")
    result = array("i")
    result.frombytes(bytes(sa_buf))
    return result


def build_pace(
    corpus_path: str | Path,
    pace_dir: str | Path,
    *,
    min_term_len: int = 2,
    progress: bool = True,
) -> PaceMetadata:
    """Build PACE's auxiliary structure over `corpus_path` into `pace_dir`:
    a singletons-first vocabulary, Elias-Fano posting lists, a pruned
    vocabulary suffix array, a separator-rank bitvector, and blocked line
    offsets. One-time, offline, single-machine: streams the corpus once in
    binary mode and accumulates postings in memory before flushing to disk.

    Terms shorter than `min_term_len` are never recorded; query words
    shorter than that are never used for narrowing either (see the module
    docstring), which keeps the two consistent.
    """
    corpus_path = Path(corpus_path).resolve()
    pace_dir = Path(pace_dir)
    pace_dir.mkdir(parents=True, exist_ok=True)

    corpus_stat = corpus_path.stat()
    corpus_size = corpus_stat.st_size
    corpus_mtime = corpus_stat.st_mtime

    postings: dict[str, array] = {}
    offsets = array("Q")

    pbar = None
    if progress:
        from tqdm import tqdm

        pbar = tqdm(total=corpus_size, unit="B", unit_scale=True, desc="building PACE structure")

    num_lines = 0
    try:
        with open(corpus_path, "rb") as f:
            for offset, raw_line in _iter_lines_with_offsets(f):
                offsets.append(offset)
                text = raw_line.decode("utf-8", errors="replace")
                for term in _tokenize(text):
                    if len(term) < min_term_len:
                        continue
                    bucket = postings.get(term)
                    if bucket is None:
                        bucket = array("I")
                        postings[term] = bucket
                    bucket.append(num_lines)
                num_lines += 1
                if pbar is not None:
                    pbar.update(len(raw_line))
        offsets.append(corpus_size)  # sentinel: line i spans [offsets[i], offsets[i+1])
    finally:
        if pbar is not None:
            pbar.close()

    terms_all = list(postings.keys())
    counts_all = np.fromiter((len(postings[t]) for t in terms_all), dtype=np.int64, count=len(terms_all))
    offsets_np = np.frombuffer(offsets.tobytes(), dtype=np.uint64)

    def fetch(term_ids: np.ndarray) -> np.ndarray:
        return np.concatenate([np.frombuffer(postings[terms_all[int(i)]].tobytes(), dtype=np.uint32) for i in term_ids]) if len(term_ids) else np.zeros(0, np.uint32)

    _write_pace_files(pace_dir, terms_all, counts_all, fetch, offsets_np, num_lines)
    metadata = PaceMetadata(
        corpus_path=str(corpus_path), corpus_size=corpus_size, corpus_mtime=corpus_mtime, num_lines=num_lines,
        num_terms=len(terms_all), min_term_len=min_term_len, built_at=time.time(), token_fold=TOKEN_FOLD_CASEFOLD,
        format_version=FORMAT_VERSION, n_singletons=int((counts_all == 1).sum()), token_class=TOKEN_CLASS_UNICODE,
    )
    (pace_dir / META_FILENAME).write_text(metadata.to_json(), encoding="utf-8")
    return metadata


def _write_separator_rank(vocab_bytes: bytes, pace_dir: Path) -> None:
    """Bitvector of 0x00 separators in vocab.bin (uint64 words, little
    endian bit order) plus cumulative popcount per word: term id of a
    suffix-array position = number of separators before it."""
    bits = np.frombuffer(vocab_bytes, dtype=np.uint8) == 0
    n_words = (len(bits) + 63) // 64
    padded = np.zeros(n_words * 64, dtype=np.uint8)
    padded[: len(bits)] = bits
    words = np.packbits(padded, bitorder="little").view("<u8")
    pop = np.bitwise_count(words).astype(np.int64)
    cum = np.concatenate(([0], np.cumsum(pop)[:-1])).astype(np.uint32)
    words.astype("<u8").tofile(pace_dir / VOCAB_SEP_WORDS_FILENAME)
    cum.tofile(pace_dir / VOCAB_SEP_CUM_FILENAME)


def _write_pace_files(pace_dir: Path, terms: list[str], counts: np.ndarray, fetch, offsets: np.ndarray, num_lines: int) -> None:
    """Write PACE's on-disk files. `terms[i]`/`counts[i]` describe term i in
    the caller's order; `fetch(term_ids)` returns the concatenated sorted
    postings of those terms (in that order); `offsets` has num_lines + 1
    uint64 line offsets."""
    if _EF_ENCODE_LISTS is None or _EF_ENCODED_SIZE is None:
        raise RuntimeError("native extension required to build PACE data (run inference/pace/_native/build.sh)")
    pace_dir = Path(pace_dir)
    counts = np.asarray(counts, dtype=np.int64)
    single_ids = np.flatnonzero(counts == 1)
    multi_ids = np.flatnonzero(counts != 1)
    order = np.concatenate((single_ids, multi_ids))  # new id -> old id
    # vocabulary in the new order
    with open(pace_dir / VOCAB_FILENAME, "wb") as f:
        for i in order.tolist():
            f.write(terms[i].encode("utf-8") + b"\x00")
    vocab_bytes = (pace_dir / VOCAB_FILENAME).read_bytes()
    if len(vocab_bytes) >= 2**31:
        raise RuntimeError("vocabulary too large for libsais int32")
    # singleton postings: one uint32 per singleton term, in new-id order
    with open(pace_dir / SINGLES_FILENAME, "wb") as f:
        for i in range(0, len(single_ids), 4 << 20):
            f.write(np.asarray(fetch(single_ids[i : i + (4 << 20)]), dtype="<u4").tobytes())
    # multi-posting lists: Elias-Fano, chunked
    multi_counts = counts[multi_ids]
    np.asarray(multi_counts, dtype="<u4").tofile(pace_dir / MULTI_COUNT_FILENAME)
    byte_offsets = np.empty(len(multi_ids), dtype=np.uint64)
    with open(pace_dir / POSTINGS_EF_FILENAME, "wb") as f:
        write_offset = 0
        i = 0
        n_multi = len(multi_ids)
        while i < n_multi:
            j = i
            acc = 0
            while j < n_multi and (acc + int(multi_counts[j]) <= (32 << 20) or j == i):
                acc += int(multi_counts[j])
                j += 1
            vals = np.ascontiguousarray(fetch(multi_ids[i:j]), dtype=np.uint32)
            cnts = np.ascontiguousarray(multi_counts[i:j], dtype=np.uint32)
            el_offsets = np.ascontiguousarray(np.cumsum(cnts, dtype=np.int64) - cnts, dtype=np.int64)
            size = sum(_EF_ENCODED_SIZE(int(c), num_lines) for c in cnts.tolist())
            out = np.empty(size, dtype=np.uint8)
            offs = np.empty(len(cnts), dtype=np.int64)
            n = _EF_ENCODE_LISTS(vals.ctypes.data, el_offsets.ctypes.data, cnts.ctypes.data, len(cnts), num_lines, out.ctypes.data, offs.ctypes.data)
            f.write(out[:n].tobytes())
            byte_offsets[i:j] = (offs + write_offset).astype(np.uint64)
            write_offset += n
            i = j
    byte_offsets.tofile(pace_dir / MULTI_EF_BYTEOFFSET_FILENAME)
    # suffix array, pruned to suffixes that can match a >= 2-char query
    sa = np.frombuffer(_build_suffix_array(vocab_bytes).tobytes(), dtype=np.int32)
    v = np.frombuffer(vocab_bytes, dtype=np.uint8)
    keep = (v[sa] != 0) & (v[np.minimum(sa + 1, len(v) - 1)] != 0)
    sa[keep].astype("<i4").tofile(pace_dir / VOCAB_SA_FILENAME)
    _write_separator_rank(vocab_bytes, pace_dir)
    # line offsets: uint64 base per block + uint32 deltas
    offsets = np.asarray(offsets, dtype=np.uint64)
    base = offsets[::OFFSETS_BLOCK]
    delta = offsets - np.repeat(base, OFFSETS_BLOCK)[: len(offsets)]
    if len(delta) and int(delta.max()) >= 2**32:
        raise RuntimeError("line offset delta exceeds uint32; lower OFFSETS_BLOCK")
    base.astype("<u8").tofile(pace_dir / OFFSETS_BASE_FILENAME)
    delta.astype("<u4").tofile(pace_dir / OFFSETS_DELTA_FILENAME)


# ---------------------------------------------------------------------------
# Budget: how much narrowing work is worth doing before a full scan is cheaper
# ---------------------------------------------------------------------------


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


@dataclass(frozen=True, slots=True)
class PaceBudget:
    """Cost caps for narrowing one query against a loaded PACE structure.
    Defaults scale with the corpus so the same code serves the 865K-line
    BrowseComp Plus corpus and the 21M-line wiki18 corpus; every field can
    be pinned via the environment variable named in its comment (used for
    calibration replays, never required).

    A word is *skipped* (dropped from the intersection, which can only widen
    the candidate set) when resolving it is too expensive; the whole call is
    *not served* only when no word is cheap enough to narrow with, or the
    final candidate set would cost more to materialize than a full scan.
    """

    # Suffix-array range above which a query word is skipped without mapping
    # its matches to vocabulary entries (a substring found in that many
    # vocabulary suffixes is never the cheapest word of a pattern, and the
    # mapping itself costs ~0.2 s per million). GREPSEEK_PACE_MAX_SA_MATCHES
    max_sa_matches: int
    # Estimated posting count (sum over matched vocabulary entries) above
    # which a word is skipped. GREPSEEK_PACE_MAX_TERM_POSTINGS
    max_term_postings: int
    # Stop intersecting further words once this few candidates remain.
    # GREPSEEK_PACE_TARGET_CANDIDATES
    target_candidates: int
    # Do not serve when the candidate lines total more bytes than this.
    # Default corpus/4: the served path streams at ~1 ns/byte (memcpy +
    # pipe + rg) while a warm sequential scan runs at ~6 GB/s, so a quarter
    # of the corpus costs at most ~1.5x a full scan; measured on wiki18
    # (922 MB -> 1.1 s vs 2.3 s) and BrowseComp Plus (437 MB -> 0.43 s vs
    # 0.5 s). GREPSEEK_PACE_MAX_MATERIALIZE_BYTES
    max_materialize_bytes: int
    # Do not serve when the candidate set has more lines than this.
    # GREPSEEK_PACE_MAX_CANDIDATE_LINES
    max_candidate_lines: int
    # Cost model used to decide whether intersecting one more word pays for
    # itself: gathering a word's postings costs ~gather_ns_per_posting per
    # posting; materializing + scanning the current candidates costs
    # ~line_ns_per_candidate per line plus ~byte_ns_per_candidate_byte per
    # byte. GREPSEEK_PACE_{GATHER_NS,LINE_NS,BYTE_NS}
    gather_ns_per_posting: int = 40
    line_ns_per_candidate: int = 300
    byte_ns_per_candidate_byte: int = 1
    # Fixed cost of materializing one word as a per-line mask (zeroing +
    # scatter over num_lines bytes). GREPSEEK_PACE_MASK_NS_PER_LINE
    mask_ns_per_line: int = 1

    @classmethod
    def for_corpus(cls, num_lines: int, corpus_size: int) -> "PaceBudget":
        return cls(
            max_sa_matches=_env_int("GREPSEEK_PACE_MAX_SA_MATCHES") or 500_000,
            max_term_postings=_env_int("GREPSEEK_PACE_MAX_TERM_POSTINGS") or max(250_000, num_lines // 8),
            target_candidates=_env_int("GREPSEEK_PACE_TARGET_CANDIDATES") or 256,
            max_materialize_bytes=_env_int("GREPSEEK_PACE_MAX_MATERIALIZE_BYTES") or max(64 << 20, corpus_size // 4),
            max_candidate_lines=_env_int("GREPSEEK_PACE_MAX_CANDIDATE_LINES") or max(250_000, num_lines // 4),
            gather_ns_per_posting=_env_int("GREPSEEK_PACE_GATHER_NS") or 40,
            line_ns_per_candidate=_env_int("GREPSEEK_PACE_LINE_NS") or 300,
            byte_ns_per_candidate_byte=_env_int("GREPSEEK_PACE_BYTE_NS") or 1,
            mask_ns_per_line=_env_int("GREPSEEK_PACE_MASK_NS_PER_LINE") or 1,
        )


class PaceStructure:
    """Read-only handle on PACE's on-disk structure.

    Every on-disk array is memory-mapped through numpy (read-only), so many
    processes opening the same pace_dir share pages via the OS page cache,
    and lookups never build Python-level containers of line numbers: posting
    unions and intersections are numpy operations on uint32 arrays.

    This build supports exactly one on-disk layout (see the module
    docstring); `__init__` raises if the loaded metadata does not match it.
    """

    def __init__(self, pace_dir: str | Path) -> None:
        self.pace_dir = Path(pace_dir)
        self.metadata = PaceMetadata.from_json((self.pace_dir / META_FILENAME).read_text(encoding="utf-8"))
        if not (
            self.metadata.format_version == FORMAT_VERSION
            and self.metadata.token_class == TOKEN_CLASS_UNICODE
            and self.metadata.token_fold == TOKEN_FOLD_CASEFOLD
        ):
            raise RuntimeError(
                "this build only supports format_version=4, token_class="
                f"{TOKEN_CLASS_UNICODE!r}, token_fold={TOKEN_FOLD_CASEFOLD!r}; "
                f"the on-disk structure at {self.pace_dir} does not match "
                f"(got format_version={self.metadata.format_version!r}, "
                f"token_class={self.metadata.token_class!r}, "
                f"token_fold={self.metadata.token_fold!r})"
            )
        self.budget = PaceBudget.for_corpus(self.metadata.num_lines, self.metadata.corpus_size)

        if _EF_DECODE_LISTS is None or _RANK_POSITIONS is None:
            raise RuntimeError("this build requires the native extension (run inference/pace/_native/build.sh)")
        self._vocab = self._memmap(VOCAB_FILENAME, np.uint8)
        self._sa = self._memmap(VOCAB_SA_FILENAME, "<i4")
        self._n_singletons = self.metadata.n_singletons
        self._off_base = self._memmap(OFFSETS_BASE_FILENAME, "<u8")
        self._off_delta = self._memmap(OFFSETS_DELTA_FILENAME, "<u4")
        self._singles = self._memmap(SINGLES_FILENAME, "<u4")
        self._multi_count = self._memmap(MULTI_COUNT_FILENAME, "<u4")
        self._multi_byteoffset = self._memmap(MULTI_EF_BYTEOFFSET_FILENAME, "<u8")
        self._postings_ef = self._memmap(POSTINGS_EF_FILENAME, np.uint8)
        self._sep_words = self._memmap(VOCAB_SEP_WORDS_FILENAME, "<u8")
        self._sep_cum = self._memmap(VOCAB_SEP_CUM_FILENAME, "<u4")

        # ctypes views for the native substring search. `data_as` only takes
        # the array's address, so a read-only memmap is fine and there is no
        # PEP 3118 export to release on close().
        self._vocab_ptr = self._vocab.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if len(self._vocab) else None
        self._sa_ptr = self._sa.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if len(self._sa) else None
        # Lazily created read-only mapping of the corpus for memcpy-based
        # materialization (see _corpus_mapping); False = mapping failed.
        self._corpus_mmap: "tuple[mmap.mmap, np.ndarray] | bool | None" = None
        # Guards the lazy corpus mapping: tool calls run in parallel threads,
        # and a mapping replaced by a racing thread would be unmapped while
        # the first thread still copies from its address.
        self._corpus_mmap_lock = threading.Lock()

    def _memmap(self, name: str, dtype) -> np.ndarray:
        path = self.pace_dir / name
        if path.stat().st_size == 0:
            return np.zeros(0, dtype=dtype)  # np.memmap refuses zero-length files
        return np.memmap(path, dtype=dtype, mode="r")

    @classmethod
    def open(
        cls,
        pace_dir: str | Path,
        *,
        expected_corpus_path: str | Path | None = None,
        verify_fresh: bool = True,
    ) -> "PaceStructure":
        pace = cls(pace_dir)
        if verify_fresh:
            pace.check_fresh(expected_corpus_path)
        return pace

    def check_fresh(self, expected_corpus_path: str | Path | None = None) -> None:
        """Raise StalePaceError if the corpus this structure was built from
        no longer matches its recorded size/mtime, or, when
        `expected_corpus_path` is given, does not resolve to the recorded
        path."""
        meta = self.metadata
        recorded_path = Path(meta.corpus_path)

        if expected_corpus_path is not None:
            resolved_expected = Path(expected_corpus_path).resolve()
            if resolved_expected != recorded_path:
                raise StalePaceError(
                    f"the PACE structure at {self.pace_dir} was built from {meta.corpus_path!r}, "
                    f"not {str(resolved_expected)!r}"
                )

        if not recorded_path.exists():
            raise StalePaceError(
                f"the PACE structure at {self.pace_dir} was built from {meta.corpus_path!r}, which no longer exists"
            )
        st = recorded_path.stat()
        if st.st_size != meta.corpus_size or st.st_mtime != meta.corpus_mtime:
            raise StalePaceError(
                f"the PACE structure at {self.pace_dir} was built from {meta.corpus_path!r} "
                f"(size={meta.corpus_size}, mtime={meta.corpus_mtime}) but it is now "
                f"(size={st.st_size}, mtime={st.st_mtime}); rebuild it."
            )

    @property
    def num_lines(self) -> int:
        return self.metadata.num_lines

    @property
    def native_available(self) -> bool:
        return _VOCAB_SEARCH_RANGE is not None and self._vocab_ptr is not None and self._sa_ptr is not None

    # -- vocabulary / postings -------------------------------------------------

    def _line_offsets(self, idx: np.ndarray) -> np.ndarray:
        """Byte offsets of the given line numbers (0..num_lines inclusive; the
        last is the sentinel end-of-file offset), as int64."""
        idx = np.asarray(idx, dtype=np.int64)
        return np.asarray(self._off_base[idx // OFFSETS_BLOCK], dtype=np.int64) + np.asarray(self._off_delta[idx], dtype=np.int64)

    def line_span(self, line_no: int) -> tuple[int, int]:
        """[start, end) byte span of `line_no` in the corpus file (end is
        exclusive and includes the trailing newline, if any)."""
        if not 0 <= line_no < self.metadata.num_lines:
            raise IndexError(line_no)
        se = self._line_offsets(np.array([line_no, line_no + 1]))
        return int(se[0]), int(se[1])

    def sa_range(self, word: str) -> Optional[tuple[int, int]]:
        """[lo, hi) range of suffix-array positions whose suffix starts with
        the (folded) `word`; None if the native extension is missing or the
        word is empty."""
        if not self.native_available:
            return None
        word_bytes = _fold(word).encode("utf-8")
        if not word_bytes:
            return None
        lo = ctypes.c_int64()
        hi = ctypes.c_int64()
        rc = _VOCAB_SEARCH_RANGE(
            self._vocab_ptr,
            len(self._vocab),
            self._sa_ptr,
            len(self._sa),
            word_bytes,
            len(word_bytes),
            ctypes.byref(lo),
            ctypes.byref(hi),
        )
        if rc != 0:
            return None
        return lo.value, hi.value

    def term_ids(self, word: str, *, max_sa_matches: Optional[int] = None) -> Optional[np.ndarray]:
        """Sorted, deduplicated ids of every vocabulary term containing
        `word` as a substring. Empty array = provably nowhere. None = cannot
        resolve (no native extension, or more suffix-array matches than
        `max_sa_matches`)."""
        rng = self.sa_range(word)
        if rng is None:
            return None
        lo, hi = rng
        if hi - lo > (max_sa_matches if max_sa_matches is not None else self.budget.max_sa_matches):
            return None
        if hi == lo:
            return np.zeros(0, dtype=np.int64)
        sa_positions = np.ascontiguousarray(self._sa[lo:hi], dtype=np.int64)
        # term id = number of 0x00 separators before the suffix position;
        # dedupe as int32 (term ids < 2^31): the sort is the cost here.
        out = np.empty(len(sa_positions), dtype=np.int64)
        _RANK_POSITIONS(self._sep_words.ctypes.data, self._sep_cum.ctypes.data, sa_positions.ctypes.data, len(sa_positions), out.ctypes.data)
        return np.unique(out.astype(np.int32)).astype(np.int64)

    def estimate_postings(self, term_ids: np.ndarray) -> int:
        """Upper bound on the size of the union of these terms' postings
        (the sum of their counts), read from the count array only."""
        if len(term_ids) == 0:
            return 0
        return int(self._counts_for(term_ids).sum())

    def _counts_for(self, term_ids: np.ndarray) -> np.ndarray:
        term_ids = np.asarray(term_ids, dtype=np.int64)
        out = np.ones(len(term_ids), dtype=np.int64)
        multi = term_ids >= self._n_singletons
        if multi.any():
            out[multi] = np.asarray(self._multi_count[term_ids[multi] - self._n_singletons], dtype=np.int64)
        return out

    def _gather_raw(self, term_ids: np.ndarray) -> tuple[np.ndarray, bool]:
        """Concatenated postings of these vocabulary terms (uint32, possibly
        with duplicates across terms) and whether the result is already
        sorted+unique (true for a single term)."""
        n = len(term_ids)
        if n == 0:
            return np.zeros(0, dtype=np.uint32), True
        term_ids = np.asarray(term_ids, dtype=np.int64)
        single = term_ids[term_ids < self._n_singletons]
        multi = np.ascontiguousarray(term_ids[term_ids >= self._n_singletons] - self._n_singletons, dtype=np.int64)
        parts = []
        if len(single):
            parts.append(np.asarray(self._singles[single], dtype=np.uint32))
        if len(multi):
            total = int(np.asarray(self._multi_count[multi], dtype=np.int64).sum())
            out = np.empty(total, dtype=np.uint32)
            written = _EF_DECODE_LISTS(self._postings_ef.ctypes.data, self._multi_byteoffset.ctypes.data,
                                       self._multi_count.ctypes.data, multi.ctypes.data, len(multi), self.metadata.num_lines, out.ctypes.data)
            if written != total:
                raise RuntimeError("corrupt Elias-Fano posting list")
            parts.append(out)
        vals = parts[0] if len(parts) == 1 else np.concatenate(parts)
        return vals, n == 1

    def gather_postings(self, term_ids: np.ndarray) -> np.ndarray:
        """Sorted, deduplicated union of the postings of these vocabulary
        terms, as a uint32 array of line numbers."""
        vals, clean = self._gather_raw(term_ids)
        return vals if clean else np.unique(vals)

    # Candidate-set representations used inside plan(): ("lines", sorted
    # unique uint32 array) for small sets, ("mask", uint8 array with one
    # byte per corpus line) for large ones. Building a mask is a scatter
    # (no sort) and intersecting two masks is a byte-wise AND, so a word
    # with millions of postings costs tens of milliseconds instead of the
    # ~0.3 s a sort-based union took.
    def _mask_threshold(self) -> int:
        return max(4096, self.metadata.num_lines // 128)

    def _word_repr(self, term_ids: np.ndarray, est: int) -> tuple[str, np.ndarray]:
        vals, clean = self._gather_raw(term_ids)
        if est > self._mask_threshold():
            mask = np.zeros(self.metadata.num_lines, dtype=np.uint8)
            mask[vals] = 1
            return "mask", mask
        return "lines", (vals if clean else np.unique(vals))

    def _repr_intersect(self, cur: tuple[str, np.ndarray], nxt: tuple[str, np.ndarray]) -> tuple[str, np.ndarray]:
        ck, ca = cur
        nk, na = nxt
        if ck == "mask" and nk == "mask":
            return "mask", ca & na
        if ck == "mask":
            return "lines", na[ca[na] != 0]
        if nk == "mask":
            return "lines", ca[na[ca] != 0]
        return "lines", self._intersect_sorted(ca, na)

    @staticmethod
    def _repr_count(rep: tuple[str, np.ndarray]) -> int:
        kind, arr = rep
        return int(np.count_nonzero(arr)) if kind == "mask" else int(len(arr))

    @staticmethod
    def _repr_lines(rep: tuple[str, np.ndarray]) -> np.ndarray:
        kind, arr = rep
        if kind != "mask":
            return np.asarray(arr, dtype=np.uint32)
        # Scan 8 lines at a time: most 64-bit words of the mask are zero, so
        # finding the non-zero words first is ~8x less work than a byte scan.
        n = len(arr)
        pad = (-n) % 8
        words = (np.concatenate((arr, np.zeros(pad, dtype=np.uint8))) if pad else arr).view(np.uint64)
        nzw = np.flatnonzero(words)
        if len(nzw) == 0:
            return np.zeros(0, dtype=np.uint32)
        cand = (nzw[:, None] * 8 + np.arange(8, dtype=np.int64)[None, :]).ravel()
        cand = cand[cand < n]
        return cand[arr[cand] != 0].astype(np.uint32)

    def lookup(self, word: str) -> Optional[np.ndarray]:
        """Sorted line numbers where `word` occurs as a substring of any
        recorded token. Empty = provably nowhere. None = cannot resolve
        within budget (callers treat as "no narrowing from this word")."""
        ids = self.term_ids(word)
        if ids is None:
            return None
        if self.estimate_postings(ids) > self.budget.max_term_postings:
            return None
        return self.gather_postings(ids)

    @staticmethod
    def _intersect_sorted(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Intersection of two sorted unique uint32 arrays."""
        if len(a) == 0 or len(b) == 0:
            return np.zeros(0, dtype=np.uint32)
        if len(a) > len(b):
            a, b = b, a
        if len(a) * 16 < len(b):
            pos = np.searchsorted(b, a)
            pos[pos >= len(b)] = len(b) - 1
            return a[b[pos] == a]
        return np.intersect1d(a, b, assume_unique=True)

    def query_words(self, patterns: Sequence[str]) -> list[str]:
        """Distinct words of `patterns` eligible for narrowing, in
        first-seen order."""
        seen: dict[str, None] = {}
        for pattern in patterns:
            for w in _WORD_RE.findall(_fold(pattern)):
                if len(w) >= self.metadata.min_term_len:
                    seen.setdefault(w, None)
        return list(seen)

    def candidate_lines(self, patterns: str | Sequence[str]) -> Optional[np.ndarray]:
        """Sorted uint32 line numbers that can contain every literal in
        `patterns` (a superset of the true matches), or None when no word is
        cheap enough to narrow with. Convenience wrapper over `plan()`."""
        plan = self.plan([patterns] if isinstance(patterns, str) else list(patterns))
        return plan.lines if plan.reason in ("served", "no_match") else None

    def plan(self, patterns: Sequence[str]) -> "PaceDecision":
        """Resolve the words of `patterns` cheapest-first and intersect their
        postings until the candidate set is small (see PaceBudget)."""
        t0 = time.perf_counter()
        words = self.query_words(patterns)
        decision = PaceDecision(served=False, reason="", patterns=tuple(patterns))
        if not words:
            decision.reason = "no_query_words"
            decision.plan_time_s = time.perf_counter() - t0
            return decision
        if not self.native_available:
            decision.reason = "native_unavailable"
            decision.plan_time_s = time.perf_counter() - t0
            return decision

        budget = self.budget
        priced: list[tuple[int, str, np.ndarray]] = []
        skipped: list[str] = []
        for w in words:
            ids = self.term_ids(w)
            if ids is None:
                skipped.append(w)
                continue
            if len(ids) == 0:
                # This word occurs in no token anywhere: exact empty answer.
                decision.reason = "no_match"
                decision.served = True
                decision.words_used = (w,)
                decision.lines = np.zeros(0, dtype=np.uint32)
                decision.plan_time_s = time.perf_counter() - t0
                return decision
            priced.append((self.estimate_postings(ids), w, ids))
        decision.words_skipped = tuple(skipped)
        if not priced:
            decision.reason = "all_words_too_common"
            decision.plan_time_s = time.perf_counter() - t0
            return decision

        priced.sort(key=lambda t: t[0])
        est0, w0, ids0 = priced[0]
        decision.cheapest_word_estimate = est0
        if est0 > budget.max_term_postings:
            decision.reason = "cheapest_word_too_common"
            decision.plan_time_s = time.perf_counter() - t0
            return decision

        cur = self._word_repr(ids0, est0)
        n_cur = self._repr_count(cur)
        used = [w0]
        avg_line_bytes = self.metadata.corpus_size / max(1, self.metadata.num_lines)
        mask_threshold = self._mask_threshold()
        for est, w, ids in priced[1:]:
            if n_cur <= budget.target_candidates:
                break
            if est > budget.max_term_postings:
                break
            # Intersect only while gathering the next word is cheaper than
            # what it could save downstream (materializing + scanning the
            # candidates we already have, estimated from the average line).
            gather_cost = est * budget.gather_ns_per_posting
            if est > mask_threshold:
                gather_cost += self.metadata.num_lines * budget.mask_ns_per_line
            current_cost = n_cur * (budget.line_ns_per_candidate + avg_line_bytes * budget.byte_ns_per_candidate_byte)
            if gather_cost > current_cost:
                break
            cur = self._repr_intersect(cur, self._word_repr(ids, est))
            n_cur = self._repr_count(cur)
            used.append(w)
            if n_cur == 0:
                break
        lines = self._repr_lines(cur)
        decision.words_used = tuple(used)
        decision.lines = lines
        decision.n_candidates = int(len(lines))
        if len(lines) == 0:
            decision.served = True
            decision.reason = "no_match"
            decision.plan_time_s = time.perf_counter() - t0
            return decision
        if len(lines) > budget.max_candidate_lines:
            decision.reason = "too_many_candidate_lines"
            decision.plan_time_s = time.perf_counter() - t0
            return decision
        total = self.candidate_bytes(lines)
        decision.total_bytes = total
        if total > budget.max_materialize_bytes:
            decision.reason = "too_many_candidate_bytes"
            decision.plan_time_s = time.perf_counter() - t0
            return decision
        decision.served = True
        decision.reason = "served"
        decision.plan_time_s = time.perf_counter() - t0
        return decision

    # -- materialization ---------------------------------------------------------

    def candidate_bytes(self, lines: np.ndarray) -> int:
        if len(lines) == 0:
            return 0
        idx = np.asarray(lines, dtype=np.int64)
        starts = self._line_offsets(idx)
        ends = self._line_offsets(idx + 1)
        return int((ends - starts).sum())

    def _coalesced_runs(self, lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(starts, lengths) of maximal contiguous byte runs covering the
        given (sorted, unique) lines. Adjacent candidate lines collapse into
        one pread."""
        idx = np.asarray(lines, dtype=np.int64)
        starts = self._line_offsets(idx)
        ends = self._line_offsets(idx + 1)
        if len(idx) <= 1:
            return starts, ends - starts
        brk = np.flatnonzero(starts[1:] != ends[:-1]) + 1
        run_first = np.concatenate(([0], brk))
        run_last = np.concatenate((brk - 1, [len(idx) - 1]))
        run_starts = starts[run_first]
        run_lengths = ends[run_last] - run_starts
        return run_starts, run_lengths

    def iter_materialized_chunks(
        self,
        lines: np.ndarray,
        *,
        corpus_path: str | Path | None = None,
        chunk_bytes: int = 4 << 20,
    ) -> Iterator[bytes]:
        """Yield the raw bytes of `lines` (ascending file order, so `head`
        semantics are preserved) in chunks of roughly `chunk_bytes`."""
        lines = np.asarray(lines, dtype=np.int64)
        if len(lines) == 0:
            return
        if not np.all(lines[1:] > lines[:-1]):
            lines = np.unique(lines)
        path = str(corpus_path or self.metadata.corpus_path)
        run_starts, run_lengths = self._coalesced_runs(lines)
        cum = np.cumsum(run_lengths)
        n_runs = len(run_starts)
        pos = 0
        while pos < n_runs:
            # last position in run_starts whose cumulative size stays within one chunk
            limit = cum[pos - 1] + chunk_bytes if pos > 0 else chunk_bytes
            end = int(np.searchsorted(cum, limit, side="right"))
            end = max(end, pos + 1)
            yield self._materialize_runs(path, run_starts[pos:end], run_lengths[pos:end])
            pos = end

    def materialize_lines(self, corpus_path: str | Path, line_numbers: Iterable[int]) -> bytes:
        """Concatenated raw bytes of exactly `line_numbers` in ascending
        order (regardless of the order given)."""
        lines = np.unique(np.fromiter(line_numbers, dtype=np.int64))
        return b"".join(self.iter_materialized_chunks(lines, corpus_path=corpus_path))

    def _corpus_mapping(self, path: str) -> Optional[tuple[int, int]]:
        """(base_address, length) of a read-only mmap of the corpus file,
        created once per PaceStructure object and kept for its lifetime;
        None if `path` is not the corpus this structure covers or mapping
        fails."""
        if path != self.metadata.corpus_path:
            return None
        if self._corpus_mmap is None:
            with self._corpus_mmap_lock:
                if self._corpus_mmap is None:
                    try:
                        with open(path, "rb") as f:
                            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    except (OSError, ValueError):
                        self._corpus_mmap = False
                        return None
                    view = np.frombuffer(mm, dtype=np.uint8)
                    self._corpus_mmap = (mm, view)
        if self._corpus_mmap is False:
            return None
        _mm, view = self._corpus_mmap
        return view.ctypes.data, len(view)

    def _materialize_runs(self, path: str, starts: np.ndarray, lengths: np.ndarray) -> bytes:
        starts = np.ascontiguousarray(starts, dtype=np.int64)
        lengths = np.ascontiguousarray(lengths, dtype=np.int64)
        total = int(lengths.sum())
        if total == 0:
            return b""
        mapping = self._corpus_mapping(path) if _NATIVE_COPY_SPANS is not None else None
        if mapping is not None:
            base, base_len = mapping
            buf = ctypes.create_string_buffer(total)
            rc = _NATIVE_COPY_SPANS(
                base,
                base_len,
                starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                len(starts),
                buf,
            )
            if rc == 0:
                return buf.raw
            logger.warning("native copy_spans returned %d; falling back to pread", rc)
        if _NATIVE_MATERIALIZE is not None:
            buf = ctypes.create_string_buffer(total)
            rc = _NATIVE_MATERIALIZE(
                path.encode("utf-8"),
                starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                len(starts),
                buf,
            )
            if rc == 0:
                return buf.raw
            logger.warning("native materialize_spans returned %d; falling back to pure Python", rc)
        chunks: list[bytes] = []
        with open(path, "rb") as f:
            for start, length in zip(starts.tolist(), lengths.tolist()):
                f.seek(start)
                chunks.append(f.read(length))
        return b"".join(chunks)

    def close(self) -> None:
        # Drop every reference; the memmaps close when garbage collected. No
        # explicit mmap.close() because outstanding numpy views would make
        # it raise BufferError, and nothing here needs deterministic unmap.
        self._vocab_ptr = None
        self._sa_ptr = None
        if isinstance(self._corpus_mmap, tuple):
            mm, _view = self._corpus_mmap
            self._corpus_mmap = None
            del _view
            try:
                mm.close()
            except BufferError:
                pass
        for name in ("_vocab", "_sa", "_sep_words", "_sep_cum", "_off_base", "_off_delta",
                     "_singles", "_multi_count", "_multi_byteoffset", "_postings_ef"):
            setattr(self, name, np.zeros(0, dtype=np.uint8))

    def __enter__(self) -> "PaceStructure":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Pipeline classification: which stages may narrow, how stage 0 is rewritten
# ---------------------------------------------------------------------------
#
# Everything below operates on a pipeline already split into per-`|`-stage
# argv lists (`Sequence[Sequence[str]]`); all call sites can produce that
# shape cheaply.

PACE_DIR_ENV_VAR = "GREPSEEK_PACE_DIR"

_PACE_CACHE: dict[str, Optional["PaceStructure"]] = {}
_PACE_CACHE_LOCK = threading.Lock()

# rg/grep flags that do not change *which* lines match, or change only the
# per-line rendering in a way that is identical for a file and for stdin.
_FLAG_FIXED = {"-F", "--fixed-strings"}
_FLAG_IGNORE_CASE = {"-i", "--ignore-case"}
_FLAG_NOOP = {
    "-w", "--word-regexp",
    "-x", "--line-regexp",
    "-s", "--case-sensitive",
    "-S", "--smart-case",
    "-N", "--no-line-number",
    "--no-config", "--mmap", "--no-mmap", "--line-buffered", "--no-messages",
}
# Output-transforming flags: the stage still filters exactly the input lines
# containing the pattern, but its *output* is no longer those lines, so no
# later stage may contribute patterns.
_FLAG_TRANSFORM = {"-c", "--count", "-o", "--only-matching"}
_FLAG_INVERT = {"-v", "--invert-match"}
_FLAG_MAX_COUNT = {"-m", "--max-count"}
_FLAG_PATTERN = {"-e", "--regexp"}
_SHORT_OK = set("iFwxsSNcov")  # chars allowed in a combined short flag group
_REGEX_META = set("\\.^$|?*+()[]{}")


@dataclass(slots=True)
class FilterStage:
    prog: str
    pattern: str
    files: list[str]
    fixed: bool
    invert: bool
    transform: bool
    positional_file_positions: list[int]
    ci: bool = False      # -i / --ignore-case
    plain: bool = True    # only -F / -i / -e / no-op flags: rg's output is the matching lines verbatim


def parse_filter_stage(argv: Sequence[str]) -> Optional[FilterStage]:
    """Interpret `argv` as an rg/grep per-line filter whose match set PACE
    can reason about. Returns None for any shape not on the allowlist
    (unknown flag, several patterns, regex metacharacters without -F, ...).
    """
    if not argv:
        return None
    prog = argv[0].split("/")[-1]
    if prog not in ("rg", "grep"):
        return None
    fixed = invert = transform = ci = False
    plain = True
    patterns: list[str] = []
    positionals: list[tuple[int, str]] = []
    i = 1
    n = len(argv)
    while i < n:
        tok = argv[i]
        if tok == "-" or not tok.startswith("-"):
            positionals.append((i, tok))
            i += 1
            continue
        if tok == "--":
            return None
        if tok in _FLAG_FIXED:
            fixed = True
        elif tok in _FLAG_IGNORE_CASE:
            ci = True
        elif tok in ("--no-config", "--mmap", "--no-mmap", "--line-buffered", "--no-messages"):
            pass
        elif tok in _FLAG_NOOP:
            pass  # -w/-x/-s/-S/-N: match set unchanged but not plain rg -F semantics
            plain = False
        elif tok in _FLAG_TRANSFORM:
            transform = True
            plain = False
        elif tok in _FLAG_INVERT:
            invert = True
            plain = False
        elif tok in _FLAG_MAX_COUNT:
            if i + 1 >= n or not argv[i + 1].isdigit():
                return None
            i += 1
            plain = False
        elif tok.startswith("--max-count="):
            if not tok.split("=", 1)[1].isdigit():
                return None
            plain = False
        elif tok in _FLAG_PATTERN:
            if i + 1 >= n:
                return None
            patterns.append(argv[i + 1])
            i += 1
        elif tok.startswith("--regexp="):
            patterns.append(tok.split("=", 1)[1])
        elif tok.startswith("--"):
            return None
        else:
            # combined short flags, e.g. -iF
            body = tok[1:]
            if not body or any(ch not in _SHORT_OK for ch in body):
                return None
            if "F" in body:
                fixed = True
            if "i" in body:
                ci = True
            if "v" in body:
                invert = True
            if "c" in body or "o" in body:
                transform = True
            if any(ch in "wxsSNcov" for ch in body):
                plain = False
        i += 1

    if not patterns:
        if not positionals:
            return None
        _, pattern = positionals.pop(0)
        patterns.append(pattern)
    if len(patterns) != 1:
        return None
    pattern = patterns[0]
    if not fixed and any(ch in _REGEX_META for ch in pattern):
        return None
    return FilterStage(
        prog=prog,
        pattern=pattern,
        files=[tok for _, tok in positionals],
        fixed=fixed,
        invert=invert,
        transform=transform,
        positional_file_positions=[pos for pos, _ in positionals],
        ci=ci,
        plain=plain,
    )


@dataclass(slots=True)
class NativeShape:
    """A pipeline whose output the in-process verifier reproduces exactly:
    a run of plain `rg`/`grep` `-F` [`-i`] filters with non-empty patterns
    (ASCII-only when `-i`), optionally followed by one `head -n N` or `wc -l`.
    `filters` holds (pattern bytes, case_insensitive) per stage in order;
    `tail` is None, ("head", N) or ("wc",)."""

    filters: tuple[tuple[bytes, bool], ...]
    tail: Optional[tuple]

    @property
    def any_ci(self) -> bool:
        return any(ci for _, ci in self.filters)


@dataclass(slots=True)
class PacePlan:
    """What the classifier concluded about a pipeline: the literals every
    output line must contain (stage 0's plus the positive per-line filters
    that directly follow it) and stage 0's argv with the corpus file
    argument removed, so it reads the narrowed buffer from stdin.
    `native` is set when the whole pipeline can be answered in-process."""

    patterns: tuple[str, ...]
    stage0_argv: tuple[str, ...]
    native: Optional[NativeShape] = None

    @property
    def pattern(self) -> str:
        return self.patterns[0]


_HEAD_N_RE = re.compile(r"^(?:-n\s*(\d+)|-(\d+))$")


def _native_shape(stages: Sequence[Sequence[str]], filters: list[FilterStage]) -> Optional[NativeShape]:
    """Return the NativeShape for `stages` if every filter stage is plain
    `-F`[`-i`] with a non-empty ASCII pattern and the remainder is nothing,
    one `head -n N`, or one `wc -l`."""
    fl: list[tuple[bytes, bool]] = []
    for st in filters:
        if not st.plain or st.invert or st.transform or not st.pattern:
            return None
        if not st.fixed and any(ch in _REGEX_META for ch in st.pattern):
            return None
        # Without -i, rg -F is a byte comparison: any UTF-8 pattern is fine.
        # With -i the folded search below handles ASCII only.
        if st.ci and not st.pattern.isascii():
            return None
        pb = st.pattern.encode("utf-8")
        fl.append((pb.lower() if st.ci else pb, st.ci))
    rest = [list(st) for st in stages[len(filters):]]
    tail: Optional[tuple] = None
    if len(rest) == 1:
        argv = rest[0]
        prog = argv[0].split("/")[-1]
        if prog == "head":
            joined = " ".join(argv[1:])
            m = _HEAD_N_RE.match(joined)
            if not m:
                return None
            tail = ("head", int(m.group(1) or m.group(2)))
        elif prog == "wc" and argv[1:] == ["-l"]:
            tail = ("wc",)
        else:
            return None
    elif len(rest) > 1:
        return None
    return NativeShape(filters=tuple(fl), tail=tail)


def classify_for_pace(
    stages: Sequence[Sequence[str]],
    *,
    cwd: str | Path,
    corpus_path: str | Path,
) -> Optional[PacePlan]:
    """Return a `PacePlan` iff `stages[0]` is an rg/grep literal filter over
    exactly the corpus file, else None.

    Stage 0 must be a positive filter (no `-v`) with exactly one pattern
    and one file argument resolving to `corpus_path`. Later stages are
    walked while they remain rg/grep per-line filters reading stdin: a
    positive one contributes its pattern to the intersection; a negated
    one contributes nothing but is stepped over (still per-line); an
    output-transforming one (`-c`, `-o`) contributes and ends the walk;
    anything else (`head`, `wc`, `sort`, `sed`, an unknown flag, a file
    argument) ends the walk. Stages themselves are never modified except
    for dropping stage 0's file argument.
    """
    if not stages or not stages[0]:
        return None
    head = parse_filter_stage(stages[0])
    if head is None or head.invert or len(head.files) != 1:
        return None
    file_arg = head.files[0]
    if file_arg == "-":
        return None
    resolved_cwd = Path(cwd).resolve()
    candidate = Path(file_arg)
    resolved_file = (candidate if candidate.is_absolute() else resolved_cwd / candidate).resolve()
    if resolved_file != Path(corpus_path).resolve():
        return None

    patterns = [head.pattern]
    filters = [head]
    if not head.transform:
        for stage in stages[1:]:
            st = parse_filter_stage(stage)
            if st is None or st.files:
                break
            filters.append(st)
            if not st.invert:
                patterns.append(st.pattern)
            if st.transform:
                break

    drop = head.positional_file_positions[0]
    stage0_argv = tuple(tok for j, tok in enumerate(stages[0]) if j != drop)
    return PacePlan(patterns=tuple(patterns), stage0_argv=stage0_argv, native=_native_shape(stages, filters))


@dataclass(slots=True)
class PaceDecision:
    """Outcome of asking PACE about one pipeline. `served` is True when
    stage 0 can read `lines` from stdin instead of the corpus file;
    `reason` says why or why not (see `plan()` / `decide_for_pace()`)."""

    served: bool
    reason: str
    patterns: tuple[str, ...] = ()
    stage0_argv: tuple[str, ...] = ()
    words_used: tuple[str, ...] = ()
    words_skipped: tuple[str, ...] = ()
    cheapest_word_estimate: int = 0
    n_candidates: int = 0
    total_bytes: int = 0
    plan_time_s: float = 0.0
    lines: Optional[np.ndarray] = field(default=None, repr=False)
    native: Optional[NativeShape] = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "served": self.served,
            "reason": self.reason,
            "patterns": list(self.patterns),
            "words_used": list(self.words_used),
            "words_skipped": list(self.words_skipped),
            "cheapest_word_estimate": self.cheapest_word_estimate,
            "n_candidates": self.n_candidates,
            "total_bytes": self.total_bytes,
            "plan_time_s": self.plan_time_s,
            "native_shape": self.native is not None,
        }


def decide_for_pace(
    stages: Sequence[Sequence[str]],
    *,
    cwd: str | Path,
    pace: "PaceStructure",
) -> PaceDecision:
    """Classify `stages` against `pace` and, if eligible, plan the candidate
    set. Never raises for a well-formed argv list."""
    plan = classify_for_pace(stages, cwd=cwd, corpus_path=pace.metadata.corpus_path)
    if plan is None:
        return PaceDecision(served=False, reason="not_classified")
    decision = pace.plan(plan.patterns)
    decision.stage0_argv = plan.stage0_argv
    decision.native = plan.native
    return decision


def verify_natively_streaming(chunks: Iterable[bytes], shape: NativeShape) -> Optional[tuple[bytes, int]]:
    """Like verify_natively() but over candidate chunks in file order (each
    chunk holds whole lines), stopping as soon as a `head -n N` tail is
    satisfied -- the in-process equivalent of rg dying on SIGPIPE. Returns
    None on a guard trip in any chunk consumed so far (caller falls back to
    the real pipeline; nothing has been emitted yet)."""
    if _VERIFY_LINES is None or _VERIFY_UNSAFE_BYTES is None:
        return None
    max_matches = 0
    if shape.tail and shape.tail[0] == "head":
        max_matches = shape.tail[1]
        if max_matches == 0:
            return b"", 0
    k = len(shape.filters)
    pats = (ctypes.c_char_p * k)(*[p for p, _ in shape.filters])
    lens = np.array([len(p) for p, _ in shape.filters], dtype=np.int64)
    cis = np.array([1 if ci else 0 for _, ci in shape.filters], dtype=np.int32)
    check_fold = 1 if shape.any_ci else 0
    found = 0
    out_parts: list[bytes] = []
    for buf in chunks:
        n = len(buf)
        if n == 0:
            continue
        if _VERIFY_UNSAFE_BYTES(buf, n, check_fold):
            return None
        remaining = (max_matches - found) if max_matches else 0
        cap = remaining if max_matches else max(1, buf.count(b"\n") + 1)
        starts = np.empty(cap, dtype=np.int64)
        ends = np.empty(cap, dtype=np.int64)
        got = int(_VERIFY_LINES(buf, n, pats, lens.ctypes.data, cis.ctypes.data, k, remaining, starts.ctypes.data, ends.ctypes.data, cap))
        got = min(got, cap)
        if got and not (shape.tail and shape.tail[0] == "wc"):
            out_parts.append(b"".join(buf[a:b] + b"\n" for a, b in zip(starts[:got].tolist(), ends[:got].tolist())))
        found += got
        if max_matches and found >= max_matches:
            break
    if shape.tail and shape.tail[0] == "wc":
        return f"{found}\n".encode(), 0
    out = b"".join(out_parts)
    if shape.tail:
        return out, 0
    return out, (0 if found else 1)


def verify_natively(buf: bytes, shape: NativeShape, n_lines: Optional[int] = None) -> Optional[tuple[bytes, int]]:
    """Run `shape` over the candidate buffer in-process. Returns
    (stdout_bytes, exit_code) exactly as the bash pipeline would produce,
    or None when the buffer contains bytes for which rg's behaviour differs
    from a plain byte comparison (NUL; U+017F / U+212A under -i) -- the
    caller then runs the real pipeline."""
    if _VERIFY_LINES is None or _VERIFY_UNSAFE_BYTES is None:
        return None
    n = len(buf)
    if n and _VERIFY_UNSAFE_BYTES(buf, n, 1 if shape.any_ci else 0):
        return None
    max_matches = 0
    if shape.tail and shape.tail[0] == "head":
        max_matches = shape.tail[1]
        if max_matches == 0:
            return b"", 0
    k = len(shape.filters)
    pats = (ctypes.c_char_p * k)(*[p for p, _ in shape.filters])
    lens = np.array([len(p) for p, _ in shape.filters], dtype=np.int64)
    cis = np.array([1 if ci else 0 for _, ci in shape.filters], dtype=np.int32)
    cap = max_matches if max_matches else max(1, (n_lines + 1) if n_lines is not None else buf.count(b"\n") + 1)
    starts = np.empty(cap, dtype=np.int64)
    ends = np.empty(cap, dtype=np.int64)
    found = _VERIFY_LINES(buf, n, pats, lens.ctypes.data, cis.ctypes.data, k, max_matches,
                          starts.ctypes.data, ends.ctypes.data, cap) if n else 0
    found = int(min(found, cap))
    if shape.tail and shape.tail[0] == "wc":
        return f"{found}\n".encode(), 0
    out = b"".join(buf[s:e] + b"\n" for s, e in zip(starts[:found].tolist(), ends[:found].tolist()))
    if shape.tail:  # head: exit status of head is 0
        return out, 0
    return out, (0 if found else 1)


_SHELL_SPECIAL_CHARS = set("$`~*?[{\\")
_SHELL_UNSAFE_CHARS = set(";&()<>\n\r")


def split_pipeline_stages(cmd: str, *, cwd: str | Path, env: Optional[dict] = None) -> Optional[tuple[list[list[str]], list[str]]]:
    """Split a validated shell pipeline into `(stage_argvs, stage_texts)`.

    Callers that execute the pipeline through `bash -c` must plan on the
    argv bash will actually produce: `rg -F "little over $500" corpus` is
    executed with the pattern `little over 00` (positional parameter 5 is
    empty). `shlex.split` does no expansion, so whenever a stage contains
    a character bash would expand (`$`, backtick, `~`, globs, braces,
    backslash) the stages are expanded by bash itself, in `cwd` with `env`,
    via `printf '%s\\0'`. Plain stages use `shlex.split`, which agrees with
    bash for quote handling. Returns None when a stage cannot be split (or
    contains characters the validators should already have rejected), in
    which case the caller must not serve the command.
    """
    import shlex
    import subprocess

    texts = [seg.strip() for seg in cmd.split("|")]
    if any(not t for t in texts):
        return None
    if not any(ch in _SHELL_SPECIAL_CHARS for ch in cmd):
        try:
            argvs = [shlex.split(t) for t in texts]
        except ValueError:
            return None
        return (argvs, texts) if all(argvs) else None
    if any(ch in _SHELL_UNSAFE_CHARS for ch in cmd):
        return None
    script = " ; ".join(f"printf '%s\\0' {t} ; printf '\\1'" for t in texts)
    try:
        proc = subprocess.run(
            ["/bin/bash", "-c", script], cwd=str(cwd), env=env, capture_output=True, timeout=5.0, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    groups = proc.stdout.split(b"\x01")
    if len(groups) != len(texts) + 1 or groups[-1] != b"":
        return None
    argvs = []
    for g in groups[:-1]:
        if not g.endswith(b"\x00"):
            return None
        argv = [a.decode("utf-8", errors="surrogateescape") for a in g[:-1].split(b"\x00")]
        if not argv or not argv[0]:
            return None
        argvs.append(argv)
    return argvs, texts


def load_pace_from_env() -> Optional["PaceStructure"]:
    """Load (and cache process-wide) the PACE structure configured via
    `GREPSEEK_PACE_DIR`. Returns None -- never raises -- whenever the env
    var is unset, the directory is missing, or the structure no longer
    matches its corpus; callers treat None as "feature off".

    One loaded structure is consulted for every command a process runs;
    `classify_for_pace`'s path comparison against
    `pace.metadata.corpus_path` decides per call whether it applies.
    """
    pace_dir = os.environ.get(PACE_DIR_ENV_VAR, "").strip()
    if not pace_dir:
        return None
    if pace_dir in _PACE_CACHE:
        return _PACE_CACHE[pace_dir]
    with _PACE_CACHE_LOCK:  # parallel tool calls must share one structure
        if pace_dir in _PACE_CACHE:
            return _PACE_CACHE[pace_dir]
        try:
            pace: Optional["PaceStructure"] = PaceStructure.open(pace_dir)
        except (FileNotFoundError, StalePaceError, OSError, ValueError, TypeError):
            logger.warning("PACE structure at %r unavailable; falling back to full-scan search", pace_dir, exc_info=True)
            pace = None
        _PACE_CACHE[pace_dir] = pace
    return pace
