"""Byte-equivalence test for ShardedSearchEngine.

Builds a SMALL synthetic corpus under a tempdir (never touches the real
wiki_corpus.jsonl), shards it, and asserts that for a battery of
representative pipeline shapes the sharded engine returns the same bytes
as the single-file fallback.

Run:
    python -m inference.parallel_search.tests.test_byte_equivalence
or:
    python -m pytest inference/parallel_search/tests/test_byte_equivalence.py
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
import unittest

from inference.parallel_search import ShardedSearchEngine
from inference.parallel_search.sharder import build_shards


def _make_synthetic_corpus(path: str, n_lines: int = 5000) -> None:
    """Write a small JSONL corpus shaped like wiki_corpus.jsonl entries.

    Includes a mix of:
      - 'Einstein' in some entries
      - 'Curie' in some entries
      - both in a few
      - some duplicate-ish lines (for uniq tests)
      - some entries with neither (so rg/grep exit codes are meaningful)
    """
    import random
    random.seed(42)
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_lines):
            doc_id = f"doc-{i:05d}"
            r = random.random()
            if r < 0.05:
                content = '"Einstein was born in 1879."'
            elif r < 0.08:
                content = '"Marie Curie discovered polonium and radium."'
            elif r < 0.09:
                content = '"Einstein corresponded with Marie Curie."'
            elif r < 0.20:
                # Duplicate content (for sort | uniq tests)
                content = '"a common boilerplate line."'
            else:
                content = f'"Random padding text number {i}."'
            f.write(json.dumps({"id": doc_id, "contents": content}, ensure_ascii=False) + "\n")


def _run_single_file(cmd: str, corpus_dir: str) -> tuple[str, int]:
    """Run the same way inference.tools.run_tool would, for comparison."""
    cmd_to_run = cmd.replace("corpus.jsonl", "wiki_corpus.jsonl")
    proc = subprocess.run(
        cmd_to_run, shell=True, executable="/bin/bash",
        cwd=corpus_dir, capture_output=True, text=True, timeout=30,
    )
    return proc.stdout or "", proc.returncode


class ByteEquivalenceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="parallel_search_test_")
        cls.corpus_dir = os.path.join(cls.tmpdir, "wiki_18_corpus")
        os.makedirs(cls.corpus_dir, exist_ok=True)
        cls.corpus_file = os.path.join(cls.corpus_dir, "wiki_corpus.jsonl")
        _make_synthetic_corpus(cls.corpus_file, n_lines=5000)

        cls.shard_dir = os.path.join(cls.tmpdir, "shards_8")
        build_shards(cls.corpus_file, cls.shard_dir, n=8)

        cls.engine = ShardedSearchEngine(
            shard_dir=cls.shard_dir,
            fallback_corpus_file=cls.corpus_file,
            timeout=30.0,
        )

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    # ----- the cases -----

    def _check(self, cmd: str, expected_strategy: str | None = None):
        sharded, sharded_rc = self.engine.execute(cmd)
        single, single_rc = _run_single_file(cmd, self.corpus_dir)
        # rg returns 1 when no matches; we treat that as success-shape but
        # the single-file path will also return 1 in that case. Just compare.
        self.assertEqual(sharded, single,
                         msg=f"\n--- cmd: {cmd}\n"
                             f"--- expected (single file, rc={single_rc}):\n{single!r}\n"
                             f"--- got (sharded, rc={sharded_rc}, strategy={self.engine.last_stats.strategy}):\n{sharded!r}")
        if expected_strategy is not None:
            self.assertEqual(self.engine.last_stats.strategy, expected_strategy,
                             msg=f"cmd: {cmd}\nexpected strategy: {expected_strategy}\n"
                                 f"got: {self.engine.last_stats.strategy}")

    # CONCAT
    def test_concat_substring(self):
        self._check('rg -F "Einstein" corpus.jsonl', expected_strategy="concat")

    def test_concat_and_narrow(self):
        self._check('rg -F "Einstein" corpus.jsonl | rg -F "Curie"', expected_strategy="concat")

    def test_concat_no_matches(self):
        # No matches: rg exits 1 in single-file; sharded should also be empty.
        self._check('rg -F "ZZZZZZ_no_such_phrase_ZZZZZZ" corpus.jsonl',
                    expected_strategy="concat")

    def test_concat_case_insensitive(self):
        self._check('rg -i -F "einstein" corpus.jsonl', expected_strategy="concat")

    # HEAD
    def test_head_3(self):
        self._check('rg -F "Einstein" corpus.jsonl | head -n 3', expected_strategy="head")

    def test_head_8_with_narrow(self):
        self._check('rg -F "Einstein" corpus.jsonl | rg -F "Curie" | head -n 8',
                    expected_strategy="head")

    def test_head_short_flag(self):
        # `head -3` form
        self._check('rg -F "Einstein" corpus.jsonl | head -3', expected_strategy="head")

    def test_head_more_than_available(self):
        # Asks for 100 lines, fewer exist → no truncation.
        self._check('rg -F "Einstein corresponded" corpus.jsonl | head -n 100',
                    expected_strategy="head")

    # COUNT
    def test_count(self):
        self._check('rg -F "Einstein" corpus.jsonl | wc -l', expected_strategy="count")

    def test_count_zero_matches(self):
        self._check('rg -F "ZZZZZZ" corpus.jsonl | wc -l', expected_strategy="count")

    # SORT_HEAD
    def test_sort_head(self):
        # sort | head — exact ordering must match (lexicographic on line bytes).
        self._check('rg -F "common boilerplate" corpus.jsonl | sort | head -n 5',
                    expected_strategy="sort_head")

    def test_sort_uniq_head(self):
        # Many duplicate "common boilerplate" lines → uniq should collapse to 1.
        self._check('rg -F "common boilerplate" corpus.jsonl | sort | uniq | head -n 3',
                    expected_strategy="sort_head")

    # FALLBACK paths must still byte-match because they run on the single file.
    def test_fallback_tail(self):
        # tail goes to single-file fallback
        self._check('rg -F "Einstein" corpus.jsonl | tail -n 3',
                    expected_strategy="fallback")

    def test_fallback_line_numbers(self):
        self._check('rg -n -F "Einstein corresponded" corpus.jsonl',
                    expected_strategy="fallback")

    def test_fallback_awk(self):
        self._check('rg -F "Einstein" corpus.jsonl | awk "{print NR}" | head -n 3',
                    expected_strategy="fallback")


if __name__ == "__main__":
    unittest.main(verbosity=2)
