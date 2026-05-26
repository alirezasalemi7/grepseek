"""ShardedSearchEngine — public entry point.

Construct once at eval start, then call .execute(command) per tool invocation.
Thread-safe (state is read-only after init).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

from .pipeline import Strategy, classify, parse_pipeline
from .runners import STRATEGY_DISPATCH

logger = logging.getLogger(__name__)

# The model writes the logical name; we expand to the actual filename in
# tools.run_tool before reaching the engine. Either name may appear in the
# command at the engine boundary, so the engine must recognize both.
_KNOWN_CORPUS_TOKENS = ("wiki_corpus.jsonl", "corpus.jsonl")


@dataclass
class EngineStats:
    """Per-call telemetry. Caller can read these via .stats after .execute."""
    strategy: str = ""
    n_shards_used: int = 0
    fallback_reason: str = ""


class ShardedSearchEngine:
    """Fan a parallel-safe pipeline across N shards; fall back to single-file
    execution for anything the classifier doesn't recognize.

    Args:
        shard_dir: directory containing `shard_NN` files. Must also contain
            `manifest.json` written by `sharder.py` describing the shard set.
        fallback_corpus_file: absolute path to the un-sharded corpus file.
            Used for FALLBACK strategy. Must exist (validated at init).
        timeout: per-call timeout in seconds (matches inference.tools default).
        log_path: optional jsonl file to append per-call telemetry to.
    """

    def __init__(
        self,
        *,
        shard_dir: str,
        fallback_corpus_file: str,
        timeout: float = 60.0,
        log_path: Optional[str] = None,
    ):
        if not os.path.isdir(shard_dir):
            raise FileNotFoundError(f"shard_dir does not exist: {shard_dir}")
        if not os.path.isfile(fallback_corpus_file):
            raise FileNotFoundError(f"fallback corpus missing: {fallback_corpus_file}")

        manifest_path = os.path.join(shard_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"manifest.json missing in {shard_dir}. "
                f"Build shards via: python -m inference.parallel_search.sharder "
                f"--src {fallback_corpus_file} --dst {shard_dir} --n <N>"
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        n_shards = int(manifest["n_shards"])
        shard_names = manifest["shards"]
        if len(shard_names) != n_shards:
            raise ValueError(f"manifest n_shards={n_shards} != len(shards)={len(shard_names)}")

        shard_files = [os.path.join(shard_dir, name) for name in shard_names]
        for sf in shard_files:
            if not os.path.isfile(sf):
                raise FileNotFoundError(f"shard listed in manifest is missing: {sf}")

        self._shard_files = shard_files
        self._n_shards = n_shards
        self._fallback_corpus_file = fallback_corpus_file
        self._fallback_dir = os.path.dirname(fallback_corpus_file)
        self._fallback_basename = os.path.basename(fallback_corpus_file)
        self._timeout = timeout
        self._log_path = log_path
        self.last_stats = EngineStats()

    @property
    def n_shards(self) -> int:
        return self._n_shards

    @property
    def shard_files(self) -> list[str]:
        return list(self._shard_files)

    def describe(self) -> str:
        return (f"ShardedSearchEngine(n_shards={self._n_shards}, "
                f"shard_dir={os.path.dirname(self._shard_files[0])!r}, "
                f"fallback={self._fallback_corpus_file!r})")

    # -----------------------------------------------------------------------

    def execute(self, command: str, env: Optional[dict] = None
                ) -> tuple[str, int]:
        """Run `command`. Returns (stdout, returncode).

        The returncode follows the same convention `subprocess.run` would
        produce on the un-sharded path (0 = success; non-zero = error or no
        matches for rg-style pipelines).
        """
        # Identify which corpus token appears in the command. If neither
        # appears (e.g. model issued `ls` without referencing the corpus), we
        # have nothing to shard — fall back.
        corpus_token = None
        for tok in _KNOWN_CORPUS_TOKENS:
            if tok in command:
                corpus_token = tok
                break
        if corpus_token is None:
            return self._run_fallback(command, env, "no corpus token in command")

        # Parse + classify.
        try:
            stages = parse_pipeline(command)
        except ValueError as exc:
            # Parse failure: the validator upstream of us should already have
            # caught this. Defensive fallback so the engine never crashes.
            return self._run_fallback(command, env, f"parse error: {exc}")
        plan = classify(stages)

        if plan.strategy == Strategy.FALLBACK:
            return self._run_fallback(command, env, plan.fallback_reason)

        runner = STRATEGY_DISPATCH[plan.strategy]
        stdout, rc = runner(
            plan,
            shard_files=self._shard_files,
            corpus_token=corpus_token,
            timeout=self._timeout,
            env=env,
        )
        self.last_stats = EngineStats(
            strategy=plan.strategy.value,
            n_shards_used=self._n_shards,
        )
        self._log(command, plan.strategy.value, rc, fallback_reason="")
        return stdout, rc

    # -----------------------------------------------------------------------

    def _run_fallback(self, command: str, env: Optional[dict], reason: str
                      ) -> tuple[str, int]:
        """Execute against the single corpus file (same path as today).

        The command references `corpus.jsonl` or `wiki_corpus.jsonl` by name;
        we cwd into the directory holding the fallback file and substitute
        the logical name → actual basename, matching inference.tools behavior.
        """
        cmd = command.replace("corpus.jsonl", self._fallback_basename)
        try:
            proc = subprocess.run(
                cmd, shell=True, executable="/bin/bash",
                cwd=self._fallback_dir,
                timeout=self._timeout, capture_output=True, text=True, env=env,
            )
            stdout, rc = proc.stdout or "", proc.returncode
        except subprocess.TimeoutExpired:
            stdout, rc = f"[command timed out after {self._timeout}s]", -1

        self.last_stats = EngineStats(
            strategy=Strategy.FALLBACK.value,
            n_shards_used=0,
            fallback_reason=reason,
        )
        self._log(command, Strategy.FALLBACK.value, rc, fallback_reason=reason)
        return stdout, rc

    def _log(self, command: str, strategy: str, rc: int, fallback_reason: str) -> None:
        if not self._log_path:
            return
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "command": command,
                    "strategy": strategy,
                    "n_shards": self._n_shards if strategy != Strategy.FALLBACK.value else 0,
                    "returncode": rc,
                    "fallback_reason": fallback_reason,
                }, ensure_ascii=False) + "\n")
        except OSError:
            # Logging is best-effort; never let it crash the eval.
            pass
