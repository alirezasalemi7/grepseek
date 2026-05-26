"""VERL BaseTool that mirrors the cold-start data-gen `shell` tool exactly.

Implementation parity with `cold_start_sft/utils/tools.py`:
  - same validator (whitelist + dangerous-pattern blocklist)
  - same `corpus.jsonl -> wiki_corpus.jsonl` rewriting before execution
  - same subprocess execution (bash, cwd=corpus_root, capture stdout/stderr)
  - same output truncation behavior

Returned payload matches the SFT trajectory's tool-message JSON shape:
  {command, stdout, stderr, exit_code, timed_out, truncated, information_lines}
so a model trained on cold-start SFT data sees identical tool behavior at RL time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _truncate_tokens(text: str, max_tokens: int, tokenizer) -> tuple[str, bool]:
    """Truncate `text` to at most `max_tokens` tokens. Returns (text, was_truncated).
    Mirrors cold_start_sft/utils/pipeline.py:truncate_tokens exactly. The
    tokenizer is the trained model's tokenizer (injected by the agent loop)."""
    if not text:
        return text, False
    # Cheap short-circuit: Qwen BPE is >=3 chars/token on natural English, so
    # any text under max_tokens*3 chars is guaranteed under the token cap.
    # Saves the tokenizer call on most tool responses (single Wikipedia
    # passage at `| head -n 3` is typically a few KB).
    if len(text) <= max_tokens * 3:
        return text, False
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text, False
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=False), True


# ---------------------------------------------------------------------------
# Validator + executor — verbatim copies of cold_start_sft/utils/tools.py
# ---------------------------------------------------------------------------

ALLOWED_COMMANDS = {
    "rg", "grep", "find", "sed", "awk", "head", "tail", "cat",
    "ls", "wc", "sort", "cut", "uniq", "tr",
}

CORPUS_FILENAME_LOGICAL = "corpus.jsonl"      # what the agent types
CORPUS_FILENAME_ACTUAL = "wiki_corpus.jsonl"  # what's on disk in corpus_root


class ToolError(Exception):
    pass


def _strip_quoted(s: str) -> str:
    s = re.sub(r"'[^']*'", "", s)
    s = re.sub(r'"[^"]*"', "", s)
    return s


_DANGEROUS_PATTERNS = [
    (re.compile(r"\bsed\b[^|]*\s-(?:[a-zA-Z]*i|-in-place)\b"), "sed -i / --in-place (in-place edit)"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\s-i\s+inplace\b"), "awk/gawk -i inplace"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\s--in-place\b"), "awk --in-place"),
    (re.compile(r"\bfind\b[^|]*\s-delete\b"), "find -delete"),
    (re.compile(r"\bfind\b[^|]*\s-exec(?:dir)?\b"), "find -exec / -execdir"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\bsystem\s*\("), "awk system() call"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\bprintf?\b[^|]*>\s*[\"']"), "awk script redirection (print > file)"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\|\s*[\"']"), "awk script pipe-to-shell-command"),
    (re.compile(r"\bsed\b[^|]*'\s*e\s+"), "sed 'e' (execute shell) command"),
    (re.compile(r"\b(?:rm|mv|cp|dd|chmod|chown|chgrp|truncate|install|tee|ln)\b"),
     "destructive/write command name"),
]


def _validate_pipeline(cmd: str) -> None:
    if "\n" in cmd:
        raise ToolError("newlines not allowed in command")
    if len(cmd) > 2000:
        raise ToolError("command too long")
    for pat, label in _DANGEROUS_PATTERNS:
        if pat.search(cmd):
            raise ToolError(f"disallowed: {label}")
    bare = _strip_quoted(cmd)
    for bad in ("`", "$(", "&&", "||", ";", ">", "<"):
        if bad in bare:
            raise ToolError(f"disallowed shell construct: {bad!r}")
    segments = [s.strip() for s in cmd.split("|")]
    for seg in segments:
        if not seg:
            raise ToolError("empty pipeline segment")
        try:
            tokens = shlex.split(seg)
        except ValueError as e:
            raise ToolError(f"could not parse segment: {e}")
        if not tokens:
            raise ToolError("empty command")
        prog = tokens[0].split("/")[-1]
        if prog not in ALLOWED_COMMANDS:
            raise ToolError(f"command '{prog}' not in whitelist")


def _execute_pipeline(
    cmd: str,
    *,
    cwd: str,
    timeout: float,
    max_tokens: int,
    tokenizer,
) -> dict:
    """Run a validated pipeline. Returns the JSON-shaped payload the SFT pipeline
    emits in the trajectory's tool message (see
    cold_start_sft/utils/pipeline.py:_verl_style_tool_message_content).

    Output is truncated to `max_tokens` tokens using the SAME tokenizer the
    trained model uses, matching the SFT pipeline's `tool_max_tokens=2048`
    semantic exactly (chars-based truncation would over-cut some queries and
    under-cut others vs what SFT trajectories saw)."""
    # Rewrite corpus.jsonl -> wiki_corpus.jsonl, exactly like the SFT pipeline.
    cmd_to_run = cmd.replace(CORPUS_FILENAME_LOGICAL, CORPUS_FILENAME_ACTUAL)
    timed_out = False
    try:
        proc = subprocess.run(
            cmd_to_run,
            shell=True,
            executable="/bin/bash",
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        stdout = f"[command timed out after {timeout}s]"
        stderr = ""
        exit_code = -1
        timed_out = True

    stdout, truncated = _truncate_tokens(stdout, max_tokens, tokenizer)
    if truncated:
        stdout += f"\n[... output truncated at {max_tokens} tokens]"

    info_lines = [ln for ln in stdout.split("\n") if ln.strip()]
    return {
        "command": cmd,                # report the AGENT's command, not the rewritten one
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "truncated": truncated,
        "information_lines": info_lines,
    }


# ---------------------------------------------------------------------------
# VERL BaseTool wrapper
# ---------------------------------------------------------------------------


class SearchCorpusTool:
    """VERL tool that runs a single shell pipeline against the local Wikipedia
    corpus, with behavior identical to the cold-start SFT data-gen pipeline."""

    def __init__(self, config: dict[str, Any], tool_schema) -> None:
        # Env var takes precedence over YAML so launchers can redirect to a
        # local corpus copy without editing the tool config file. run_rl.sh sets
        # VERL_TOOL_CORPUS_ROOT from GREPSEEK_CORPUS_ROOT (point it at a local-disk
        # copy for rg speed). Falls back to the YAML corpus_root when unset.
        env_corpus_root = os.environ.get("VERL_TOOL_CORPUS_ROOT", "").strip()
        corpus_root = env_corpus_root or config.get("corpus_root")
        if not corpus_root:
            raise ValueError("SearchCorpusTool requires 'corpus_root' in config "
                             "or VERL_TOOL_CORPUS_ROOT env var")
        self.corpus_root = str(corpus_root)
        # Hard check the file is actually there — fail fast at tool-init rather
        # than producing empty stdout on every rollout.
        _corpus_file = os.path.join(self.corpus_root, "wiki_corpus.jsonl")
        if not os.path.isfile(_corpus_file):
            raise ValueError(
                f"SearchCorpusTool: corpus file not found at {_corpus_file!r}. "
                f"corpus_root={self.corpus_root!r} "
                f"(VERL_TOOL_CORPUS_ROOT={env_corpus_root!r}, "
                f"yaml corpus_root={config.get('corpus_root')!r})"
            )
        self.timeout_seconds: float = float(config.get("timeout_seconds", 180.0))
        self.max_output_tokens: int = int(config.get("max_output_tokens", 2048))

        # Tokenizer is INJECTED by GrepSeekAgentLoop.__init__ after tool
        # construction (see set_tokenizer below) — it uses the same tokenizer
        # the agent loop loaded from actor_rollout_ref.model.path. Zero config
        # duplication: whatever model is being trained, that's the tokenizer
        # used for token-based stdout truncation, identical to what SFT
        # trajectories were generated with.
        self._tokenizer = None

        self.config = config
        self.tool_schema = tool_schema
        self.name: str = tool_schema.function.name if tool_schema else "shell"
        self._instances: dict[str, dict] = {}

    def set_tokenizer(self, tokenizer) -> None:
        """Called once by GrepSeekAgentLoop.__init__ to bind the trained
        model's tokenizer for token-based stdout truncation."""
        self._tokenizer = tokenizer

    def get_openai_tool_schema(self):
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, Any]:
        from verl.tools.schemas import ToolResponse

        iid = instance_id if instance_id is not None else str(uuid4())
        self._instances[iid] = {}
        return iid, ToolResponse()

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs: Any,
    ) -> tuple[Any, float, dict]:
        from verl.tools.schemas import ToolResponse

        command = (parameters.get("command") or "").strip()
        if not command:
            return ToolResponse(text="Error: empty command"), 0.0, {}

        # Validate (synchronous — pure string checks).
        try:
            _validate_pipeline(command)
        except ToolError as exc:
            err_payload = json.dumps(
                {
                    "command": command,
                    "stdout": "",
                    "stderr": f"validation error: {exc}",
                    "exit_code": -2,
                    "timed_out": False,
                    "truncated": False,
                    "information_lines": [],
                },
                ensure_ascii=False,
            )
            return ToolResponse(text=err_payload), 0.0, {"validation_error": str(exc)}

        if self._tokenizer is None:
            raise RuntimeError(
                "SearchCorpusTool was invoked before set_tokenizer() was "
                "called. GrepSeekAgentLoop.__init__ should have injected "
                "the tokenizer from self.tokenizer; check that the agent loop "
                "is initialized correctly."
            )

        # Execute the pipeline in a thread to avoid blocking the asyncio event loop.
        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(
            None,
            lambda: _execute_pipeline(
                command,
                cwd=self.corpus_root,
                timeout=self.timeout_seconds,
                max_tokens=self.max_output_tokens,
                tokenizer=self._tokenizer,
            ),
        )

        response_text = json.dumps(payload, ensure_ascii=False)
        metrics = {
            "exit_code": payload["exit_code"],
            "timed_out": payload["timed_out"],
            "truncated": payload["truncated"],
        }
        return ToolResponse(text=response_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        self._instances.pop(instance_id, None)
