"""Whitelisted shell-tool executor for the search agent.

Allows pipelines of whitelisted commands. No redirection, no command
chaining, no command substitution. Output is truncated for trace fidelity.
"""
import re
import shlex
import subprocess
from dataclasses import dataclass

ALLOWED_COMMANDS = {
    "rg", "grep", "find", "sed", "awk", "head", "tail", "cat",
    "ls", "wc", "sort", "cut", "uniq", "tr",
}

DEFAULT_TIMEOUT = 60
DEFAULT_OUTPUT_CHARS = 4000


class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    stdout: str
    stderr: str
    returncode: int
    truncated: bool

    def display(self) -> str:
        if self.stdout:
            body = self.stdout
        elif self.stderr:
            body = f"[no stdout]\nstderr: {self.stderr.strip()[:400]}"
        else:
            body = "[empty output]"
        if self.truncated:
            body += "\n[... output truncated]"
        return body


def _strip_quoted(s: str) -> str:
    # Remove single- and double-quoted regions to make metachar checks safer.
    s = re.sub(r"'[^']*'", "", s)
    s = re.sub(r'"[^"]*"', "", s)
    return s


# Patterns that allow file-writing/destruction even inside an otherwise-allowed
# command (sed -i, awk -i inplace, find -delete, find -exec, awk system()/pipe,
# awk redirection inside the script, etc.). We check these on the raw command
# (BEFORE stripping quoted regions), since they often live inside a quoted awk
# or sed script.
_DANGEROUS_PATTERNS = [
    # sed in-place editing: -i, -i'.bak', --in-place
    (re.compile(r"\bsed\b[^|]*\s-(?:[a-zA-Z]*i|-in-place)\b"), "sed -i / --in-place (in-place edit)"),
    # awk/gawk -i inplace
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\s-i\s+inplace\b"), "awk/gawk -i inplace"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\s--in-place\b"), "awk --in-place"),
    # find destructive flags
    (re.compile(r"\bfind\b[^|]*\s-delete\b"), "find -delete"),
    (re.compile(r"\bfind\b[^|]*\s-exec(?:dir)?\b"), "find -exec / -execdir"),
    # awk script side-effects: system() calls, output redirection inside script,
    # pipe-to-command from awk script.
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\bsystem\s*\("), "awk system() call"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\bprintf?\b[^|]*>\s*[\"']"), "awk script redirection (print > file)"),
    (re.compile(r"\b(?:awk|gawk)\b[^|]*\|\s*[\"']"), "awk script pipe-to-shell-command"),
    # sed `e` (execute shell) command; covers `sed 'e cmd'` and `sed -e 'e cmd'`
    (re.compile(r"\bsed\b[^|]*'\s*e\s+"), "sed 'e' (execute shell) command"),
    # Anything that tries to chmod / chown / mv / rm / cp / dd / truncate / install,
    # even though they're not in the whitelist — defense in depth in case the
    # whitelist check is ever loosened.
    (re.compile(r"\b(?:rm|mv|cp|dd|chmod|chown|chgrp|truncate|install|tee|ln)\b"),
     "destructive/write command name"),
]


def _validate_pipeline(cmd: str) -> None:
    if "\n" in cmd:
        raise ToolError("newlines not allowed in command")
    if len(cmd) > 2000:
        raise ToolError("command too long")

    # Check destructive patterns on the RAW command (so we still see >, system(,
    # etc. that would otherwise be hidden inside an awk/sed script's quotes).
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


def run_tool(
    cmd: str,
    timeout: int = DEFAULT_TIMEOUT,
    max_chars: int = DEFAULT_OUTPUT_CHARS,
    cwd: str = None,
) -> ToolResult:
    """Execute a whitelisted shell pipeline. Raises ToolError on validation failure.

    If `cwd` is given, the command runs in that directory — convenient for the
    agent to refer to the corpus by a short relative filename instead of an
    absolute path.
    """
    _validate_pipeline(cmd)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            cwd=cwd,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            stdout=f"[command timed out after {timeout}s]",
            stderr="",
            returncode=-1,
            truncated=False,
        )
    out = proc.stdout or ""
    err = proc.stderr or ""
    truncated = False
    if len(out) > max_chars:
        out = out[:max_chars]
        truncated = True
    return ToolResult(stdout=out, stderr=err, returncode=proc.returncode, truncated=truncated)
