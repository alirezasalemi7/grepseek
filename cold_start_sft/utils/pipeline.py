"""Cold-start trajectory generation pipeline.

High-level flow per HotpotQA example:

  A. Decomposition (tutor)
     question + gold_answer  -->  ordered list of sub-questions

  B. Backward construction (tutor only)
     For i = N, N-1, ..., 1:
       - Determine expected_answer_i:
            i == N:  gold_answer
            i  < N:  bridge entity extracted from doc_{i+1}
       - Iteratively propose a shell command and a doc retrieved by it,
         judged by an LLM judge, until verdict == YES (or M tries used).
       - Save (cmd_i, doc_i).
     Hard rule for backward command writers: must NOT use the expected
     answer as a search term (the agent at inference doesn't know it).

  C. Forward trajectory assembly
     For i = 1, ..., N:
       - planner draft: think + cmd from (question, prior history)
                        [planner sees no answer / no target]
       - tutor edit:    rewrite planner's think to lead to cmd_i
                        while staying grounded in observed history
       - step recorded as (edited_think, cmd_i, doc_i)

  D. Final answer (planner)
     Produces <answer>; scored with normalized exact match.

Output is a deeply-nested record per example with all intermediate inputs,
raw LLM responses, and decisions, so a failing run can be diagnosed offline.
"""
import json
import os
import random
import re
import string
from typing import Optional

import json5
from tokenizers import Tokenizer

from .llm import ServerLLM, SamplingParams
from .tools import run_tool, ToolError
from . import prompts


# Directory that contains the corpus file (CORPUS_FILENAME_ACTUAL below).
# Override with the CORPUS_DIR env var or create_data.py's --corpus_dir flag.
# Default: the `data/` directory next to this package.
CORPUS_DIR = os.environ.get(
    "CORPUS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
)
CORPUS_FILENAME_LOGICAL = "corpus.jsonl"     # what the agent sees in commands
CORPUS_FILENAME_ACTUAL = "wiki_corpus.jsonl"  # what's actually on disk
CORPUS_PATH = f"{CORPUS_DIR}/{CORPUS_FILENAME_ACTUAL}"  # full absolute path (back-compat)

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


# ---------------------------------------------------------------
# Qwen3.5-27B tokenizer (used to cap tool output by token count)
# ---------------------------------------------------------------
# Lazily loaded — first call downloads the tokenizer.json into HF_HOME and
# caches it for the rest of the session.

# Tokenizer name to use for token counting. Defaults to the teacher model; set
# the TOKENIZER_NAME env var to any tokenizer available locally / on the Hub.
_TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "Qwen/Qwen3.5-27B")
_TOKENIZER = None
_TOKENIZER_TRIED = False


def _get_tokenizer():
    """Load the configured HF tokenizer once. It is used only to cap tool output
    by token count, so if it cannot be downloaded we fall back to a ~4-chars/token
    estimate and keep running (set TOKENIZER_NAME to a local tokenizer to avoid
    the network call)."""
    global _TOKENIZER, _TOKENIZER_TRIED
    if not _TOKENIZER_TRIED:
        _TOKENIZER_TRIED = True
        try:
            _TOKENIZER = Tokenizer.from_pretrained(_TOKENIZER_NAME)
        except Exception as e:  # offline / gated / unknown repo
            print(f"[tokenizer] could not load {_TOKENIZER_NAME!r} ({e}); "
                  f"falling back to a character-based length estimate.")
            _TOKENIZER = None
    return _TOKENIZER


def count_tokens(text: str) -> int:
    if not text:
        return 0
    tok = _get_tokenizer()
    if tok is None:
        return max(1, len(text) // 4)
    return len(tok.encode(text).ids)


def truncate_tokens(text: str, max_tokens: int) -> tuple:
    """Returns (truncated_text, was_truncated), keeping the first max_tokens tokens."""
    if not text:
        return text, False
    tok = _get_tokenizer()
    if tok is None:
        limit = max_tokens * 4  # ~4 chars/token
        return (text, False) if len(text) <= limit else (text[:limit], True)
    ids = tok.encode(text).ids
    if len(ids) <= max_tokens:
        return text, False
    return tok.decode(ids[:max_tokens]), True


# ============================================================
# Parsing helpers
# ============================================================

_TAG_RE_CACHE = {}


def _tag(name: str) -> re.Pattern:
    if name not in _TAG_RE_CACHE:
        _TAG_RE_CACHE[name] = re.compile(rf"<{name}>\s*(.*?)\s*</{name}>", re.DOTALL)
    return _TAG_RE_CACHE[name]


def extract_tag(text: str, name: str) -> Optional[str]:
    if not text:
        return None
    m = _tag(name).search(text)
    return m.group(1).strip() if m else None


def parse_planner_step(text: str) -> tuple:
    """Parse planner output -> (prose_think, tool_call, answer).

    The planner is instructed to write a prose paragraph, then exactly one of
    <tool_call> or <answer>. Prose-before-block is used as the think.
    """
    if not text:
        return None, None, None
    call_match = _tag("tool_call").search(text)
    answer_match = _tag("answer").search(text)
    anchor = None
    for m in (call_match, answer_match):
        if m and (anchor is None or m.start() < anchor):
            anchor = m.start()
    prose = text[:anchor] if anchor is not None else text
    prose = re.sub(r"</?think>", "", prose).strip()
    return (
        prose or None,
        call_match.group(1).strip() if call_match else None,
        answer_match.group(1).strip() if answer_match else None,
    )


# ============================================================
# Trace rendering
# ============================================================

def render_history(steps: list) -> str:
    if not steps:
        return "(no commands run yet)"
    out = []
    for s in steps:
        out.append(f"<think>\n{s['think']}\n</think>")
        out.append(f"<tool_call>\n{s['tool_call']}\n</tool_call>")
        out.append(f"<tool_response>\n{s['tool_response']}\n</tool_response>")
    return "\n".join(out)


# ============================================================
# Tool execution
# ============================================================

def execute_safe(
    cmd: str,
    max_tokens: int = 2048,
    timeout: int = 60,
    corpus_dir: str = CORPUS_DIR,
) -> tuple:
    """Run a shell pipeline. Returns (display_str, ok).

    The command runs with cwd=corpus_dir, and any occurrence of the logical
    filename `corpus.jsonl` is rewritten to the actual file on disk
    (`wiki_corpus.jsonl`) right before execution. The output is truncated to
    `max_tokens` tokens of the Qwen3.5-27B tokenizer (the same vocabulary the
    trained model will use), with a generous byte-level safety cap up front so
    the tokenizer never has to chew on a runaway gigabyte of stdout.
    """
    if not cmd:
        return "[no command produced]", False
    cmd_to_run = cmd.replace(CORPUS_FILENAME_LOGICAL, CORPUS_FILENAME_ACTUAL)
    # Byte cap as a fast first defense — assume max 8 chars/token in worst case.
    safety_chars = max(max_tokens * 8, 4000)
    try:
        result = run_tool(cmd_to_run, max_chars=safety_chars, timeout=timeout, cwd=corpus_dir)
    except ToolError as e:
        return f"[ToolError: {e}]", False
    text = result.display()
    truncated, was_truncated = truncate_tokens(text, max_tokens)
    if was_truncated and "[... output truncated]" not in truncated:
        truncated += f"\n[... output truncated at {max_tokens} tokens]"
    return truncated, True


# ============================================================
# Phase A: decomposition (tutor) — retry-until-parseable
# ============================================================

def _try_parse_decomposition(text: str) -> Optional[list]:
    if not text:
        return None
    candidates = [text.strip()]
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    fence = re.search(r"```(?:json5?|JSON)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    for cand in candidates:
        for parser in (json.loads, json5.loads):
            try:
                chain = parser(cand)
                break
            except (json.JSONDecodeError, ValueError):
                chain = None
        if not isinstance(chain, list) or not chain:
            continue
        cleaned = []
        for item in chain:
            if not isinstance(item, dict):
                cleaned = None
                break
            sq = str(item.get("sub_question") or "").strip()
            if not sq:
                cleaned = None
                break
            cleaned.append({"sub_question": sq})
        if cleaned:
            return cleaned
    return None


def decompose(
    llm: ServerLLM,
    question: str,
    gold_answer: str,
    max_attempts: int = 4,
) -> dict:
    """Returns {"chain": [{sub_question:...}, ...], "raw_responses": [...]}.

    chain is None if parsing failed across all attempts.
    """
    prompt = prompts.DECOMPOSE_PROMPT.format(question=question, answer=gold_answer)
    raw_responses = []
    for attempt in range(max_attempts):
        temp = 0.0 if attempt == 0 else 0.6
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=SamplingParams(temperature=temp, max_tokens=512, extra_body=NO_THINK),
        )
        raw = resp.text or ""
        raw_responses.append(raw)
        chain = _try_parse_decomposition(raw)
        if chain:
            return {"chain": chain, "raw_responses": raw_responses}
    return {"chain": None, "raw_responses": raw_responses}


# ============================================================
# Phase B helpers: bridge extraction (tutor)
# ============================================================

def extract_bridge_entity(
    llm: ServerLLM,
    question: str,
    sub_q_prev: str,
    sub_q_next: str,
    expected_next: str,
    doc_next: str,
    max_attempts: int = 3,
) -> dict:
    """Reads doc_next and predicts the answer to sub_q_prev (the bridge entity).

    Returns {"bridge_entity": str|None, "aliases": [str], "alternates": [str],
             "evidence": str|None, "raw_responses": [str], "prompts": [str]}.

    `aliases` are alternative names of the SAME underlying bridge entity
    explicitly mentioned in doc_next (stage names, real names, "also known as"
    constructions). `alternates` are at most 2 OTHER plausible candidate
    entities for cases where the passage genuinely supports multiple readings.
    Both lists feed into the judge for sub_q_prev as acceptable forms.
    """
    prompt = prompts.BRIDGE_EXTRACT_PROMPT.format(
        question=question,
        sub_q_prev=sub_q_prev,
        sub_q_next=sub_q_next,
        expected_next=expected_next,
        doc_next=doc_next[:6000],
    )

    def _empty(raw_responses):
        return {
            "bridge_entity": None, "aliases": [], "alternates": [],
            "evidence": None,
            "raw_responses": raw_responses, "prompts": [prompt],
        }

    def _norm_list(value):
        if not isinstance(value, list):
            return []
        out = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        # Cap alternates at 2; aliases at 5 (model can be exuberant).
        return out

    raw_responses = []
    for attempt in range(max_attempts):
        temp = 0.0 if attempt == 0 else 0.5
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=SamplingParams(temperature=temp, max_tokens=2048, extra_body=NO_THINK),
        )
        raw = resp.text or ""
        raw_responses.append(raw)
        for parser in (json.loads, json5.loads):
            try:
                obj = parser(raw.strip())
            except (json.JSONDecodeError, ValueError):
                # Try grabbing first {...} block
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    continue
                try:
                    obj = parser(m.group(0))
                except (json.JSONDecodeError, ValueError):
                    continue
            if isinstance(obj, dict):
                be = obj.get("bridge_entity")
                ev = obj.get("evidence")
                if be is None or (isinstance(be, str) and not be.strip()):
                    return _empty(raw_responses)
                aliases = _norm_list(obj.get("aliases"))[:5]
                alternates = _norm_list(obj.get("alternates"))[:2]
                return {
                    "bridge_entity": str(be).strip(),
                    "aliases": aliases,
                    "alternates": alternates,
                    "evidence": str(ev).strip() if ev else None,
                    "raw_responses": raw_responses, "prompts": [prompt],
                }
            break
    return _empty(raw_responses)


# ============================================================
# Phase B helpers: judge (tutor)
# ============================================================

# Generic answer tokens whose literal-string presence in tool output is not a
# meaningful signal (the string appears everywhere, judge must reason).
_GENERIC_ANSWER_TOKENS = {
    "yes", "no", "true", "false",
    "both", "all", "none", "neither", "either", "every", "any", "some",
    "many", "most", "few", "more", "less",
    "same", "different",
    "one", "two", "three", "four", "five",
    "first", "second", "third", "fourth",
    "left", "right", "top", "bottom", "middle",
}


def _is_generic_answer(s: str) -> bool:
    """An answer too generic for substring-presence to carry signal."""
    s_clean = (s or "").strip().lower().rstrip(".!?,;:")
    if len(s_clean) < 3:
        return True
    return s_clean in _GENERIC_ANSWER_TOKENS


def _tool_output_is_empty_or_errored(tool_output: str) -> bool:
    """The tool returned no useful content. Judge would always say NO."""
    s = (tool_output or "").strip()
    if not s:
        return True
    sentinels = ("[empty output]", "[no stdout]", "[command timed out", "[ToolError")
    return any(sentinel in s for sentinel in sentinels)


def llm_judge(
    llm: ServerLLM,
    sub_question: str,
    expected_answer: str,
    tool_output: str,
    acceptable_forms: Optional[list] = None,
    max_attempts: int = 3,
) -> dict:
    """Returns {"verdict": "YES"|"NO", "reasoning": str, "raw_responses": [...],
                "skip_reason": str|None}.

    `acceptable_forms` is an optional list of alternative names/aliases for
    the same expected answer; the judge accepts a passage that confirms via
    any of those forms as YES. Used for bridge entities with multiple known
    aliases (e.g., "El-P" and "Jaime Meline" for the same person).

    Two cheap pre-judge checks short-circuit before any LLM call:
      Check 1: tool output is empty / errored / timed-out → NO (always).
      Check 2: none of the concrete acceptable forms appear in the tool
               output as a case-insensitive substring → NO. Skipped when
               every candidate form is a generic word ("yes", "no", "both",
               etc.) where literal absence carries no signal.
    Roughly ~64% of judge calls are eliminated this way (per analysis.md).
    """
    # ---------- Check 1: empty / errored output ----------
    if _tool_output_is_empty_or_errored(tool_output):
        return {
            "verdict": "NO",
            "reasoning": "[pre-judge: tool output empty or errored — judge skipped]",
            "raw_responses": [],
            "skip_reason": "empty_output",
        }

    # ---------- Check 2: literal absence of all concrete acceptable forms ----
    all_forms = [expected_answer] + list(acceptable_forms or [])
    concrete_forms = [f for f in all_forms if f and not _is_generic_answer(f)]
    if concrete_forms:
        out_lc = tool_output.lower()
        if not any(f.lower() in out_lc for f in concrete_forms):
            return {
                "verdict": "NO",
                "reasoning": "[pre-judge: none of the expected answer forms appear in tool output — judge skipped]",
                "raw_responses": [],
                "skip_reason": "answer_absent",
            }

    # ---------- Otherwise: real LLM judge call ----------
    forms_str = (
        "(none — only the primary expected answer counts)"
        if not acceptable_forms
        else ", ".join(repr(f) for f in acceptable_forms)
    )
    prompt = prompts.JUDGE_PROMPT.format(
        sub_question=sub_question,
        expected_answer=expected_answer,
        acceptable_forms=forms_str,
        tool_output=tool_output[:6000],
    )
    raw_responses = []
    for attempt in range(max_attempts):
        temp = 0.0 if attempt == 0 else 0.4
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=SamplingParams(temperature=temp, max_tokens=256, extra_body=NO_THINK),
        )
        raw = (resp.text or "").strip()
        raw_responses.append(raw)
        for parser in (json.loads, json5.loads):
            try:
                obj = parser(raw)
            except (json.JSONDecodeError, ValueError):
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if not m:
                    continue
                try:
                    obj = parser(m.group(0))
                except (json.JSONDecodeError, ValueError):
                    continue
            if isinstance(obj, dict) and "verdict" in obj:
                v = str(obj["verdict"]).strip().upper()
                if v in ("YES", "NO"):
                    return {
                        "verdict": v,
                        "reasoning": str(obj.get("reasoning") or "").strip(),
                        "raw_responses": raw_responses,
                        "skip_reason": None,
                    }
            break
    # Default to NO when judge can't be parsed (conservative).
    return {
        "verdict": "NO",
        "reasoning": "[judge parse failed]",
        "raw_responses": raw_responses,
        "skip_reason": None,
    }


# ============================================================
# Phase B helpers: backward command discovery (tutor)
# ============================================================

def _parse_command_block(text: str) -> tuple:
    """Returns (reasoning, command). Either may be None."""
    return extract_tag(text, "reasoning"), extract_tag(text, "command")


def discover_backward_command(
    llm: ServerLLM,
    sub_question: str,
    expected_answer: str,
    downstream_docs: list,
    acceptable_forms: Optional[list] = None,
    corpus_dir: str = CORPUS_DIR,
    max_iterations: int = 6,
    tool_max_tokens: int = 2048,
    tool_timeout: int = 60,
) -> dict:
    """Iteratively produce (cmd, doc) where doc confirms expected_answer for
    sub_question. Returns a dict with full iteration log:
        {"success": bool, "final_command": str|None, "final_doc": str|None,
         "iterations": [
             {"prompt": str, "raw_response": str, "reasoning": str|None,
              "command": str|None, "tool_response": str|None,
              "judge": {...}, "judge_prompt": ...},
             ...
         ]}

    `acceptable_forms` is the list of alternative names/aliases for the same
    expected answer. The judge accepts any of them as confirming, and the
    tutor's anti-leak rule forbids ALL of them as search terms.
    """
    system_msg = {
        "role": "system",
        "content": prompts.BACKWARD_COMMAND_SYSTEM.format(
            corpus_description=prompts.CORPUS_DESCRIPTION,
        ),
    }

    if downstream_docs:
        downstream_str = "\n\n".join(
            f"[passage retrieved for sub_q {idx + 1} downstream]\n{d[:1500]}"
            for idx, d in enumerate(downstream_docs)
        )
    else:
        downstream_str = "(none — this is the final sub-question)"

    forbidden_forms_str = (
        "(none — only the primary expected answer is forbidden)"
        if not acceptable_forms
        else ", ".join(repr(f) for f in acceptable_forms)
    )

    iterations = []
    final_command = None
    final_doc = None

    for it in range(max_iterations):
        if it == 0:
            user_content = prompts.BACKWARD_COMMAND_USER_INITIAL.format(
                sub_question=sub_question,
                expected_answer=expected_answer,
                forbidden_forms=forbidden_forms_str,
                downstream_docs=downstream_str,
            )
        else:
            prior_attempts_str = "\n\n".join(
                f"Attempt {j + 1}:\n  command: {p['command']}\n  judge: {p['judge']['verdict']} ({p['judge']['reasoning']})"
                for j, p in enumerate(iterations)
            )
            last = iterations[-1]
            last_output_str = (last["tool_response"] or "")[:2500]

            # Layer-1 recovery hint: when the previous command returned empty
            # output, tell the tutor to broaden — never to add more piped
            # filters. AND-narrowing is fine on the irrelevant-doc case but is
            # the wrong move after empty output (more filters → still empty).
            last_was_empty = (
                last_output_str.strip() == ""
                or "[empty output]" in last_output_str
                or "[no stdout]" in last_output_str
            )
            recovery_hint = ""
            if last_was_empty:
                recovery_hint = (
                    "YOUR PREVIOUS COMMAND RETURNED EMPTY OUTPUT. The next command MUST broaden, "
                    "not narrow further. Drop one of the piped `rg` filters, shorten one of the "
                    "phrases, or replace a phrase with a synonym. Adding more `| rg` filters to "
                    "an already-empty result will keep returning empty.\n\n"
                )

            user_content = recovery_hint + prompts.BACKWARD_COMMAND_USER_REFINE.format(
                sub_question=sub_question,
                expected_answer=expected_answer,
                forbidden_forms=forbidden_forms_str,
                prior_attempts=prior_attempts_str,
                last_command=last["command"] or "(unparseable)",
                last_output=last_output_str,
                judge_reasoning=last["judge"]["reasoning"],
            )

        resp = llm.chat(
            [system_msg, {"role": "user", "content": user_content}],
            sampling_params=SamplingParams(
                # Very generous: the tutor sometimes rambles for 1-2k tokens
                # of "is the writer X? Y?..." before emitting <command>. 8192
                # essentially eliminates token-cap parse failures (cost is
                # ~0 because vLLM uses paged attention and gen latency
                # scales with tokens emitted, not the cap).
                temperature=0.4 + 0.1 * it, max_tokens=8192, extra_body=NO_THINK,
            ),
        )
        raw = resp.text or ""
        reasoning, command = _parse_command_block(raw)

        iter_record = {
            "prompt": user_content,
            "raw_response": raw,
            "reasoning": reasoning,
            "command": command,
            "tool_response": None,
            "judge": None,
        }

        if not command:
            iter_record["judge"] = {
                "verdict": "NO", "reasoning": "[no <command> tag parsed from response]",
                "raw_responses": [],
            }
            iterations.append(iter_record)
            continue

        tool_response, _ok = execute_safe(
            command, max_tokens=tool_max_tokens, timeout=tool_timeout, corpus_dir=corpus_dir,
        )
        iter_record["tool_response"] = tool_response

        verdict = llm_judge(llm, sub_question, expected_answer, tool_response,
                            acceptable_forms=acceptable_forms)
        iter_record["judge"] = verdict
        iterations.append(iter_record)

        if verdict["verdict"] == "YES":
            final_command = command
            final_doc = tool_response
            return {
                "success": True,
                "final_command": final_command,
                "final_doc": final_doc,
                "iterations": iterations,
            }

    return {
        "success": False,
        "final_command": None,
        "final_doc": None,
        "iterations": iterations,
    }


# ============================================================
# Phase B: backward construction (orchestration)
# ============================================================

def run_backward(
    llm: ServerLLM,
    question: str,
    gold_answer: str,
    chain: list,
    corpus_dir: str = CORPUS_DIR,
    max_iterations: int = 6,
    tool_max_tokens: int = 2048,
    tool_timeout: int = 60,
) -> dict:
    """Returns {"steps": [BackwardStep, ...], "success": bool, "abort_reason": str}.

    Each BackwardStep:
        {"i": int, "sub_question": str, "expected_answer": str,
         "expected_answer_source": "gold" | "extracted_from_doc_i+1",
         "extraction": dict|None, "discovery": <discover_backward_command return>,
         "final_command": str, "final_doc": str}
    Steps are returned in forward order (i = 1, 2, ..., N).
    """
    N = len(chain)
    steps_reverse = []  # filled i = N..1; reversed at end
    docs_collected = []  # docs in reverse order of discovery

    expected_answer = gold_answer
    expected_answer_source = "gold"
    acceptable_forms: list = []  # aliases + alternates for the current expected_answer
    extraction = None

    for i_zero in range(N - 1, -1, -1):
        sub_q = chain[i_zero]["sub_question"]
        discovery = discover_backward_command(
            llm,
            sub_question=sub_q,
            expected_answer=expected_answer,
            downstream_docs=docs_collected,
            acceptable_forms=acceptable_forms,
            corpus_dir=corpus_dir,
            max_iterations=max_iterations,
            tool_max_tokens=tool_max_tokens,
            tool_timeout=tool_timeout,
        )

        step_record = {
            "i": i_zero + 1,
            "sub_question": sub_q,
            "expected_answer": expected_answer,
            "expected_answer_source": expected_answer_source,
            "acceptable_forms": list(acceptable_forms),
            "extraction": extraction,
            "discovery": discovery,
            "final_command": discovery["final_command"],
            "final_doc": discovery["final_doc"],
        }
        steps_reverse.append(step_record)

        if not discovery["success"]:
            return {
                "steps": list(reversed(steps_reverse)),
                "success": False,
                "abort_reason": f"backward_discovery_failed_i_{i_zero + 1}",
            }

        docs_collected.append(discovery["final_doc"])

        # If there is a previous sub_q, derive its expected_answer from this doc.
        if i_zero > 0:
            prev_sub_q = chain[i_zero - 1]["sub_question"]
            extraction = extract_bridge_entity(
                llm,
                question=question,
                sub_q_prev=prev_sub_q,
                sub_q_next=sub_q,
                expected_next=expected_answer,
                doc_next=discovery["final_doc"],
            )
            if not extraction["bridge_entity"]:
                return {
                    "steps": list(reversed(steps_reverse)),
                    "success": False,
                    "abort_reason": f"bridge_extraction_failed_i_{i_zero}",
                }
            expected_answer = extraction["bridge_entity"]
            # Aliases (alt names of same entity) + alternates (other plausible
            # candidates) both feed into the next iteration's acceptable_forms.
            acceptable_forms = list(extraction.get("aliases") or []) + list(extraction.get("alternates") or [])
            expected_answer_source = f"extracted_from_doc_i_{i_zero + 1}"
        else:
            extraction = None
            acceptable_forms = []

    return {
        "steps": list(reversed(steps_reverse)),
        "success": True,
        "abort_reason": "",
    }


# ============================================================
# Phase C: forward step assembly (planner draft + tutor edit)
# ============================================================

def planner_draft(
    llm: ServerLLM,
    question: str,
    history: str,
) -> dict:
    """Returns {"raw_response": str, "think": str|None, "command": str|None,
                "answer": str|None}."""
    system = prompts.PLANNER_SYSTEM.format(corpus_description=prompts.CORPUS_DESCRIPTION)
    user = prompts.PLANNER_USER.format(question=question, history=history)
    resp = llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        sampling_params=SamplingParams(temperature=0.7, max_tokens=2048, extra_body=NO_THINK),
    )
    raw = resp.text or ""
    think, cmd, ans = parse_planner_step(raw)
    return {"raw_response": raw, "think": think, "command": cmd, "answer": ans}


def tutor_edit_think(
    llm: ServerLLM,
    question: str,
    history: str,
    draft_think: str,
    draft_command: str,
    target_command: str,
    target_doc: str,
    max_attempts: int = 3,
) -> dict:
    """Returns {"raw_response": str, "edited_think": str|None}."""
    prompt = prompts.TUTOR_EDIT_PROMPT.format(
        question=question,
        history=history,
        draft_think=draft_think or "(none)",
        draft_command=draft_command or "(none)",
        target_command=target_command,
        target_doc_preview=(target_doc or "")[:2000],
    )
    raw_responses = []
    for attempt in range(max_attempts):
        temp = 0.3 if attempt == 0 else 0.7
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=SamplingParams(temperature=temp, max_tokens=2048, extra_body=NO_THINK),
        )
        raw = resp.text or ""
        raw_responses.append(raw)
        edited = extract_tag(raw, "edited_reasoning")
        if edited:
            return {"raw_response": raw_responses[-1], "raw_responses": raw_responses, "edited_think": edited}
    return {"raw_response": raw_responses[-1] if raw_responses else "",
            "raw_responses": raw_responses, "edited_think": None}


# ============================================================
# Phase D: final answer (planner)
# ============================================================

def generate_final_answer(
    llm: ServerLLM,
    question: str,
    history: str,
    max_attempts: int = 3,
) -> dict:
    system = prompts.PLANNER_SYSTEM.format(corpus_description=prompts.CORPUS_DESCRIPTION)
    user = prompts.FINAL_ANSWER_USER.format(question=question, history=history)
    raw_responses = []
    for attempt in range(max_attempts):
        temp = 0.0 if attempt == 0 else 0.5
        resp = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            sampling_params=SamplingParams(temperature=temp, max_tokens=600, extra_body=NO_THINK),
        )
        raw = resp.text or ""
        raw_responses.append(raw)
        think, _, answer = parse_planner_step(raw)
        if answer:
            return {"raw_responses": raw_responses, "think": think or "", "answer": answer}
    return {"raw_responses": raw_responses, "think": None, "answer": None}


# ============================================================
# Phase E: Quality judge (post-hoc filter on the assembled trajectory)
# ============================================================

def _indent(text: str, n: int) -> str:
    pad = " " * n
    return "\n".join(pad + line for line in (text or "").split("\n"))


def render_for_judge(record: dict) -> str:
    """Render the agent-side trajectory in a turn-numbered plain-text format
    that uses none of the agent's own special tags. This is what the quality
    judge sees — keeping it free of <think>/<tool_call>/<tool_response>/<answer>
    avoids confusing the judge LLM about whose reasoning is whose.
    """
    steps = record.get("history_steps") or []
    final = record.get("final") or {}
    n_real_turns = len(steps) + (1 if final.get("answer") else 0)
    lines = []

    for i, s in enumerate(steps, start=1):
        lines.append(f"TURN {i}")
        lines.append("  REASONING:")
        lines.append(_indent((s.get("think") or "").strip(), 4))
        lines.append("  COMMAND:")
        lines.append(_indent((s.get("tool_call") or "").strip(), 4))
        lines.append("  OUTPUT:")
        lines.append(_indent((s.get("tool_response") or "").strip(), 4))
        lines.append("")

    if final.get("answer"):
        lines.append(f"TURN {n_real_turns} (FINAL)")
        lines.append("  REASONING:")
        lines.append(_indent((final.get("think") or "").strip(), 4))
        lines.append("  FINAL ANSWER:")
        lines.append(_indent((final.get("answer") or "").strip(), 4))

    return "\n".join(lines)


_VALID_VERDICTS = {"PASS", "FAIL"}


def _try_parse_judge_verdict(text: str) -> Optional[dict]:
    if not text:
        return None
    candidates = [text.strip()]
    fence = re.search(r"```(?:json5?|JSON)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        for parser in (json.loads, json5.loads):
            try:
                obj = parser(cand)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(obj, dict):
                continue
            verdict = str(obj.get("verdict") or "").strip().upper()
            if verdict not in _VALID_VERDICTS:
                continue
            failing_check = obj.get("failing_check")
            first_failing_turn = obj.get("first_failing_turn")
            reasoning = str(obj.get("reasoning") or "").strip()
            return {
                "verdict": verdict,
                "failing_check": failing_check,
                "first_failing_turn": first_failing_turn,
                "reasoning": reasoning,
            }
    return None


def quality_judge(
    llm: ServerLLM,
    record: dict,
    max_attempts: int = 3,
    max_input_tokens: int = 7500,
) -> dict:
    """Run the LLM coherence judge over a fully-assembled trajectory.

    Returns a dict suitable for record["quality_check"]:
        {"verdict": "PASS"|"FAIL"|"UNKNOWN",
         "failing_check": int|None, "first_failing_turn": int|None,
         "reasoning": str, "raw_responses": [str], "rendered_input": str,
         "prompt": str}

    "UNKNOWN" is set if the judge can't be parsed across all retries; callers
    can decide whether to treat that as PASS or FAIL (we recommend FAIL).
    """
    rendered = render_for_judge(record)
    # Guard against runaway-long trajectories pushing past the model's context.
    rendered_capped, _ = truncate_tokens(rendered, max_input_tokens)
    prompt = prompts.QUALITY_JUDGE_PROMPT.format(
        question=record.get("question", ""),
        trajectory_text=rendered_capped,
    )

    raw_responses = []
    for attempt in range(max_attempts):
        temp = 0.0 if attempt == 0 else 0.4
        resp = llm.chat(
            [{"role": "user", "content": prompt}],
            sampling_params=SamplingParams(
                temperature=temp, max_tokens=2048, extra_body=NO_THINK,
            ),
        )
        raw = resp.text or ""
        raw_responses.append(raw)
        parsed = _try_parse_judge_verdict(raw)
        if parsed:
            return {
                **parsed,
                "raw_responses": raw_responses,
                "rendered_input": rendered_capped,
                "prompt": prompt,
            }

    return {
        "verdict": "UNKNOWN",
        "failing_check": None,
        "first_failing_turn": None,
        "reasoning": "[judge response could not be parsed across retries]",
        "raw_responses": raw_responses,
        "rendered_input": rendered_capped,
        "prompt": prompt,
    }


# ============================================================
# QA-style normalized exact match
# ============================================================

_PUNCT = set(string.punctuation)


def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def qa_exact_match(prediction: str, gold_answers) -> bool:
    if not prediction:
        return False
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    pred_n = _normalize_answer(prediction)
    return any(pred_n == _normalize_answer(g) for g in gold_answers if g)


def _f1_against(pred_norm: str, gold_norm: str) -> float:
    """Standard SQuAD/HotpotQA token-overlap F1 between two normalized strings."""
    if not pred_norm or not gold_norm:
        # Convention from the SQuAD eval script: if either is empty, F1 is 1.0
        # iff they're both empty, else 0.0.
        return 1.0 if pred_norm == gold_norm else 0.0
    pred_toks = pred_norm.split()
    gold_toks = gold_norm.split()
    if not pred_toks or not gold_toks:
        return 0.0
    common = {}
    pred_count = {}
    for t in pred_toks:
        pred_count[t] = pred_count.get(t, 0) + 1
    for t in gold_toks:
        if pred_count.get(t, 0) > 0:
            pred_count[t] -= 1
            common[t] = common.get(t, 0) + 1
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def qa_f1_score(prediction: str, gold_answers) -> float:
    """Max F1 across all gold answers, after SQuAD-style normalization."""
    if not prediction:
        return 0.0
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    pred_norm = _normalize_answer(prediction)
    best = 0.0
    for g in gold_answers:
        if not g:
            continue
        f1 = _f1_against(pred_norm, _normalize_answer(g))
        if f1 > best:
            best = f1
    return best


# ============================================================
# Main entry point
# ============================================================

def generate_trajectory(
    llm: ServerLLM,
    example: dict,
    corpus_dir: str = CORPUS_DIR,
    backward_max_iterations: int = 6,
    tool_max_tokens: int = 2048,
    tool_timeout: int = 60,
    max_tool_calls: int = 10,
    enable_quality_judge: bool = True,
    rng: Optional[random.Random] = None,
    verbose: bool = False,
) -> dict:
    """Generate one cold-start trajectory.

    `max_tool_calls` caps the total number of <tool_call> turns in the
    agent-side trace. If the trajectory would exceed it, the example is dropped
    with abort_reason="too_many_tool_calls".

    Returns a richly-nested dict with all intermediate inputs/outputs/decisions
    so that any failing example can be diagnosed offline.
    """
    if rng is None:
        rng = random.Random()

    question = example["question"]
    gold_answers = example.get("golden_answers") or []
    gold_answer_str = gold_answers[0] if gold_answers else ""

    record = {
        "id": str(example.get("id", "")),
        "question": question,
        "gold_answer": list(gold_answers),
        "decomposition": None,
        "backward": None,
        "forward": [],
        "final": None,
        "success": False,
        "abort_reason": "",
    }

    # ---- A. Decomposition ----
    decomp = decompose(llm, question, gold_answer_str)
    record["decomposition"] = decomp
    if not decomp["chain"]:
        record["abort_reason"] = "decomposition_failed"
        return record
    if len(decomp["chain"]) > max_tool_calls:
        record["abort_reason"] = (
            f"decomposition_too_long_{len(decomp['chain'])}_gt_{max_tool_calls}"
        )
        return record
    if verbose:
        print(f"  decomposed into {len(decomp['chain'])} sub-questions")

    # ---- B. Backward construction ----
    backward = run_backward(
        llm,
        question=question,
        gold_answer=gold_answer_str,
        chain=decomp["chain"],
        corpus_dir=corpus_dir,
        max_iterations=backward_max_iterations,
        tool_max_tokens=tool_max_tokens,
        tool_timeout=tool_timeout,
    )
    record["backward"] = backward
    if not backward["success"]:
        record["abort_reason"] = backward["abort_reason"]
        if verbose:
            print(f"  backward failed: {backward['abort_reason']}")
        return record
    if verbose:
        print(f"  backward succeeded: {len(backward['steps'])} steps")

    # ---- C. Forward assembly ----
    history_steps = []  # list of dicts: {think, tool_call, tool_response}
    for fw_idx, b_step in enumerate(backward["steps"]):
        target_command = b_step["final_command"]
        target_doc = b_step["final_doc"]
        sub_q = b_step["sub_question"]

        forward_record = {
            "i": fw_idx + 1,
            "sub_question": sub_q,
            "planner_draft": None,
            "tutor_edit": None,
            "final_step": None,
        }

        # Build the real step: the planner drafts a (think, command) from the
        # question and observed history only; the tutor then rewrites the think
        # so it leads to the known-good target command while staying grounded.
        history_str = render_history(history_steps)

        draft = planner_draft(llm, question, history_str)
        forward_record["planner_draft"] = draft
        edit = tutor_edit_think(
            llm,
            question=question,
            history=history_str,
            draft_think=draft["think"] or "",
            draft_command=draft["command"] or "",
            target_command=target_command,
            target_doc=target_doc,
        )
        forward_record["tutor_edit"] = edit
        edited_think = edit["edited_think"]
        if not edited_think:
            record["abort_reason"] = f"tutor_edit_failed_step_{fw_idx + 1}"
            record["forward"].append(forward_record)
            return record

        final_step = {
            "think": edited_think,
            "tool_call": target_command,
            "tool_response": target_doc,
        }
        history_steps.append(final_step)
        forward_record["final_step"] = final_step
        record["forward"].append(forward_record)
        if verbose:
            print(f"  step {fw_idx + 1}: assembled (think={len(edited_think)} chars)")

    # Hard post-check: drop trajectories that exceed the max-tool-calls budget.
    if len(history_steps) > max_tool_calls:
        record["abort_reason"] = (
            f"too_many_tool_calls_{len(history_steps)}_gt_{max_tool_calls}"
        )
        record["history_steps"] = history_steps
        record["agent_trajectory"] = render_agent_trajectory(record)
        return record

    # ---- D. Final answer ----
    history_str = render_history(history_steps)
    final = generate_final_answer(llm, question, history_str)
    record["final"] = final
    if not final["answer"]:
        record["abort_reason"] = "final_answer_parse_failed"
        return record

    f1 = qa_f1_score(final["answer"], gold_answers)
    em = qa_exact_match(final["answer"], gold_answers)
    record["f1"] = f1
    record["em"] = em
    # We keep any trajectory with non-zero F1 against the gold answer(s).
    record["success"] = f1 > 0
    record["history_steps"] = history_steps  # convenient flat copy for SFT export
    record["agent_trajectory"] = render_agent_trajectory(record)

    # ---- E. Quality judge (post-hoc filter) ----
    # Only run on trajectories that scored f1 > 0 — there's no point judging a
    # trajectory we'd already discard anyway.
    if enable_quality_judge and record["success"]:
        record["quality_check"] = quality_judge(llm, record)
        record["quality_pass"] = (record["quality_check"]["verdict"] == "PASS")
        if verbose:
            qc = record["quality_check"]
            print(
                f"  quality_judge: {qc['verdict']}"
                + (f"  failing_check={qc['failing_check']} "
                   f"first_failing_turn={qc['first_failing_turn']}"
                   f"  reason={qc['reasoning']!r}"
                   if qc["verdict"] != "PASS" else "")
            )
    else:
        record["quality_check"] = None
        # If we didn't run the judge (judge disabled, or trajectory not f1>0),
        # default quality_pass to whatever success says — i.e. we don't add an
        # extra filter on top.
        record["quality_pass"] = record["success"]

    return record


# ============================================================
# SFT export helper
# ============================================================

# ---------------------------------------------------------------
# Output trace serialization (the format the model is trained on)
# ---------------------------------------------------------------
# Assistant turn: <think>...</think> + a JSON tool_call payload.
# Single available tool: "shell", which takes one argument "command".
#
#   <think>...</think>
#   <tool_call>
#   {"name": "shell", "arguments": {"command": "rg -F ... | head -n 3"}}
#   </tool_call>
#
# Tool responses: JSON-encoded inside <tool_response>...</tool_response>:
#
#   <tool_response>
#   {"result": "<raw stdout>"}
#   </tool_response>
#
# Final assistant turn: <think>...</think> + <answer>...</answer>.

TOOL_NAME = "shell"
TOOL_ARG_NAME = "command"


def format_tool_call(command: str) -> str:
    payload = json.dumps(
        {"name": TOOL_NAME, "arguments": {TOOL_ARG_NAME: command}},
        ensure_ascii=False,
    )
    return f"<tool_call>\n{payload}\n</tool_call>"


def format_tool_response(output: str) -> str:
    payload = json.dumps({"result": output}, ensure_ascii=False)
    return f"<tool_response>\n{payload}\n</tool_response>"


def format_assistant_step(think: str, command: str) -> str:
    think = (think or "").strip()
    return f"<think>\n{think}\n</think>\n{format_tool_call((command or '').strip())}"


def format_assistant_final(think: str, answer: str) -> str:
    think = (think or "").strip()
    return f"<think>\n{think}\n</think>\n<answer>\n{(answer or '').strip()}\n</answer>"


def render_agent_trajectory(record: dict) -> str:
    """Render the agent-side conversation in human-readable form: just what the
    trained model will see/produce — the question, every (think + tool_call +
    tool_response) round, and the final answer.

    Useful for eyeballing whether the SFT data makes sense before training.
    """
    lines = []
    lines.append("=" * 72)
    lines.append(f"id:           {record.get('id', '')}")
    lines.append(f"question:     {record.get('question', '')}")
    lines.append(f"gold_answer:  {record.get('gold_answer', [])}")
    lines.append(f"success:      {record.get('success', False)}")
    if record.get("abort_reason"):
        lines.append(f"abort_reason: {record['abort_reason']}")
    lines.append("=" * 72)

    steps = record.get("history_steps") or []
    for i, s in enumerate(steps, start=1):
        lines.append(f"\n--- assistant turn {i} ---")
        lines.append(format_assistant_step(s.get("think") or "", s.get("tool_call") or ""))
        lines.append(f"\n--- user turn {i} (tool result) ---")
        lines.append(format_tool_response((s.get("tool_response") or "").strip()))

    final = record.get("final") or {}
    if final.get("answer"):
        lines.append("\n--- final assistant turn ---")
        lines.append(format_assistant_final(final.get("think") or "", final["answer"]))
    elif record.get("abort_reason"):
        lines.append(f"\n[DID NOT FINISH — {record['abort_reason']}]")

    return "\n".join(lines)


def _verl_style_tool_message_content(command: str, tool_response: str) -> str:
    """Build the structured JSON content for a tool-role message, matching the
    payload shape an agent loop / tool runtime expects.

    We don't track stderr / exit_code / timed_out internally, so they are
    filled with sensible defaults at export time: the run is treated as a
    successful, non-timed-out execution; truncation is detected from a
    "[... output truncated" sentinel in the tool_response; information_lines
    is the non-empty lines of stdout.
    """
    stdout = tool_response or ""
    truncated = "[... output truncated" in stdout or "[... truncated" in stdout
    info_lines = [ln for ln in stdout.split("\n") if ln.strip()]
    return json.dumps(
        {
            "command": command or "",
            "stdout": stdout,
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
            "truncated": truncated,
            "information_lines": info_lines,
        },
        ensure_ascii=False,
    )


def trajectory_to_chatml(record: dict) -> Optional[list]:
    """Build a flat ChatML message list for SFT.

    Tool-side messages match verl's format: role="tool" and a structured JSON
    payload {command, stdout, stderr, exit_code, timed_out, truncated,
    information_lines}. The rest of the trajectory keeps our own format
    (system prompt, "shell" tool name in the <tool_call>, <think> + <answer>
    on the final turn, etc.).

    Returns None if the record is incomplete.
    """
    steps = record.get("history_steps")
    if not steps:
        return None
    final = record.get("final") or {}
    final_answer = final.get("answer")
    if not final_answer:
        return None

    system = prompts.PLANNER_SYSTEM.format(corpus_description=prompts.CORPUS_DESCRIPTION)
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {record['question']}"},
    ]
    for s in steps:
        cmd = (s.get("tool_call") or "").strip()
        msgs.append({
            "role": "assistant",
            "content": format_assistant_step(s.get("think") or "", cmd),
        })
        msgs.append({
            "role": "tool",
            "content": _verl_style_tool_message_content(cmd, s.get("tool_response") or ""),
        })
    msgs.append({
        "role": "assistant",
        "content": format_assistant_final(final.get("think") or "", final_answer),
    })
    return msgs
