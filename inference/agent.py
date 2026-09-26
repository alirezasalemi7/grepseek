"""Multi-turn agent that mirrors the SFT trajectory format exactly.

The SFT data was generated as a flat ChatML transcript:

    system   : PLANNER_SYSTEM (with CORPUS_DESCRIPTION)
    user     : "Query: {question}"
    assistant: "<think>...</think>\\n<tool_call>\\n{json}\\n</tool_call>"
    tool     : JSON {command, stdout, stderr, exit_code, timed_out,
                     truncated, information_lines}
    ...
    assistant: "<think>...</think>\\n<answer>\\n...\\n</answer>"

`run_agent_on_example` replays that loop at eval time against a vLLM server
serving a checkpoint. The tool side uses `tools.run_tool`, which validates
the command, rewrites `corpus.jsonl` -> `wiki_corpus.jsonl`, and truncates
stdout by token count using the same tokenizer the model uses.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .tools import run_tool, ToolResult


# ---------------------------------------------------------------------------
# Constants taken from cold_start_data_gen/utils/prompts.py + pipeline.py so
# inference matches training byte-for-byte. Hard-coded (rather than imported)
# so this package has no dependency on cold_start_data_gen at runtime.
# ---------------------------------------------------------------------------

CORPUS_DESCRIPTION = """Corpus file: corpus.jsonl

Treat the corpus as a large text file with one Wikipedia passage per line. The file has 21 million lines, so always cap output (use `| head -n 3` for narrow searches, up to `| head -n 8` when you need to scan more chunks of the same article).

Useful command patterns:
- Substring search:
    rg -F "distinctive phrase" corpus.jsonl | head -n 3
- AND-narrow with a second grep:
    rg -F "phrase1" corpus.jsonl | rg -i -F "phrase2" | head -n 3
- Count first, then narrow if too many results:
    rg -F "pattern" corpus.jsonl | wc -l
- Useful flags: -F (fixed string, no regex), -i (case-insensitive), -w (whole word).

Allowed shell tools: rg, grep, find, sed, awk, head, tail, cat, ls, wc, sort, cut, uniq, tr.
You MAY pipe with |. You MAY NOT use redirection (>, <), command chaining (; && ||), or command substitution ($(...), `...`)."""

PLANNER_SYSTEM_TEMPLATE = """You are a research agent that searches a Wikipedia corpus to answer multi-hop questions by issuing shell commands.

{corpus_description}

For every step, write a short paragraph of reasoning in plain prose (2-5 sentences) — what you have learned from prior tool outputs, what's still missing, what you'll search for next. Then output exactly one of:

<tool_call>
{{"name": "shell", "arguments": {{"command": "your single-pipeline shell command, no newlines"}}}}
</tool_call>

OR (only when you have enough information to answer):

<answer>
the final answer (concise — typically a short noun phrase, name, or date)
</answer>

Always reason first, then exactly one action block. Do not skip the reasoning."""


def build_system_prompt() -> str:
    return PLANNER_SYSTEM_TEMPLATE.format(corpus_description=CORPUS_DESCRIPTION)


# ---------------------------------------------------------------------------
# Parsing the assistant's response
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


@dataclass
class AssistantParse:
    raw: str
    tool_call_json: Optional[str]
    parsed_command: Optional[str]
    answer: Optional[str]
    error: Optional[str]


def parse_assistant(text: str) -> AssistantParse:
    """Pull <tool_call> JSON or <answer> out of the assistant turn.

    Whichever block appears first wins. Returns an error string when a
    tool_call block is present but its JSON doesn't have name='shell' with a
    string 'command' argument.
    """
    if not text:
        return AssistantParse(raw="", tool_call_json=None, parsed_command=None,
                              answer=None, error="empty assistant response")
    tool_m = _TOOL_CALL_RE.search(text)
    ans_m = _ANSWER_RE.search(text)

    # Resolve which block fires first.
    if tool_m and ans_m:
        first = "tool" if tool_m.start() < ans_m.start() else "answer"
    elif tool_m:
        first = "tool"
    elif ans_m:
        first = "answer"
    else:
        return AssistantParse(raw=text, tool_call_json=None, parsed_command=None,
                              answer=None, error="no <tool_call> or <answer> found")

    if first == "answer":
        return AssistantParse(raw=text, tool_call_json=None, parsed_command=None,
                              answer=ans_m.group(1).strip(), error=None)

    tc_text = tool_m.group(1).strip()
    try:
        tc_obj = json.loads(tc_text)
    except Exception as e:
        return AssistantParse(raw=text, tool_call_json=tc_text, parsed_command=None,
                              answer=None, error=f"tool_call JSON parse error: {e}")
    if not isinstance(tc_obj, dict):
        return AssistantParse(raw=text, tool_call_json=tc_text, parsed_command=None,
                              answer=None, error="tool_call JSON is not a dict")
    name = tc_obj.get("name")
    if name != "shell":
        return AssistantParse(raw=text, tool_call_json=tc_text, parsed_command=None,
                              answer=None, error=f"unknown tool: {name!r}")
    args = tc_obj.get("arguments") or {}
    if not isinstance(args, dict):
        return AssistantParse(raw=text, tool_call_json=tc_text, parsed_command=None,
                              answer=None, error="tool_call arguments not a dict")
    cmd = args.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        return AssistantParse(raw=text, tool_call_json=tc_text, parsed_command=None,
                              answer=None, error="missing/empty command arg")

    return AssistantParse(raw=text, tool_call_json=tc_text,
                          parsed_command=cmd.strip(), answer=None, error=None)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@dataclass
class TurnRecord:
    role: str
    content: str
    # For tool turns: keep a handle to the structured ToolResult too, so
    # downstream code can inspect exit_code, timed_out, etc.
    tool_result: Optional[dict] = None
    # For assistant turns: the parsed view.
    parse: Optional[dict] = None


@dataclass
class AgentRecord:
    id: str
    question: str
    gold_answers: list
    prediction: Optional[str] = None
    finish_reason: str = ""          # "answer" | "max_turns" | "tool_validation" | "parse_error" | "api_error"
    n_assistant_turns: int = 0
    n_tool_calls: int = 0
    error: Optional[str] = None
    turns: list = field(default_factory=list)   # list[TurnRecord-as-dict]
    messages: list = field(default_factory=list)  # final ChatML for replay
    # Per-example timing (seconds). Populated by run_agent_on_example.
    # llm_time_s = sum of _chat_completion latency across all assistant turns.
    # tool_time_s = sum of run_tool latency across all tool calls (includes
    # subprocess, engine fan-out, AND tokenizer truncation).
    # total_time_s = wall-clock from entering run_agent_on_example to leaving.
    llm_time_s: float = 0.0
    tool_time_s: float = 0.0
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "gold_answers": list(self.gold_answers),
            "prediction": self.prediction,
            "finish_reason": self.finish_reason,
            "n_assistant_turns": self.n_assistant_turns,
            "n_tool_calls": self.n_tool_calls,
            "error": self.error,
            "turns": self.turns,
            "messages": self.messages,
            "llm_time_s": self.llm_time_s,
            "tool_time_s": self.tool_time_s,
            "total_time_s": self.total_time_s,
        }


def _chat_completion(
    client,
    *,
    model: str,
    messages: list,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stop: Optional[list],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (text, finish_reason, error). On API error: (None, None, error_msg).

    Reconstructs the full assistant text including the <think>...</think> block.
    Why this matters: the Qwen3.5 chat template injects "<think>\\n" before
    generation, and vLLM's `--reasoning-parser qwen3` splits the response so:
      - choice.message.reasoning_content  = the thinking text
      - choice.message.content            = everything after </think>
    The SFT'd model was trained with <think> blocks in every assistant turn,
    so we want them visible in the recorded trajectory AND fed back as context
    to subsequent turns. Rebuild the full <think>...</think>\\n{content} string.
    """
    def _create(mt):
        return client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=mt,
            stop=stop,
            # NOTE: do NOT pass enable_thinking=False. The SFT data is
            # think-block-heavy and the model wants the prefix injection.
        )
    try:
        resp = _create(max_tokens)
    except Exception as e:
        # Context-length overflow: once the accumulated trajectory (system +
        # question + prior <think>/tool turns) is large, requesting a fixed
        # `max_tokens` output can exceed the model's context window and vLLM
        # rejects with a 400: "You passed N input tokens and requested M output
        # tokens, but the model's context length is only C". Don't kill the
        # example — clamp the output request to the room that's actually left
        # and retry once. Fires only on that specific error.
        import re as _re
        m = _re.search(
            r"passed\s+(\d+)\s+input tokens.*?context length is only\s+(\d+)",
            str(e), _re.DOTALL,
        )
        if m:
            n_in, ctx = int(m.group(1)), int(m.group(2))
            room = ctx - n_in - 8  # tiny safety margin
            if room <= 0:
                return None, None, f"API error: input {n_in} tokens >= context {ctx}, no room to generate"
            try:
                resp = _create(room)
            except Exception as e2:
                return None, None, f"API error (after clamp to {room}): {type(e2).__name__}: {e2}"
        else:
            return None, None, f"API error: {type(e).__name__}: {e}"
    if not resp.choices:
        return None, None, "API returned no choices"
    choice = resp.choices[0]
    content = choice.message.content or ""
    # vLLM versions disagree on the attribute name: older puts it in
    # `reasoning_content`, newer (0.17+) puts it in `reasoning`. Try both.
    reasoning = (getattr(choice.message, "reasoning_content", None)
                 or getattr(choice.message, "reasoning", None)
                 or "")
    if reasoning.strip():
        full = f"<think>\n{reasoning.strip()}\n</think>\n{content}"
    else:
        full = content
    return (full, choice.finish_reason or "", None)


def run_agent_on_example(
    example: dict,
    *,
    client,
    model: str,
    tokenizer,
    corpus_dir: str,
    max_assistant_turns: int = 6,
    max_tokens_per_turn: int = 2048,
    tool_max_tokens: int = 2048,
    tool_timeout: float = 60.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    engine=None,
) -> AgentRecord:
    """Run the agent on one example until <answer> or `max_assistant_turns`.

    The conversation matches SFT format:
      system: build_system_prompt()
      user:   "Query: {question}"
      assistant <-> tool turns
      (final assistant emits <answer>...</answer>)

    A malformed assistant turn (missing block, bad JSON, unknown tool) is fed
    back as a synthetic tool turn carrying a validation-error message — the
    same shape `run_tool` returns when `validate_pipeline` rejects a command.
    The model can then self-correct on the next turn. If it fails twice in a
    row we stop with finish_reason="parse_error".
    """
    ex_id = str(example.get("id", ""))
    question = example.get("question") or ""
    gold = list(example.get("golden_answers") or [])

    record = AgentRecord(id=ex_id, question=question, gold_answers=gold)
    _t_overall_start = time.perf_counter()

    system_prompt = build_system_prompt()
    messages: list = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Query: {question}"},
    ]
    record.turns.append({"role": "system", "content": system_prompt})
    record.turns.append({"role": "user", "content": messages[1]["content"]})

    consecutive_parse_errors = 0

    for turn_idx in range(max_assistant_turns):
        _t_llm = time.perf_counter()
        text, finish_reason, err = _chat_completion(
            client,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            # max_tokens_per_turn <= 0 means "no per-turn cap": pass None so vLLM
            # lets the turn generate up to the remaining context window. This
            # matches training, where the rollout had no per-turn cap (only the
            # 15360-token total response budget) — a fixed 2048 cap is a
            # train/inference mismatch that truncates long <think> turns
            # (finish_reason=length) into parse_errors.
            max_tokens=(max_tokens_per_turn if (max_tokens_per_turn or 0) > 0 else None),
            stop=None,
        )
        _llm_elapsed = time.perf_counter() - _t_llm
        record.llm_time_s += _llm_elapsed
        if err is not None:
            record.finish_reason = "api_error"
            record.error = err
            record.total_time_s = time.perf_counter() - _t_overall_start
            return record

        record.n_assistant_turns += 1
        messages.append({"role": "assistant", "content": text})

        parse = parse_assistant(text)
        record.turns.append({
            "role": "assistant",
            "content": text,
            "finish_reason": finish_reason,
            "elapsed_s": _llm_elapsed,
            "parse": {
                "tool_call_json": parse.tool_call_json,
                "parsed_command": parse.parsed_command,
                "answer": parse.answer,
                "error": parse.error,
            },
        })

        if parse.answer is not None:
            record.prediction = parse.answer
            record.finish_reason = "answer"
            record.messages = messages
            record.total_time_s = time.perf_counter() - _t_overall_start
            return record

        if parse.error is not None or parse.parsed_command is None:
            consecutive_parse_errors += 1
            if consecutive_parse_errors >= 2:
                record.finish_reason = "parse_error"
                record.error = parse.error or "unparseable assistant turn"
                record.messages = messages
                record.total_time_s = time.perf_counter() - _t_overall_start
                return record
            # Feed the model a synthetic tool-validation message so it can
            # self-correct. Same JSON shape as a real tool turn.
            err_payload = json.dumps({
                "command": "",
                "stdout": "",
                "stderr": f"format error: {parse.error or 'no tool_call or answer block'}",
                "exit_code": -2,
                "timed_out": False,
                "truncated": False,
                "information_lines": [],
            }, ensure_ascii=False)
            messages.append({"role": "tool", "content": err_payload})
            record.turns.append({"role": "tool", "content": err_payload, "synthetic": True})
            continue

        # Execute the tool.
        consecutive_parse_errors = 0
        _t_tool = time.perf_counter()
        tool_res: ToolResult = run_tool(
            parse.parsed_command,
            corpus_dir=corpus_dir,
            timeout=tool_timeout,
            max_tokens=tool_max_tokens,
            tokenizer=tokenizer,
            engine=engine,
        )
        _tool_elapsed = time.perf_counter() - _t_tool
        record.tool_time_s += _tool_elapsed
        record.n_tool_calls += 1
        payload = tool_res.to_payload()
        messages.append({"role": "tool", "content": payload})
        record.turns.append({
            "role": "tool",
            "content": payload,
            "elapsed_s": _tool_elapsed,
            "tool_result": {
                "command": tool_res.command,
                "exit_code": tool_res.exit_code,
                "timed_out": tool_res.timed_out,
                "truncated": tool_res.truncated,
            },
        })

    # Hit the turn cap.
    record.finish_reason = "max_turns"
    record.messages = messages
    record.total_time_s = time.perf_counter() - _t_overall_start
    return record
