from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InvalidToolRequest:
    reason: str
    detail: str
    raw_tool_request: str = ""


def classify_invalid_tool_request(
    response_text: str,
    tool_calls: list[Any],
    active_tools: dict[str, Any],
) -> InvalidToolRequest | None:
    if not tool_calls:
        if looks_like_attempted_tool_call(response_text):
            return InvalidToolRequest(
                reason="malformed_tool_call",
                detail="malformed <tool_call> block",
                raw_tool_request=extract_attempted_tool_request(response_text),
            )
        return None

    tool_call = tool_calls[0]
    if tool_call.name != "shell":
        return InvalidToolRequest(
            reason="wrong_tool_name",
            detail=f"wrong tool name {tool_call.name!r}; expected 'shell'",
            raw_tool_request=format_parsed_tool_call(tool_call),
        )

    try:
        arguments = json.loads(tool_call.arguments)
    except (json.JSONDecodeError, TypeError):
        return InvalidToolRequest(
            reason="invalid_arguments",
            detail="arguments must be valid JSON",
            raw_tool_request=format_parsed_tool_call(tool_call),
        )

    if not isinstance(arguments, dict):
        return InvalidToolRequest(
            reason="invalid_arguments",
            detail="arguments must be an object",
            raw_tool_request=format_parsed_tool_call(tool_call),
        )

    if "command" not in arguments:
        return InvalidToolRequest(
            reason="missing_command",
            detail="missing arguments.command",
            raw_tool_request=format_parsed_tool_call(tool_call),
        )

    command = arguments["command"]
    if not isinstance(command, str):
        return InvalidToolRequest(
            reason="non_string_command",
            detail="arguments.command must be a string",
            raw_tool_request=format_parsed_tool_call(tool_call),
        )
    if not command.strip():
        return InvalidToolRequest(
            reason="empty_command",
            detail="arguments.command must be non-empty",
            raw_tool_request=format_parsed_tool_call(tool_call),
        )

    # SearchCorpusTool validates the pipeline inside execute() via
    # _validate_pipeline (errors surface as an exit_code=-2 JSON payload), so
    # there is no command-level pre-check here — invalid commands are caught at
    # exec time rather than at request-classification time.
    return None


async def append_terminal_invalid_tool_response(
    agent_data: Any,
    invalid: InvalidToolRequest,
    *,
    apply_chat_template: Callable[..., Awaitable[list[int]]],
    response_length: int,
) -> None:
    count = int(agent_data.extra_fields.get("invalid_tool_call_count", 0) or 0) + 1
    logger.warning(
        "Invalid tool request: reason=%s detail=%s raw_tool_request=%r",
        invalid.reason,
        invalid.detail,
        invalid.raw_tool_request,
    )
    agent_data.extra_fields.update(
        {
            "invalid_tool_request": True,
            "invalid_tool_call_count": count,
            "invalid_tool_request_reason": invalid.reason,
            "invalid_tool_request_detail": invalid.detail,
        }
    )
    agent_data.tool_calls = []

    response_ids = await apply_chat_template(
        [{"role": "tool", "content": format_invalid_tool_response(invalid)}],
        remove_system_prompt=True,
    )
    remaining = response_length - len(agent_data.response_mask)
    if remaining <= 0:
        return

    response_ids = response_ids[:remaining]
    agent_data.prompt_ids += response_ids
    agent_data.response_mask += [0] * len(response_ids)
    if agent_data.response_logprobs:
        agent_data.response_logprobs += [0.0] * len(response_ids)
    agent_data.user_turns += 1


def looks_like_attempted_tool_call(response_text: str) -> bool:
    return "<tool_call" in response_text or "</tool_call>" in response_text


def extract_attempted_tool_request(response_text: str) -> str:
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", response_text, flags=re.DOTALL)
    if matches:
        return matches[-1]

    start = response_text.rfind("<tool_call")
    if start >= 0:
        return response_text[start:]

    end = response_text.rfind("</tool_call>")
    if end >= 0:
        return response_text[: end + len("</tool_call>")]

    return response_text


def format_parsed_tool_call(tool_call: Any) -> str:
    return json.dumps(
        {
            "name": getattr(tool_call, "name", None),
            "arguments": getattr(tool_call, "arguments", None),
        },
        ensure_ascii=False,
    )


def format_invalid_tool_response(invalid: InvalidToolRequest) -> str:
    return f"Error: invalid tool request: {invalid.detail}."
