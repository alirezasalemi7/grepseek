"""
Custom VERL agent loop for grepseek GRPO training.

Subclasses ToolAgentLoop with two modifications:
1. _handle_pending_state: passes tools=[] to apply_chat_template so the tool spec
   is NOT injected via Qwen3's native tool calling (it is already embedded in the
   system prompt text, identical to the inference pipeline).
2. run: records len(output.response_ids) as "response_token_count" in extra_fields
   so the reward function can compute the length-decay f(L).
"""

from __future__ import annotations

import logging
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, register
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentLoopOutput,
    AgentState,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.utils.rollout_trace import rollout_trace_op

from grepseek.prompting import build_final_user_message
from grepseek.trainer.verl_integration.invalid_tool_request import (
    InvalidToolRequest,
    append_terminal_invalid_tool_response,
    classify_invalid_tool_request,
)

logger = logging.getLogger(__name__)


@register("grepseek_agent")
class GrepSeekAgentLoop(ToolAgentLoop):
    """Agent loop for grepseek that keeps the tool spec in the system prompt.

    The system prompt already contains the shell tool spec (built identically
    to the inference pipeline via render_system_prompt). Passing tools=[] prevents
    the Qwen3 chat template from injecting a duplicate tool spec block.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The SFT-aligned SearchCorpusTool needs the trained model's tokenizer
        # to do token-based stdout truncation (matching SFT's tool_max_tokens=2048
        # behavior). The agent loop already loaded that tokenizer from
        # actor_rollout_ref.model.path during super().__init__, so just hand it
        # to each tool that asks for it. Zero config duplication.
        for tool in self.tools.values():
            if hasattr(tool, "set_tokenizer"):
                tool.set_tokenizer(self.tokenizer)

    async def process_vision_info(self, messages: list[dict]) -> dict:
        # Text-only task: skip image/video processing regardless of processor.
        return {}

    async def _handle_pending_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
    ) -> AgentState:
        """Tokenize the initial prompt without injecting tool schemas into the template."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=[],  # tool spec is in system prompt text; skip native injection
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
    ) -> AgentState:
        state = await super()._handle_generating_state(agent_data, sampling_params, ignore_termination)

        response_text = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
        )
        tool_calls = agent_data.tool_calls if state == AgentState.PROCESSING_TOOLS else []
        if not tool_calls and "<tool_call" in response_text:
            active_tools = getattr(agent_data, "_active_tools", self.tools)
            tools = [tool.tool_schema for tool in active_tools.values()]
            _, tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)
        invalid = self._classify_invalid_tool_request(
            response_text,
            tool_calls,
            getattr(agent_data, "_active_tools", self.tools),
        )
        if invalid is None:
            # Rescue prompt disabled. Previously, when a rollout hit the turn
            # budget with its last assistant turn ending in <tool_call> (no
            # <answer>), we appended a synthetic user turn "Retrieval stops
            # here. Give your single best answer..." to force one more
            # generation. Removed for the SFT-warm-start regime because:
            #   - SFT data always produces <answer> within the turn budget,
            #     so the model is trained to self-terminate;
            #   - empirically the rescued generations are mostly wrong
            #     (~6% conditional EM vs ~51% for self-terminated);
            #   - the injected "user\nRetrieval..." text confuses the format
            #     gate, silently zeroing 24% of rollouts' rewards.
            # Now a turn-budget exhaustion flows through as TERMINATED with no
            # <answer>, which the reward fn correctly classifies as
            # `failed_rollout` (score = failed_rollout_reward, default 0).
            return state

        return await self._terminate_invalid_tool_request(agent_data, invalid)

    def _classify_invalid_tool_request(
        self,
        response_text: str,
        tool_calls: list[FunctionCall],
        active_tools: dict[str, Any],
    ) -> InvalidToolRequest | None:
        return classify_invalid_tool_request(response_text, tool_calls, active_tools)

    async def _terminate_invalid_tool_request(
        self,
        agent_data: AgentData,
        invalid: InvalidToolRequest,
    ) -> AgentState:
        await append_terminal_invalid_tool_response(
            agent_data,
            invalid,
            apply_chat_template=self.apply_chat_template,
            response_length=self.response_length,
        )
        return AgentState.TERMINATED

    async def _append_final_answer_prompt(self, agent_data: AgentData) -> AgentState:
        response_ids = await self.apply_chat_template(
            [{"role": "user", "content": build_final_user_message(sentinel="INSUFFICIENT_EVIDENCE")}],
            remove_system_prompt=True,
        )
        remaining = self.response_length - len(agent_data.response_mask)
        if remaining <= 0:
            return AgentState.TERMINATED

        response_ids = response_ids[:remaining]
        logger.info("Appending forced final-answer prompt after valid tool call at turn budget.")
        agent_data.extra_fields["forced_final_prompt_appended"] = True
        agent_data.messages.append(
            {"role": "user", "content": build_final_user_message(sentinel="INSUFFICIENT_EVIDENCE")}
        )
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        agent_data.tool_calls = []
        return AgentState.GENERATING

    @rollout_trace_op
    async def run(
        self,
        sampling_params: dict[str, Any],
        **kwargs: Any,
    ) -> AgentLoopOutput:
        """Run the agent loop and record response token stats in extra_fields."""
        output: AgentLoopOutput = await super().run(sampling_params, **kwargs)
        n_tokens = len(output.response_ids)
        output.extra_fields["response_token_count"] = n_tokens
        # Flag whether the trajectory was cut off at the response_length budget.
        # The reward function uses this to distinguish a natural termination from
        # a hard truncation (model was still mid-search when context ran out).
        output.extra_fields["hit_max_context"] = n_tokens >= self.response_length
        output.extra_fields.setdefault("invalid_tool_request", False)
        output.extra_fields.setdefault("invalid_tool_call_count", 0)
        output.extra_fields.setdefault("invalid_tool_request_reason", "")
        output.extra_fields.setdefault("invalid_tool_request_detail", "")
        return output
