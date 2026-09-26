# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    """apply_chat_template to messages with special attention to template requiring
    at least one user message, e.g. Qwen3.5.

    Args:
        processor: tokenizer or processor.
        messages: list[dict], messages.
        tokenize: bool, whether to tokenize the output.
        add_generation_prompt: bool, whether to add generation prompt.
        tools: list[dict], tools schema.
        return_dict: bool, whether to return a dict.
        **kwargs: additional arguments for apply_chat_template.

    Returns:
        list[int] | str: tokenized ids or text string.
    """
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
    except Exception:
        # Some templates (notably Qwen3.5) require both:
        #   (a) at least one user message in the messages list, AND
        #   (b) any system message to be at index 0.
        # Single-message tokenization (used by MultiTurnSFTDataset's per-turn
        # rendering) violates (a) for system/assistant/tool-only turns. The
        # original fallback prepended a dummy user, which works for assistant
        # / tool single-message turns but breaks (b) when the input starts
        # with a system message — putting system at index 1 raises "System
        # message must be at the beginning."
        # Fix: pick prepend vs append based on the leading role.
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        starts_with_system = (
            bool(messages)
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "system"
        )
        if starts_with_system:
            # Append the dummy user so system stays at index 0. The dummy's
            # trailing contribution (including any generation prompt) is
            # measured by rendering it alone with the SAME add_generation_prompt
            # setting and stripped from the end of the full render.
            extended = list(messages) + dummy_user_message
            dummy_render = processor.apply_chat_template(
                dummy_user_message,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )
        else:
            # Original behavior: prepend the dummy user. Its leading
            # contribution does NOT include a generation prompt (the gen
            # prompt only appears at the very end of the full render).
            extended = dummy_user_message + list(messages)
            dummy_render = processor.apply_chat_template(
                dummy_user_message,
                tokenize=tokenize,
                add_generation_prompt=False,
                tools=tools,
                return_dict=return_dict,
                **kwargs,
            )
        output = processor.apply_chat_template(
            extended,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )

        def _strip(seq, dummy):
            return seq[: len(seq) - len(dummy)] if starts_with_system else seq[len(dummy):]

        if not tokenize:  # tokenize=False -> str
            return _strip(output, dummy_render)
        elif not return_dict:  # tokenize=True and return_dict=False -> list[int]
            if isinstance(output[0], list):  # transformers>=5
                assert len(output) == 1, "output must be a list[int] or list[list[int]]"
                dummy_render = dummy_render[0]
                output = output[0]
            return _strip(output, dummy_render)
        else:  # tokenize=True and return_dict=True and return_tensors="pt"
            dummy_render = dict(dummy_render)
            output = dict(output)
            dummy_len = dummy_render["input_ids"].shape[1]
            if starts_with_system:
                n = output["input_ids"].shape[1]
                output["input_ids"] = output["input_ids"][:, : n - dummy_len]
                output["attention_mask"] = output["attention_mask"][:, : n - dummy_len]
                if "mm_token_type_ids" in output:
                    output["mm_token_type_ids"] = output["mm_token_type_ids"][:, : n - dummy_len]
            else:
                output["input_ids"] = output["input_ids"][:, dummy_len:]
                output["attention_mask"] = output["attention_mask"][:, dummy_len:]
                if "mm_token_type_ids" in output:
                    output["mm_token_type_ids"] = output["mm_token_type_ids"][:, dummy_len:]
            return output
