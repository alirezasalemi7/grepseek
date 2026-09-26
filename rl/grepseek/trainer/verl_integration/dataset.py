"""
Custom VERL dataset for grepseek GRPO training.

Loads JSONL QA examples (HotpotQA, MuSiQue, NQ, or any other dataset whose
records contain `id`/`qid`, `question`/`query`, and `golden_answers` fields)
and formats each one as a VERL-compatible dict with raw_prompt, reward_model,
and extra_info. The system prompt is built identically to the inference
pipeline (tool spec embedded in system prompt text).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _build_tool_spec_json() -> str:
    """Return the `shell` tool spec JSON string. Renamed from search_corpus to
    match the tool name the SFT'd model was trained to emit."""
    tool_spec = {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run a single-pipeline shell command against the local Wikipedia "
                "corpus (corpus.jsonl in the cwd). Allowed commands: rg, grep, "
                "find, sed, awk, head, tail, cat, ls, wc, sort, cut, uniq, tr. "
                "Pipes (|) are allowed. Redirection, command chaining, and "
                "command substitution are not."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "A single shell pipeline to execute against corpus.jsonl.",
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
    return json.dumps(tool_spec, ensure_ascii=False, indent=2)


class GrepSeekDataset(Dataset):
    """Custom Dataset for grepseek GRPO training that bypasses VERL's default
    RLHFDataset. Dataset-agnostic — works on HotpotQA, MuSiQue, NQ, or any JSONL
    whose records have `id`/`qid`, `question`/`query`, and `golden_answers`.

    Implements the torch.utils.data.Dataset interface required by VERL's custom_cls
    config. Builds prompts identically to the inference pipeline: tool spec is
    embedded in the system prompt text (not injected via the chat template's
    native tool calling).
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer,
        config,
        processor=None,
        max_samples: int = -1,
        is_train: bool = True,
    ) -> None:
        """
        Args:
            data_files: Path to a QA JSONL file (HotpotQA, MuSiQue, NQ, etc.).
            tokenizer: HuggingFace tokenizer (required by VERL interface; not used directly).
            config: DictConfig from Hydra containing dataset configuration keys.
            processor: Unused (required by VERL interface).
            max_samples: Max samples from VERL's side (-1 = no limit).
            is_train: True for training dataset, False for validation.

        Config keys (read from YAML data section):
            system_prompt_path: Path to prompts/system_prompt.txt
            retrieval_instruction_path: Path to prompts/output_format_instruction.txt
            tool_spec_template_path: Path to prompts/tool_spec_template.txt
            answerable_only: bool — filter out unanswerable examples (default True)
            debug_mode_train / debug_mode_val: bool — use small subset
            debug_num_samples_train / debug_num_samples_val: int — subset size
        """
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train

        # Resolve data file path
        if isinstance(data_files, list):
            data_path = data_files[0]
        else:
            data_path = data_files

        # Generic JSONL loader — accepts any record with id/qid + question/query +
        # golden_answers (FlashRAG-style). Works for NQ, HotpotQA, MuSiQue, etc.
        # without modification; the paper's RL uses NQ + HotpotQA.
        from grepseek.data.qa import load_qa_examples

        raw_examples = load_qa_examples(data_path)

        # Filter unanswerable examples
        answerable_only = bool(config.get("answerable_only", True))
        if answerable_only:
            before = len(raw_examples)
            raw_examples = [ex for ex in raw_examples if ex.golden_answers]
            logger.info(f"answerable_only filter: {before} → {len(raw_examples)} examples")

        # Debug-mode subsetting
        debug_mode_key = "debug_mode_train" if is_train else "debug_mode_val"
        debug_num_key = "debug_num_samples_train" if is_train else "debug_num_samples_val"
        debug_mode = bool(config.get(debug_mode_key, False))
        if debug_mode:
            debug_num = int(config.get(debug_num_key, config.get("debug_num_samples", 32)))
            raw_examples = raw_examples[:debug_num]
            logger.info(f"debug_mode: using {len(raw_examples)} examples")

        self.examples = raw_examples
        logger.info(f"GrepSeekDataset ({'train' if is_train else 'val'}): {len(self.examples)} examples from {data_path}")

        # Build system prompt (done once; same for all examples)
        self._system_prompt = self._build_system_prompt(config)

    def _build_system_prompt(self, config) -> str:
        """Build system prompt with tool spec embedded, matching inference pipeline format."""
        from grepseek.prompting import PromptAssets, render_system_prompt

        system_prompt_path = Path(config.get("system_prompt_path"))
        retrieval_instruction_path = Path(config.get("retrieval_instruction_path"))
        tool_spec_template_path = Path(config.get("tool_spec_template_path"))

        assets = PromptAssets.from_files(
            system_prompt_path=system_prompt_path,
            retrieval_instruction_path=retrieval_instruction_path,
            tool_spec_template_path=tool_spec_template_path,
        )
        tool_spec_json = _build_tool_spec_json()
        return render_system_prompt(assets=assets, tool_spec=tool_spec_json)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from grepseek.prompting import build_initial_messages

        example = self.examples[idx]

        raw_prompt = build_initial_messages(
            query=example.query,
            system_prompt=self._system_prompt,
            retrieval_instruction="",  # already included in system_prompt via render_system_prompt
        )

        return {
            "raw_prompt": raw_prompt,
            "data_source": example.qid,
            "extra_info": {
                "qid": example.qid,
                "golden_answers": example.golden_answers,
                "sample_index": idx,
                # Mark val items so the reward fn can disable length decay at
                # eval time (validation should report raw EM only).
                "is_validation": not self.is_train,
            },
            "reward_model": {
                "ground_truth": {
                    "qid": example.qid,
                    "golden_answers": example.golden_answers,
                }
            },
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "index": idx,
        }
