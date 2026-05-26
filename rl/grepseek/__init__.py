"""GrepSeek RL integration package.

Custom components that plug into verl (used as the GRPO engine):
  - trainer.verl_integration.agent_loop   — the multi-turn search agent loop
  - trainer.verl_integration.search_corpus_tool — the `shell`/grep tool over the corpus
  - trainer.verl_integration.reward_function — the (format-gated) EM/F1 − length-penalty reward
  - trainer.verl_integration.dataset      — the QA-JSONL dataset adapter

These are registered with verl purely via the YAML configs under
`trainer/config/`; verl itself is vendored separately (see ../../verl).
"""
