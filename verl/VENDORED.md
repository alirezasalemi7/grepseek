# Vendored verl

This directory is a **vendored copy of [verl](https://github.com/volcengine/verl)**
(Volcano Engine Reinforcement Learning for LLMs), licensed under Apache-2.0 (see
[`LICENSE`](LICENSE)).

It is included verbatim (not as a submodule) so the GrepSeek training stages
reproduce **exactly the verl version used in the paper**, with no dependency on
an external fork remaining available. It is the same code that produced the
paper's SFT and RL checkpoints, including our local fixes (e.g. the
Ulysses sequence-parallel / `remove_padding` path).

**Modifications vs. upstream verl:** small, project-specific changes in the
worker/trainer paths. GrepSeek's own additions (the parallel-grep search tool,
the F1 reward, and the agent loop) live **outside** this directory in the
`grepseek` package under [`../rl`](../rl) and are registered with verl purely via
config — verl is used as an engine.

To keep the repository lean, non-essential upstream directories were removed
from this copy: `docs/`, `examples/`, `recipe/`, `tests/`, `.github/`,
`docker/`. The `verl/` Python package itself is complete and unmodified in
structure (405 source files). For those materials, see upstream verl.
