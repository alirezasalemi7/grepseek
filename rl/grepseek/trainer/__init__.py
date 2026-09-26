# Importing agent_loop triggers @register("grepseek_agent") at import time.
# verl's AgentLoopWorker looks up the agent loop by name from the registry, so
# this import must run before the lookup. The training launcher does
# `import grepseek.trainer` (or relies on the YAML _target_ in agent_loops.yaml).
from grepseek.trainer.verl_integration import agent_loop  # noqa: F401
