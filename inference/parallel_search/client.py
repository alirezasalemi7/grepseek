"""DaemonClient — duck-typed substitute for ShardedSearchEngine that proxies
requests to a daemon over a Unix socket.

tools.py passes either an in-process ShardedSearchEngine or a DaemonClient as
the `engine` kwarg; both expose the same `.execute(command, env=None)` API,
so the call site in tools.run_tool is identical for both backends.
"""
from __future__ import annotations

import os
import socket
from typing import Optional

from .daemon import recv_msg, send_msg
from .engine import EngineStats


class DaemonClient:
    """Connect to a search daemon. Each .execute() opens a fresh connection
    (Unix-socket connect is ~50µs — negligible vs the engine call latency).

    Why per-call connects instead of a pool: the eval uses many worker
    threads in parallel; a connection pool would need synchronization and
    bounded-pool eviction. Per-call is dead-simple and the overhead is
    well below the noise floor of a tool call.
    """

    def __init__(self, socket_path: str, timeout: float = 65.0):
        if not os.path.exists(socket_path):
            raise FileNotFoundError(
                f"search daemon socket not found at {socket_path}. "
                f"Either the daemon isn't running, or the path is wrong. "
                f"Launch via scripts/rl/eval_rl_fast.sh (which auto-spawns "
                f"a daemon when ENGINE_MODE=daemon)."
            )
        # Fail-fast: open a probe connection so misconfigurations surface
        # at run_eval startup rather than mid-eval.
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(2.0)
        try:
            probe.connect(socket_path)
        except OSError as exc:
            raise ConnectionError(
                f"search daemon socket exists at {socket_path} but connect "
                f"failed: {exc}. Is the daemon healthy?"
            )
        probe.close()
        self._socket_path = socket_path
        self._timeout = timeout
        self.last_stats = EngineStats()

    @property
    def n_shards(self) -> int:
        # Client doesn't know the daemon's shard count without a round-trip
        # to a metadata endpoint (not implemented; not needed by callers).
        return 0

    def describe(self) -> str:
        return f"DaemonClient(socket={self._socket_path!r})"

    def execute(self, command: str, env: Optional[dict] = None
                ) -> tuple[str, int]:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect(self._socket_path)
        try:
            # Only ship env vars that DIFFER from the daemon's environment
            # (or are missing from os.environ). Shipping all of os.environ
            # on every call wastes bytes and could leak credentials to a log.
            env_overrides = None
            if env:
                env_overrides = {
                    k: v for k, v in env.items()
                    if os.environ.get(k) != v
                }
                if not env_overrides:
                    env_overrides = None
            send_msg(sock, {"command": command, "env_overrides": env_overrides})
            resp = recv_msg(sock)
            if resp is None:
                # Daemon closed without sending a reply.
                self.last_stats = EngineStats(
                    strategy="error", fallback_reason="daemon closed connection"
                )
                return "", -1
            self.last_stats = EngineStats(
                strategy=resp.get("strategy", ""),
                n_shards_used=0,  # daemon-side info; not surfaced over the wire
                fallback_reason=resp.get("fallback_reason", ""),
            )
            return resp.get("stdout", ""), int(resp.get("returncode", -1))
        except socket.timeout:
            self.last_stats = EngineStats(
                strategy="error", fallback_reason="client socket timeout"
            )
            return "", -1
        finally:
            try:
                sock.close()
            except OSError:
                pass
