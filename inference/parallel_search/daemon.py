"""Long-running search-engine daemon.

Holds one ShardedSearchEngine in-process; accepts shell-command requests over
a Unix domain socket; returns results in the same shape the in-proc engine
produces. Threaded handler — the engine is already thread-safe (each call
spawns its own subprocess set).

Why it exists: scaffolding for an eventual indexed search backend (Tantivy /
Zoekt). Without an index, the daemon adds ~1-3ms of Python wrapper-overhead
savings per call vs in-proc, which is real but not headline. The architectural
win is that swapping in an index later requires no client-side changes.

Protocol: length-prefixed JSON, both directions.
    <8-byte big-endian uint64 length><utf-8 JSON body>

Request body:
    {"command": str, "env_overrides": dict | None}
Response body:
    {"stdout": str, "returncode": int, "strategy": str, "fallback_reason": str}

Launch standalone:
    python -m inference.parallel_search.daemon \\
        --socket /tmp/search.sock \\
        --shard_dir /scratch/.../shards_16 \\
        --corpus /scratch/.../wiki_corpus.jsonl

Or auto-spawned by scripts/rl/eval_rl_fast.sh.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import socketserver
import struct
import sys
import threading
from typing import Optional

from .engine import ShardedSearchEngine

logger = logging.getLogger(__name__)

# Wire-protocol: 8-byte unsigned big-endian length prefix + utf-8 JSON body.
_LEN_FMT = "!Q"
_LEN_SIZE = struct.calcsize(_LEN_FMT)


# ---------------------------------------------------------------------------
# Wire helpers (used by both daemon and client)
# ---------------------------------------------------------------------------

def send_msg(sock: socket.socket, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    sock.sendall(struct.pack(_LEN_FMT, len(body)))
    sock.sendall(body)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed mid-message")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock: socket.socket) -> Optional[dict]:
    """Read one length-prefixed JSON message. Returns None on clean EOF."""
    try:
        header = sock.recv(_LEN_SIZE, socket.MSG_WAITALL)
    except (ConnectionError, OSError):
        return None
    if len(header) == 0:
        return None
    if len(header) < _LEN_SIZE:
        # Partial header → finish with the exact-read helper.
        header = bytes(header) + _recv_exact(sock, _LEN_SIZE - len(header))
    (length,) = struct.unpack(_LEN_FMT, header)
    body = _recv_exact(sock, length)
    return json.loads(body.decode("utf-8"))


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class _Handler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sock: socket.socket = self.request
        # Keep the connection open for many requests (typical eval worker
        # reuses the same socket). A dropped peer closes the loop cleanly.
        while True:
            try:
                msg = recv_msg(sock)
            except (json.JSONDecodeError, ConnectionError, OSError) as exc:
                logger.debug(f"daemon: recv failed: {exc}")
                return
            if msg is None:
                return
            cmd = msg.get("command")
            env_overrides = msg.get("env_overrides")  # dict or None
            if not isinstance(cmd, str):
                self._reply_error(sock, "missing or non-string 'command' field")
                continue
            env = None
            if env_overrides:
                env = {**os.environ, **env_overrides}
            try:
                stdout, rc = self.server.engine.execute(cmd, env=env)
                stats = self.server.engine.last_stats
                send_msg(sock, {
                    "stdout": stdout,
                    "returncode": rc,
                    "strategy": stats.strategy,
                    "fallback_reason": stats.fallback_reason,
                })
            except Exception as exc:
                logger.exception("daemon: engine.execute raised")
                self._reply_error(sock, f"engine error: {type(exc).__name__}: {exc}")

    @staticmethod
    def _reply_error(sock: socket.socket, reason: str) -> None:
        try:
            send_msg(sock, {
                "stdout": "",
                "returncode": -2,
                "strategy": "error",
                "fallback_reason": reason,
            })
        except (OSError, BrokenPipeError):
            pass


class _ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: str, RequestHandlerClass, engine):
        super().__init__(server_address, RequestHandlerClass)
        self.engine = engine


def serve(socket_path: str, engine: ShardedSearchEngine,
          pid_file: Optional[str] = None) -> _ThreadedUnixServer:
    """Start serving on `socket_path`. Blocks until SIGTERM/SIGINT.

    Returns the server object (for tests that want to call .shutdown() from
    another thread). In production, this function does not return — the
    signal handlers call .shutdown() and then we clean up.
    """
    # If a stale socket file exists (previous crashed run), remove it. Don't
    # touch a socket that's actively in use — bind would fail and we'd notice.
    if os.path.exists(socket_path):
        try:
            os.unlink(socket_path)
        except OSError:
            pass

    server = _ThreadedUnixServer(socket_path, _Handler, engine)
    # Lock down the socket so other cluster users can't connect.
    try:
        os.chmod(socket_path, 0o600)
    except OSError:
        pass

    if pid_file:
        try:
            with open(pid_file, "w", encoding="utf-8") as f:
                f.write(f"{os.getpid()}\n")
        except OSError as exc:
            logger.warning(f"daemon: failed to write pid_file {pid_file}: {exc}")

    def _shutdown(signum, _frame) -> None:
        logger.info(f"daemon: signal {signum} received, shutting down")
        # serve_forever's shutdown must run on a non-server thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(f"daemon: serving on {socket_path} ({engine.describe()})")
    try:
        server.serve_forever()
    finally:
        try:
            os.unlink(socket_path)
        except OSError:
            pass
        if pid_file:
            try:
                os.unlink(pid_file)
            except OSError:
                pass
        logger.info("daemon: stopped")
    return server


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--socket", required=True, help="Unix socket path to bind")
    p.add_argument("--shard_dir", required=True,
                   help="directory containing pre-built shards + manifest.json")
    p.add_argument("--corpus", required=True,
                   help="absolute path to fallback corpus file (e.g. wiki_corpus.jsonl)")
    p.add_argument("--timeout", type=float, default=60.0,
                   help="per-call subprocess timeout (seconds)")
    p.add_argument("--engine_log", default=None,
                   help="optional jsonl path for per-call engine telemetry")
    p.add_argument("--pid_file", default=None,
                   help="optional path to write the daemon's PID")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    engine = ShardedSearchEngine(
        shard_dir=args.shard_dir,
        fallback_corpus_file=args.corpus,
        timeout=args.timeout,
        log_path=args.engine_log,
    )
    serve(args.socket, engine, pid_file=args.pid_file)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
