"""Multi-server LLM pool with hot-reload, per-server caps, health tracking, and
sticky per-example affinity.

The pool replaces the single-client path in `ServerLLM` so the pipeline can talk
to N vLLM servers. Servers are listed in a JSON file:

    {
      "reload_interval_s": 3600,
      "servers": [
        {"host": "...", "port": 9346, "model": "qwen3.5-27b", "max_in_flight": 9},
        ...
      ]
    }

A daemon thread re-reads the file periodically (mtime-checked) and applies a
diff: new entries are added, removed entries are drained, changed `max_in_flight`
caps are resized live.

Per-server in-flight throttling is implemented with a single `Condition` over
plain ints (`in_flight`, `cap`) — `Semaphore` cannot be resized, this can.

On a retryable error (transport/timeout/5xx/408/429) the pool marks the server
unhealthy with an exponential cooldown (30, 60, 120, 240, 480, 600 s) and the
caller's `chat()` loop retries on a different healthy server. A successful call
clears the streak.

Sticky affinity: the caller (`process_one`) sets `_example_id_var` (a
ContextVar) at the top of each example. The pool reads it during selection and
prefers the server previously used for that example, when that server is
eligible. Soft only — never waits for the preferred server when others are
free. The mapping is dropped via `pool.forget(id)` when the example finishes.
"""
import contextvars
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI


# ContextVar set by process_one() at the top of each example. Read by chat()
# for sticky-affinity routing.
_example_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_example_id_var", default=None
)


@dataclass
class _ServerEntry:
    host: str
    port: int
    model: str
    cap: int
    in_flight: int = 0
    client: Optional[OpenAI] = None
    cooldown_until: float = 0.0   # monotonic time; 0 means healthy
    fail_streak: int = 0
    draining: bool = False


def _server_key(host: str, port: int, model: str) -> str:
    return f"{host}:{port}:{model}"


def _build_client(host: str, port: int, api_key: str = "EMPTY") -> OpenAI:
    return OpenAI(api_key=api_key, base_url=f"http://{host}:{port}/v1")


def _is_retryable(exc: Exception) -> bool:
    """Transport errors, timeouts, 5xx, 408, 429 -> retryable. Other 4xx -> not."""
    code = getattr(exc, "status_code", None)
    if code is None:
        # No HTTP status: connection/timeout/parse-level error. Retry.
        return True
    if code >= 500 or code in (408, 429):
        return True
    return False


class ServerPool:
    """A pool of vLLM servers with health tracking and per-example affinity.

    The public surface is `chat(messages, sampling_params)` which mirrors
    `ServerLLM.chat(...)` so callers don't change. `forget(id)` is called by
    `process_one` after each example to release that example's affinity slot.
    """

    def __init__(
        self,
        servers_path: Optional[str],
        fallback_servers: Optional[list] = None,
        default_reload_interval_s: int = 3600,
        cooldown_schedule: tuple = (30, 60, 120, 240, 480, 600),
        affinity_max: int = 4096,
        max_total_wait_s: int = 600,
    ):
        self._servers_path = servers_path
        self._reload_interval_s = int(default_reload_interval_s)
        self.cooldown_schedule = tuple(int(x) for x in cooldown_schedule)
        if not self.cooldown_schedule:
            raise ValueError("cooldown_schedule must be non-empty")
        self.affinity_max = int(affinity_max)
        self.max_total_wait_s = float(max_total_wait_s)

        self._cv = threading.Condition()
        self._servers: "OrderedDict[str, _ServerEntry]" = OrderedDict()
        self._affinity: "OrderedDict[str, str]" = OrderedDict()

        self._stop = threading.Event()
        self._reloader_thread: Optional[threading.Thread] = None
        self._last_mtime: Optional[float] = None

        if servers_path is not None:
            cfg = self._read_config_file(servers_path)
            interval = cfg.get("reload_interval_s", default_reload_interval_s)
            if isinstance(interval, int) and interval >= 60:
                self._reload_interval_s = interval
            self._validate_and_seed(cfg.get("servers") or [])
            self._last_mtime = os.path.getmtime(servers_path)
        elif fallback_servers:
            self._validate_and_seed(fallback_servers)
        else:
            raise ValueError("ServerPool: provide servers_path or fallback_servers")

    # ------------------------------------------------------------------ public

    def chat(self, messages: list, sampling_params) -> "ChatResponse":
        from .llm import _post_process_response  # local import to avoid cycle
        example_id = _example_id_var.get(None)
        last_err: Optional[Exception] = None
        deadline = time.monotonic() + self.max_total_wait_s
        attempts = 0

        while time.monotonic() < deadline:
            attempts += 1
            entry: Optional[_ServerEntry] = None
            with self._cv:
                while True:
                    self._sweep_drained_locked()
                    cand = self._pick_locked(example_id)
                    if cand is not None:
                        cand.in_flight += 1
                        entry = cand
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=min(remaining, 1.0))
                if entry is None:
                    break

            try:
                raw = self._do_request(entry, messages, sampling_params)
            except Exception as e:
                retryable = _is_retryable(e)
                with self._cv:
                    entry.in_flight -= 1
                    if retryable:
                        self._mark_unhealthy_locked(entry)
                    self._maybe_drop_drained_locked(entry)
                    self._cv.notify_all()
                if not retryable:
                    print(
                        f"[ServerPool] {self._entry_key(entry)} non-retryable "
                        f"error: {type(e).__name__}: {e}"
                    )
                    return _post_process_response(None)
                cd = self.cooldown_schedule[
                    min(entry.fail_streak - 1, len(self.cooldown_schedule) - 1)
                ]
                print(
                    f"[ServerPool] {self._entry_key(entry)} failed "
                    f"({type(e).__name__}); cooldown {cd}s; trying another server"
                )
                last_err = e
                continue

            with self._cv:
                entry.in_flight -= 1
                if entry.cooldown_until > 0:
                    print(f"[ServerPool] {self._entry_key(entry)} recovered")
                    entry.cooldown_until = 0.0
                entry.fail_streak = 0
                self._maybe_drop_drained_locked(entry)
                self._cv.notify_all()
            return _post_process_response(raw)

        print(
            f"[ServerPool] giving up after {attempts} attempt(s) / "
            f"{self.max_total_wait_s}s wait: {last_err}"
        )
        return _post_process_response(None)

    def forget(self, example_id: str) -> None:
        if not example_id:
            return
        with self._cv:
            self._affinity.pop(example_id, None)

    def start_reloader(self) -> None:
        if self._servers_path is None or self._reloader_thread is not None:
            return
        self._reloader_thread = threading.Thread(
            target=self._reloader_loop, daemon=True, name="ServerPoolReloader"
        )
        self._reloader_thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._reloader_thread is not None:
            self._reloader_thread.join(timeout=5)

    def describe(self) -> str:
        with self._cv:
            n = len(self._servers)
            cap_total = sum(e.cap for e in self._servers.values() if not e.draining)
            parts = [f"{self._entry_key(e)}(cap={e.cap})" for e in self._servers.values()]
        return f"{n} server(s), total cap={cap_total} [{', '.join(parts)}]"

    def total_capacity(self) -> int:
        """Sum of max_in_flight across all non-draining servers — i.e. the
        maximum number of concurrent LLM calls the pool can have in flight
        right now."""
        with self._cv:
            return sum(e.cap for e in self._servers.values() if not e.draining)

    def total_in_flight(self) -> int:
        """Sum of in-flight LLM calls across all non-draining servers right now."""
        with self._cv:
            return sum(e.in_flight for e in self._servers.values() if not e.draining)

    def primary_model_name(self) -> str:
        with self._cv:
            for e in self._servers.values():
                return e.model
        return ""

    @property
    def reload_interval_s(self) -> int:
        return self._reload_interval_s

    # ------------------------------------------------------------- internals

    @staticmethod
    def _entry_key(e: _ServerEntry) -> str:
        return _server_key(e.host, e.port, e.model)

    @staticmethod
    def _read_config_file(path: str) -> dict:
        with open(path) as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"servers config {path}: top level must be a dict")
        if not isinstance(cfg.get("servers"), list) or not cfg.get("servers"):
            raise ValueError(f"servers config {path}: 'servers' must be a non-empty list")
        return cfg

    def _validate_and_seed(self, server_list: list) -> None:
        seen = set()
        for s in server_list:
            host = s["host"]
            port = int(s["port"])
            model = s["model"]
            cap = int(s.get("max_in_flight", 1))
            if not isinstance(host, str) or not host:
                raise ValueError(f"servers: bad host {host!r}")
            if not (1 <= port <= 65535):
                raise ValueError(f"servers: bad port {port}")
            if not isinstance(model, str) or not model:
                raise ValueError(f"servers: bad model {model!r}")
            if cap < 1:
                raise ValueError(f"servers: max_in_flight must be >= 1 (got {cap})")
            key = _server_key(host, port, model)
            if key in seen:
                raise ValueError(f"servers: duplicate entry {key}")
            seen.add(key)
            self._servers[key] = _ServerEntry(
                host=host, port=port, model=model, cap=cap,
                client=_build_client(host, port),
            )

    def _pick_locked(self, example_id: Optional[str]) -> Optional[_ServerEntry]:
        now = time.monotonic()
        eligible = [
            e for e in self._servers.values()
            if not e.draining and now >= e.cooldown_until and e.in_flight < e.cap
        ]
        if not eligible:
            return None
        if example_id is not None:
            sticky_key = self._affinity.get(example_id)
            if sticky_key is not None:
                for e in eligible:
                    if self._entry_key(e) == sticky_key:
                        self._affinity.move_to_end(example_id)
                        return e
        eligible.sort(key=lambda e: (e.in_flight, self._entry_key(e)))
        chosen = eligible[0]
        if example_id is not None:
            self._affinity[example_id] = self._entry_key(chosen)
            self._affinity.move_to_end(example_id)
            while len(self._affinity) > self.affinity_max:
                self._affinity.popitem(last=False)
        return chosen

    def _mark_unhealthy_locked(self, entry: _ServerEntry) -> None:
        entry.fail_streak = min(entry.fail_streak + 1, len(self.cooldown_schedule))
        delay = self.cooldown_schedule[entry.fail_streak - 1]
        entry.cooldown_until = time.monotonic() + delay

    def _sweep_drained_locked(self) -> None:
        drop = [k for k, e in self._servers.items() if e.draining and e.in_flight == 0]
        for k in drop:
            del self._servers[k]
            print(f"[ServerPool] removed {k} (drained)")

    def _maybe_drop_drained_locked(self, entry: _ServerEntry) -> None:
        if entry.draining and entry.in_flight == 0:
            key = self._entry_key(entry)
            self._servers.pop(key, None)
            print(f"[ServerPool] removed {key} (drained)")

    def _do_request(self, entry: _ServerEntry, messages: list, sp):
        return entry.client.chat.completions.create(
            model=entry.model,
            messages=messages,
            max_tokens=sp.max_tokens,
            temperature=sp.temperature,
            top_p=sp.top_p,
            n=sp.n,
            stop=sp.stop,
            presence_penalty=sp.presence_penalty,
            frequency_penalty=sp.frequency_penalty,
            extra_body=sp.extra_body or None,
        )

    # ------------------------------------------------------------ reload

    def _reloader_loop(self) -> None:
        while not self._stop.is_set():
            if self._stop.wait(self._reload_interval_s):
                return
            try:
                mtime = os.path.getmtime(self._servers_path)
                if self._last_mtime is not None and mtime == self._last_mtime:
                    continue
                cfg = self._read_config_file(self._servers_path)
                new_interval = cfg.get("reload_interval_s", self._reload_interval_s)
                if not isinstance(new_interval, int) or new_interval < 60:
                    new_interval = self._reload_interval_s
                new_list = cfg["servers"]
                with self._cv:
                    self._apply_config_locked(new_interval, new_list)
                    self._cv.notify_all()
                self._last_mtime = mtime
            except Exception as e:
                print(f"[ServerPool] reload failed, keeping current state: {e}")

    def _apply_config_locked(self, new_interval: int, new_list: list) -> None:
        new_keys: dict = {}
        for s in new_list:
            try:
                host = s["host"]
                port = int(s["port"])
                model = s["model"]
                cap = int(s.get("max_in_flight", 1))
            except (KeyError, TypeError, ValueError):
                print(f"[ServerPool] reload: skipping malformed entry {s!r}")
                continue
            if cap < 1 or not (1 <= port <= 65535) or not host or not model:
                print(f"[ServerPool] reload: skipping invalid entry {s!r}")
                continue
            key = _server_key(host, port, model)
            if key in new_keys:
                print(f"[ServerPool] reload: duplicate {key}, keeping first")
                continue
            new_keys[key] = (host, port, model, cap)

        if not new_keys:
            print("[ServerPool] reload: no valid servers in file; keeping current state")
            return

        # Mark removed as draining.
        for key, entry in list(self._servers.items()):
            if key not in new_keys and not entry.draining:
                entry.draining = True
                print(f"[ServerPool] draining {key} (in_flight={entry.in_flight})")

        # Add or update.
        for key, (host, port, model, cap) in new_keys.items():
            if key in self._servers:
                entry = self._servers[key]
                if entry.draining:
                    entry.draining = False
                    entry.cap = cap
                    print(f"[ServerPool] {key} un-drained, cap={cap}")
                elif entry.cap != cap:
                    old = entry.cap
                    entry.cap = cap
                    print(f"[ServerPool] {key} cap {old}->{cap}")
            else:
                self._servers[key] = _ServerEntry(
                    host=host, port=port, model=model, cap=cap,
                    client=_build_client(host, port),
                )
                print(f"[ServerPool] added {key} cap={cap}")

        self._sweep_drained_locked()

        if new_interval != self._reload_interval_s:
            print(
                f"[ServerPool] reload_interval_s {self._reload_interval_s} -> {new_interval}"
            )
            self._reload_interval_s = new_interval
