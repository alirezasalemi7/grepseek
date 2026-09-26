"""End-to-end test: start a daemon in a thread, connect a client, verify
byte-equivalence between client.execute() and engine.execute().

Synthetic corpus under /tmp; real wiki_corpus.jsonl is NOT touched.
"""
from __future__ import annotations

import os
import socket
import tempfile
import threading
import time
import unittest

from inference.parallel_search import DaemonClient, ShardedSearchEngine
from inference.parallel_search.daemon import serve
from inference.parallel_search.sharder import build_shards
from inference.parallel_search.tests.test_byte_equivalence import _make_synthetic_corpus


def _wait_for_socket(path: str, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(path)
                s.close()
                return True
            except OSError:
                pass
        time.sleep(0.05)
    return False


class DaemonRoundTripTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="parallel_search_daemon_test_")
        cls.corpus_dir = os.path.join(cls.tmpdir, "wiki_18_corpus")
        os.makedirs(cls.corpus_dir, exist_ok=True)
        cls.corpus_file = os.path.join(cls.corpus_dir, "wiki_corpus.jsonl")
        _make_synthetic_corpus(cls.corpus_file, n_lines=4000)

        cls.shard_dir = os.path.join(cls.tmpdir, "shards_8")
        build_shards(cls.corpus_file, cls.shard_dir, n=8)

        cls.engine = ShardedSearchEngine(
            shard_dir=cls.shard_dir,
            fallback_corpus_file=cls.corpus_file,
            timeout=15.0,
        )
        cls.socket_path = os.path.join(cls.tmpdir, "daemon.sock")

        # Start the daemon in a background thread. serve() blocks on
        # serve_forever; we need to call .shutdown() on the server from
        # outside, so we capture the server object via a holder list and
        # call serve() in a wrapper that exits cleanly.
        cls._server_holder = {"server": None, "stopped": False}

        def _run():
            # serve() registers SIGTERM/SIGINT handlers but those don't
            # interrupt the call from another thread. Instead, monkey-patch
            # serve_forever to expose the server so we can call .shutdown().
            from socketserver import ThreadingMixIn, UnixStreamServer
            from inference.parallel_search.daemon import _Handler, _ThreadedUnixServer

            if os.path.exists(cls.socket_path):
                os.unlink(cls.socket_path)
            srv = _ThreadedUnixServer(cls.socket_path, _Handler, cls.engine)
            os.chmod(cls.socket_path, 0o600)
            cls._server_holder["server"] = srv
            try:
                srv.serve_forever()
            finally:
                try:
                    os.unlink(cls.socket_path)
                except OSError:
                    pass
                cls._server_holder["stopped"] = True

        cls._daemon_thread = threading.Thread(target=_run, daemon=True)
        cls._daemon_thread.start()
        if not _wait_for_socket(cls.socket_path):
            raise RuntimeError("daemon socket did not appear within timeout")

        cls.client = DaemonClient(socket_path=cls.socket_path, timeout=20.0)

    @classmethod
    def tearDownClass(cls):
        srv = cls._server_holder.get("server")
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        cls._daemon_thread.join(timeout=5.0)
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _check_pair(self, cmd: str, expected_strategy: str | None = None):
        eng_out, eng_rc = self.engine.execute(cmd)
        cli_out, cli_rc = self.client.execute(cmd)
        self.assertEqual(eng_out, cli_out,
                         msg=f"\ncmd: {cmd}\nengine: {eng_out!r} (rc={eng_rc})\nclient: {cli_out!r} (rc={cli_rc})")
        self.assertEqual(eng_rc, cli_rc, msg=f"returncode mismatch on cmd: {cmd}")
        if expected_strategy is not None:
            self.assertEqual(self.client.last_stats.strategy, expected_strategy,
                             msg=f"client strategy mismatch on cmd: {cmd}")

    def test_concat(self):
        self._check_pair('rg -F "Einstein" corpus.jsonl', expected_strategy="concat")

    def test_head(self):
        self._check_pair('rg -F "Einstein" corpus.jsonl | head -n 3',
                         expected_strategy="head")

    def test_count(self):
        self._check_pair('rg -F "Einstein" corpus.jsonl | wc -l',
                         expected_strategy="count")

    def test_sort_uniq_head(self):
        self._check_pair('rg -F "common boilerplate" corpus.jsonl | sort | uniq | head -n 3',
                         expected_strategy="sort_head")

    def test_fallback_tail(self):
        self._check_pair('rg -F "Einstein" corpus.jsonl | tail -n 3',
                         expected_strategy="fallback")

    def test_no_matches(self):
        self._check_pair('rg -F "ZZZZ_nothing_ZZZZ" corpus.jsonl')

    def test_env_overrides_passed_through(self):
        # If env overrides reach the daemon, LC_ALL=C should be inherited
        # in the per-shard subprocess invocations. We can't observe LC_ALL
        # easily from stdout, but verify the call doesn't error.
        out_a, rc_a = self.client.execute('rg -F "Einstein" corpus.jsonl',
                                           env={**os.environ, "LC_ALL": "C"})
        out_b, rc_b = self.client.execute('rg -F "Einstein" corpus.jsonl')
        # Same output expected regardless of locale override for -F patterns.
        self.assertEqual(out_a, out_b)
        self.assertEqual(rc_a, rc_b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
