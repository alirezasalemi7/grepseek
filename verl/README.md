# verl (vendored)

This is a vendored, trimmed copy of [verl](https://github.com/volcengine/verl)
(Apache-2.0) used as the training engine for GrepSeek. See
[`VENDORED.md`](VENDORED.md) for provenance and the list of removed upstream
directories.

In GrepSeek it is used **via `PYTHONPATH`** (the SFT/RL launchers add this
directory automatically) — no installation step is required. It can also be
installed editable with `pip install -e . --no-deps` if you prefer.
