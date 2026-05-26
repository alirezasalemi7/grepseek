"""Convert cold-start SFT JSONL (the `--out_chatml` output of create_data.py)
into train/val parquet files for supervised fine-tuning.

Each input line looks like:
    {"id": ..., "question": ..., "messages": [ {role, content}, ... ], ...}

We keep the `messages` column (and `id` for traceability), do a deterministic
hash-based train/val split (so re-running with more data preserves the split
for already-seen ids), and optionally attach a `tools` column carrying the
function-call schema for the `shell` tool.

Example:
    python to_parquet.py --in 'output/*_sft.jsonl' --out_dir output/sft_parquet \\
        --val_frac 0.02 --include_tools
"""
import argparse
import glob
import hashlib
import json

import pandas as pd

# Function-call schema for the `shell` tool, matching what the trajectories emit
# inside <tool_call>: {"name": "shell", "arguments": {"command": "..."}}.
SHELL_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": (
                "Run a single-pipeline shell command against the local Wikipedia "
                "corpus (corpus.jsonl). Pipes (|) are allowed; redirection, command "
                "chaining, and command substitution are not. Allowed tools: rg, grep, "
                "find, sed, awk, head, tail, cat, ls, wc, sort, cut, uniq, tr."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            'The single-pipeline shell command to execute, e.g. '
                            'rg -F "phrase" corpus.jsonl | head -n 3'
                        ),
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def in_val(example_id: str, val_frac: float) -> bool:
    """Deterministic split: hash the id to [0,1) and compare to val_frac."""
    h = hashlib.sha1(str(example_id).encode()).hexdigest()
    return (int(h[:8], 16) / 0xFFFFFFFF) < val_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", nargs="+", required=True,
                    help="one or more glob patterns of *_sft.jsonl files")
    ap.add_argument("--out_dir", required=True, help="directory for train.parquet / val.parquet")
    ap.add_argument("--val_frac", type=float, default=0.02, help="fraction held out for validation")
    ap.add_argument("--include_tools", action="store_true",
                    help="attach a `tools` column with the shell function-call schema")
    args = ap.parse_args()

    paths = sorted({p for pat in args.inputs for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no files matched: {args.inputs}")
    print(f"reading {len(paths)} file(s): {paths}")

    rows, seen = [], set()
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated last line
                msgs = rec.get("messages")
                rid = str(rec.get("id", ""))
                if not msgs or rid in seen:
                    continue
                seen.add(rid)
                row = {"id": rid, "messages": msgs}
                if args.include_tools:
                    row["tools"] = SHELL_TOOL_SCHEMA
                rows.append(row)

    if not rows:
        raise SystemExit("no usable rows (need a non-empty `messages` field)")

    train = [r for r in rows if not in_val(r["id"], args.val_frac)]
    val = [r for r in rows if in_val(r["id"], args.val_frac)]

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "val.parquet")
    pd.DataFrame(train).to_parquet(train_path, index=False)
    pd.DataFrame(val).to_parquet(val_path, index=False)
    print(f"wrote {len(train)} train -> {train_path}")
    print(f"wrote {len(val)} val   -> {val_path}")


if __name__ == "__main__":
    main()
