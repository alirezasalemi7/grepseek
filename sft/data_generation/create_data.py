"""Generate cold-start ReAct trajectories for HotpotQA training queries.

Implements the backward-construction + tutor-edit forward generation pipeline
described in the README. Each output line is a richly-nested record with all
intermediate prompts, raw LLM responses, and decisions for offline debugging.

Run from the repository root, with an OpenAI-compatible teacher server reachable:
    python sft/data_generation/create_data.py --dataset hotpotqa --n 20 \\
        --host 127.0.0.1 --port 8000 --model <served-model-name> \\
        --corpus_dir data/wiki_18_corpus \\
        --out sft/data_generation/output/traces.jsonl \\
        --out_chatml sft/data_generation/output/sft.jsonl

See README.md for how to launch the teacher (e.g. vLLM) and obtain the corpus.
"""
import argparse
import json
import glob
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm

from utils.llm import ServerLLM
from utils.load_hotpotqa import load_hotpotqa
from utils.load_nq import load_nq
from utils.pipeline import (
    CORPUS_DIR,
    generate_trajectory,
    render_agent_trajectory,
    trajectory_to_chatml,
)
from utils.server_pool import ServerPool, _example_id_var


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("LLM_HOST", "127.0.0.1"),
                   help="host of the OpenAI-compatible teacher server, e.g. a local vLLM (env: LLM_HOST)")
    p.add_argument("--port", type=int, default=int(os.environ.get("LLM_PORT", "8000")),
                   help="port of the teacher server (env: LLM_PORT)")
    p.add_argument("--model", default=os.environ.get("LLM_MODEL", ""),
                   help="served model name to request; must match the server's --served-model-name (env: LLM_MODEL)")
    p.add_argument("--servers", default=None,
                   help="path to servers.json for multi-server pool mode. When set, "
                        "--host/--port/--model are ignored. Schema: "
                        "{reload_interval_s, servers:[{host, port, model, max_in_flight}, ...]}. "
                        "The pool is hot-reloaded by mtime; per-server in-flight caps are "
                        "honored; failures route around dead servers; calls for the same "
                        "example id stick to the same server when capacity allows.")
    p.add_argument("--dataset", default="hotpotqa", choices=["hotpotqa", "nq"],
                   help="dataset to load: 'hotpotqa' (multi-hop) or 'nq' (Natural "
                        "Questions / NQ-Open, mostly single-hop). NQ ids are prefixed "
                        "with 'nq_' so they don't collide with HotpotQA in cross-run "
                        "dedup files.")
    p.add_argument("--split", default="train", choices=["train", "dev"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default="output/traces.jsonl",
                   help="rich JSONL output (one trajectory per line, all debug info)")
    p.add_argument("--out_chatml", default=None,
                   help="optional ChatML SFT output (only successful traces)")
    p.add_argument("--out_pretty", default=None,
                   help="optional human-readable text dump of the agent-side trajectories "
                        "(question + think + tool_call + tool_response + answer per example) "
                        "for visual review of what the model would be trained on")
    p.add_argument("--corpus_dir", default=CORPUS_DIR,
                   help="directory containing wiki_corpus.jsonl; commands are run with cwd=this and the agent refers to the file as `corpus.jsonl`")
    p.add_argument("--parallel_examples", type=int, default=9,
                   help="examples processed in parallel = max concurrent vLLM calls")
    p.add_argument("--backward_max_iterations", type=int, default=6,
                   help="max command-refinement iterations per backward sub-question")
    p.add_argument("--tool_max_tokens", type=int, default=2048,
                   help="cap on tool output length, in Qwen3.5-27B tokens (defense against unbounded stdout)")
    p.add_argument("--tool_timeout", type=int, default=60)
    p.add_argument("--max_tool_calls", type=int, default=10,
                   help="hard cap on the number of <tool_call> turns in the agent trace; "
                        "trajectories exceeding this are dropped with abort_reason=too_many_tool_calls / decomposition_too_long")
    p.add_argument("--no_quality_filter", action="store_true",
                   help="disable the post-hoc LLM coherence judge (default: judge runs and "
                        "successful traces that don't pass the judge are excluded from SFT export)")
    p.add_argument("--resume", action="store_true",
                   help="resume from a previous run: skip examples whose id is already present in --out, "
                        "and open --out / --out_pretty / --out_chatml in append mode. SFT chatml is "
                        "rebuilt only for newly-processed examples (existing chatml lines are preserved).")
    p.add_argument("--resume_from", nargs="*", default=[],
                   help="additional JSONL paths or glob patterns whose ids should ALSO be skipped. "
                        "Use this when running multiple overlapping ranges in parallel — list the "
                        "other runs' --out paths (or a glob like 'output/test_*.jsonl') so the same "
                        "example isn't processed twice. Read once at startup; tolerates truncated "
                        "last lines in still-running peer files.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _load_processed_ids(path: Path) -> set:
    """Read a JSONL file and return the set of `id` fields seen. Tolerates a
    truncated last line."""
    ids = set()
    if not path.exists():
        return ids
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip truncated/corrupted line (likely from a kill mid-flush)
            ids.add(str(rec.get("id", "")))
    return ids


def process_one(args, llm, ex, seed_offset, pool):
    example_id = str(ex.get("id", ""))
    token = _example_id_var.set(example_id)
    rng = random.Random(args.seed + seed_offset)
    t0 = time.time()
    try:
        try:
            rec = generate_trajectory(
                llm,
                ex,
                corpus_dir=args.corpus_dir,
                backward_max_iterations=args.backward_max_iterations,
                tool_max_tokens=args.tool_max_tokens,
                tool_timeout=args.tool_timeout,
                max_tool_calls=args.max_tool_calls,
                enable_quality_judge=not args.no_quality_filter,
                rng=rng,
                verbose=args.verbose,
            )
        except Exception as e:
            rec = {
                "id": example_id,
                "question": ex["question"],
                "gold_answer": list(ex.get("golden_answers") or []),
                "decomposition": None, "backward": None, "forward": [], "final": None,
                "success": False, "abort_reason": f"exception: {type(e).__name__}: {e}",
            }
        # Always ensure agent_trajectory is rendered (some early-abort paths don't set it).
        if "agent_trajectory" not in rec:
            rec["agent_trajectory"] = render_agent_trajectory(rec)
        return rec, time.time() - t0
    finally:
        _example_id_var.reset(token)
        pool.forget(example_id)


def short_status(rec) -> str:
    if rec["success"]:
        em = "EM" if rec.get("em") else "f1"
        qc = rec.get("quality_check") or {}
        qtag = ""
        if qc:
            v = qc.get("verdict")
            if v == "FAIL":
                qtag = (f"  qjudge=FAIL[c{qc.get('failing_check')},t{qc.get('first_failing_turn')}]"
                        f" reason={qc.get('reasoning', '')!r}")
            elif v == "UNKNOWN":
                qtag = "  qjudge=UNKNOWN"
        return (f"PASS[{em}]  f1={rec.get('f1', 0.0):.2f}  "
                f"ans={rec['final']['answer']!r}{qtag}")
    if rec.get("final") and rec["final"].get("answer"):
        return (f"ZERO_F1  ans={rec['final']['answer']!r}  "
                f"gold={rec['gold_answer']!r}")
    return f"ABORT[{rec['abort_reason']}]"


def main():
    args = parse_args()

    print(f"[1/3] loading {args.split} {args.dataset} (offset={args.start}, n={args.n})")
    if args.dataset == "hotpotqa":
        full = load_hotpotqa(split=args.split, limit=args.start + args.n)
    elif args.dataset == "nq":
        full = load_nq(split=args.split, limit=args.start + args.n)
    else:
        raise ValueError(f"unknown --dataset {args.dataset!r}")
    examples = full[args.start: args.start + args.n]
    print(f"      got {len(examples)} examples in slice")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chatml_path = Path(args.out_chatml) if args.out_chatml else None
    if chatml_path:
        chatml_path.parent.mkdir(parents=True, exist_ok=True)
    pretty_path = Path(args.out_pretty) if args.out_pretty else None
    if pretty_path:
        pretty_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Resume support ----
    # Skip ids already present in the rich JSONL; open output files in append mode.
    if args.resume:
        processed_ids = _load_processed_ids(out_path)
        before = len(examples)
        examples = [ex for ex in examples if str(ex.get("id", "")) not in processed_ids]
        print(f"      resume: {len(processed_ids)} ids already in {out_path}; "
              f"{before - len(examples)} skipped, {len(examples)} remain to process")
    file_mode = "a" if args.resume else "w"

    # ---- Cross-run dedup via --resume_from ----
    # Glob-expand each pattern; union all ids; drop any example already done in
    # any peer run. Useful for running multiple overlapping ranges in parallel.
    if args.resume_from:
        peer_paths = []
        for pat in args.resume_from:
            matches = sorted(glob.glob(pat))
            if not matches and Path(pat).exists():
                matches = [pat]
            for m in matches:
                p_ = Path(m).resolve()
                if p_ == out_path.resolve():
                    continue  # already counted via --resume
                peer_paths.append(p_)
        peer_ids = set()
        for p_ in peer_paths:
            peer_ids |= _load_processed_ids(p_)
        if peer_ids:
            before = len(examples)
            examples = [ex for ex in examples if str(ex.get("id", "")) not in peer_ids]
            print(f"      resume_from: loaded {len(peer_ids)} ids from {len(peer_paths)} peer file(s); "
                  f"{before - len(examples)} additional skips, {len(examples)} remain to process")
        else:
            print(f"      resume_from: 0 ids loaded from {len(peer_paths)} peer file(s)")

    # ---- Skip boolean / yes-no questions ----
    # Empirically these have a 100% backward-discovery failure rate: the
    # decomposition produces a final sub-question like "Are both X and Y...?"
    # with expected_answer = "yes" / "no", which no single corpus passage can
    # confirm jointly. Each one burns 4-12 iterations before aborting.
    _BOOLEAN_GOLD = {"yes", "no", "true", "false"}
    before = len(examples)
    examples = [
        ex for ex in examples
        if not (
            ex.get("golden_answers")
            and any(
                (g or "").strip().lower().rstrip(".!?") in _BOOLEAN_GOLD
                for g in ex["golden_answers"]
            )
        )
    ]
    skipped = before - len(examples)
    if skipped:
        print(f"      boolean-skip: dropped {skipped} yes/no examples (100% failure rate empirically)")

    if args.servers:
        print(f"[2/3] loading server pool from {args.servers}")
        pool = ServerPool(servers_path=args.servers)
        pool.start_reloader()
        print(f"      pool: {pool.describe()}  reload_interval={pool.reload_interval_s}s")
    else:
        if not args.model:
            raise SystemExit(
                "error: --model (or env LLM_MODEL) is required unless --servers is used. "
                "It must match the teacher server's served model name "
                "(vLLM: --served-model-name)."
            )
        print(f"[2/3] connecting to vLLM at http://{args.host}:{args.port}/v1 model={args.model}")
        pool = ServerPool(
            servers_path=None,
            fallback_servers=[{
                "host": args.host, "port": args.port, "model": args.model,
                "max_in_flight": max(1, args.parallel_examples),
            }],
        )
    llm = ServerLLM(pool=pool)
    print(f"      executor max_workers={args.parallel_examples}")

    extras = []
    if chatml_path:
        extras.append(f"chatml -> {chatml_path}")
    if pretty_path:
        extras.append(f"pretty -> {pretty_path}")
    suffix = ("  (+ " + ", ".join(extras) + ")") if extras else ""
    print(f"[3/3] generating -> {out_path}{suffix}  (mode={file_mode!r})")

    n_success = 0
    n_em = 0
    n_aborted = 0
    n_zero_f1 = 0
    n_qpass = 0
    n_qfail = 0
    n_qunknown = 0
    f1_sum = 0.0

    f_full = out_path.open(file_mode)
    f_chat = chatml_path.open(file_mode) if chatml_path else None
    f_pretty = pretty_path.open(file_mode) if pretty_path else None

    try:
        with ThreadPoolExecutor(max_workers=args.parallel_examples) as ex_pool:
            futures = [
                ex_pool.submit(process_one, args, llm, ex, i, pool)
                for i, ex in enumerate(examples)
            ]
            pbar = tqdm.tqdm(
                as_completed(futures),
                total=len(futures),
                desc="trajectories",
                dynamic_ncols=True,
            )
            for fut in pbar:
                rec, dt = fut.result()
                f_full.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f_full.flush()
                if f_pretty:
                    f_pretty.write(rec.get("agent_trajectory", "") + "\n\n")
                    f_pretty.flush()
                if rec["success"]:
                    n_success += 1
                    f1_sum += rec.get("f1", 0.0)
                    if rec.get("em"):
                        n_em += 1
                    qc = rec.get("quality_check") or {}
                    qverdict = qc.get("verdict")  # "PASS" / "FAIL" / "UNKNOWN" / None
                    if qverdict == "PASS":
                        n_qpass += 1
                    elif qverdict == "FAIL":
                        n_qfail += 1
                    elif qverdict == "UNKNOWN":
                        n_qunknown += 1
                    # SFT export only if both f1>0 AND quality_pass.
                    # When the judge is disabled (quality_pass defaults to success),
                    # this collapses to just f1>0.
                    if f_chat and rec.get("quality_pass", True):
                        msgs = trajectory_to_chatml(rec)
                        if msgs:
                            f_chat.write(json.dumps(
                                {"id": rec["id"], "question": rec["question"],
                                 "gold_answer": rec["gold_answer"],
                                 "f1": rec.get("f1"), "em": rec.get("em"),
                                 "quality_check": qc or None,
                                 "messages": msgs},
                                ensure_ascii=False,
                            ) + "\n")
                            f_chat.flush()
                elif rec.get("final") and rec["final"].get("answer"):
                    n_zero_f1 += 1
                else:
                    n_aborted += 1
                avg_f1 = (f1_sum / n_success) if n_success else 0.0
                pbar.set_postfix(
                    pass_=n_success, em=n_em,
                    qpass=n_qpass, qfail=n_qfail,
                    zero_f1=n_zero_f1, abort=n_aborted,
                    load=f"{pool.total_in_flight()}/{pool.total_capacity()}",
                    avg_f1=f"{avg_f1:.2f}",
                    last=f"{rec['id']}({dt:.0f}s)",
                )
                if args.verbose:
                    pbar.write(f"  {rec['id']}  {short_status(rec)}  ({dt:.1f}s)")
    finally:
        f_full.close()
        if f_chat:
            f_chat.close()
        if f_pretty:
            f_pretty.close()
        pool.close()

    avg_f1 = (f1_sum / n_success) if n_success else 0.0
    print(
        f"\ndone. success(f1>0)={n_success}/{len(examples)}  "
        f"em={n_em}  zero_f1={n_zero_f1}  aborted={n_aborted}  "
        f"avg_f1_among_kept={avg_f1:.3f}"
    )
    if not args.no_quality_filter:
        kept_for_sft = n_qpass
        print(
            f"      quality_judge: pass={n_qpass}  fail={n_qfail}  "
            f"unknown={n_qunknown}  -> {kept_for_sft} kept for SFT"
        )
    else:
        print("      quality_judge: disabled")
    print(f"      written -> {out_path}")


if __name__ == "__main__":
    main()
