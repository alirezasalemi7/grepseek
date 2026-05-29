"""GrepSeek inference harness — generation and (optional) evaluation.

The agent searches the Wikipedia corpus with shell/grep tool calls against a
served vLLM model. Two modes:

  1. Generation (no benchmark, gold optional) — run the agent on your own
     questions and write the trajectories + answers:
        python -m inference.run \\
            --base_url http://HOST:PORT/v1 --model grepseek \\
            --input my_questions.jsonl --corpus_dir /path/to/wiki_18_corpus \\
            --out_dir output/gen
     (each input row needs a `question`/`query`; `id` and `golden_answers`
     optional. Add --no_eval to never score, even if gold is present.)

  2. Benchmark eval — run + score on the Search-R1 QA suite (NQ, HotpotQA, ...):
        python -m inference.run \\
            --base_url http://HOST:PORT/v1 --model grepseek \\
            --datasets nq,hotpotqa,bamboogle --limit 200 \\
            --corpus_dir /path/to/wiki_18_corpus --out_dir output/eval

Per task it writes <out_dir>/<name>/predictions.jsonl + summary.json. When
scoring is on it also reports per-dataset and overall EM / F1 (reported
separately, no combined metric).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from openai import OpenAI
from transformers import AutoTokenizer

# Package-relative imports if invoked as `-m inference.run`, otherwise fall back
# to absolute imports when run as a plain script.
try:
    from .agent import run_agent_on_example
    from .load_dataset import (
        SEARCH_R1_DATASETS,
        default_split,
        load_flashrag_dataset,
        load_questions_file,
    )
    from .scoring import aggregate, score
except ImportError:  # pragma: no cover - script-mode fallback
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from agent import run_agent_on_example  # type: ignore
    from load_dataset import (  # type: ignore
        SEARCH_R1_DATASETS,
        default_split,
        load_flashrag_dataset,
        load_questions_file,
    )
    from scoring import aggregate, score  # type: ignore


# Corpus dir defaults to $GREPSEEK_CORPUS_ROOT when set, otherwise the repo-local
# default data/wiki_18_corpus. Cache dir defaults to $HF_HOME (HF default if unset).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CORPUS_DIR = (
    os.environ.get("GREPSEEK_CORPUS_ROOT")
    or os.path.join(_REPO_ROOT, "data", "wiki_18_corpus")
)
DEFAULT_CACHE_DIR = os.environ.get("HF_HOME")


def _parse_datasets(arg: Optional[str]) -> list[str]:
    if not arg or arg.lower() == "all":
        return sorted(SEARCH_R1_DATASETS.keys())
    names = [s.strip().lower() for s in arg.split(",") if s.strip()]
    unknown = [n for n in names if n not in SEARCH_R1_DATASETS]
    if unknown:
        raise SystemExit(
            f"unknown datasets {unknown}; supported: {sorted(SEARCH_R1_DATASETS)}"
        )
    return names


def _build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GrepSeek inference: run the search agent for generation, or "
                    "evaluate on the Search-R1 QA benchmarks.")
    # Server
    p.add_argument("--host", default=None, help="vLLM server host (used if --base_url not set)")
    p.add_argument("--port", type=int, default=None, help="vLLM server port (used if --base_url not set)")
    p.add_argument("--base_url", default=None,
                   help="Full OpenAI base URL, e.g. http://host:port/v1. Takes precedence over --host/--port.")
    p.add_argument("--api_key", default="EMPTY", help="API key for the vLLM server (any string).")
    p.add_argument("--model", required=True,
                   help="model id as registered on the vLLM server (e.g. grepseek).")
    # Tokenizer (used to truncate tool output by token count)
    p.add_argument("--tokenizer", default=None,
                   help="HF model id or local path for the tokenizer. Defaults to --model.")
    # What to run — generation input OR benchmark datasets (mutually exclusive).
    p.add_argument("--input", default=None,
                   help="GENERATION mode: path to a JSON/JSONL of your own questions "
                        "(each row needs `question`/`query`; `id` + `golden_answers` "
                        "optional). Mutually exclusive with --datasets.")
    p.add_argument("--datasets", default=None,
                   help=f"BENCHMARK mode: comma-separated subset, or 'all'. "
                        f"Supported: {sorted(SEARCH_R1_DATASETS)}")
    p.add_argument("--no_eval", action="store_true",
                   help="Generation only: never score, even when gold answers are present.")
    p.add_argument("--split", default=None,
                   help="Override split; otherwise per-dataset default is used (dev or test).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap rows per task (after offset).")
    p.add_argument("--offset", type=int, default=0)
    # Agent
    p.add_argument("--corpus_dir", default=DEFAULT_CORPUS_DIR,
                   help="Directory containing wiki_corpus.jsonl (tool commands cwd into it). "
                        "Defaults to $GREPSEEK_CORPUS_ROOT, else data/wiki_18_corpus.")
    p.add_argument("--max_assistant_turns", type=int, default=6)
    p.add_argument("--max_tokens_per_turn", type=int, default=2048)
    p.add_argument("--tool_max_tokens", type=int, default=2048,
                   help="Truncate tool stdout to this many tokens (model tokenizer).")
    p.add_argument("--tool_timeout", type=float, default=60.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    # Sharded parallel search (optional speedup; off by default).
    p.add_argument("--engine_mode", choices=["none", "inproc", "daemon"], default="none",
                   help="Search-engine backend: 'none' = single-file (current behavior, default); "
                        "'inproc' = sharded engine instantiated in this process; "
                        "'daemon' = connect to an existing daemon over Unix socket "
                        "(see scripts/rl/eval_rl_fast.sh which auto-spawns one).")
    p.add_argument("--parallel_shards", type=int, default=0,
                   help="DEPRECATED ALIAS. Equivalent to --engine_mode=inproc when > 0 and "
                        "--engine_mode is not explicitly set. Kept for back-compat with "
                        "earlier eval scripts. New scripts should use --engine_mode.")
    p.add_argument("--shard_dir", default=None,
                   help="Directory containing pre-built shards + manifest.json. "
                        "Required when --engine_mode is 'inproc' or 'daemon'.")
    p.add_argument("--socket", default=None,
                   help="Unix-socket path of a running search daemon. "
                        "Required when --engine_mode=daemon.")
    p.add_argument("--engine_log", default=None,
                   help="Optional jsonl path for per-tool-call engine telemetry. "
                        "When --engine_mode=daemon, ignored client-side (the daemon "
                        "writes its own log via its --engine_log flag).")
    # Runner
    p.add_argument("--parallel", type=int, default=8,
                   help="Examples in flight against the server.")
    p.add_argument("--out_dir", required=True,
                   help="Output directory; results land under out_dir/<dataset>/...")
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR,
                   help="HuggingFace cache dir for datasets + tokenizer.")
    p.add_argument("--resume", action="store_true",
                   help="Skip examples whose id is already in <out_dir>/<dataset>/predictions.jsonl.")
    return p.parse_args()


def _resolve_base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        return args.base_url
    if args.host and args.port:
        return f"http://{args.host}:{args.port}/v1"
    raise SystemExit("Need either --base_url, or both --host and --port.")


def _load_existing_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    seen: set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ex_id = obj.get("id")
            if ex_id:
                seen.add(ex_id)
    return seen


def _run_example_safe(example, *, client, model, tokenizer, agent_kwargs):
    """Wrap run_agent_on_example to never propagate exceptions to the executor."""
    try:
        rec = run_agent_on_example(
            example,
            client=client,
            model=model,
            tokenizer=tokenizer,
            **agent_kwargs,
        )
        return rec.to_dict()
    except Exception as e:
        return {
            "id": str(example.get("id", "")),
            "question": example.get("question") or "",
            "gold_answers": list(example.get("golden_answers") or []),
            "prediction": None,
            "finish_reason": "exception",
            "n_assistant_turns": 0,
            "n_tool_calls": 0,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "turns": [],
            "messages": [],
        }


def run_task(
    name: str,
    examples: list,
    *,
    client,
    model: str,
    tokenizer,
    out_dir: str,
    parallel: int,
    resume: bool,
    do_eval: bool,
    agent_kwargs: dict,
    split_label: Optional[str] = None,
) -> dict:
    """Run the agent over `examples` (already loaded). Scores each trajectory with
    EM/F1 only when `do_eval` and the example has gold answers; otherwise it's a
    pure generation pass (predictions written, no scoring)."""
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)
    preds_path = os.path.join(ds_dir, "predictions.jsonl")
    summary_path = os.path.join(ds_dir, "summary.json")

    hdr = f"\n=== {name}" + (f" (split={split_label})" if split_label else "") + " ==="
    print(hdr, flush=True)
    print(f"loaded {len(examples)} examples"
          + ("" if do_eval else "   [generation only — no scoring]"), flush=True)

    seen = _load_existing_ids(preds_path) if resume else set()
    if seen:
        examples = [ex for ex in examples if ex["id"] not in seen]
        print(f"resume: skipping {len(seen)} already-done ids; {len(examples)} remaining",
              flush=True)

    # Open output file for append (resume) or overwrite (fresh).
    write_mode = "a" if (resume and seen) else "w"
    fout = open(preds_path, write_mode, encoding="utf-8")

    t0 = time.time()
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            futures = {
                pool.submit(
                    _run_example_safe,
                    ex,
                    client=client,
                    model=model,
                    tokenizer=tokenizer,
                    agent_kwargs=agent_kwargs,
                ): ex for ex in examples
            }
            for fut in as_completed(futures):
                rec = fut.result()
                ex = futures[fut]
                # Score only when evaluating AND the example has gold answers.
                if do_eval and ex.get("golden_answers"):
                    rec_score = score(rec.get("prediction") or "", ex["golden_answers"])
                    rec["em"] = bool(rec_score["em"])
                    rec["f1"] = float(rec_score["f1"])
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                completed += 1
                if completed % 25 == 0 or completed == len(examples):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    print(f"  [{name}] {completed}/{len(examples)} ({rate:.2f} ex/s)", flush=True)
    finally:
        fout.close()

    # Re-read everything (resume case included): always aggregate timing; aggregate
    # EM/F1 only over rows that were actually scored.
    all_scores: list = []
    n_rows = 0
    llm_times: list = []
    tool_times: list = []
    total_times: list = []
    n_assistant_turns_list: list = []
    n_tool_calls_list: list = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n_rows += 1
            if obj.get("em") is not None:
                all_scores.append({"em": bool(obj.get("em")), "f1": float(obj.get("f1") or 0.0)})
            # Timing fields are present from the agent loop; default to 0 for
            # back-compat with predictions.jsonl rows written by older runs.
            llm_times.append(float(obj.get("llm_time_s") or 0.0))
            tool_times.append(float(obj.get("tool_time_s") or 0.0))
            total_times.append(float(obj.get("total_time_s") or 0.0))
            n_assistant_turns_list.append(int(obj.get("n_assistant_turns") or 0))
            n_tool_calls_list.append(int(obj.get("n_tool_calls") or 0))

    def _pcts(xs: list) -> dict:
        if not xs:
            return {"sum": 0.0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
        s = sorted(xs)
        n = len(s)
        def _q(p): return s[min(n - 1, int(p * n))]
        return {
            "sum": float(sum(xs)),
            "mean": float(sum(xs) / n),
            "median": float(_q(0.50)),
            "p90": float(_q(0.90)),
            "p99": float(_q(0.99)),
            "min": float(s[0]),
            "max": float(s[-1]),
        }

    summary = {
        "dataset": name,
        "model": model,
        "n_total": n_rows,
        # Timing breakdown — what fraction of wall-clock was LLM vs retrieval.
        # All times in seconds. Aggregated over all examples.
        "timing": {
            "n_examples": len(total_times),
            "llm_time_s":   _pcts(llm_times),
            "tool_time_s":  _pcts(tool_times),
            "total_time_s": _pcts(total_times),
            "n_assistant_turns": _pcts([float(x) for x in n_assistant_turns_list]),
            "n_tool_calls":      _pcts([float(x) for x in n_tool_calls_list]),
            # Headline ratios (computed from sums to be sequencing-independent).
            "llm_fraction_of_total":  (sum(llm_times)  / sum(total_times)) if sum(total_times) > 0 else 0.0,
            "tool_fraction_of_total": (sum(tool_times) / sum(total_times)) if sum(total_times) > 0 else 0.0,
        },
    }
    if split_label:
        summary["split"] = split_label
    if all_scores:
        agg = aggregate(all_scores)
        summary["n_scored"] = len(all_scores)
        summary["em"] = agg["em"]
        summary["f1"] = agg["f1"]

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _t = summary["timing"]
    if all_scores:
        print(f"  -> EM={summary['em']:.4f}  F1={summary['f1']:.4f}  n={len(all_scores)}", flush=True)
    else:
        print(f"  -> generated {n_rows} trajectories (no scoring)", flush=True)
    print(f"     timing  LLM={_t['llm_time_s']['sum']:.1f}s  "
          f"tool={_t['tool_time_s']['sum']:.1f}s  "
          f"total={_t['total_time_s']['sum']:.1f}s  "
          f"(LLM={_t['llm_fraction_of_total']*100:.1f}%  "
          f"tool={_t['tool_fraction_of_total']*100:.1f}%)", flush=True)
    return summary


def main():
    args = _build_args()
    base_url = _resolve_base_url(args)
    os.makedirs(args.out_dir, exist_ok=True)

    # Cache dir defaults — applies to both datasets and tokenizer.
    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)

    # The agent always needs the corpus directory (it cwds there to run grep).
    if not args.corpus_dir:
        raise SystemExit("set --corpus_dir (or $GREPSEEK_CORPUS_ROOT) to the dir "
                         "containing wiki_corpus.jsonl")
    if not os.path.isfile(os.path.join(args.corpus_dir, "wiki_corpus.jsonl")):
        raise SystemExit(f"{args.corpus_dir}/wiki_corpus.jsonl not found "
                         "(download it with sft/data_generation/download_corpus.py)")

    # Pick the work: generation (--input) XOR benchmark (--datasets).
    if args.input and args.datasets:
        raise SystemExit("pass either --input (generation) or --datasets (benchmark), not both")
    if not args.input and not args.datasets:
        raise SystemExit("nothing to do: pass --input <questions.jsonl> (generation) "
                         "or --datasets <names|all> (benchmark eval)")
    do_eval = not args.no_eval

    print(f"server:      {base_url}", flush=True)
    print(f"model:       {args.model}", flush=True)
    tok_id = args.tokenizer or args.model
    print(f"tokenizer:   {tok_id}", flush=True)
    print(f"corpus_dir:  {args.corpus_dir}", flush=True)
    print(f"out_dir:     {args.out_dir}", flush=True)
    print(f"mode:        {'generation' if args.input else 'benchmark eval'}"
          f"{'' if do_eval else ' (no scoring)'}", flush=True)

    client = OpenAI(api_key=args.api_key, base_url=base_url)
    tokenizer = AutoTokenizer.from_pretrained(tok_id, cache_dir=args.cache_dir)

    # Search-engine backend selection. Resolve the deprecated
    # --parallel_shards alias first, then pick a backend.
    engine_mode = args.engine_mode
    if engine_mode == "none" and args.parallel_shards and args.parallel_shards > 0:
        engine_mode = "inproc"  # legacy alias
    engine = None
    if engine_mode == "inproc":
        if not args.shard_dir:
            raise SystemExit("--engine_mode=inproc requires --shard_dir")
        from .parallel_search import ShardedSearchEngine
        engine = ShardedSearchEngine(
            shard_dir=args.shard_dir,
            fallback_corpus_file=os.path.join(args.corpus_dir, "wiki_corpus.jsonl"),
            timeout=args.tool_timeout,
            log_path=args.engine_log,
        )
        print(f"engine:      {engine.describe()}", flush=True)
    elif engine_mode == "daemon":
        if not args.socket:
            raise SystemExit("--engine_mode=daemon requires --socket")
        from .parallel_search import DaemonClient
        engine = DaemonClient(
            socket_path=args.socket,
            timeout=args.tool_timeout + 5.0,  # slack vs server-side per-call timeout
        )
        print(f"engine:      {engine.describe()}", flush=True)
    else:
        print("engine:      disabled (single-file path)", flush=True)

    agent_kwargs = dict(
        corpus_dir=args.corpus_dir,
        max_assistant_turns=args.max_assistant_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        tool_max_tokens=args.tool_max_tokens,
        tool_timeout=args.tool_timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        engine=engine,
    )

    # Build the list of (name, examples, split_label) tasks.
    tasks: list = []
    if args.input:
        name = os.path.splitext(os.path.basename(args.input))[0] or "generation"
        examples = load_questions_file(args.input, limit=args.limit, offset=args.offset)
        tasks.append((name, examples, None))
    else:
        for ds in _parse_datasets(args.datasets):
            examples = load_flashrag_dataset(
                ds, split=args.split, cache_dir=args.cache_dir,
                limit=args.limit, offset=args.offset,
            )
            tasks.append((ds, examples, args.split or default_split(ds)))

    summaries = []
    for name, examples, split_label in tasks:
        try:
            s = run_task(
                name, examples,
                client=client,
                model=args.model,
                tokenizer=tokenizer,
                out_dir=args.out_dir,
                parallel=args.parallel,
                resume=args.resume,
                do_eval=do_eval,
                agent_kwargs=agent_kwargs,
                split_label=split_label,
            )
            summaries.append(s)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    # Overall summary across datasets (per-dataset EM/F1 averaged equally).
    def _t_sum(s, key):  # safe getter — older summaries may lack 'timing'
        return float(((s.get("timing") or {}).get(key) or {}).get("sum") or 0.0)

    total_llm  = sum(_t_sum(s, "llm_time_s")   for s in summaries)
    total_tool = sum(_t_sum(s, "tool_time_s")  for s in summaries)
    total_wall = sum(_t_sum(s, "total_time_s") for s in summaries)
    total_ex   = sum((s.get("timing") or {}).get("n_examples") or 0 for s in summaries)

    scored = [s for s in summaries if "em" in s]  # generation tasks have no em/f1
    # macro = datasets weighted equally; micro = examples pooled across datasets
    # (each scored example weighted equally), computed from per-dataset mean×n.
    n_scored_total = sum(s.get("n_scored", 0) for s in scored)
    overall = {
        "datasets": [s["dataset"] for s in summaries],
        "per_dataset": {s["dataset"]: {"em": s.get("em"), "f1": s.get("f1"), "n": s["n_total"]}
                        for s in summaries},
        "macro_em": (sum(s["em"] for s in scored) / len(scored)) if scored else None,
        "macro_f1": (sum(s["f1"] for s in scored) / len(scored)) if scored else None,
        "micro_em": (sum(s["em"] * s["n_scored"] for s in scored) / n_scored_total) if n_scored_total else None,
        "micro_f1": (sum(s["f1"] * s["n_scored"] for s in scored) / n_scored_total) if n_scored_total else None,
        "n_scored_total": n_scored_total,
        # Cross-dataset timing — sums in seconds, plus the headline LLM-vs-tool
        # split that justifies the search-engine optimizations in the paper.
        "timing": {
            "n_examples_total": total_ex,
            "llm_time_s":   total_llm,
            "tool_time_s":  total_tool,
            "total_time_s": total_wall,
            "llm_fraction_of_total":  (total_llm  / total_wall) if total_wall > 0 else 0.0,
            "tool_fraction_of_total": (total_tool / total_wall) if total_wall > 0 else 0.0,
            "mean_llm_time_per_example_s":  (total_llm  / total_ex) if total_ex > 0 else 0.0,
            "mean_tool_time_per_example_s": (total_tool / total_ex) if total_ex > 0 else 0.0,
        },
    }
    with open(os.path.join(args.out_dir, "overall.json"), "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    print("\n=== Overall ===", flush=True)
    print(f"{'dataset':<22}  {'EM':>8}  {'F1':>8}  {'n':>6}", flush=True)
    for s in summaries:
        em = f"{s['em']:>8.4f}" if "em" in s else f"{'—':>8}"
        f1 = f"{s['f1']:>8.4f}" if "f1" in s else f"{'—':>8}"
        print(f"{s['dataset']:<22}  {em}  {f1}  {s['n_total']:>6}", flush=True)
    if scored:
        print(f"{'macro':<22}  {overall['macro_em']:>8.4f}  {overall['macro_f1']:>8.4f}",
              flush=True)
        print(f"{'micro':<22}  {overall['micro_em']:>8.4f}  {overall['micro_f1']:>8.4f}  "
              f"{n_scored_total:>6}", flush=True)
    if summaries:
        ov_t = overall["timing"]
        print(f"\n=== Timing ===", flush=True)
        print(f"  n_examples_total : {ov_t['n_examples_total']}", flush=True)
        print(f"  LLM time         : {ov_t['llm_time_s']:.1f}s  "
              f"({ov_t['llm_fraction_of_total']*100:.1f}% of total, "
              f"{ov_t['mean_llm_time_per_example_s']:.2f}s/ex)", flush=True)
        print(f"  Tool time        : {ov_t['tool_time_s']:.1f}s  "
              f"({ov_t['tool_fraction_of_total']*100:.1f}% of total, "
              f"{ov_t['mean_tool_time_per_example_s']:.2f}s/ex)", flush=True)
        print(f"  Total time       : {ov_t['total_time_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
