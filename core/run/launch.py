#!/usr/bin/env python
"""core/run/launch.py — autonomous sweep driver (CLAUDE.md §8).

Drives the reliable per-task runner core/run/eval_task.py: one process per LIBERO task
(one EGL context = the stable rendering mode on this stack), each process sweeping
all requested conditions. Idempotent/resumable: a (task, per-condition) cell is
skipped if its results/.../<cond>/seed<k>/task<tid>.jsonl already has >= requested
episodes. A global --deadline_s wall budget stops starting new tasks so a compute
cap is never blown (§12). After the sweep, run core/run/aggregate.py to build the
per-condition summary.json/episodes.jsonl + MANIFEST + scene-fixed check.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def task_conditions_done(results_root, model, suite, seed, task_id, conditions, n_episodes):
    """True iff every scene-valid condition for this task already has >= n_episodes."""
    for cond in conditions:
        f = results_root / model / suite / cond / f"seed{seed}" / f"task{task_id}.jsonl"
        if not f.exists():
            return False
        n = sum(1 for _ in f.read_text().splitlines() if _.strip())
        if n < n_episodes:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--conditions", required=True, help="comma-sep conditions")
    ap.add_argument("--task_ids", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_episodes", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--obs_hw", type=int, default=360)
    ap.add_argument("--deadline_s", type=float, default=None)
    ap.add_argument("--results_root", default=str(REPO_ROOT / "data" / "results"))
    ap.add_argument("--aggregate", action="store_true", help="run aggregate.py at the end")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    task_ids = [int(x) for x in args.task_ids.split(",")]
    results_root = Path(args.results_root)
    t0 = time.time()

    for tid in task_ids:
        elapsed = time.time() - t0
        if args.deadline_s is not None and elapsed > args.deadline_s:
            print(f"[launch] DEADLINE ({elapsed:.0f}s); stopping before task {tid}", flush=True)
            break
        if task_conditions_done(results_root, args.model, args.suite, args.seed, tid, conditions, args.n_episodes):
            print(f"[launch] SKIP task {tid} (done)", flush=True)
            continue
        cmd = [
            sys.executable, str(REPO_ROOT / "core" / "run" / "eval_task.py"),
            "--model", args.model, "--suite", args.suite, "--task_id", str(tid),
            "--conditions", ",".join(conditions), "--seed", str(args.seed),
            "--n_episodes", str(args.n_episodes), "--max_steps", str(args.max_steps),
            "--obs_hw", str(args.obs_hw), "--results_root", str(results_root),
        ]
        print(f"[launch] RUN task {tid} (elapsed {elapsed:.0f}s)", flush=True)
        rc = subprocess.run(cmd).returncode
        print(f"[launch] task {tid} rc={rc} (elapsed {time.time()-t0:.0f}s)", flush=True)

    if args.aggregate:
        print("[launch] aggregating...", flush=True)
        subprocess.run([sys.executable, str(REPO_ROOT / "core" / "run" / "aggregate.py"),
                        "--model", args.model, "--suite", args.suite,
                        "--results_root", str(results_root)])
    print(f"[launch] DONE total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
