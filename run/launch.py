#!/usr/bin/env python
"""run/launch.py — autonomous sweep driver (CLAUDE.md §8).

Reads a list of (model, suite, condition, seed, n_episodes) cells (from
--conditions on the CLI; GATE 4 extends to configs/grid.yaml), and for each cell:
  - SKIPS it if results/.../seed<k>/summary.json already has n_episodes >= requested
    (idempotent / resumable, §8.2),
  - else dispatches run/run_one.py in the current conda env,
  - on error, run_one writes error.json and returns nonzero; we log and CONTINUE
    (§8.6 — one bad cell never halts the sweep).

A global --deadline_s wall budget stops STARTING new cells once exceeded, so a
compute-time cap is never blown; completed cells remain valid (§12).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def cell_done(results_root: Path, model, suite, condition, seed, n_episodes) -> bool:
    s = results_root / model / suite / condition / f"seed{seed}" / "summary.json"
    if not s.exists():
        return False
    try:
        d = json.loads(s.read_text())
        return d.get("n_episodes", 0) >= n_episodes and d.get("tsr") is not None
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--conditions", required=True, help="comma-sep ordered conditions")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_episodes", type=int, default=2)
    ap.add_argument("--task_ids", default=None)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--deadline_s", type=float, default=None, help="stop starting cells past this wall budget")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    results_root = Path(args.results_root)
    t0 = time.time()

    log = []
    for cond in conditions:
        elapsed = time.time() - t0
        if args.deadline_s is not None and elapsed > args.deadline_s:
            done = [l["condition"] for l in log if l["status"] in ("OK", "SKIP")]
            print(f"[launch] DEADLINE reached ({elapsed:.0f}s > {args.deadline_s}s); "
                  f"stopping before condition={cond}. Completed: {done}")
            break
        if cell_done(results_root, args.model, args.suite, cond, args.seed, args.n_episodes):
            print(f"[launch] SKIP (done): {args.model}/{args.suite}/{cond}/seed{args.seed}")
            log.append({"condition": cond, "status": "SKIP"})
            continue

        cmd = [
            sys.executable, str(REPO_ROOT / "run" / "run_one.py"),
            "--model", args.model, "--suite", args.suite, "--condition", cond,
            "--seed", str(args.seed), "--n_episodes", str(args.n_episodes),
            "--max_steps", str(args.max_steps), "--results_root", str(results_root),
        ]
        if args.task_ids:
            cmd += ["--task_ids", args.task_ids]
        print(f"[launch] RUN {cond} (elapsed {elapsed:.0f}s)", flush=True)
        r = subprocess.run(cmd)
        log.append({"condition": cond, "status": "OK" if r.returncode == 0 else "FAILED"})
        print(f"[launch] {cond} -> rc={r.returncode} (elapsed {time.time()-t0:.0f}s)", flush=True)

    print(f"[launch] DONE. total elapsed {time.time()-t0:.0f}s. summary: {log}")


if __name__ == "__main__":
    main()
