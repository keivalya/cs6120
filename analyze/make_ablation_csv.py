#!/usr/bin/env python3
"""analyze/make_ablation_csv.py — the word-class ablation, the paper's decisive test.

Section~\\ref{sec:info} predicts, from the benchmark's information structure alone:

    verb_dropped   ("the bowl on the stove")        removes 0.00 bits -> ~no cost
    nouns_masked   ("put the thing on the thing")   removes 3.32 bits -> collapse

This is a REMOVAL test, not a substitution test, and that is the point: the standard
verb-substitution probe cannot detect verb-insensitivity on this suite because seven
of its ten substitutions leave the noun set (and hence the task) intact.

Writes report/rq_ablation.csv with one row per (model, condition), including the
scene-matched baseline restricted to the episodes the ablation actually ran, since
the ablation conditions are produced in the same process as `original` and so share
initial scenes -- unlike the paraphrase pools.

Usage: python analyze/make_ablation_csv.py [--suite libero_goal]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ["smolvla", "openvla", "openvla_oft"]
ABLATIONS = ["verb_dropped", "nouns_masked"]
# Reference points: `original` is the ceiling, `blank` the floor. nouns_masked keeps
# the syntactic frame and the verb while destroying the object references, so it is
# informative to see where it lands between the two.
REFS = ["original", "blank"]


def load(base: Path, cond: str) -> list[dict]:
    recs = []
    for f in sorted(base.glob(f"{cond}/seed*/task*.jsonl")):
        seed = f.parent.name
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            r["_seed"] = seed
            recs.append(r)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    out_rows = []
    for model in MODELS:
        base = Path(args.results_root) / model / args.suite
        if not base.exists():
            continue
        byc = {c: load(base, c) for c in REFS + ABLATIONS}
        orig = byc.get("original") or []
        # (task, episode, seed) -> success, for the scene-matched baseline
        orig_key = {(r["task_id"], r["episode"], r["_seed"]): int(r["success"]) for r in orig}

        for cond in REFS + ABLATIONS:
            recs = byc.get(cond) or []
            if not recs:
                continue
            n = len(recs)
            tsr = sum(int(r["success"]) for r in recs) / n
            tasks = sorted({r["task_id"] for r in recs})
            # Baseline over exactly the episodes this condition covers. For the
            # ablations this is the honest comparison: same process, same scenes.
            matched = [orig_key[k] for k in
                       ((r["task_id"], r["episode"], r["_seed"]) for r in recs)
                       if k in orig_key]
            base_tsr = sum(matched) / len(matched) if matched else None
            out_rows.append({
                "model": model,
                "condition": cond,
                "n": n,
                "n_tasks": len(tasks),
                "tasks": " ".join(str(t) for t in tasks),
                "TSR": f"{tsr:.4f}",
                "n_matched_baseline": len(matched),
                "TSR_matched_baseline": "" if base_tsr is None else f"{base_tsr:.4f}",
                "delta_pp_matched": "" if base_tsr is None else f"{100 * (base_tsr - tsr):.2f}",
                "bits_removed": {"verb_dropped": "0.00", "nouns_masked": "3.32",
                                 "blank": "3.32", "original": "0.00"}[cond],
            })

    out = REPO_ROOT / "report" / "rq_ablation.csv"
    if not out_rows:
        print("[ablation] no data yet — nothing written")
        return 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[ablation] wrote {out.relative_to(REPO_ROOT)} ({len(out_rows)} rows)")
    for r in out_rows:
        print(f"  {r['model']:<12} {r['condition']:<14} n={r['n']:<4} "
              f"tasks={r['n_tasks']:<3} TSR={float(r['TSR']):6.1%} "
              f"delta_vs_matched={r['delta_pp_matched'] or '--':>7} pp "
              f"(bits removed {r['bits_removed']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
