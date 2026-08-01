#!/usr/bin/env python
"""core/perturb/make_paraphrases.py — RQ2 paraphrase-condition generator (CLAUDE.md §3,§7).

Builds the meaning-preserving paraphrase conditions from the LIBERO-Para suite
(held-out eval strings only — we never author new demos, §3). Joins
`LIBERO-Para/metrics/libero_para_metadata.csv` to our 10 LIBERO-Goal tasks on
`original_instruction == task.language`, keeping the two RQ2 axes + compositional:

  high == "obj"  -> para_object          (object-referring paraphrases, RQ2.2)
  high == "act"  -> para_action          (action-referring paraphrases, RQ2.2)
  high == "comp" -> para_compositional   (both axes; the bulk of the suite)

Each emitted row carries the LIBERO-Para similarity fields (keyword_similarity SK,
structural_similarity ST) and the operation label (`mid`, RQ2.3) so PRIDE / PD and
the per-operation breakdown can be computed downstream without re-joining
(core/analyze/pride_wrap.py, core/analyze/make_rq2_csv.py). Output (one row per paraphrase):
  core/perturb/generated/<suite>_paraphrases.jsonl

Sampling (§2 budget, §11 no-silent-truncation): `--max_per_axis N` caps paraphrases
per (task, axis), **stratified by operation** (round-robin over `mid` groups) so a
capped 7B run still spans the PD range instead of one operation type. Default None =
keep all (SmolVLA full). Kept/total counts are printed per axis.

Usage:
  python core/perturb/make_paraphrases.py --suite libero_goal [--max_per_axis 15] [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = REPO_ROOT / "LIBERO-Para" / "metrics" / "libero_para_metadata.csv"

AXIS = {"obj": "para_object", "act": "para_action", "comp": "para_compositional"}


def stratified_sample(rows, k, rng):
    """Up to k rows, round-robin across `operation` groups for a PD/op spread."""
    if k is None or len(rows) <= k:
        return rows
    groups = defaultdict(list)
    for r in rows:
        groups[r["operation"]].append(r)
    for g in groups.values():
        rng.shuffle(g)
    out, order = [], sorted(groups)
    while len(out) < k and any(groups[g] for g in order):
        for g in order:
            if groups[g]:
                out.append(groups[g].pop())
                if len(out) >= k:
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--max_per_axis", type=int, default=None,
                    help="cap paraphrases per (task,axis); None=all (SmolVLA full)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from libero.libero import benchmark  # noqa: E402

    ts = benchmark.get_benchmark_dict()[args.suite]()
    lang_to_task = {}
    for i in range(ts.get_num_tasks()):
        t = ts.get_task(i)
        lang_to_task[t.language.strip().lower()] = {"task_id": i, "task_name": t.name}

    # group CSV rows by (task_id, axis)
    by_cell = defaultdict(list)
    unmatched = 0
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            axis_key = row.get("high")
            if axis_key not in AXIS:
                continue
            orig = row["original_instruction"].strip().lower()
            tinfo = lang_to_task.get(orig)
            if tinfo is None:
                unmatched += 1
                continue
            by_cell[(tinfo["task_id"], AXIS[axis_key])].append({
                "suite": args.suite,
                "task_id": tinfo["task_id"],
                "task_name": tinfo["task_name"],
                "axis": AXIS[axis_key],
                "instruction": row["new_instruction"].strip(),
                "original_instruction": row["original_instruction"].strip(),
                "keyword_similarity": float(row["keyword_similarity"]),
                "structural_similarity": float(row["structural_similarity"]),
                "operation": row.get("mid", ""),
            })

    out_path = REPO_ROOT / "data" / "instructions" / f"{args.suite}_paraphrases.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept_counts = defaultdict(lambda: [0, 0])  # axis -> [kept, total]
    all_rows = []
    for (tid, axis), rows in sorted(by_cell.items()):
        rng = random.Random(f"{args.suite}:{tid}:{axis}:{args.seed}")
        sampled = stratified_sample(rows, args.max_per_axis, rng)
        for j, r in enumerate(sampled):
            r["para_idx"] = j
            all_rows.append(r)
        kept_counts[axis][0] += len(sampled)
        kept_counts[axis][1] += len(rows)

    with open(out_path, "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(all_rows)} paraphrase rows to {out_path}")
    if unmatched:
        print(f"  (note: {unmatched} CSV rows had no matching {args.suite} task language)")
    for axis in ("para_object", "para_action", "para_compositional"):
        k, tot = kept_counts[axis]
        cap = "all" if args.max_per_axis is None else f"cap={args.max_per_axis}/task"
        print(f"  {axis:20s} kept={k:4d} / total={tot:4d}  ({cap})")


if __name__ == "__main__":
    main()
