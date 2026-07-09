#!/usr/bin/env python
"""analyze/locus.py — planning-level vs execution-level failure classifier (§7).

For each FAILED episode, decide from the eef trajectory + object poses:
  - Planning-level: the gripper never approached the correct target object
    (min eef-to-target distance over the trajectory > APPROACH_THRESH_M) → wrong
    *what*.
  - Execution-level: the gripper reached/engaged the correct object but the
    manipulation failed (min distance <= threshold) → right *what*, bad *how*.

Target object = the movable object in the task's bddl :obj_of_interest, whose
init pose we read from the episode's init_object_poses. Distance uses the
subsampled eef_traj recorded during rollout.

THRESHOLD is documented here and MUST be sanity-checked on >=10 hand-labeled
episodes (see validate()) before trusting at scale (§7).
"""
from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APPROACH_THRESH_M = 0.08  # eef within 8cm of target's init pos == "approached"

# task_id -> target movable object's pose key (from bddl :obj_of_interest).
# Fixture-only targets (drawer/stove) have no movable target -> use None and fall
# back to "planning if never moved anything".
TASK_TARGET_POSE_KEY = {
    0: None,                         # open middle drawer (fixture)
    1: "akita_black_bowl_1_pos",     # bowl -> stove
    2: "wine_bottle_1_pos",          # wine bottle -> cabinet top
    3: "akita_black_bowl_1_pos",     # open drawer + bowl inside
    4: "akita_black_bowl_1_pos",     # bowl -> cabinet top
    5: "plate_1_pos",                # push plate
    6: "cream_cheese_1_pos",         # cream cheese -> bowl
    7: None,                         # turn on stove (fixture)
    8: "akita_black_bowl_1_pos",     # bowl -> plate
    9: "wine_bottle_1_pos",          # wine bottle -> rack
}


def classify_episode(rec) -> str | None:
    """Return 'planning' | 'execution' | None (None if success or unclassifiable)."""
    if rec.get("success"):
        return None
    key = TASK_TARGET_POSE_KEY.get(int(rec["task_id"]))
    traj = rec.get("eef_traj") or []
    if not traj:
        return None
    if key is None:
        # fixture task: no movable target; treat as planning unless eef moved a lot
        return "planning"
    tgt = rec.get("init_object_poses", {}).get(key)
    if tgt is None:
        return None
    tgt = np.asarray(tgt[:3])
    dmin = min(float(np.linalg.norm(np.asarray(p[:3]) - tgt)) for p in traj)
    return "execution" if dmin <= APPROACH_THRESH_M else "planning"


def load_failed(results_root: Path, model: str, suite: str, condition: str):
    recs = []
    base = results_root / model / suite / condition
    if not base.exists():
        return recs
    for seed_dir in sorted(base.glob("seed*")):
        ep = seed_dir / "episodes.jsonl"
        if ep.exists():
            recs += [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    return [r for r in recs if not r.get("success")]


def locus_distribution(results_root, model, suite, condition):
    failed = load_failed(results_root, model, suite, condition)
    counts = {"planning": 0, "execution": 0, "unclassified": 0}
    for r in failed:
        c = classify_episode(r)
        counts["unclassified" if c is None else c] += 1
    counts["n_failed"] = len(failed)
    counts["condition"] = condition
    return counts


def validate(results_root, model, suite, condition, labels_path):
    """Compare classifier vs a hand-labeled JSON:
    {"<task_id>_<episode>_<seed>": "planning"|"execution"}. Prints agreement."""
    labels = json.loads(Path(labels_path).read_text())
    failed = load_failed(Path(results_root), model, suite, condition)
    by_key = {f"{r['task_id']}_{r['episode']}_{r['seed']}": r for r in failed}
    ok = tot = 0
    for k, gold in labels.items():
        if k in by_key:
            pred = classify_episode(by_key[k])
            tot += 1
            ok += int(pred == gold)
            print(f"  {k}: pred={pred} gold={gold} {'OK' if pred==gold else 'MISS'}")
    print(f"agreement: {ok}/{tot}" + (f" = {ok/tot:.2f}" if tot else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--condition", default="original")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
    ap.add_argument("--validate", default=None, help="path to hand-labeled JSON")
    args = ap.parse_args()
    if args.validate:
        validate(args.results_root, args.model, args.suite, args.condition, args.validate)
    else:
        print(json.dumps(locus_distribution(Path(args.results_root), args.model, args.suite, args.condition), indent=2))


if __name__ == "__main__":
    main()
