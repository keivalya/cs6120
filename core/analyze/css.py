#!/usr/bin/env python
"""core/analyze/css.py — Causal Sensitivity Score + Follow/Ignore/Fail + OAR (§7).

From raw episodes.jsonl:

- CSS(cond) = (TSR_original − TSR_cond) / max(TSR_original, eps), for
  cond ∈ {blank, nonsense}. CSS≈0 ⇒ language ignored; CSS≈1 ⇒ strongly causal.
  Reported per condition; blank and nonsense are NOT averaged together (§7).

- Follow / Ignore / Fail for wrong_* conditions, via the §7 goal-predicate method
  when the episode carries `achieved_task_ids` (all 10 LIBERO-Goal goals evaluated
  against the final sim state by the runner; see core/analyze/goal_eval.py). Because the
  env goal always encodes the TRUE task:
    * Ignore (→ OAR) = the TRUE task's id is in achieved_task_ids (did the true
      task despite the wrong instruction). EXACT.
    * Follow = the WRONGLY-INSTRUCTED task's id is in achieved_task_ids and the
      true task is not. EXACT for `wrong_task` (the wrong instruction IS another
      Goal task, id recorded as `wrong_task_id`). For `wrong_object`/`wrong_action`
      the instructed target is NOT one of the 10 Goal tasks, so goal-predicate
      Follow is undefined; we fall back to the documented >MOVE_THRESH_M
      displacement heuristic for `wrong_object` and report Follow as approximate.
    * Fail = neither.
  Back-compat: episodes without `achieved_task_ids` (pre-goal-logging runs) fall
  back to env `success` for Ignore + the geometry heuristic for Follow (flagged).
  OAR = fraction Ignore. High OAR is the strongest evidence the policy reads
  pixels, not words (§7).
"""
from __future__ import annotations
import argparse, json
import numpy as np
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-9
MOVE_THRESH_M = 0.05  # a movable object displaced >5cm counts as "manipulated"

# object noun (as used in perturbations) -> pose key prefix in the raw obs
NOUN_TO_POSE_KEY = {
    "bowl": "akita_black_bowl_1_pos",
    "cream cheese": "cream_cheese_1_pos",
    "wine bottle": "wine_bottle_1_pos",
    "plate": "plate_1_pos",
}


def load_episodes(results_root: Path, model: str, suite: str, condition: str):
    recs = []
    base = results_root / model / suite / condition
    if not base.exists():
        return recs
    for seed_dir in sorted(base.glob("seed*")):
        ep = seed_dir / "episodes.jsonl"
        if ep.exists():
            recs += [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]
    return recs


def tsr_of(recs):
    if not recs:
        return None
    return sum(int(r["success"]) for r in recs) / len(recs)


def css(results_root, model, suite):
    orig = load_episodes(results_root, model, suite, "original")
    t_orig = tsr_of(orig)
    out = {"tsr_original": t_orig}
    for cond in ("blank", "nonsense"):
        recs = load_episodes(results_root, model, suite, cond)
        t = tsr_of(recs)
        out[cond] = {
            "tsr": t,
            "css": ((t_orig - t) / max(t_orig, EPS)) if (t_orig is not None and t is not None) else None,
            "n": len(recs),
        }
    return out


def _disp(rec, pose_key):
    """Displacement (m) of an object between init and final poses."""
    ip = rec.get("init_object_poses", {}).get(pose_key)
    fp = rec.get("final_object_poses", {}).get(pose_key)
    if ip is None or fp is None:
        return None
    return float(np.linalg.norm(np.asarray(fp[:3]) - np.asarray(ip[:3])))


def follow_ignore_fail(results_root, model, suite, condition):
    """Trichotomy for a wrong_* condition (goal-predicate method; §7). Counts + OAR."""
    recs = load_episodes(results_root, model, suite, condition)
    # per task_id: wrongly-referenced object noun (wrong_object) + substituted
    # task id (wrong_task), from the perturbation provenance.
    gen = REPO_ROOT / "data" / "instructions" / f"{suite}.jsonl"
    wrong_obj_noun, wrong_tid = {}, {}
    if gen.exists():
        for line in gen.read_text().splitlines():
            r = json.loads(line)
            if r["condition"] != condition:
                continue
            if r.get("object_to"):
                wrong_obj_noun[int(r["task_id"])] = r["object_to"]
            if r.get("wrong_task_id") is not None:
                wrong_tid[int(r["task_id"])] = int(r["wrong_task_id"])
    n = len(recs)
    ignore = follow = fail = 0
    n_exact = 0  # episodes classified via achieved_task_ids (vs fallback)
    for r in recs:
        tid = int(r["task_id"])
        achieved = r.get("achieved_task_ids")
        if achieved is not None:
            n_exact += 1
            true_done = tid in achieved
        else:
            true_done = bool(r["success"])  # back-compat fallback
        if true_done:
            ignore += 1
            continue
        # not Ignore -> Follow or Fail
        did_follow = False
        if condition == "wrong_task" and achieved is not None and tid in wrong_tid:
            did_follow = wrong_tid[tid] in achieved          # EXACT
        else:
            # wrong_object/action (or fallback): >MOVE_THRESH_M displacement of the
            # wrongly-named movable object. Approximate.
            noun = wrong_obj_noun.get(tid)
            if noun and noun in NOUN_TO_POSE_KEY:
                d = _disp(r, NOUN_TO_POSE_KEY[noun])
                did_follow = d is not None and d > MOVE_THRESH_M
        follow += int(did_follow)
        fail += int(not did_follow)
    exact = (n_exact == n and n > 0)
    if condition == "wrong_task":
        method = "goal-predicate (exact)" if exact else "mixed/fallback (success+heuristic)"
    else:
        method = "OAR exact (goal-predicate); Follow via >5cm displacement heuristic (approx)"
    return {
        "condition": condition, "n": n, "n_goal_predicate": n_exact,
        "ignore": ignore, "follow": follow, "fail": fail,
        "OAR": (ignore / n) if n else None,
        "follow_rate": (follow / n) if n else None,
        "fail_rate": (fail / n) if n else None,
        "method": method,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "data" / "results"))
    args = ap.parse_args()
    rr = Path(args.results_root)
    c = css(rr, args.model, args.suite)
    print("== RQ1.1 CSS ==")
    print(json.dumps(c, indent=2))
    print("== RQ1.2 Follow/Ignore/Fail + OAR ==")
    for cond in ("wrong_object", "wrong_action", "wrong_task"):
        print(json.dumps(follow_ignore_fail(rr, args.model, args.suite, cond), indent=2))


if __name__ == "__main__":
    main()
