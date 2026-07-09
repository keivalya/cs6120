#!/usr/bin/env python
"""analyze/css.py — Causal Sensitivity Score + Follow/Ignore/Fail + OAR (§7).

From raw episodes.jsonl:

- CSS(cond) = (TSR_original − TSR_cond) / max(TSR_original, eps), for
  cond ∈ {blank, nonsense}. CSS≈0 ⇒ language ignored; CSS≈1 ⇒ strongly causal.
  Reported per condition; blank and nonsense are NOT averaged together (§7).

- Follow / Ignore / Fail for wrong_* conditions. Because the bddl (and thus the
  env success check) always encodes the TRUE task, an episode that "succeeds"
  under a wrong instruction means the robot did the TRUE task:
    * Ignore (→ OAR) = env success under the wrong instruction. EXACT.
    * Follow = achieved the wrongly-instructed task instead. Detected by a
      documented geometry heuristic: the wrongly-referenced movable object moved
      substantially (> MOVE_THRESH_M) from its init pose AND the true task was
      not achieved. HEURISTIC — validate on hand-labeled episodes before trusting.
    * Fail = neither.
  OAR = fraction Ignore. High OAR under a wrong instruction is the strongest
  evidence the policy reads pixels, not words (§7).
"""
from __future__ import annotations
import argparse, json
import numpy as np
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    """Trichotomy for a wrong_* condition. Returns counts + OAR."""
    recs = load_episodes(results_root, model, suite, condition)
    # map task_id -> wrongly-referenced object noun (for wrong_object)
    gen = REPO_ROOT / "perturb" / "generated" / f"{suite}.jsonl"
    wrong_obj_noun = {}
    if gen.exists():
        for line in gen.read_text().splitlines():
            r = json.loads(line)
            if r["condition"] == condition and r.get("object_to"):
                wrong_obj_noun[int(r["task_id"])] = r["object_to"]
    n = len(recs)
    ignore = follow = fail = 0
    for r in recs:
        if r["success"]:            # did the TRUE task
            ignore += 1
            continue
        # Follow heuristic: the wrongly-named object was manipulated
        did_follow = False
        noun = wrong_obj_noun.get(int(r["task_id"]))
        if noun and noun in NOUN_TO_POSE_KEY:
            d = _disp(r, NOUN_TO_POSE_KEY[noun])
            if d is not None and d > MOVE_THRESH_M:
                did_follow = True
        follow += int(did_follow)
        fail += int(not did_follow)
    return {
        "condition": condition, "n": n,
        "ignore": ignore, "follow": follow, "fail": fail,
        "OAR": (ignore / n) if n else None,
        "follow_rate": (follow / n) if n else None,
        "fail_rate": (fail / n) if n else None,
        "note": "OAR/Ignore exact; Follow via >5cm displacement heuristic (validate on hand-labeled eps)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
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
