#!/usr/bin/env python
"""run/aggregate.py — combine per-task eval_task outputs into the standard
per-condition contract, and verify the scene-fixed invariant (CLAUDE.md §7,§8,§12).

eval_task.py writes results/<model>/<suite>/<cond>/seed<k>/task<tid>.jsonl (one file
per task). This aggregates them, per (model, suite, condition, seed) dir, into:
  - episodes.jsonl  (concatenation of task*.jsonl, so analyze/* work unchanged)
  - summary.json    (n_success, n_total, tsr, per_task, checkpoint_hash)
and appends a MANIFEST.json record.

Scene-fixed assertion (§5/§12): for each (task_id, episode), the reset_state_hash
must be IDENTICAL across all conditions (only the instruction string changes;
init state is chosen by episode index alone). Result ->
report/scene_fixed_check_<model>_<suite>.json, and a FAILING check exits 2 — the
causal claims are only valid while it passes.

Usage: python run/aggregate.py [--model smolvla --suite libero_goal --results_root ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "run"))
from run_one import append_manifest, atomic_write_json, git_sha  # noqa: E402


def read_task_files(seed_dir: Path):
    recs = []
    for tf in sorted(seed_dir.glob("task*.jsonl")):
        recs += [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
    return recs


def aggregate_condition(seed_dir: Path, model, suite, condition, seed) -> dict | None:
    recs = read_task_files(seed_dir)
    if not recs:
        return None
    # write concatenated episodes.jsonl (sorted by task then episode)
    recs.sort(key=lambda r: (r["task_id"], r["episode"]))
    (seed_dir / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    n_total = len(recs)
    n_success = sum(int(r["success"]) for r in recs)
    per_task = defaultdict(lambda: {"n": 0, "success": 0})
    for r in recs:
        per_task[r["task_id"]]["n"] += 1
        per_task[r["task_id"]]["success"] += int(r["success"])
    n_episodes = max((v["n"] for v in per_task.values()), default=0)
    import yaml
    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        mc = yaml.safe_load(f).get(model, {})
    ckpt_name = next((r.get("checkpoint") for r in recs if r.get("checkpoint")), mc.get("hf_repo"))
    ckpt_hash = next((r.get("checkpoint_hash") for r in recs if r.get("checkpoint_hash")), None)
    framework = mc.get("framework", "lerobot")
    runner = "eval_task_oft.py" if model == "openvla_oft" else ("eval_task_openvla.py" if model == "openvla" else "eval_task.py")
    summary = {
        "model": model, "suite": suite, "condition": condition, "seed": seed,
        "checkpoint": ckpt_name, "checkpoint_hash": ckpt_hash,
        "n_episodes": n_episodes, "n_total_episodes": n_total, "n_success": n_success,
        "tsr": (n_success / n_total) if n_total else None,
        "per_task": {int(k): dict(v) for k, v in per_task.items()},
        "framework": framework, "runner": runner,
    }
    atomic_write_json(seed_dir / "summary.json", summary)
    return summary


def scene_fixed_check(base: Path):
    """reset_state_hash must match across core conditions per (task_id, episode)."""
    hashes = defaultdict(dict)
    core_conds = {"original", "blank", "nonsense", "wrong_action", "wrong_object", "wrong_task", "repeated"}
    for cond_dir in sorted(base.iterdir()):
        if not cond_dir.is_dir() or cond_dir.name not in core_conds:
            continue
        for seed_dir in cond_dir.glob("seed*"):
            for tf in seed_dir.glob("task*.jsonl"):
                lines = [l for l in tf.read_text().splitlines() if l.strip()]
                for line in lines:
                    try:
                        r = json.loads(line.replace("\x00", "").strip())
                        # Key on the RECORDED episode, not the line index. Line
                        # position only equals the episode number when the file is
                        # a gapless, duplicate-free, in-order prefix — which is
                        # exactly what the resume bug in eval_task.py broke. A
                        # position-keyed check silently compares different
                        # episodes to each other.
                        key = f"task{r['task_id']}_ep{r['episode']}_{seed_dir.name}"
                        if r.get("reset_state_hash") is not None:
                            hashes[key][cond_dir.name] = r["reset_state_hash"]
                    except Exception:
                        continue
    # A key holding ONE condition cannot mismatch -- there is nothing to compare it
    # against -- so counting it as "checked" inflates the check's apparent coverage.
    # Measured 2026-07-31 on openvla_oft: 200 keys reported, but 160 of them held
    # only `original` (its 10-episode rerun) while the other conditions sat at 2
    # episodes. The honest coverage was 40 keys, not 200. Reporting the larger number
    # is the same silent-success pattern as the rest of this pipeline's history, so
    # comparable and vacuous keys are now counted separately and a tree with nothing
    # to compare FAILS instead of passing.
    mismatches = []
    comparable = 0
    vacuous = 0
    depth: dict[int, int] = defaultdict(int)
    for key, cond_hash in hashes.items():
        depth[len(cond_hash)] += 1
        if len(cond_hash) < 2:
            vacuous += 1
            continue
        comparable += 1
        if len(set(cond_hash.values())) > 1:
            mismatches.append({"key": key, "hashes": cond_hash})
    return {
        "pass": comparable > 0 and not mismatches,
        "n_keys_comparable": comparable,
        "n_keys_vacuous": vacuous,
        "n_keys_total": comparable + vacuous,
        "conds_per_key": dict(sorted(depth.items())),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "note": ("scene provably fixed iff pass; only keys carrying >=2 conditions "
                 "are evidence, and n_keys_vacuous are single-condition keys that "
                 "cannot mismatch"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
    ap.add_argument("--allow-scene-mismatch", action="store_true",
                    help="report a scene-fixed failure but still exit 0 (for "
                         "inspecting a known-broken tree; never for paper data)")
    args = ap.parse_args()
    base = Path(args.results_root) / args.model / args.suite
    if not base.exists():
        print(f"[aggregate] nothing at {base}")
        return 0
    manifest = REPO_ROOT / "MANIFEST.json"
    for cond_dir in sorted(base.iterdir()):
        if not cond_dir.is_dir():
            continue
        for seed_dir in sorted(cond_dir.glob("seed*")):
            seed = int(seed_dir.name.replace("seed", ""))
            s = aggregate_condition(seed_dir, args.model, args.suite, cond_dir.name, seed)
            if s is None:
                continue
            append_manifest(manifest, {
                "model": args.model, "suite": args.suite, "condition": cond_dir.name,
                "seed": seed, "n_episodes": s["n_episodes"], "status": "OK",
                "n_success": s["n_success"], "tsr": s["tsr"],
                "checkpoint_hash": s["checkpoint_hash"], "repo_sha": git_sha(REPO_ROOT),
                "lerobot_sha": git_sha(REPO_ROOT / "lerobot"),
                "runner": "eval_task.py+aggregate", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            print(f"[aggregate] {cond_dir.name}/seed{seed}: TSR={s['tsr']} ({s['n_success']}/{s['n_total_episodes']})")
    check = scene_fixed_check(base)
    check["model"], check["suite"] = args.model, args.suite
    (REPO_ROOT / "report").mkdir(exist_ok=True)
    # PER-MODEL path. This used to write one shared scene_fixed_check.json, so
    # each model's aggregate run overwrote the previous one and the committed file
    # described only whichever model happened to run last — the same
    # one-output-overwritten-per-invocation bug that produced RQ4's duplicated
    # block. The unsuffixed name is kept as a pointer to the per-model files.
    atomic_write_json(REPO_ROOT / "report" / f"scene_fixed_check_{args.model}_{args.suite}.json", check)
    print(f"[aggregate] scene_fixed_check: pass={check['pass']} "
          f"({check['n_keys_comparable']} comparable keys, "
          f"{check['n_mismatches']} mismatches; "
          f"{check['n_keys_vacuous']} single-condition keys carry no evidence; "
          f"conds/key={check['conds_per_key']})")
    if check["pass"] and check["n_keys_vacuous"] > check["n_keys_comparable"]:
        print(f"[aggregate] WARNING: most keys ({check['n_keys_vacuous']} of "
              f"{check['n_keys_total']}) hold a single condition, so this pass rests "
              f"on {check['n_keys_comparable']} keys. Thin, not clean -- the grid is "
              f"incomplete rather than verified.", file=sys.stderr)

    if not check["pass"]:
        # Do NOT let this pass silently. The scene-fixed property is the entire
        # basis for the causal claims (§5/§12): if two conditions ran from
        # different initial scenes, their TSR difference is confounded and cannot
        # be attributed to the instruction. This sat at pass=false in the repo
        # while the paper asserted the opposite.
        print(f"\n!!! SCENE-FIXED CHECK FAILED for {args.model}/{args.suite}: "
              f"{check['n_mismatches']}/{check['n_keys_comparable']} comparable "
              f"episodes ran from different scenes across conditions "
              f"({check['n_keys_vacuous']} further keys held one condition and "
              f"could not be checked at all).\n"
              f"!!! Causal claims for this model are NOT valid until this passes.\n"
              f"!!! All 7 conditions for a (task, seed) must be produced by ONE "
              f"runner process; cross-process cells do not share scenes.",
              file=sys.stderr)
        if not args.allow_scene_mismatch:
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
