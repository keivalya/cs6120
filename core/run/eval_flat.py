#!/usr/bin/env python
"""core/run/eval_flat.py — flat, module-level SmolVLA causal eval (CLAUDE.md §5,§7,§8).

Mirrors the proven standalone structure (paper/patches/working_eval_reference.py):
fork the AsyncVectorEnv render workers at module level, BEFORE CUDA/policy, then
load the policy once and iterate conditions × tasks × episodes REUSING the same
forked envs (only the injected `task_description` string changes; the bddl file
and init_states are never touched, so the scene is provably fixed — §5, §12).

This exists because the equivalent logic wrapped inside run_one.py's main()/
run_lerobot() call stack aborts the EGL render worker (SIGABRT in robosuite
read_pixels); the flat form does not. Same output contract as run_one:
results/<model>/<suite>/<condition>/seed<k>/{episodes.jsonl,summary.json} + MANIFEST.

Usage:
  python core/run/eval_flat.py --model smolvla --suite libero_goal \
    --conditions original,blank,nonsense --seed 7 --n_episodes 2 [--task_ids 1,7]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core" / "run"))

# --------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="smolvla")
ap.add_argument("--suite", default="libero_goal")
ap.add_argument("--conditions", required=True)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--n_episodes", type=int, default=2)
ap.add_argument("--task_ids", default=None)
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--deadline_s", type=float, default=None)
ap.add_argument("--results_root", default=str(REPO_ROOT / "data" / "results"))
args = ap.parse_args()

conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
task_ids = [int(x) for x in args.task_ids.split(",")] if args.task_ids else None
results_root = Path(args.results_root)

# ---- read all condition instructions + task metadata from the generated jsonl
# (plain json only — NO yaml/run_one imports before the fork, to keep the
# pre-fork parent state minimal, matching paper/patches/working_eval_reference.py;
# heavier imports before forking abort the EGL render worker).
gen = REPO_ROOT / "data" / "instructions" / f"{args.suite}.jsonl"
by_cond: dict[str, dict[int, str]] = {}
true_langs: dict[int, str] = {}
task_names: dict[int, str] = {}
for line in gen.read_text().splitlines():
    r = json.loads(line)
    if r.get("skip") or r.get("instruction") is None:
        if r["condition"] != "original":
            continue
    by_cond.setdefault(r["condition"], {})[int(r["task_id"])] = r["instruction"]
    if r["condition"] == "original":
        true_langs[int(r["task_id"])] = r["instruction"]
        task_names[int(r["task_id"])] = r["task_name"]

all_ids = sorted(true_langs) if task_ids is None else task_ids

# --------------------------------------------------------------------------- #
# Import order matches the known-good standalone path: probe env + lerobot
# modules imported in the PARENT before forking; only make_policy() (CUDA) after.
# --------------------------------------------------------------------------- #
from _libero_probe import make_probe_vec  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs import make_env_pre_post_processors, preprocess_observation  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

# FORK the render workers (before CUDA/make_policy)
probe_vecs = {i: make_probe_vec(args.suite, int(i), args.max_steps) for i in all_ids}
for v in probe_vecs.values():
    v.reset(seed=args.seed)
print(f"[eval_flat] forked {len(probe_vecs)} render workers", flush=True)

# ---- now (post-fork) it's safe to import yaml/run_one and read config
import yaml  # noqa: E402
from run_one import append_manifest, atomic_write_json, git_sha, set_all_seeds  # noqa: E402

with open(REPO_ROOT / "configs" / "models.yaml") as f:
    model_cfg = yaml.safe_load(f)[args.model]
hf_repo = model_cfg["hf_repo"]
device = model_cfg.get("eval_flags", {}).get("device", "cuda:0")

# ---- CUDA + policy + processors
env_cfg = LiberoEnvConfig(task=args.suite, task_ids=all_ids)
policy_cfg = PreTrainedConfig.from_pretrained(hf_repo)
policy_cfg.pretrained_path = hf_repo
policy_cfg.device = device
policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
policy.eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy_cfg, pretrained_path=hf_repo,
    preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
)
env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
print("[eval_flat] policy + processors ready", flush=True)

try:
    from huggingface_hub import HfApi
    ckpt_hash = HfApi().model_info(hf_repo).sha
except Exception:
    ckpt_hash = None


def _eef(obs):
    try:
        return np.asarray(obs["robot_state"]["eef"]["pos"]).reshape(-1)[:3].astype(float).tolist()
    except Exception:
        return None


def run_episode(vec, instr, seed, ep):
    vec.set_attr("init_state_id", ep)
    vec.set_attr("task_description", instr)
    policy.reset()
    set_all_seeds(seed + ep)
    obs, _ = vec.reset(seed=seed + ep)
    probe = vec.call("probe_state")[0]
    done, step, success, eef_traj = False, 0, False, []
    t0 = time.time()
    while not done and step < args.max_steps:
        o = preprocess_observation(obs)
        o["task"] = list(vec.call("task_description"))
        o = env_preprocessor(o)
        o = preprocessor(o)
        with torch.inference_mode():
            action = policy.select_action(o)
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        obs, _, term, trunc, info = vec.step(action.to("cpu").numpy())
        success = success or bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0])
        if step % 10 == 0 and (e := _eef(obs)) is not None:
            eef_traj.append(e)
        done = bool(np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0])
        step += 1
    final = vec.call("probe_state")[0]
    return {
        "success": bool(success), "steps": step,
        "reset_state_hash": probe.get("hash"),
        "init_object_poses": probe.get("poses", {}),
        "final_object_poses": final.get("poses", {}),
        "eef_traj": eef_traj, "wall_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------- #
t_start = time.time()
manifest = REPO_ROOT / "MANIFEST.json"
node = __import__("os").environ.get("SLURMD_NODENAME") or __import__("os").uname().nodename
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

for cond in conditions:
    if args.deadline_s is not None and (time.time() - t_start) > args.deadline_s:
        print(f"[eval_flat] DEADLINE — stopping before {cond}", flush=True)
        break
    instr_map = None if cond == "original" else by_cond.get(cond, {})
    run_ids = [i for i in all_ids if instr_map is None or i in instr_map]
    out_dir = results_root / args.model / args.suite / cond / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_file = open(out_dir / "episodes.jsonl", "w")
    n_succ = n_tot = 0
    per_task = {}
    for tid in run_ids:
        instr = true_langs[tid] if instr_map is None else instr_map[tid]
        vec = probe_vecs[tid]
        ts = 0
        for ep in range(args.n_episodes):
            r = run_episode(vec, instr, args.seed, ep)
            rec = {
                "model": args.model, "suite": args.suite, "condition": cond, "seed": args.seed,
                "task_id": tid, "task_name": task_names[tid], "episode": ep,
                "instruction": instr, "true_instruction": true_langs[tid], **r,
            }
            ep_file.write(json.dumps(rec) + "\n"); ep_file.flush()
            n_tot += 1; ts += int(r["success"]); n_succ += int(r["success"])
            print(f"[{cond} task {tid} ep {ep}] success={r['success']} steps={r['steps']}", flush=True)
        per_task[tid] = {"n": args.n_episodes, "success": ts}
    ep_file.close()
    summary = {
        "model": args.model, "checkpoint": hf_repo, "checkpoint_hash": ckpt_hash,
        "suite": args.suite, "condition": cond, "seed": args.seed,
        "n_episodes": args.n_episodes, "n_total_episodes": n_tot, "n_success": n_succ,
        "tsr": (n_succ / n_tot) if n_tot else None, "per_task": per_task,
        "max_steps": args.max_steps, "framework": "lerobot",
    }
    atomic_write_json(out_dir / "summary.json", summary)
    append_manifest(manifest, {
        "model": args.model, "suite": args.suite, "condition": cond, "seed": args.seed,
        "n_episodes": args.n_episodes, "status": "OK", "n_success": n_succ,
        "tsr": summary["tsr"], "checkpoint_hash": ckpt_hash,
        "repo_sha": git_sha(REPO_ROOT), "lerobot_sha": git_sha(REPO_ROOT / "lerobot"),
        "node": node, "gpu": gpu, "wall_s": round(time.time() - t_start, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    print(f"[eval_flat] {cond} TSR={summary['tsr']} ({n_succ}/{n_tot})", flush=True)

for v in probe_vecs.values():
    v.close()
print(f"[eval_flat] DONE total {time.time()-t_start:.0f}s", flush=True)
