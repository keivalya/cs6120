#!/usr/bin/env python
"""run/eval_task.py — one LIBERO task, all conditions, SmolVLA (CLAUDE.md §5,§7).

Reliable rendering architecture discovered empirically on this node's
MuJoCo-EGL + CUDA stack:
  * A single SYNC OffScreenRenderEnv, created BEFORE CUDA is initialized, renders
    full-length episodes reliably (proven in GATE 2).
  * Multiple simultaneous EGL contexts (many tasks in one process) abort.
  * AsyncVectorEnv workers abort after ~49 renders.
So we run ONE process per task: build the single env first, load the policy, then
sweep all conditions REUSING that one env (only the injected task_description
changes — bddl/init_states untouched, scene provably fixed, §5/§12).

Output: results/<model>/<suite>/<condition>/seed<k>/task<tid>.jsonl  (one file per
task, so per-task processes never clash). analyze/* aggregate across task*.jsonl.

Usage:
  python run/eval_task.py --task_id 1 --conditions original,blank,nonsense \
    --seed 7 --n_episodes 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "run"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="smolvla")
ap.add_argument("--suite", default="libero_goal")
ap.add_argument("--task_id", type=int, required=True)
ap.add_argument("--conditions", required=True)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--n_episodes", type=int, default=2)
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--obs_hw", type=int, default=256, help="render resolution (H=W)")
ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
args = ap.parse_args()

conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
results_root = Path(args.results_root)
tid = args.task_id

# instructions per condition for THIS task (plain json; no heavy imports pre-env)
gen = REPO_ROOT / "perturb" / "generated" / f"{args.suite}.jsonl"
instr = {}  # condition -> instruction (None/absent => skip)
true_lang = task_name = None
for line in gen.read_text().splitlines():
    r = json.loads(line)
    if int(r["task_id"]) != tid:
        continue
    if r["condition"] == "original":
        true_lang = r["instruction"]
        task_name = r["task_name"]
    if not (r.get("skip") or r.get("instruction") is None):
        instr[r["condition"]] = r["instruction"]

# --------------------------------------------------------------------------- #
# Order matches the proven GATE 2 configuration (single process, SYNC env, full
# episodes, worked 6/6): load the policy (CUDA) FIRST, then make_env for this ONE
# task, then run. One process per task => one EGL context => reliable.
# --------------------------------------------------------------------------- #
import yaml  # noqa: E402

from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

with open(REPO_ROOT / "configs" / "models.yaml") as f:
    model_cfg = yaml.safe_load(f)[args.model]
hf_repo = model_cfg["hf_repo"]
device = model_cfg.get("eval_flags", {}).get("device", "cuda:0")

# Render at --obs_hw (default 256), not the lerobot default 360: on a degraded
# GPU channel, 360 triggered NVRM Xid 31 (graphics-engine MMU fault) mid-episode
# once CUDA was resident. 256 is a standard policy input size; the policy resizes
# internally so scene content is unchanged. On a healthy GPU 360 also works (it
# gave TSR=1.0 in the GATE 2 smoke) — bump --obs_hw if preferred.
env_cfg = LiberoEnvConfig(
    task=args.suite, task_ids=[tid],
    observation_height=args.obs_hw, observation_width=args.obs_hw,
)
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
print(f"[eval_task {tid}] policy ready", flush=True)

# make_env AFTER the policy (proven GATE 2 order)
envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
vec = envs[args.suite][tid]
sub = vec.envs[0]
print(f"[eval_task {tid}] env ready", flush=True)

from run_one import atomic_write_json, set_all_seeds  # noqa: E402

try:
    from huggingface_hub import HfApi
    ckpt_hash = HfApi().model_info(hf_repo).sha
except Exception:
    ckpt_hash = None


def poses_of():
    try:
        raw = sub._env.env._get_observations()
        return {k: np.asarray(v, dtype=np.float64).tolist()
                for k, v in raw.items() if k.endswith("_pos") or k.endswith("_quat")}
    except Exception:
        return {}


def reset_hash():
    try:
        st = sub._env.env.sim.get_state()
        return hashlib.sha256(
            np.concatenate([np.asarray(st.qpos), np.asarray(st.qvel)]).tobytes()
        ).hexdigest()[:16]
    except Exception:
        return None


def eef_of():
    try:
        return np.asarray(sub._env.env._get_observations()["robot0_eef_pos"], dtype=np.float64).tolist()
    except Exception:
        return None


for cond in conditions:
    if cond != "original" and cond not in instr:
        print(f"[eval_task {tid}] skip {cond} (no scene-valid probe)", flush=True)
        continue
    text = true_lang if cond == "original" else instr[cond]
    out_dir = results_root / args.model / args.suite / cond / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"task{tid}.jsonl", "w") as ep_file:
        for ep in range(args.n_episodes):
            vec.set_attr("init_state_id", ep)
            vec.set_attr("task_description", text)
            policy.reset()
            set_all_seeds(args.seed + ep)
            obs, _ = vec.reset(seed=args.seed + ep)
            rhash = reset_hash()
            init_poses = poses_of()
            done = step = 0
            success = False
            eef_traj = []
            t0 = time.time()
            done = False
            while not done and step < args.max_steps:
                o = preprocess_observation(obs)
                o["task"] = list(vec.call("task_description"))
                o = env_preprocessor(o)
                o = preprocessor(o)
                with torch.inference_mode():
                    a = policy.select_action(o)
                a = postprocessor(a)
                a = env_postprocessor({ACTION: a})[ACTION]
                obs, _, term, trunc, info = vec.step(a.to("cpu").numpy())
                success = success or bool(np.asarray(info.get("is_success", [False])).reshape(-1)[0])
                if step % 10 == 0 and (e := eef_of()) is not None:
                    eef_traj.append(e)
                done = bool(np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0])
                step += 1
            rec = {
                "model": args.model, "suite": args.suite, "condition": cond, "seed": args.seed,
                "task_id": tid, "task_name": task_name, "episode": ep,
                "instruction": text, "true_instruction": true_lang,
                "success": bool(success), "steps": step, "reset_state_hash": rhash,
                "init_object_poses": init_poses, "final_object_poses": poses_of(),
                "eef_traj": eef_traj, "wall_s": round(time.time() - t0, 2),
                "checkpoint_hash": ckpt_hash,
            }
            ep_file.write(json.dumps(rec) + "\n")
            ep_file.flush()
            print(f"[eval_task {tid} {cond} ep{ep}] success={success} steps={step}", flush=True)

vec.close()
print(f"[eval_task {tid}] DONE", flush=True)
