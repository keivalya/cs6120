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
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "run"))

# Before any libero import (LIBERO reads LIBERO_CONFIG_PATH at import time).
from libero_paths import assert_libero_config, use_smolvla_config  # noqa: E402

use_smolvla_config()
assert_libero_config()

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="smolvla")
ap.add_argument("--suite", default="libero_goal")
ap.add_argument("--task_id", type=int, required=True)
ap.add_argument("--conditions", default="")
ap.add_argument("--paraphrase_axis", default=None,
                help="para_object|para_action|para_compositional — RQ2 mode (overrides --conditions)")
ap.add_argument("--max_per_axis", type=int, default=None, help="cap paraphrases/axis (sampling)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--n_episodes", type=int, default=2, help="episodes per condition OR per paraphrase")
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

# RQ1.2 (§7): parse ALL Goal task goals once for achieved_task_ids (analyze/goal_eval.py).
# smolvla uses the lerobot/hf-libero env; sub._env is the OffScreenRenderEnv whose
# .env is the problem env exposing _eval_predicate (same as poses_of uses).
sys.path.insert(0, str(REPO_ROOT / "analyze"))
import goal_eval as GOAL  # noqa: E402
from libero.libero import get_libero_path as _glp  # noqa: E402
from libero.libero import benchmark as _bm  # noqa: E402
try:
    _ts = _bm.get_benchmark_dict()[args.suite]()
    _bddl_dir = os.path.join(_glp("bddl_files"), args.suite)
    goal_states = GOAL.load_goal_states(
        args.suite, _bddl_dir, {i: _ts.get_task(i).name for i in range(_ts.get_num_tasks())}
    )
except Exception as _e:
    print(f"[eval_task {tid}] goal_states unavailable ({_e}); achieved_task_ids=None", flush=True)
    goal_states = None
print(f"[eval_task {tid}] env ready", flush=True)

# Monkey-patch LiberoEnv.step to capture achieved task goals BEFORE auto-reset
from lerobot.envs.libero import LiberoEnv
_orig_step = LiberoEnv.step

def _patched_step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
    self._ensure_env()
    assert self._env is not None
    if action.ndim != 1:
        raise ValueError(
            f"Expected action to be 1-D (shape (action_dim,)), "
            f"but got shape {action.shape} with ndim={action.ndim}"
        )
    raw_obs, reward, done, info = self._env.step(action)
    is_success = self._env.check_success()
    terminated = done or is_success
    
    achieved = []
    if goal_states is not None:
        try:
            achieved = [tid for tid, gs in goal_states.items() if GOAL.eval_goal_on_env(self._env, gs)]
        except Exception:
            pass
            
    info.update(
        {
            "task": self.task,
            "task_id": self.task_id,
            "done": done,
            "is_success": is_success,
            "achieved_task_ids": achieved,
        }
    )
    observation = self._format_raw_obs(raw_obs)
    if terminated:
        self.reset()
    truncated = False
    return observation, reward, terminated, truncated, info

LiberoEnv.step = _patched_step

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


# work groups: conditions -> one item/file each; paraphrase -> one axis group with
# the task's paraphrases appended to one file (run/para_util.py).
if args.paraphrase_axis:
    sys.path.insert(0, str(REPO_ROOT / "run"))
    import para_util  # noqa: E402
    paras = para_util.load_task_paraphrases(args.suite, tid, args.paraphrase_axis,
                                            args.max_per_axis, args.seed)
    items = [(p["instruction"], {"axis": args.paraphrase_axis, "para_idx": p["para_idx"],
                                 "keyword_similarity": p["keyword_similarity"],
                                 "structural_similarity": p["structural_similarity"],
                                 "operation": p.get("operation", "")}) for p in paras]
    groups = [(args.paraphrase_axis, items)]
    print(f"[eval_task {tid}] paraphrase mode axis={args.paraphrase_axis} n_paraphrases={len(items)}", flush=True)
else:
    groups = [(c, [(true_lang if c == "original" else instr[c], {})])
              for c in conditions if (c == "original" or c in instr)]

for label, items in groups:
    out_dir = results_root / args.model / args.suite / label / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"task{tid}.jsonl"
    # Resume by KEY, not by line count. A count-based fast-forward assumes the
    # file on disk is an exact prefix of this work list; when it is not, it lands
    # at the wrong offset and re-rolls episodes that are already present. That is
    # how para_compositional/seed7/task7.jsonl grew 21 duplicate
    # (para_idx, episode) pairs — 321 rows for 300 distinct episodes, silently
    # double-counted by PRIDE (see analyze/dedupe_results.py). Every record
    # already carries the key it belongs to, so match on that instead: correct
    # regardless of what the file holds or what order it was written in.
    expected_keys, _ep = [], 0
    for _text, _extra in items:
        for _ in range(args.n_episodes):
            expected_keys.append((_extra.get("para_idx"), _ep))
            _ep += 1

    existing_keys = set()
    if out_file.exists():
        try:
            with open(out_file) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        existing_keys.add((r.get("para_idx"), r.get("episode")))
        except Exception:
            pass

    if out_file.exists() and all(k in existing_keys for k in expected_keys):
        print(f"[{args.model} {tid} {label}] skip (all {len(expected_keys)} eps present)", flush=True)
        continue

    ep_global = 0
    with open(out_file, "a" if existing_keys else "w") as ep_file:
        for text, extra in items:
          for ep in range(args.n_episodes):
            if (extra.get("para_idx"), ep_global) in existing_keys:
                ep_global += 1
                continue
            # Episode k -> init state k (50 available per LIBERO task). ep_global
            # restarts at 0 for every condition group, so episode k sees the SAME
            # scene in every condition — that matching is what makes the causal
            # claims scene-fixed. This used to be `ep_global % 2`, which pinned
            # every episode to one of just 2 scenes: at 10 eps/task that is 5
            # repeats of 2 scenes, so extra episodes bought policy-seed variance
            # only, left the CIs scene-limited, and kept RQ3's pair count at 2.
            # The 7B runners already index all init states this way.
            vec.set_attr("init_state_id", ep_global)
            vec.set_attr("task_description", text)
            policy.reset()
            set_all_seeds(args.seed + ep_global)
            obs, _ = vec.reset(seed=args.seed + ep_global)
            rhash = reset_hash()
            init_poses = poses_of()
            done = step = 0
            success = False
            eef_traj = []
            t0 = time.time()
            done = False
            last_achieved = []
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
                if info and "achieved_task_ids" in info:
                    last_achieved = info["achieved_task_ids"][0]
                if step % 10 == 0 and (e := eef_of()) is not None:
                    eef_traj.append(e)
                done = bool(np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0])
                step += 1
            achieved = last_achieved if goal_states is not None else None
            rec = {
                "model": args.model, "suite": args.suite, "condition": label, "seed": args.seed,
                "task_id": tid, "task_name": task_name, "episode": ep_global,
                "instruction": text, "true_instruction": true_lang,
                "success": bool(success), "steps": step, "reset_state_hash": rhash,
                "achieved_task_ids": achieved, **extra,
                "init_object_poses": init_poses, "final_object_poses": poses_of(),
                "eef_traj": eef_traj, "wall_s": round(time.time() - t0, 2),
                "checkpoint_hash": ckpt_hash,
            }
            ep_file.write(json.dumps(rec) + "\n")
            ep_file.flush()
            print(f"[eval_task {tid} {label} ep{ep_global}] success={success} steps={step}", flush=True)
            ep_global += 1

vec.close()
print(f"[eval_task {tid}] DONE", flush=True)
