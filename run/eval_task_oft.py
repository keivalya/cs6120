#!/usr/bin/env python
"""run/eval_task_oft.py — one LIBERO task, all conditions, OpenVLA-OFT 7.5B (RQ1.3).

Third scale point. OFT is heavier than plain OpenVLA: 2 images (3rd-person +
wrist), 8-dim proprio, an L1-regression action head, and **8-step action chunks**
executed open-loop. So instead of inlining (as run/eval_task_openvla.py does for
plain OpenVLA), we REUSE the OFT `experiments.robot.*` utilities directly — they
import cleanly in vla-oft once the dlimp/wandb/protobuf conflicts are resolved
(tensorflow-metadata==1.14.0 + protobuf 3.20.3 + wandb==0.16.6, and a lazy
prismatic/__init__.py; see report/GATE4_STATUS.md / report/patches/).

Same fixed-scene protocol + OUTPUT CONTRACT as run/eval_task{,_openvla}.py:
one process per task (single EGL context), same task bddl + init_state per
(task,episode), only the language to the model changes; env `done` = TRUE task's
goal, so success under blank/wrong_task == OAR. Output:
results/openvla_oft/<suite>/<condition>/seed<k>/task<tid>.jsonl (identical fields),
so run/aggregate.py + analyze/* work unchanged.

Env: vla-oft. unnorm_key="libero_goal_no_noops". use_l1_regression=True,
num_images_in_input=2, use_proprio=True, center_crop=True, use_film=False (plain
OFT checkpoint, §6.3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "run"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openvla_oft")
ap.add_argument("--suite", default="libero_goal")
ap.add_argument("--task_id", type=int, required=True)
ap.add_argument("--conditions", required=True)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--n_episodes", type=int, default=2)
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--obs_hw", type=int, default=256)
ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
args = ap.parse_args()

conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
results_root = Path(args.results_root)
tid = args.task_id

os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_openvla"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
assert Path(os.environ["LIBERO_CONFIG_PATH"], "config.yaml").exists()

gen = REPO_ROOT / "perturb" / "generated" / f"{args.suite}.jsonl"
instr = {}
true_lang = task_name = None
for line in gen.read_text().splitlines():
    r = json.loads(line)
    if int(r["task_id"]) != tid:
        continue
    if r["condition"] == "original":
        true_lang = r["instruction"]; task_name = r["task_name"]
    if not (r.get("skip") or r.get("instruction") is None):
        instr[r["condition"]] = r["instruction"]
assert true_lang is not None

import yaml  # noqa: E402

with open(REPO_ROOT / "configs" / "models.yaml") as f:
    mc = yaml.safe_load(f)[args.model]
ckpt = mc["hf_repo"]
ef = mc.get("eval_flags", {})
device = ef.get("device", "cuda:0")

# OFT config object consumed by experiments.robot.* utilities.
cfg = SimpleNamespace(
    model_family="openvla",
    pretrained_checkpoint=ckpt,
    use_l1_regression=ef.get("use_l1_regression", True),
    use_diffusion=ef.get("use_diffusion", False),
    use_film=ef.get("use_film", False),
    num_images_in_input=ef.get("num_images_in_input", 2),
    use_proprio=ef.get("use_proprio", True),
    center_crop=ef.get("center_crop", True),
    num_open_loop_steps=ef.get("num_open_loop_steps", 8),
    lora_rank=ef.get("lora_rank", 32),
    unnorm_key=ef.get("unnorm_key", "libero_goal_no_noops"),
    load_in_8bit=False,
    load_in_4bit=False,
    num_steps_wait=10,
    task_suite_name=args.suite,
)

# proven order: model (CUDA) first, then env
import torch  # noqa: E402

# OFT's get_model needs a LOCAL DIR (it loads action_head/proprio_projector .pt by
# filename from the checkpoint dir), not a HF repo id — resolve to the cached snapshot.
if not os.path.isdir(str(cfg.pretrained_checkpoint)):
    from huggingface_hub import snapshot_download
    cfg.pretrained_checkpoint = snapshot_download(str(cfg.pretrained_checkpoint))
ckpt = cfg.pretrained_checkpoint

sys.path.insert(0, str(REPO_ROOT / "openvla-oft"))
from experiments.robot.robot_utils import (  # noqa: E402
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
)
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)

model = get_model(cfg)
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
action_head = get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
processor = get_processor(cfg)
resize_size = get_image_resize_size(cfg)
print(f"[oft {tid}] model ready (l1={cfg.use_l1_regression}, imgs={cfg.num_images_in_input}, "
      f"chunk={cfg.num_open_loop_steps}, unnorm={cfg.unnorm_key})", flush=True)

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

bench = get_benchmark(args.suite)()
task = bench.get_task(tid)
assert task.language.strip().lower() == true_lang.strip().lower()
init_states = bench.get_task_init_states(tid)
task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=args.obs_hw, camera_widths=args.obs_hw)
env.seed(0)

# RQ1.2 (§7): parse ALL Goal task goals once for achieved_task_ids (see analyze/goal_eval.py).
sys.path.insert(0, str(REPO_ROOT / "analyze"))
import goal_eval as GOAL  # noqa: E402
_bddl_dir = os.path.join(get_libero_path("bddl_files"), task.problem_folder)
goal_states = GOAL.load_goal_states(
    args.suite, _bddl_dir, {i: bench.get_task(i).bddl_file for i in range(bench.n_tasks)}
)
print(f"[oft {tid}] env ready", flush=True)

try:
    from huggingface_hub import HfApi
    ckpt_hash = HfApi().model_info(ckpt).sha
except Exception:
    ckpt_hash = None


def raw_obs():
    try:
        return env.env._get_observations()
    except Exception:
        return {}


def poses_of(o):
    return {k: np.asarray(v, dtype=np.float64).tolist()
            for k, v in o.items() if k.endswith("_pos") or k.endswith("_quat")}


def reset_hash():
    try:
        st = env.sim.get_state()
        return hashlib.sha256(
            np.concatenate([np.asarray(st.qpos), np.asarray(st.qvel)]).tobytes()
        ).hexdigest()[:16]
    except Exception:
        return None


def prepare_observation(obs):
    img = resize_image_for_policy(get_libero_image(obs), resize_size)
    wrist = resize_image_for_policy(get_libero_wrist_image(obs), resize_size)
    return {
        "full_image": img,
        "wrist_image": wrist,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


def set_all_seeds(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


NUM_WAIT = 10

for cond in conditions:
    if cond != "original" and cond not in instr:
        print(f"[oft {tid}] skip {cond} (no scene-valid probe)", flush=True)
        continue
    text = true_lang if cond == "original" else instr[cond]
    out_dir = results_root / args.model / args.suite / cond / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"task{tid}.jsonl", "w") as ep_file:
        for ep in range(args.n_episodes):
            set_all_seeds(args.seed + ep)
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(NUM_WAIT):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
            rhash = reset_hash()
            init_poses = poses_of(raw_obs())
            t0 = time.time()
            success = False
            eef_traj = []
            step = 0
            done = False
            action_queue = deque(maxlen=cfg.num_open_loop_steps)
            while not done and step < args.max_steps:
                observation = prepare_observation(obs)
                if len(action_queue) == 0:
                    actions = get_action(
                        cfg, model, observation, text, processor=processor,
                        action_head=action_head, proprio_projector=proprio_projector,
                        noisy_action_projector=None, use_film=cfg.use_film,
                    )
                    action_queue.extend(actions)
                action = action_queue.popleft()
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                obs, _r, done, _i = env.step(action.tolist())
                success = success or bool(done)
                if step % 10 == 0:
                    eef_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist())
                step += 1
            achieved = GOAL.achieved_task_ids(env, goal_states)  # §7 final-state goal check
            rec = {
                "model": args.model, "suite": args.suite, "condition": cond, "seed": args.seed,
                "task_id": tid, "task_name": task_name, "episode": ep,
                "instruction": text, "true_instruction": true_lang,
                "success": bool(success), "steps": step, "reset_state_hash": rhash,
                "achieved_task_ids": achieved,
                "init_object_poses": init_poses, "final_object_poses": poses_of(raw_obs()),
                "eef_traj": eef_traj, "wall_s": round(time.time() - t0, 2),
                "checkpoint_hash": ckpt_hash,
            }
            ep_file.write(json.dumps(rec) + "\n"); ep_file.flush()
            print(f"[oft {tid} {cond} ep{ep}] success={success} steps={step}", flush=True)

env.close()
print(f"[oft {tid}] DONE", flush=True)
