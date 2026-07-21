#!/usr/bin/env python
"""run/eval_task_openvla.py — one LIBERO task, all conditions, OpenVLA-7B (RQ1.3).

Mirrors run/eval_task.py (SmolVLA) exactly in OUTPUT CONTRACT and the reliable
ONE-process-per-task / single-EGL-context discipline, but swaps the model+env
plumbing to OpenVLA's original-LIBERO stack:

  * Model: HF `trust_remote_code` load (the checkpoint bundles modeling_prismatic),
    attn_implementation="sdpa" — NO flash-attn, NO `import prismatic` (avoids the
    dlimp/protobuf import chain). Verified: loads in ~170s, predict_action works.
  * LIBERO: original LIBERO (openvla/LIBERO), config isolated via
    LIBERO_CONFIG_PATH=~/.libero_openvla so it does NOT collide with the smolvla
    env's hf-libero paths (CLAUDE.md §6 package-name collision).
  * unnorm_key="libero_goal" (this checkpoint's norm_stats has no _no_noops key).

Scene is provably fixed across conditions: same task bddl + same init_state per
(task,episode); only the language passed to the model changes. The env's `done`
is the TRUE task's goal predicate, so success under blank/wrong_task == OAR
(robot did the original task despite a bad instruction) — same semantics as the
SmolVLA analysis, so analyze/* work unchanged.

Output: results/openvla/<suite>/<condition>/seed<k>/task<tid>.jsonl  (same fields
as eval_task.py). Usage mirrors eval_task.py:
  python run/eval_task_openvla.py --task_id 7 --conditions original,blank,wrong_task \
    --seed 7 --n_episodes 10
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "run"))

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="openvla")
ap.add_argument("--suite", default="libero_goal")
ap.add_argument("--task_id", type=int, required=True)
ap.add_argument("--conditions", default="")
ap.add_argument("--paraphrase_axis", default=None,
                help="para_object|para_action|para_compositional — RQ2 mode (overrides --conditions)")
ap.add_argument("--max_per_axis", type=int, default=None, help="cap paraphrases/axis (7B sampling)")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--n_episodes", type=int, default=10, help="episodes per condition OR per paraphrase")
ap.add_argument("--max_steps", type=int, default=300)
ap.add_argument("--obs_hw", type=int, default=256, help="LIBERO render resolution")
ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
args = ap.parse_args()

conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
results_root = Path(args.results_root)
tid = args.task_id

# Isolated LIBERO config (original LIBERO, not smolvla's hf-libero). Must be set
# BEFORE importing libero. Written once by setup; assert it exists.
os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_openvla"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
assert Path(os.environ["LIBERO_CONFIG_PATH"], "config.yaml").exists(), (
    f"missing LIBERO config at {os.environ['LIBERO_CONFIG_PATH']}; seed it first"
)

# instructions per condition for THIS task (same generated file as SmolVLA).
gen = REPO_ROOT / "perturb" / "generated" / f"{args.suite}.jsonl"
instr = {}
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
assert true_lang is not None, f"no original instruction for task {tid} in {gen}"

import yaml  # noqa: E402

with open(REPO_ROOT / "configs" / "models.yaml") as f:
    model_cfg = yaml.safe_load(f)[args.model]
ckpt = model_cfg["hf_repo"]
unnorm_key = model_cfg.get("eval_flags", {}).get("unnorm_key", "libero_goal")
device = model_cfg.get("eval_flags", {}).get("device", "cuda:0")

# --------------------------------------------------------------------------- #
# Proven order (eval_task.py): load model (CUDA) FIRST, then build the ONE env.
# --------------------------------------------------------------------------- #
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
vla = (
    AutoModelForVision2Seq.from_pretrained(
        ckpt, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    )
    .to(device)
    .eval()
)
assert unnorm_key in vla.norm_stats, (
    f"unnorm_key {unnorm_key} not in {list(vla.norm_stats.keys())}"
)
print(f"[ov {tid}] model ready (sdpa, unnorm_key={unnorm_key})", flush=True)

# --------------------------------------------------------------------------- #
# We do NOT import experiments.robot.{libero_utils,openvla_utils,robot_utils}:
# those transitively `import prismatic` -> dlimp -> a protobuf>=5.26 `_pb2`
# (`runtime_version`) that is incompatible with tensorflow 2.15's protobuf 4.25.
# tensorflow ITSELF imports fine, so we inline the (small, verbatim) helper
# bodies from OpenVLA's libero_utils/robot_utils/openvla_utils and use only the
# clean `libero` primitives + tf. Logic is copied 1:1 to match trained
# preprocessing (resize scheme, 180° rotate, center-crop, gripper normalize).
# Provenance: openvla/experiments/robot/{libero/libero_utils.py,robot_utils.py,
# openvla_utils.py}. See report/patches/ note in GATE4_STATUS.md.
# --------------------------------------------------------------------------- #
import math  # noqa: E402

import tensorflow as tf  # noqa: E402  (imports fine; only dlimp's proto is broken)
tf.config.set_visible_devices([], 'GPU')
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def get_libero_dummy_action():
    return [0, 0, 0, 0, 0, 0, -1]


def quat2axisangle(quat):  # verbatim from libero_utils (robosuite transform)
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def resize_image(img, resize_size):  # verbatim (Octo/RLDS resize scheme)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def get_libero_image(obs, resize_size):  # verbatim (180° rotate to match train)
    rs = (resize_size, resize_size) if isinstance(resize_size, int) else resize_size
    img = obs["agentview_image"][::-1, ::-1]
    return resize_image(img, rs)


def crop_and_resize(image, crop_scale, batch_size):  # verbatim from openvla_utils
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded = True
    new_h = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_w = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    h_off = (1 - new_h) / 2
    w_off = (1 - new_w) / 2
    boxes = tf.stack([h_off, w_off, h_off + new_h, w_off + new_w], axis=1)
    image = tf.image.crop_and_resize(image, boxes, tf.range(batch_size), (224, 224))
    if expanded:
        image = image[0]
    return image


def predict_action(full_image, task_label, center_crop=True):  # inlined get_vla_action
    from PIL import Image as _Image
    image = _Image.fromarray(full_image).convert("RGB")
    if center_crop:
        t = tf.convert_to_tensor(np.array(image))
        od = t.dtype
        t = tf.image.convert_image_dtype(t, tf.float32)
        t = crop_and_resize(t, 0.9, 1)
        t = tf.clip_by_value(t, 0, 1)
        t = tf.image.convert_image_dtype(t, od, saturate=True)
        image = _Image.fromarray(t.numpy()).convert("RGB")
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inp = processor(prompt, image).to(device, dtype=torch.bfloat16)
    return vla.predict_action(**inp, unnorm_key=unnorm_key, do_sample=False)


def normalize_gripper_action(action, binarize=True):  # verbatim
    action[..., -1] = 2 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action):  # verbatim
    action[..., -1] = action[..., -1] * -1.0
    return action


bench = get_benchmark(args.suite)()
task = bench.get_task(tid)
assert task.language.strip().lower() == true_lang.strip().lower(), (
    f"task {tid} language mismatch: LIBERO={task.language!r} vs generated={true_lang!r}"
)
init_states = bench.get_task_init_states(tid)
task_bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=args.obs_hw, camera_widths=args.obs_hw)
env.seed(0)  # affects object poses even with fixed init_state (per libero_utils)

# RQ1.2 (§7): parse ALL Goal task goals once, to record which task(s) the final
# state satisfies (achieved_task_ids) -> exact Follow/Ignore/Fail in analyze/css.py.
sys.path.insert(0, str(REPO_ROOT / "analyze"))
import goal_eval as GOAL  # noqa: E402
_bddl_dir = os.path.join(get_libero_path("bddl_files"), task.problem_folder)
goal_states = GOAL.load_goal_states(
    args.suite, _bddl_dir, {i: bench.get_task(i).bddl_file for i in range(bench.n_tasks)}
)
print(f"[ov {tid}] env ready", flush=True)

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


def set_all_seeds(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


NUM_WAIT = 10  # steps to let objects settle (OpenVLA default)

# Build work groups: each = (out_label, [(text, extra_fields), ...]). Conditions ->
# one item per condition (one file each). Paraphrase mode -> one group (the axis)
# whose items are the task's paraphrases (all appended to one file). See run/para_util.py.
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
    print(f"[ov {tid}] paraphrase mode axis={args.paraphrase_axis} n_paraphrases={len(items)}", flush=True)
else:
    groups = [(c, [(true_lang if c == "original" else instr[c], {})])
              for c in conditions if (c == "original" or c in instr)]

for label, items in groups:
    out_dir = results_root / args.model / args.suite / label / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / f"task{tid}.jsonl"
    expected_lines = len(items) * args.n_episodes
    if out_file.exists():
        try:
            with open(out_file) as f:
                lines = [l.strip() for l in f if l.strip()]
            if len(lines) >= expected_lines:
                print(f"[{args.model} {tid} {label}] skip (already has {len(lines)}/{expected_lines} eps)", flush=True)
                continue
        except Exception:
            pass

    ep_global = 0
    with open(out_file, "w") as ep_file:
        for text, extra in items:
          for ep in range(args.n_episodes):
            set_all_seeds(args.seed + ep_global)
            env.reset()
            init_id = ep_global % len(init_states)
            env.set_init_state(init_states[init_id])
            # settle
            for _ in range(NUM_WAIT):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            rhash = reset_hash()
            init_poses = poses_of(raw_obs())
            t0 = time.time()
            success = False
            eef_traj = []
            step = 0
            done = False
            while not done and step < args.max_steps:
                img = get_libero_image(obs, 224)
                action = predict_action(img, text, center_crop=True)
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                obs, _reward, done, _info = env.step(action.tolist())
                success = success or bool(done)
                if step % 10 == 0:
                    eef_traj.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float64).tolist())
                step += 1
            achieved = GOAL.achieved_task_ids(env, goal_states)  # §7 final-state goal check
            rec = {
                "model": args.model, "suite": args.suite, "condition": label, "seed": args.seed,
                "task_id": tid, "task_name": task_name, "episode": ep_global,
                "instruction": text, "true_instruction": true_lang,
                "success": bool(success), "steps": step, "reset_state_hash": rhash,
                "achieved_task_ids": achieved, **extra,
                "init_object_poses": init_poses, "final_object_poses": poses_of(raw_obs()),
                "eef_traj": eef_traj, "wall_s": round(time.time() - t0, 2),
                "checkpoint_hash": ckpt_hash,
            }
            ep_file.write(json.dumps(rec) + "\n")
            ep_file.flush()
            print(f"[ov {tid} {label} ep{ep_global}] success={success} steps={step}", flush=True)
            ep_global += 1

env.close()
print(f"[ov {tid}] DONE", flush=True)
