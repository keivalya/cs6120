#!/usr/bin/env python
"""core/run/preflight_libero.py — model-free render/GPU gate that works in ANY of the envs.

core/run/preflight.py loads SmolVLA, so it can only ever validate the vla-smolvla env.
The 7B jobs were gating themselves on it anyway (core/run/slurm/p2_rq1_{openvla,oft}.sbatch
called it with the smolvla python), which means a SmolVLA-side problem aborted every
7B job and a healthy result said nothing about the 7B stack. This check loads no
checkpoint: it just proves CUDA is visible and that this env can open an EGL
LIBERO env and render, which is the failure mode that actually kills these jobs.

Run it with the env's OWN python:
    $E/vla-openvla/bin/python core/run/preflight_libero.py
    $E/vla-oft/bin/python     core/run/preflight_libero.py

Keep core/run/preflight.py for the SmolVLA job — it is the only check that reproduces
the CUDA-resident-policy + EGL interaction behind the Xid 31 history.
"""
from __future__ import annotations

import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_goal")
ap.add_argument("--task_id", type=int, default=0)
ap.add_argument("--obs_hw", type=int, default=256)
ap.add_argument("--steps", type=int, default=60)
args = ap.parse_args()

import numpy as np  # noqa: E402
import torch  # noqa: E402

if not torch.cuda.is_available():
    print("PREFLIGHT FAILED: torch.cuda.is_available() is False (CPU-only allocation?)",
          file=sys.stderr)
    raise SystemExit(1)
print(f"cuda OK: {torch.cuda.get_device_name(0)} | MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
      f"| MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID')}", flush=True)

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

bench = get_benchmark(args.suite)()
task = bench.get_task(args.task_id)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
if not os.path.exists(bddl):
    print(f"PREFLIGHT FAILED: bddl missing at {bddl} — this is a stale LIBERO config "
          f"(LIBERO_CONFIG_PATH={os.environ.get('LIBERO_CONFIG_PATH')}), NOT a bad GPU.",
          file=sys.stderr)
    raise SystemExit(1)

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=args.obs_hw,
                        camera_widths=args.obs_hw)
try:
    env.seed(0)
    env.reset()
    env.set_init_state(bench.get_task_init_states(args.task_id)[0])
    for _ in range(args.steps):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    img = np.asarray(obs["agentview_image"])
    assert img.shape[:2] == (args.obs_hw, args.obs_hw), img.shape
    assert img.any(), "rendered frame is all zeros — EGL produced no image"
finally:
    env.close()

print(f"PREFLIGHT HEALTHY: {args.steps} EGL render steps at {args.obs_hw}x{args.obs_hw} "
      f"({task.language!r})")
