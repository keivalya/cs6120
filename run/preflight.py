#!/usr/bin/env python
"""run/preflight.py — fail-fast GPU render health check (CLAUDE.md §12).

On this MuJoCo-EGL + CUDA stack a degraded GPU channel throws NVRM Xid 31
(graphics-engine MMU fault) mid-episode once the SmolVLA policy is CUDA-resident
(see report/GATE3_STATUS.md). This check loads the policy + one LIBERO env and
runs ~60 policy-driven render/step iterations. If it completes it prints HEALTHY
and exits 0; if the GPU is bad the process SIGABRTs (nonzero exit) — so a driver
(launch.py / sbatch) can bail before wasting the allocation.

Usage: python run/preflight.py [--obs_hw 360]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "run"))

# Must happen before any libero/lerobot.envs.libero import: LIBERO reads
# LIBERO_CONFIG_PATH at import time. A stale config is what this check used to
# misreport as a degraded GPU channel.
from libero_paths import assert_libero_config, use_smolvla_config  # noqa: E402

use_smolvla_config()
assert_libero_config()

ap = argparse.ArgumentParser()
ap.add_argument("--task_id", type=int, default=1)
ap.add_argument("--obs_hw", type=int, default=360)
ap.add_argument("--steps", type=int, default=60)
args = ap.parse_args()

import yaml  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.constants import ACTION  # noqa: E402

with open(REPO_ROOT / "configs" / "models.yaml") as f:
    mc = yaml.safe_load(f)["smolvla"]
hf = mc["hf_repo"]

env_cfg = LiberoEnvConfig(task="libero_goal", task_ids=[args.task_id],
                          observation_height=args.obs_hw, observation_width=args.obs_hw)
pc = PreTrainedConfig.from_pretrained(hf)
pc.pretrained_path = hf
pc.device = "cuda:0"
policy = make_policy(cfg=pc, env_cfg=env_cfg)
policy.eval()
pre, post = make_pre_post_processors(policy_cfg=pc, pretrained_path=hf,
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})
epre, epost = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=pc)
vec = make_env(env_cfg, n_envs=1, use_async_envs=False)["libero_goal"][args.task_id]
policy.reset()
obs, _ = vec.reset(seed=7)
for i in range(args.steps):
    o = preprocess_observation(obs)
    o["task"] = list(vec.call("task_description"))
    o = epre(o)
    o = pre(o)
    with torch.inference_mode():
        a = policy.select_action(o)
    a = post(a)
    a = epost({ACTION: a})[ACTION]
    obs, _, term, trunc, _ = vec.step(a.to("cpu").numpy())
    if bool(np.asarray(term).reshape(-1)[0]):
        obs, _ = vec.reset(seed=7)
vec.close()
print(f"PREFLIGHT HEALTHY: {args.steps} policy+render steps at {args.obs_hw}x{args.obs_hw}", flush=True)
