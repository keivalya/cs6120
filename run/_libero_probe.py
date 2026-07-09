#!/usr/bin/env python
"""run/_libero_probe.py — AsyncVectorEnv LIBERO envs with a state-probe hook.

Why async: MuJoCo/EGL keeps a single *global current* rendering context per
process. With SyncVectorEnv, once CUDA (the policy) is initialized, rendering a
context that isn't current aborts (SIGABRT in robosuite read_pixels). LeRobot's
own fix (see LiberoEnv._ensure_env docstring) is to run each env in a worker
SUBPROCESS via AsyncVectorEnv, so each worker owns exactly one EGL context and
CUDA lives only in the main process. We follow that.

But async workers hold the raw obs (object poses / sim state) out of reach of the
main process. So we subclass LeRobot's LiberoEnv to add `probe_state()`, callable
from the main process via `vec.call("probe_state")`, returning the settled sim
hash + object poses (needed for the reset-fixed assertion §5/§12, Follow/Ignore/
Fail §7, and locus §7).
"""
from __future__ import annotations

import hashlib

import gymnasium as gym
import numpy as np
from libero.libero import benchmark
from lerobot.envs.libero import LiberoEnv

# Match configs.LiberoEnv (the eval config that scored TSR=1.0 in the smoke test).
_OBS_TYPE = "pixels_agent_pos"
_CAMS = "agentview_image,robot0_eye_in_hand_image"
_H = _W = 360


class ProbeLiberoEnv(LiberoEnv):
    def probe_state(self) -> dict:
        """Return settled sim-state hash + object/eef poses from the raw obs."""
        try:
            raw = self._env.env._get_observations()
        except Exception:
            raw = {}
        try:
            st = self._env.env.sim.get_state()
            h = hashlib.sha256(
                np.concatenate([np.asarray(st.qpos), np.asarray(st.qvel)]).tobytes()
            ).hexdigest()[:16]
        except Exception:
            h = None
        poses = {}
        for k, v in raw.items():
            if k.endswith("_pos") or k.endswith("_quat"):
                try:
                    poses[k] = np.asarray(v, dtype=np.float64).tolist()
                except Exception:
                    pass
        return {"hash": h, "poses": poses}


def _make_probe_env(suite_name: str, task_id: int, max_steps: int | None):
    def _fn():
        suite = benchmark.get_benchmark_dict()[suite_name]()
        return ProbeLiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=_CAMS,
            obs_type=_OBS_TYPE,
            observation_height=_H,
            observation_width=_W,
            init_states=True,
            episode_index=0,
            n_envs=1,
            control_mode="relative",
            episode_length=max_steps,
        )

    return _fn


def make_probe_vec(suite_name: str, task_id: int, max_steps: int | None = None):
    """One AsyncVectorEnv (single worker) for a task, exposing probe_state().

    Uses the 'spawn' start method so each render worker is a fresh Python process,
    immune to any parent-process state (imported CUDA/torch, GL handles) that can
    otherwise corrupt a forked worker's EGL context and abort in read_pixels.
    """
    return gym.vector.AsyncVectorEnv([_make_probe_env(suite_name, task_id, max_steps)])
