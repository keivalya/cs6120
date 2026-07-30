#!/usr/bin/env python
"""run/smoke_oft.py — prove the vla-oft env can actually produce an action chunk.

Mirrors run/eval_task_oft.py: it reuses the OFT repo's experiments.robot.* helpers,
so `import prismatic` and `import dlimp` MUST work under protobuf 3.20.3. Those two
imports are the whole point of this file — every protobuf / tensorflow-metadata /
wandb pin in run/rebuild_oft_minimal.sh exists to make them succeed, and if one has
drifted this fails here in seconds instead of three hours into a production job.

Run on a GPU node with the vla-oft python:
    $E/vla-oft/bin/python run/smoke_oft.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_openvla"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

import tensorflow as tf  # noqa: E402
tf.config.set_visible_devices([], "GPU")

assert torch.cuda.is_available(), "no CUDA — run this on a GPU node"
assert np.__version__.startswith("1.26"), f"numpy drifted to {np.__version__}"

# The regression tests for this env's whole reason to exist.
import prismatic  # noqa: E402
import dlimp  # noqa: E402
print(f"prismatic OK ({prismatic.__file__})\ndlimp OK", flush=True)

sys.path.insert(0, str(REPO_ROOT / "openvla-oft"))
from experiments.robot.robot_utils import (  # noqa: E402
    get_action, get_image_resize_size, get_model,
)
from experiments.robot.openvla_utils import (  # noqa: E402
    get_action_head, get_processor, get_proprio_projector,
)
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action, get_libero_image, get_libero_wrist_image, quat2axisangle,
)

# Same eval_flags the runner uses, straight from configs/models.yaml — so this
# validates the config we actually evaluate, not a hand-written guess.
ef = yaml.safe_load((REPO_ROOT / "configs" / "models.yaml").read_text())["openvla_oft"]
hf_repo = ef["hf_repo"]
flags = ef["eval_flags"]
# get_model needs a LOCAL dir: it loads the action-head / proprio-projector .pt
# files by name from the checkpoint directory.
ckpt = snapshot_download(hf_repo)

cfg = SimpleNamespace(
    model_family="openvla", pretrained_checkpoint=ckpt,
    use_l1_regression=flags["use_l1_regression"], use_diffusion=flags["use_diffusion"],
    use_film=flags["use_film"], num_images_in_input=flags["num_images_in_input"],
    use_proprio=flags["use_proprio"], center_crop=flags["center_crop"],
    num_open_loop_steps=flags.get("num_open_loop_steps", 8), lora_rank=32,
    unnorm_key=flags["unnorm_key"], load_in_8bit=False, load_in_4bit=False,
    num_steps_wait=10, task_suite_name="libero_goal",
)

model = get_model(cfg)
proprio_projector = get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
action_head = get_action_head(cfg, model.llm_dim)
processor = get_processor(cfg)
resize_size = get_image_resize_size(cfg)
print(f"model OK: l1={cfg.use_l1_regression} imgs={cfg.num_images_in_input} "
      f"chunk={cfg.num_open_loop_steps} unnorm={cfg.unnorm_key}", flush=True)

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

bench = get_benchmark("libero_goal")()
task = bench.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
try:
    env.seed(0)
    env.reset()
    obs = env.set_init_state(bench.get_task_init_states(0)[0])
    for _ in range(10):
        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
    chunk = get_action(
        cfg, model,
        {
            "full_image": get_libero_image(obs, resize_size),
            "wrist_image": get_libero_wrist_image(obs, resize_size),
            "state": np.concatenate([
                obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            ]),
        },
        task.language, proprio_projector=proprio_projector,
        action_head=action_head, processor=processor,
    )
finally:
    env.close()

assert len(chunk) == cfg.num_open_loop_steps, f"expected an 8-step chunk, got {len(chunk)}"
print(f"SMOKE OK oft: prismatic+dlimp import clean, wrist cam + 8-dim proprio, "
      f"{len(chunk)}-step chunk")
