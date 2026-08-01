#!/usr/bin/env python
"""core/run/smoke_openvla.py — prove the vla-openvla env can actually do a rollout step.

Mirrors core/run/eval_task_openvla.py's import order and does real work: bf16 sdpa load
from the offline HF cache, EGL render, the two tf image ops the runner calls, and
one predict_action. An `import numpy` smoke test is worthless here — the env that
shipped last time imported fine and then failed in production.

Run on a GPU node with the vla-openvla python:
    $E/vla-openvla/bin/python core/run/smoke_openvla.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_openvla"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

import tensorflow as tf  # noqa: E402
tf.config.set_visible_devices([], "GPU")  # never contend with the policy for VRAM

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.benchmark import get_benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

assert torch.cuda.is_available(), "no CUDA — run this on a GPU node"
assert np.__version__.startswith("1.26"), f"numpy drifted to {np.__version__} (tf 2.15 needs <2)"

# The design decision at core/run/eval_task_openvla.py:115-125 is that prismatic must
# NEVER be imported here: it pulls dlimp -> protobuf>=5.26, which conflicts with
# this env's tf 2.15 / protobuf 4.25. This is the regression test for that.
assert "prismatic" not in sys.modules, "prismatic leaked in -> dlimp/protobuf conflict ahead"

# The two tf ops core/run/eval_task_openvla.py:148-153 calls on every frame.
_img = tf.zeros([256, 256, 3], tf.uint8)
_d = tf.io.decode_image(tf.image.encode_jpeg(_img), expand_animations=False, dtype=tf.uint8)
tf.image.resize(_d, (224, 224), method="lanczos3", antialias=True)
print("tf image ops OK", flush=True)

CKPT = "openvla/openvla-7b-finetuned-libero-goal"
processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    CKPT, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True, trust_remote_code=True,
).to("cuda:0").eval()
assert "libero_goal" in vla.norm_stats, \
    f"unnorm_key libero_goal missing; have {sorted(vla.norm_stats)[:5]}"
print(f"model OK: {CKPT} bf16/sdpa, norm_stats has libero_goal", flush=True)

bench = get_benchmark("libero_goal")()
task = bench.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
try:
    env.seed(0)
    env.reset()
    obs = env.set_init_state(bench.get_task_init_states(0)[0])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    frame = np.asarray(obs["agentview_image"])
    assert frame.any(), "EGL rendered an all-zero frame"
finally:
    env.close()
print("EGL render OK", flush=True)

img = Image.fromarray(frame[::-1, ::-1]).convert("RGB").resize((224, 224))
inp = processor(f"In: What action should the robot take to {task.language.lower()}?\nOut:",
                img).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inp, unnorm_key="libero_goal", do_sample=False)
assert np.asarray(action).shape == (7,), f"unexpected action shape {np.asarray(action).shape}"

print("SMOKE OK openvla: bf16 sdpa + offline ckpt + norm_stats + EGL render + 7-dim action")
