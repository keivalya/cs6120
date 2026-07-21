#!/bin/env python
"""run/generate_videos.py — Generate 5 success and 5 failure videos for each task with instruction overlays.

Saves videos into output_videos/task{tid}_{task_name}/:
  - success_ep1.mp4 ... success_ep5.mp4
  - failure_ep1.mp4 ... failure_ep5.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy
from lerobot.configs.policies import PreTrainedConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--model", default="smolvla")
    parser.add_argument("--num_videos", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--output_dir", default="output_videos")
    return parser.parse_args()


def extract_frame_from_obs(obs: dict) -> np.ndarray:
    """Extract uint8 RGB (256, 256, 3) frame from LeRobot observation dict."""
    img_t = None
    for key in ("observation.images.image", "observation.images.agentview", "agentview_image"):
        if isinstance(obs, dict) and key in obs:
            img_t = obs[key]
            break
    
    if img_t is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    if torch.is_tensor(img_t):
        img_t = img_t.squeeze(0).detach().cpu().numpy()
    if isinstance(img_t, np.ndarray):
        if img_t.ndim == 3 and img_t.shape[0] == 3:
            img_t = np.transpose(img_t, (1, 2, 0))
        if img_t.dtype != np.uint8:
            img_t = (img_t * 255.0).clip(0, 255).astype(np.uint8)
        return img_t

    return np.zeros((256, 256, 3), dtype=np.uint8)


def draw_overlay(frame: np.ndarray, task_name: str, task_id: int, instruction: str, step: int, max_steps: int, is_success: bool, status_label: str) -> np.ndarray:
    """Draw clean semi-transparent banners and text onto the frame using PIL."""
    img = Image.fromarray(frame)
    if img.height < 512:
        img = img.resize((512, 512), Image.Resampling.BICUBIC)
    
    w, h = img.size

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    draw.rectangle([0, 0, w, 85], fill=(15, 15, 15, 190))
    draw.rectangle([0, h - 50, w, h], fill=(15, 15, 15, 190))

    title_text = f"Task {task_id}: {task_name.replace('_', ' ').title()}"
    instr_text = f'Prompt: "{instruction}"'
    if len(instr_text) > 48:
        instr_text = instr_text[:45] + "..."

    badge_color = (0, 255, 0, 255) if is_success else (230, 50, 50, 255)
    badge_text = f"STATUS: {status_label}"
    step_text = f"Step: {step}/{max_steps}"

    draw.text((12, 12), title_text, fill=(255, 255, 255, 255))
    draw.text((12, 45), instr_text, fill=(230, 230, 100, 255))
    draw.text((12, h - 35), badge_text, fill=badge_color)
    draw.text((w - 140, h - 35), step_text, fill=(200, 200, 200, 255))

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return np.array(img)


def run_episode(env, policy, instruction: str, max_steps: int):
    """Execute a single episode and return (frames_list, is_success)."""
    try:
        env.set_attr("task_description", instruction)
    except Exception:
        pass

    obs, info = env.reset()
    policy.reset()
    
    frames = []
    is_success = False

    for step in range(1, max_steps + 1):
        img = extract_frame_from_obs(obs)
        frames.append(img.copy())

        action = policy.select_action(obs)
        step_res = env.step(action)
        if len(step_res) == 5:
            obs, reward, terminated, truncated, info = step_res
        else:
            obs, reward, terminated, info = step_res
            truncated = False

        inf = info[0] if isinstance(info, (list, tuple)) else info
        if inf.get("is_success", False):
            is_success = True
            frames.append(extract_frame_from_obs(obs))
            break

        if terminated or truncated:
            break

    return frames, is_success


def main():
    args = parse_args()
    tid = args.task_id
    
    gen_file = REPO_ROOT / "perturb" / "generated" / f"{args.suite}.jsonl"
    instr_map = {}
    task_name = f"task_{tid}"
    for line in gen_file.read_text().splitlines():
        r = json.loads(line)
        if int(r["task_id"]) == tid:
            if r["condition"] == "original":
                task_name = r["task_name"]
            instr_map[r["condition"]] = r["instruction"]

    out_folder = Path(args.output_dir) / f"task{tid}_{task_name}"
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"=== Generating Videos for Task {tid} ({task_name}) ===", flush=True)

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        model_cfg = yaml.safe_load(f)[args.model]
    hf_repo = model_cfg["hf_repo"]
    device = model_cfg.get("eval_flags", {}).get("device", "cuda:0")

    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=[tid], observation_height=256, observation_width=256)
    policy_cfg = PreTrainedConfig.from_pretrained(hf_repo)
    policy_cfg.pretrained_path = hf_repo
    policy_cfg.device = device
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)

    vec = make_env(env_cfg, n_envs=1)
    env = vec[args.suite] if isinstance(vec, dict) else vec

    orig_prompt = instr_map.get("original", f"execute task {tid}")
    fail_prompt = instr_map.get("wrong_object", instr_map.get("blank", "do invalid task"))

    success_count = 0
    fail_count = 0
    target_num = args.num_videos

    for ep in range(1, 50):
        if success_count >= target_num and fail_count >= target_num:
            break

        prompt = orig_prompt if success_count < target_num else fail_prompt
        set_seed = 100 + ep
        np.random.seed(set_seed)
        torch.manual_seed(set_seed)

        print(f"[{task_name}] Running episode {ep} (prompt: '{prompt[:30]}...')...", flush=True)
        frames, is_success = run_episode(env, policy, prompt, args.max_steps)

        if is_success and success_count < target_num:
            success_count += 1
            vid_path = out_folder / f"success_ep{success_count}.mp4"
            writer = imageio.get_writer(vid_path, fps=20)
            for i, f in enumerate(frames):
                ann = draw_overlay(f, task_name, tid, prompt, i + 1, len(frames), True, "SUCCESS")
                writer.append_data(ann)
            writer.close()
            print(f"  --> Saved {vid_path} ({len(frames)} frames)", flush=True)

        elif (not is_success) and fail_count < target_num:
            fail_count += 1
            vid_path = out_folder / f"failure_ep{fail_count}.mp4"
            writer = imageio.get_writer(vid_path, fps=20)
            for i, f in enumerate(frames):
                ann = draw_overlay(f, task_name, tid, prompt, i + 1, len(frames), False, "FAILURE")
                writer.append_data(ann)
            writer.close()
            print(f"  --> Saved {vid_path} ({len(frames)} frames)", flush=True)

    print(f"=== Task {tid} Videos Finished! Total Saved in {out_folder}: Successes={success_count}, Failures={fail_count} ===", flush=True)


if __name__ == "__main__":
    main()
