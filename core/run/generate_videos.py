#!/bin/env python
"""core/run/generate_videos.py — Generate 5 success and 5 failure videos for each task with instruction overlays.

Saves videos into data/videos/task{tid}_{task_name}/:
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
import fcntl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.constants import ACTION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--model", default="smolvla")
    parser.add_argument("--num_videos", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--output_dir", default="data/videos")
    parser.add_argument("--condition", default="original")
    parser.add_argument("--paraphrase_axis", default=None)
    return parser.parse_args()


def extract_frame_from_obs(obs: dict) -> np.ndarray:
    """Extract uint8 RGB (256, 256, 3) frame from LeRobot observation dict."""
    img_t = None
    
    # 1. Try to find inside obs["pixels"]
    if isinstance(obs, dict) and "pixels" in obs:
        pixels = obs["pixels"]
        if isinstance(pixels, dict):
            # Check common camera keys in libero/robosuite
            for key in ("agentview_image", "agentview", "image", "camera1", "robot0_agentview_left"):
                if key in pixels:
                    img_t = pixels[key]
                    break
        else:
            img_t = pixels

    # 2. Try to find at the root of obs
    if img_t is None and isinstance(obs, dict):
        for key in ("observation.images.image", "observation.images.agentview", "agentview_image", "image", "pixels"):
            if key in obs:
                img_t = obs[key]
                break

    if img_t is None:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    # Convert PyTorch tensor to numpy array if needed
    if torch.is_tensor(img_t):
        img_t = img_t.detach().cpu().numpy()

    # Handle batch dimension if present
    if isinstance(img_t, np.ndarray):
        if img_t.ndim == 4: # batch dim present: (b, h, w, c) or (b, c, h, w)
            img_t = img_t[0] # take the first environment's frame
        
        # Handle channel-first if present: (c, h, w) -> (h, w, c)
        if img_t.ndim == 3 and img_t.shape[0] == 3:
            img_t = np.transpose(img_t, (1, 2, 0))

        # Convert float to uint8 if needed
        if img_t.dtype != np.uint8:
            if img_t.max() <= 1.0:
                img_t = (img_t * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_t = img_t.clip(0, 255).astype(np.uint8)
                
        return img_t

    return np.zeros((256, 256, 3), dtype=np.uint8)


def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize uint8 RGB frame to 512x512 for better video visibility."""
    img = Image.fromarray(frame)
    if img.height != 512 or img.width != 512:
        img = img.resize((512, 512), Image.Resampling.BICUBIC)
    return np.array(img)


def append_to_global_readme(output_dir: Path, video_path: str, status: str, instruction: str):
    readme_path = output_dir / "README.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(readme_path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                f.write("# VLA Simulation Videos Index\n\n")
                f.write("| Video Path | Status | VLA Instruction |\n")
                f.write("| --- | --- | --- |\n")
            
            # Format row (path relative to repo root)
            relative_path = os.path.relpath(video_path, output_dir.parent)
            f.write(f"| `{relative_path}` | {status} | `{instruction}` |\n")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_episode(env, policy, instruction: str, max_steps: int, env_preprocessor, preprocessor, postprocessor, env_postprocessor):
    """Execute a single episode and return (frames_list, is_success)."""
    try:
        env.set_attr("task_description", instruction)
    except Exception:
        pass

    obs, _ = env.reset()
    policy.reset()
    
    frames = []
    is_success = False

    for step in range(1, max_steps + 1):
        img = extract_frame_from_obs(obs)
        frames.append(img.copy())

        # Preprocess observation
        o = preprocess_observation(obs)
        o["task"] = [instruction]
        o = env_preprocessor(o)
        o = preprocessor(o)

        with torch.inference_mode():
            action = policy.select_action(o)

        # Postprocess action
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]

        step_res = env.step(action.to("cpu").numpy())
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
    
    # 1. Get task name from original condition row in libero_goal.jsonl
    gen_file = REPO_ROOT / "data" / "instructions" / f"{args.suite}.jsonl"
    task_name = f"task_{tid}"
    for line in gen_file.read_text().splitlines():
        r = json.loads(line)
        if int(r["task_id"]) == tid and r["condition"] == "original":
            task_name = r["task_name"]
            break

    # 2. Load instructions based on parameters
    instructions = []
    condition_label = ""
    if args.paraphrase_axis:
        para_file = REPO_ROOT / "data" / "instructions" / f"{args.suite}_paraphrases.jsonl"
        for line in para_file.read_text().splitlines():
            r = json.loads(line)
            if int(r["task_id"]) == tid and r["axis"] == args.paraphrase_axis:
                if r.get("instruction"):
                    instructions.append(r["instruction"])
        condition_label = args.paraphrase_axis
        if not instructions:
            print(f"=== No instructions found for task {tid} axis {args.paraphrase_axis}. Exiting ===", flush=True)
            return
    else:
        target_instruction = None
        for line in gen_file.read_text().splitlines():
            r = json.loads(line)
            if int(r["task_id"]) == tid and r["condition"] == args.condition:
                target_instruction = r.get("instruction")
                break
        condition_label = args.condition
        if target_instruction is None:
            print(f"=== Condition {args.condition} is skipped/null/not found for task {tid}. Exiting ===", flush=True)
            return
        instructions = [target_instruction]

    # Create distinct directory for this condition
    out_folder = Path(args.output_dir) / f"task{tid}_{task_name}" / condition_label
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"=== Generating Videos for Task {tid} ({task_name}) under {condition_label} ===", flush=True)

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        model_cfg = yaml.safe_load(f)[args.model]
    hf_repo = model_cfg["hf_repo"]
    device = model_cfg.get("eval_flags", {}).get("device", "cuda:0")

    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=[tid], observation_height=256, observation_width=256)
    policy_cfg = PreTrainedConfig.from_pretrained(hf_repo)
    policy_cfg.pretrained_path = hf_repo
    policy_cfg.device = device
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)

    vec = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = vec[args.suite][tid]

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)

    success_count = 0
    fail_count = 0
    target_num = args.num_videos

    for ep in range(1, 50):
        if success_count >= target_num and fail_count >= target_num:
            break

        # Select prompt
        prompt = instructions[(ep - 1) % len(instructions)]
        set_seed = 100 + ep
        np.random.seed(set_seed)
        torch.manual_seed(set_seed)

        print(f"[{task_name}] Running episode {ep} (prompt: '{prompt[:30]}...')...", flush=True)
        frames, is_success = run_episode(env, policy, prompt, args.max_steps, env_preprocessor, preprocessor, postprocessor, env_postprocessor)

        if is_success and success_count < target_num:
            success_count += 1
            vid_path = out_folder / f"success_ep{success_count}.mp4"
            writer = imageio.get_writer(vid_path, fps=20)
            for i, f in enumerate(frames):
                ann = resize_frame(f)
                writer.append_data(ann)
            writer.close()
            print(f"  --> Saved {vid_path} ({len(frames)} frames)", flush=True)
            append_to_global_readme(Path(args.output_dir), str(vid_path), "Success", prompt)

        elif (not is_success) and fail_count < target_num:
            fail_count += 1
            vid_path = out_folder / f"failure_ep{fail_count}.mp4"
            writer = imageio.get_writer(vid_path, fps=20)
            for i, f in enumerate(frames):
                ann = resize_frame(f)
                writer.append_data(ann)
            writer.close()
            print(f"  --> Saved {vid_path} ({len(frames)} frames)", flush=True)
            append_to_global_readme(Path(args.output_dir), str(vid_path), "Failure", prompt)

    print(f"=== Task {tid} Videos Finished! Total Saved in {out_folder}: Successes={success_count}, Failures={fail_count} ===", flush=True)


if __name__ == "__main__":
    main()
