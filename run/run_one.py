#!/usr/bin/env python
"""run/run_one.py — single (model, suite, condition, seed) eval -> logs.

See CLAUDE.md §8. Reuses LeRobot's own env/policy/processor stack (so observation
preprocessing + normalization match training exactly — avoids the §9 obs-key
mangling trap). The instruction is injected per-episode by overriding each
sub-env's `task_description` attribute (read by the rollout via
`env.call("task_description")`); the bddl file and init_states are NEVER touched,
so the physical scene is provably fixed (§5, §12).

Outputs (under results/<model>/<suite>/<condition>/seed<k>/):
  - episodes.jsonl   : one record per episode, streamed as it completes
  - summary.json     : written atomically at the end
and appends one line to MANIFEST.json (atomic + flock).

Currently implements the `lerobot` framework (SmolVLA). 7B frameworks are added
in GATE 4 behind the same CLI + output contract.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def set_all_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_sha(path: Path) -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def sha256_short(obj) -> str:
    return hashlib.sha256(np.asarray(obj, dtype=np.float64).tobytes()).hexdigest()[:16]


def atomic_write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def append_manifest(manifest_path: Path, record: dict) -> None:
    """Append one run record to MANIFEST.json['runs'] atomically under flock."""
    lock = manifest_path.with_suffix(".lock")
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if manifest_path.exists():
                with open(manifest_path) as f:
                    data = json.load(f)
            else:
                data = {"runs": []}
            data["runs"].append(record)
            atomic_write_json(manifest_path, data)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def extract_object_poses(raw_obs: dict) -> dict:
    """Pull object/eef positions+quats from a raw LIBERO observation (used by
    analyze/css.py Follow/Ignore/Fail and analyze/locus.py)."""
    out = {}
    for k, v in raw_obs.items():
        if k.endswith("_pos") or k.endswith("_quat"):
            try:
                out[k] = np.asarray(v, dtype=np.float64).tolist()
            except Exception:
                pass
    return out


def sim_state_hash(sub_env) -> str | None:
    """Hash the settled physical sim state (qpos+qvel) after reset. Identical
    across conditions for the same (task, episode) proves the scene is fixed."""
    try:
        st = sub_env._env.env.sim.get_state()
        return sha256_short(np.concatenate([np.asarray(st.qpos), np.asarray(st.qvel)]))
    except Exception:
        try:
            i = sub_env.init_state_id % len(sub_env._init_states)
            return sha256_short(sub_env._init_states[i])
        except Exception:
            return None


# --------------------------------------------------------------------------- #
# instruction resolution
# --------------------------------------------------------------------------- #
def load_instructions(suite: str, condition: str) -> dict[int, str] | None:
    """Return {task_id: instruction} for a perturbation condition, from
    perturb/generated/<suite>.jsonl (produced by make_instructions.py, GATE 3).
    For `original`, returns None -> use the env's true task.language."""
    if condition == "original":
        return None
    gen = REPO_ROOT / "perturb" / "generated" / f"{suite}.jsonl"
    if not gen.exists():
        raise FileNotFoundError(
            f"No generated instructions for condition={condition!r}; run "
            f"perturb/make_instructions.py first (expected {gen})."
        )
    mapping: dict[int, str] = {}
    with open(gen) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("condition") == condition:
                # skip probes that could not be made scene-valid (instruction=null)
                if rec.get("skip") or rec.get("instruction") is None:
                    continue
                mapping[int(rec["task_id"])] = rec["instruction"]
    if not mapping:
        raise ValueError(f"No entries for condition={condition!r} in {gen}")
    return mapping


# --------------------------------------------------------------------------- #
# lerobot framework (SmolVLA)
# --------------------------------------------------------------------------- #
def run_lerobot(args, model_cfg: dict) -> dict:
    # Import the LIBERO-backed probe env (which imports `libero`) FIRST, before
    # any lerobot.policies/transformers imports, and matching the module-order of
    # the known-good standalone path. Importing heavy torch/policy modules into
    # the parent before `libero` can perturb GL global state inherited by forked
    # AsyncVectorEnv workers (SIGABRT in robosuite read_pixels).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _libero_probe import make_probe_vec

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import make_env_pre_post_processors, preprocess_observation
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.constants import ACTION

    device = model_cfg.get("eval_flags", {}).get("device", "cuda:0")
    hf_repo = model_cfg["hf_repo"]
    max_steps = args.max_steps

    instr_map = load_instructions(args.suite, args.condition)

    # Task metadata (true language, name) comes from the generated instructions
    # jsonl — NOT by instantiating the LIBERO benchmark in this (parent) process.
    # Instantiating LIBERO here touches MuJoCo/GL global state that a forked
    # AsyncVectorEnv worker then inherits, corrupting its EGL context and aborting
    # in robosuite read_pixels. The benchmark is only ever built inside workers.
    gen = REPO_ROOT / "perturb" / "generated" / f"{args.suite}.jsonl"
    true_langs, task_names = {}, {}
    with open(gen) as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] == "original":
                true_langs[int(r["task_id"])] = r["instruction"]
                task_names[int(r["task_id"])] = r["task_name"]
    all_ids = sorted(true_langs) if args.task_ids is None else list(args.task_ids)
    run_ids = [i for i in all_ids if instr_map is None or int(i) in instr_map]

    # CRITICAL ORDERING (§9-style fix): fork ALL AsyncVectorEnv rendering workers
    # BEFORE initializing CUDA in this process. Each worker owns its own EGL
    # context in its own process; if the policy inits CUDA first, forked workers
    # inherit a broken CUDA/GL state and abort (SIGABRT in robosuite read_pixels).
    # So env-workers first, policy (CUDA) second.
    probe_vecs = {i: make_probe_vec(args.suite, int(i), max_steps) for i in run_ids}
    for _v in probe_vecs.values():
        _v.reset(seed=args.seed)  # spin up workers + their EGL contexts pre-CUDA

    # now bring up CUDA + policy + processors (env_cfg built here, AFTER the fork:
    # constructing the LiberoEnv config earlier can perturb global state inherited
    # by forked workers)
    env_cfg = LiberoEnvConfig(task=args.suite, task_ids=args.task_ids)
    policy_cfg = PreTrainedConfig.from_pretrained(hf_repo)
    policy_cfg.pretrained_path = hf_repo
    policy_cfg.device = device
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=hf_repo,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg, policy_cfg=policy_cfg
    )

    try:
        from huggingface_hub import HfApi

        ckpt_hash = HfApi().model_info(hf_repo).sha
    except Exception:
        ckpt_hash = None

    out_dir = args.out_dir
    ep_path = out_dir / "episodes.jsonl"
    ep_file = open(ep_path, "w")

    n_success = 0
    n_total = 0
    per_task = {}

    def _eef_from_obs(obs):
        try:
            return np.asarray(obs["robot_state"]["eef"]["pos"]).reshape(-1)[:3].astype(float).tolist()
        except Exception:
            return None

    for task_id in run_ids:
        true_lang = true_langs[task_id]
        instr = true_lang if instr_map is None else instr_map[int(task_id)]
        vec = probe_vecs[task_id]
        task_succ = 0
        for ep in range(args.n_episodes):
            vec.set_attr("init_state_id", ep)  # deterministic init state per episode
            vec.set_attr("task_description", instr)  # inject (possibly perturbed) instruction
            policy.reset()
            set_all_seeds(args.seed + ep)
            obs, info = vec.reset(seed=args.seed + ep)

            probe = vec.call("probe_state")[0]
            reset_hash = probe.get("hash")
            init_poses = probe.get("poses", {})

            done = False
            step = 0
            success = False
            eef_traj = []
            t0 = time.time()
            while not done and step < max_steps:
                o = preprocess_observation(obs)
                o["task"] = list(vec.call("task_description"))
                o = env_preprocessor(o)
                o = preprocessor(o)
                with torch.inference_mode():
                    action = policy.select_action(o)
                action = postprocessor(action)
                action = env_postprocessor({ACTION: action})[ACTION]
                action_np = action.to("cpu").numpy()
                obs, reward, term, trunc, info = vec.step(action_np)
                is_succ = info.get("is_success", [False])
                success = success or bool(np.asarray(is_succ).reshape(-1)[0])
                if step % 10 == 0:
                    eef = _eef_from_obs(obs)
                    if eef is not None:
                        eef_traj.append(eef)
                done = bool(np.asarray(term).reshape(-1)[0] or np.asarray(trunc).reshape(-1)[0])
                step += 1

            final_poses = vec.call("probe_state")[0].get("poses", {})

            n_total += 1
            task_succ += int(success)
            n_success += int(success)

            rec = {
                "model": args.model,
                "suite": args.suite,
                "condition": args.condition,
                "seed": args.seed,
                "task_id": int(task_id),
                "task_name": task_names[task_id],
                "episode": ep,
                "instruction": instr,
                "true_instruction": true_lang,
                "success": bool(success),
                "steps": step,
                "reset_state_hash": reset_hash,
                "init_object_poses": init_poses,
                "final_object_poses": final_poses,
                "eef_traj": eef_traj,
                "wall_s": round(time.time() - t0, 2),
            }
            ep_file.write(json.dumps(rec) + "\n")
            ep_file.flush()
            print(f"[task {task_id} ep {ep}] success={success} steps={step}", flush=True)

        per_task[int(task_id)] = {"n": args.n_episodes, "success": task_succ}
        vec.close()

    ep_file.close()

    return {
        "model": args.model,
        "checkpoint": hf_repo,
        "checkpoint_hash": ckpt_hash,
        "suite": args.suite,
        "condition": args.condition,
        "seed": args.seed,
        "n_episodes": args.n_episodes,
        "n_total_episodes": n_total,
        "n_success": n_success,
        "tsr": (n_success / n_total) if n_total else None,
        "per_task": per_task,
        "max_steps": max_steps,
        "framework": "lerobot",
    }


FRAMEWORKS = {"lerobot": run_lerobot}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_episodes", type=int, required=True)
    ap.add_argument("--task_ids", type=str, default=None, help="comma-sep task ids; default all")
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--results_root", type=str, default=str(REPO_ROOT / "results"))
    args = ap.parse_args()

    args.task_ids = [int(x) for x in args.task_ids.split(",")] if args.task_ids else None

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        models = yaml.safe_load(f)
    model_cfg = models[args.model]
    framework = model_cfg["framework"]

    args.out_dir = Path(args.results_root) / args.model / args.suite / args.condition / f"seed{args.seed}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = REPO_ROOT / "MANIFEST.json"
    node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename

    t_start = time.time()
    try:
        runner = FRAMEWORKS.get(framework)
        if runner is None:
            raise NotImplementedError(f"framework {framework!r} not implemented yet (GATE 4)")
        summary = runner(args, model_cfg)
        atomic_write_json(args.out_dir / "summary.json", summary)
        status = "OK"
    except Exception:
        err = traceback.format_exc()
        atomic_write_json(
            args.out_dir / "error.json",
            {"cell": {k: str(v) for k, v in vars(args).items()}, "traceback": err},
        )
        status = "FAILED"
        summary = {"n_success": None, "tsr": None, "n_episodes": args.n_episodes}
        print(err, file=sys.stderr)

    # Query GPU name only AFTER the run: torch.cuda.* initializes a CUDA context,
    # which must NOT happen before the AsyncVectorEnv render workers are forked
    # (forked workers would inherit a broken CUDA state and SIGABRT in EGL).
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "unknown"

    append_manifest(
        manifest,
        {
            "model": args.model,
            "suite": args.suite,
            "condition": args.condition,
            "seed": args.seed,
            "n_episodes": args.n_episodes,
            "status": status,
            "n_success": summary.get("n_success"),
            "tsr": summary.get("tsr"),
            "checkpoint_hash": summary.get("checkpoint_hash"),
            "repo_sha": git_sha(REPO_ROOT),
            "lerobot_sha": git_sha(REPO_ROOT / "lerobot"),
            "node": node,
            "gpu": gpu,
            "wall_s": round(time.time() - t_start, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    print(
        f"[run_one] {status} model={args.model} suite={args.suite} "
        f"condition={args.condition} seed={args.seed} tsr={summary.get('tsr')}"
    )
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
