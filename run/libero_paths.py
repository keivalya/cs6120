"""run/libero_paths.py — pin and validate the LIBERO config paths.

LIBERO resolves its benchmark files (bddl_files, init_files, assets) through a
config.yaml holding ABSOLUTE paths, written once at first import and never
refreshed. When the conda envs moved from ~/.conda/envs to /scratch, the stale
~/.libero/config.yaml kept pointing at the old prefix, so every run died deep in
torch.load with

    FileNotFoundError: .../libero_goal/put_the_bowl_on_the_stove.pruned_init

which the sbatch scripts reported as "channel degraded, need a fresh node". That
misdiagnosis cost ~21 job submissions before anyone read the real log.

So: each model family gets its OWN config dir (smolvla here, ~/.libero_openvla
for the 7B runners — see run/eval_task_openvla.py), and we assert the paths
actually resolve BEFORE loading a model or an env.

Call use_smolvla_config() at the top of a module, before importing libero or
lerobot.envs.libero — LIBERO reads LIBERO_CONFIG_PATH at import time.
"""
from __future__ import annotations

import os
from pathlib import Path

# Keys that must point at real directories for a rollout to work. `assets` is
# intentionally NOT here: hf_libero ships no assets dir and pulls them from the
# HF cache (datasets--lerobot--libero-assets) instead. `datasets` holds
# demonstrations we never read (we evaluate policies, we don't train).
REQUIRED_KEYS = ("benchmark_root", "bddl_files", "init_states")


def use_smolvla_config() -> str:
    """Point LIBERO at the smolvla config dir (isolated from the 7B runners)."""
    return os.environ.setdefault(
        "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_smolvla")
    )


def assert_libero_config() -> dict:
    """Fail with one actionable line if the LIBERO config is stale or missing."""
    import yaml

    cfg_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")))
    cfg_file = cfg_dir / "config.yaml"
    if not cfg_file.exists():
        raise SystemExit(
            f"FATAL: no LIBERO config at {cfg_file}. Seed it first — do NOT let libero "
            "create it, it prompts on stdin via input() and a batch job dies on EOFError."
        )
    cfg = yaml.safe_load(cfg_file.read_text())
    stale = [(k, cfg.get(k)) for k in REQUIRED_KEYS if not Path(str(cfg.get(k, ""))).exists()]
    if stale:
        detail = "\n".join(f"    {k} -> {v}" for k, v in stale)
        raise SystemExit(
            f"FATAL: {cfg_file} points at paths that do not exist:\n{detail}\n"
            "  The conda env almost certainly moved. Rewrite those keys to the libero "
            "package inside the env you are running "
            "(<env>/lib/python*/site-packages/libero/libero). This is NOT a GPU problem."
        )
    return cfg
