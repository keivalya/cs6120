#!/usr/bin/env python
"""core/run/scene_atomic.py — keep the scene-fixed invariant intact across resumes.

THE INVARIANT (CLAUDE.md §5/§12): for a given (task, episode), every causal
condition must start from the SAME initial scene. Only the instruction string may
differ. Without it, a TSR gap between `original` and `blank` is confounded by the
scene and cannot be read as a language effect — which is the whole paper.

THE FAILURE IT GUARDS AGAINST: the initial scene is NOT a pure function of
(task, episode). It also depends on how many resets the env has already served in
that OS process — the runners call env.seed(...) once at construction, and each
reset advances that RNG, which perturbs object poses even though init_state pins
qpos/qvel (see the env.seed(0) comment in eval_task_openvla.py). So a cell finished
by a second process lands on a different scene than its siblings, silently.

Measured on the real tree before this guard existed:
  * smolvla  task0/seed7 — a job hit the wall after `nonsense` ep8; the resuming
    process wrote nonsense ep9 + wrong_action + wrong_task onto DIFFERENT scenes.
    8 of 10 episodes broken. Every task finished by a single process passed.
  * openvla  — every task, both seeds. Its conditions were split across a Jul-10
    and a Jul-15 job.
  * openvla_oft — every task on seed7, because `original` alone was re-run Jul-25.
The count-based resume made this invisible: the files look complete and ordered.

THE RULE: the unit of atomicity is the TASK, not the episode and not the
condition. Either one process produces all conditions x all episodes for a task,
or that task's output is discarded and regenerated. Per-task jobs make this cheap
(one task, 7 conditions, 10 episodes is ~3 h for smolvla/openvla against an 8 h
wall), and a wall-clock kill then costs one task rather than corrupting the grid.
"""
from __future__ import annotations

import json
from pathlib import Path


def _episodes_in(path: Path) -> set:
    if not path.exists():
        return set()
    eps = set()
    try:
        with path.open() as fh:
            for line in fh:
                if line.strip():
                    eps.add(json.loads(line).get("episode"))
    except Exception:
        return set()
    return eps


def enforce_task_atomicity(results_root: Path, model: str, suite: str, seed: int,
                           tid: int, labels, n_episodes: int, verbose=True) -> str:
    """Return 'complete' | 'fresh'. Wipes a partially-written task.

    'complete' -> every label already has all n_episodes; the caller may skip.
    'fresh'    -> nothing on disk for this task (any partial output was deleted);
                  the caller must produce every label in THIS process.
    """
    want = set(range(n_episodes))
    files = {lab: results_root / model / suite / lab / f"seed{seed}" / f"task{tid}.jsonl"
             for lab in labels}
    have = {lab: _episodes_in(p) for lab, p in files.items()}

    if all(want <= have[lab] for lab in labels):
        if verbose:
            print(f"[scene_atomic] {model} task{tid} seed{seed}: complete "
                  f"({n_episodes} eps x {len(labels)} conditions) — skipping", flush=True)
        return "complete"

    partial = {lab: len(have[lab]) for lab in labels if have[lab]}
    if partial:
        # Deleting paid-for episodes is the POINT: they cannot be combined with
        # episodes produced by this process without breaking the scene-fixed
        # invariant, and a silently-confounded grid is worth less than the GPU
        # hours it costs to redo. Log exactly what goes, so the loss is auditable.
        total = sum(partial.values())
        if verbose:
            print(f"[scene_atomic] {model} task{tid} seed{seed}: PARTIAL "
                  f"({total} episodes across {len(partial)} conditions: "
                  f"{', '.join(f'{k}={v}' for k, v in sorted(partial.items()))}).",
                  flush=True)
            print(f"[scene_atomic] discarding them — all conditions of a task must "
                  f"come from ONE process or they land on different scenes.", flush=True)
        for lab in labels:
            if files[lab].exists():
                files[lab].unlink()
    return "fresh"
