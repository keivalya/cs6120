#!/usr/bin/env python
"""analyze/goal_eval.py — evaluate which LIBERO-Goal task goals a live env satisfies.

RQ1.2 Follow/Ignore/Fail needs the §7-spec method: after a rollout under a *wrong*
instruction (scene fixed), determine which task was actually achieved by checking
ALL candidate LIBERO-Goal goal predicates against the final sim state. LIBERO-Goal
shares one scene, so every goal references in-scene objects and resolves against any
task's env.

API (verified against openvla-oft/LIBERO/libero/libero/):
  * `BDDLUtils.robosuite_parse_problem(bddl_path)["goal_state"]` -> list of predicate
    tuples e.g. ["on","akita_black_bowl_1","plate_1"] (file parse; no env needed).
  * `<OffScreenRenderEnv>.env._eval_predicate(state)` evaluates one predicate against
    the env's live `object_states_dict` (pure reads — no sim mutation).
  * `_check_success` == AND over the task's own goal_state. We replicate that for an
    ARBITRARY task's goal_state, guarding on object_states_dict membership so a
    foreign predicate over a missing name is treated as unsatisfiable (never KeyErrors).

Used by the runners (called at each episode's final state) to record
`achieved_task_ids`, and by analyze/css.py to classify Follow/Ignore/Fail.
"""
from __future__ import annotations

import os


def load_goal_states(suite, bddl_dir, task_name_by_id):
    """Return {task_id: goal_state_list} by parsing each task's bddl file.

    task_name_by_id: {tid: task.name} (bddl basename without extension). bddl_dir is
    get_libero_path("bddl_files")/<suite>.
    """
    from libero.libero.envs import bddl_utils as BDDLUtils  # noqa: E402

    goals = {}
    for tid, name in task_name_by_id.items():
        fname = name if name.endswith(".bddl") else name + ".bddl"
        path = os.path.join(bddl_dir, fname)
        goals[tid] = BDDLUtils.robosuite_parse_problem(path)["goal_state"]
    return goals


def _problem_env(env):
    """Unwrap to the robosuite problem env exposing _eval_predicate/object_states_dict.

    Runners pass an OffScreenRenderEnv (ControlEnv): its `.env` is the problem env.
    """
    return env.env


def eval_goal_on_env(env, goal_state):
    """True iff the env's current state satisfies every predicate in goal_state.

    Guards on object_states_dict membership (a predicate over an absent object name
    -> not satisfiable) so foreign/cross-suite goals never raise (see module docstring).
    """
    try:
        pe = _problem_env(env)
        osd = getattr(pe, "object_states_dict", {})
        if not goal_state:
            return False
        for state in goal_state:
            # state = [predicate, arg1, (arg2)]; args are object/site names.
            names = state[1:]
            if any(n not in osd for n in names):
                return False
            if not pe._eval_predicate(state):
                return False
        return True
    except Exception:
        # Never let a predicate quirk crash the rollout; treat as unsatisfied.
        return False


def achieved_task_ids(env, goal_states):
    """List of task_ids whose full goal is satisfied by the env's current state."""
    return [tid for tid, gs in goal_states.items() if eval_goal_on_env(env, gs)]
