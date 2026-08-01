#!/usr/bin/env python
"""core/analyze/goal_eval.py — evaluate which LIBERO-Goal task goals a live env satisfies.

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
`achieved_task_ids`, and by core/analyze/css.py to classify Follow/Ignore/Fail.
"""
from __future__ import annotations

import os
import sys

_WARNED = False  # print the first predicate-evaluation failure, then stay quiet


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

    A bare `return env.env` only works when the caller hands us an OffScreenRenderEnv
    (ControlEnv), which is true for the 7B runners but not in general — the SmolVLA
    path goes through lerobot wrappers, and one wrong hop returns an object with no
    _eval_predicate, whereupon eval_goal_on_env's except-clause silently reports "goal
    not achieved". That is what produced empty achieved_task_ids on successful SmolVLA
    episodes (12 such records remain in the committed data vs 0 for both 7B models).
    So walk the wrapper chain and stop at the first object that can actually answer.
    """
    seen = set()
    cur = env
    for _ in range(8):  # bounded: wrapper chains are short, and .env can self-reference
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "_eval_predicate") and hasattr(cur, "object_states_dict"):
            return cur
        cur = getattr(cur, "env", None) or getattr(cur, "unwrapped", None)
    # Nothing in the chain can evaluate predicates. Say so instead of letting the
    # caller's except-clause turn it into a quiet False.
    raise AttributeError(
        f"no robosuite problem env with _eval_predicate/object_states_dict found by "
        f"unwrapping {type(env).__name__}"
    )


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
    except Exception as e:
        # Never let a predicate quirk crash the rollout; treat as unsatisfied. But do
        # NOT swallow it silently — a mis-resolved env used to look exactly like "goal
        # not achieved", which is how the SmolVLA goal-logging bug stayed invisible.
        global _WARNED
        if not _WARNED:
            _WARNED = True
            print(f"[goal_eval] WARNING: predicate evaluation failed ({type(e).__name__}: "
                  f"{e}); reporting goals as unachieved. Further warnings suppressed.",
                  file=sys.stderr, flush=True)
        return False


def achieved_task_ids(env, goal_states):
    """List of task_ids whose full goal is satisfied by the env's current state."""
    return [tid for tid, gs in goal_states.items() if eval_goal_on_env(env, gs)]
