# GATE 3 status — code complete, RQ1 data collection paused (GPU budget)

## Done (no GPU)
- `perturb/make_instructions.py` + `perturb/generated/libero_goal.jsonl` (70 probes,
  7 causal conditions, bddl-grounded O/A, provenance; 3 `wrong_object` skipped as
  fixture-only tasks 0/3/7).
- `analyze/tsr.py`, `analyze/css.py`, `analyze/locus.py` — TSR/ΔTSR, CSS(blank)/
  CSS(nonsense), Follow/Ignore/Fail + exact OAR, planning-vs-execution locus
  (with `validate()` for the 10-hand-label check). All parse; read raw episodes.jsonl.
- `run/launch.py` — idempotent, resumable, wall-deadline-guarded driver.
- `run/_libero_probe.py` + `run/run_one.py` — refactored to AsyncVectorEnv probe
  architecture (render in worker subprocess, CUDA in main).

## BLOCKER (open)
`run_one.py` render worker SIGABRTs in `robosuite/utils/binding_utils.py:read_pixels`
(EGL). Root-cause class: MuJoCo-EGL single global GL context vs PyTorch CUDA vs
multiprocessing-fork. A flat standalone script with identical eval logic runs fine
(`report/patches/working_eval_reference.py`, produces real success + poses + reset
hash `04481935…`). Same logic inside run_one still aborts the worker.

Ruled out (each tested): CUDA-before-fork (moved GPU-name query after run),
env_cfg construction order, parent LIBERO benchmark instantiation, lerobot-vs-libero
import order, spawn-vs-fork. Exact trigger not found within GPU budget.

No results fabricated (§12): `rq1_causal.csv` NOT produced — no eval data collected.

## Agreed next step (when GPU available)
1. Refactor `run_one` eval to mirror `working_eval_reference.py` flat structure
   (chosen path: "deliver code only, pause GPU" now; fix later).
2. Verify on 1–2 tasks, then run **RQ1.1 core**: conditions `original, blank,
   nonsense`, all 10 tasks, `n_episodes=2` (~30–45 min GPU). Add `wrong_*` if budget
   remains.
3. Compute `rq1_causal.csv` via `analyze/tsr.py` + `analyze/css.py`; hand-validate
   `locus.py` on 10 episodes.

## Reset-fixed assertion (§5/§12) — how it will be checked
`run_one` records `reset_state_hash` per (task, episode). Cross-condition check:
for each (task, episode) the hash must be identical across all conditions (the
injected instruction is a string only; init_state selected by episode index alone).
