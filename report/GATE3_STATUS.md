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

## BLOCKER (open) — GPU-level rendering fault (NVRM Xid 31)
MuJoCo EGL rendering aborts (SIGABRT in robosuite `read_pixels`) **reproducibly
around render #~49** of an episode. `dmesg` shows the true cause:

    NVRM: Xid 31 MMU Fault: ENGINE GRAPHICS GPC0 ... ACCESS_TYPE_VIRT_READ

i.e. a **GPU graphics-engine memory page fault**, not a Python bug. GPU is
otherwise idle (1 MiB used, no other processes).

Key evidence it is environmental, not a code defect:
- The SAME architecture (SYNC env, single process, policy-then-make_env) ran full
  85-step episodes **6/6** in GATE 2, early in the session.
- A standalone script rendered fine for **3** steps but faulted at step **49**
  (`report/patches/working_eval_reference.py`).
- Every configuration tried now faults ~step 49: sync/async, env-before-CUDA vs
  after, 1 vs many envs, fork vs spawn.

Most likely: the node's GPU/EGL context state degraded over a long debugging
session (many core-dumped render processes). A **fresh compute node** should
restore reliable rendering.

Attempts made (all still fault at ~render 49): CUDA-before-fork ordering,
env_cfg construction order, parent LIBERO benchmark instantiation, import order,
AsyncVectorEnv spawn-vs-fork, single-context sync, GATE-2-exact order.

No results fabricated (§12): `rq1_causal.csv` NOT produced — no eval data collected.

## Runner ready for a fresh node
`run/eval_task.py` — ONE process per task (single EGL context = the reliable
mode), SYNC env, policy-then-make_env (GATE 2 order), sweeps all conditions
reusing the one env, writes `results/.../<condition>/seed<k>/task<tid>.jsonl`.
Drive it over tasks × RQ1.1 conditions. If a FRESH node still faults at ~render
49, it is an inherent mujoco-EGL/driver issue → switch render backend (try
`MUJOCO_GL=osmesa`, slower) or a different driver/torch build.

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
