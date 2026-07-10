# GATE 4 status — multi-model (RQ1.3)

## DONE
- **GATE 3 (SmolVLA/libero_goal):** RQ1.1 CSS=1.0, RQ1.2 OAR, locus. report/report.md.
- **GATE 4 RQ1.3 — OpenVLA-7B COMPLETE (RQ1.1/1.2/1.3, 2 seeds):**
  `openvla/openvla-7b-finetuned-libero-goal`, libero_goal, 10 tasks, 2 ep/task.
  Core causal conds on **seeds 7+42 (40 ep/cond)**; RQ1.2 conds on seed 7.
  Result: **original 0.70 (std 0.00 over seeds), blank/nonsense/wrong_task 0.00 →
  CSS(blank)=CSS(nonsense)=1.00, OAR(wrong_task)=0.00.** RQ1.2: wrong_object
  OAR=0.14, wrong_action OAR=0.55, repeated TSR=0.60. scene_fixed_check pass
  (0/40 mismatches). Matches SmolVLA (CSS=1.0) → **causal language reliance is
  complete at BOTH 0.45B and 7B (16× scale)**; object-noun≫action-verb binding
  and (new) redundancy-robustness-scales-up both hold.
  - report/rq1_scale.csv, report/rq1_causal_openvla.csv, report/rq1_causal.csv
    (combined), report/rq1_scale.png, report/rq1_causal_bars.png.
  - Runner: run/eval_task_openvla.py. Aggregation/analysis reused unchanged.
  - MANIFEST.json updated (4 openvla runs, checkpoint + repo SHAs, sdpa).
- **OpenVLA RQ1.2 extension (wrong_object/wrong_action/repeated): RUNNING**
  (same runner/grid); aggregate + fold into report when complete.

## How the OpenVLA env was made to work (envs/vla-openvla.lock.txt)
- torch 2.2.0+cu121, transformers 4.40.1, numpy 1.26.4, mujoco **2.3.2** (robosuite
  1.4.1 API — 3.x breaks `mj_fullM`), robosuite 1.4.1, tensorflow 2.15, imageio.
- **Load via HF trust_remote_code + attn_implementation="sdpa"** — NO flash-attn
  (wheel build fails; sdpa is identical attention math), NO `import prismatic`.
- **Do NOT import experiments.robot.{libero_utils,openvla_utils,robot_utils}** —
  they transitively `import prismatic` → dlimp → a protobuf≥5.26 `_pb2`
  (`runtime_version`) that conflicts with tensorflow 2.15's protobuf 4.25.
  eval_task_openvla.py **inlines** the (verbatim) helper bodies instead and uses
  only the clean `libero` primitives + tf (tf itself imports fine).
- **unnorm_key="libero_goal"** (this ckpt's norm_stats has no _no_noops key).
- **Isolated LIBERO config:** LIBERO_CONFIG_PATH=~/.libero_openvla (config.yaml
  points at openvla/LIBERO/libero/libero/{bddl_files,init_files,assets}) so it does
  NOT collide with the vla-smolvla hf-libero paths (§6). LIBERO installed with
  `-e LIBERO --config-settings editable_mode=compat` (default editable produced an
  empty module map → import failed).
- Checkpoint (~15G) on scratch: $HF_HOME/hub/models--openvla--openvla-7b-finetuned-libero-goal.

## OFT-7.5B (3rd scale point) — UNBLOCKED + RUNNING
- Import poison RESOLVED: `tensorflow-metadata==1.14.0` (its anomalies_pb2 was the
  proto≥5.26 `runtime_version` culprit) → protobuf 3.20.3 → then `wandb==0.16.6`
  (0.28's pb2 broke under proto 3.20) + a lazy `prismatic/__init__.py`
  (report/patches/oft_prismatic_init_lazy.diff). Now `experiments.robot.*` import
  cleanly, so run/eval_task_oft.py REUSES them (get_model/get_action_head/
  get_proprio_projector/get_action) rather than inlining.
- Checkpoint dl'd (15G, scratch): base VLA + action_head--50000 + proprio_projector
  --50000 + dataset_statistics (unnorm_key=libero_goal_no_noops). get_model needs a
  LOCAL DIR (loads .pt heads by filename) → runner resolves via snapshot_download.
- Smoke PASS: task7 original success (71 steps), blank fail → causal signature holds.
- **RQ1.3 grid RUNNING** (original/blank/nonsense/wrong_task, seed7, 2ep/task);
  aggregate + fold into rq1_scale as the 7.5B row when complete.

## Remaining (optional / next)
- `openvla_oft` (7.5B, 3rd scale point): env **vla-oft is BUILT**
  (`envs/vla-oft.lock.txt`: torch 2.2.0+cu121, moojink transformers fork,
  dlimp 0.0.1, LIBERO + robosuite 1.4.1/mujoco 2.3.2, numpy<2; PYOK). **Blocker:**
  `import prismatic` → `dlimp` → `google.protobuf runtime_version` (needs proto
  ≥5.26, conflicts w/ tf 2.15's 4.25). For OFT this must be *resolved* (its
  parallel-decoding/L1-head/proprio path needs the prismatic classes — the
  inline-around used for plain OpenVLA won't work). Fix options for next session:
  (a) patch openvla-oft `prismatic/__init__.py` to lazy-import `.models` so the
  dlimp/RLDS training path isn't pulled at import (record diff per §9); or
  (b) resolve the proto/tf version conflict. Then: download OFT ckpt
  (`moojink/openvla-7b-oft-finetuned-libero-goal`), write run/eval_task_oft.py
  (use_l1_regression=True, num_images_in_input=2, use_proprio=True, center_crop,
  unnorm_key), run reduced grid.
- `openvla_oft_film`: confirm the FiLM-trained ckpt id first (§2/§6.3) — leave
  null until verified (missing > invented).
- More seeds/episodes for tighter CIs if HPC time allows (SmolVLA + RQ1.2 conds
  are still seed-7-only).

## Infra notes
- Home per-user quota is tight; HF cache on scratch (HF_HOME); clear
  ~/.cache/{huggingface,uv} on "disk quota exceeded" (FS itself has 112T free).
- H200 nodes work; P100 (sm_60) unsupported by torch — preflight first. One
  process per task (single EGL context); never SIGKILL render procs.
