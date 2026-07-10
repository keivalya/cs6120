# GATE 4 status — multi-model (RQ1.3)

## DONE
- **GATE 3 (SmolVLA/libero_goal):** RQ1.1 CSS=1.0, RQ1.2 OAR, locus. report/report.md.
- **GATE 4 RQ1.3 — OpenVLA-7B COMPLETE (RQ1.1/1.3 causal grid):**
  `openvla/openvla-7b-finetuned-libero-goal`, libero_goal, 10 tasks, seed 7,
  2 ep/task (20/condition). Result: **original 0.70, blank 0.00, nonsense 0.00,
  wrong_task 0.00 → CSS(blank)=CSS(nonsense)=1.00, OAR(wrong_task)=0.00.**
  scene_fixed_check pass (0/20 mismatches). Matches SmolVLA (CSS=1.0) →
  **causal language reliance is complete at BOTH 0.45B and 7B (16× scale).**
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

## Remaining (optional / next)
- `openvla_oft` (7.5B): build env vla-oft (§6.3 — forked transformers + dlimp);
  gives a 3rd scale point. `openvla_oft_film`: confirm the FiLM-trained ckpt id
  first (§2/§6.3) — leave null until verified (missing > invented).
- More seeds/episodes for tighter CIs if HPC time allows.

## Infra notes
- Home per-user quota is tight; HF cache on scratch (HF_HOME); clear
  ~/.cache/{huggingface,uv} on "disk quota exceeded" (FS itself has 112T free).
- H200 nodes work; P100 (sm_60) unsupported by torch — preflight first. One
  process per task (single EGL context); never SIGKILL render procs.
