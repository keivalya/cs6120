# GATE 4 status — multi-model (RQ1.3), OpenVLA-7B env in progress

## Done
- **GATE 3 COMPLETE** (SmolVLA/libero_goal): RQ1.1 CSS=1.0, RQ1.2 OAR
  (wrong_object 0.07 / wrong_action 0.45 / wrong_task 0.00; repeated 0.10), locus
  = planning-level. See report/report.md, report/rq1_causal.csv.
- **vla-openvla env core built** (on home; quota freed by clearing ~23G of stray
  ~/.cache/{huggingface,uv}): openvla 0.0.3, torch 2.2.0+cu121 (verified on H200),
  transformers 4.40.1, timm 0.9.10, numpy 1.26.4, draccus 0.8.0; original LIBERO
  installed (`openvla/LIBERO`, editable). Lock: envs/vla-openvla.lock.txt.

## Open blockers for OpenVLA eval (each fixable; do in order)
1. **protobuf too old** — `import prismatic` fails: `cannot import name
   'runtime_version' from google.protobuf` (dlimp/tf-datasets needs newer proto).
   Fix: `pip install 'protobuf>=4.25'` in vla-openvla (watch tf/dlimp pins).
2. **flash-attn 2.5.5 wheel build failed** — NOT required. Load OpenVLA with
   `attn_implementation="sdpa"` (mathematically identical attention; the fused
   kernel is only a speedup). Avoids the compile entirely.
3. **Cross-env LIBERO config collision (§6)** — `~/.libero/config.yaml` currently
   points at the vla-smolvla *hf-libero* bddl paths; openvla uses *original*
   LIBERO with a different layout. Fix: set `LIBERO_CONFIG_PATH` to an
   openvla-specific dir (e.g. export LIBERO_CONFIG_PATH=$HOME/.libero_openvla) and
   seed it once from the openvla env so it points at openvla/LIBERO/libero bddl.

## Remaining work (after blockers)
4. Download `openvla/openvla-7b-finetuned-libero-goal` (~14 GB) to $HF_HOME (scratch).
5. **New eval runner** `run/eval_task_openvla.py` — OpenVLA has a different
   obs/action interface than lerobot SmolVLA: load via AutoModelForVision2Seq +
   AutoProcessor (attn=sdpa, bf16), `unnorm_key="libero_goal_no_noops"`,
   center_crop=True, single 3rd-person cam. Reuse the RELIABLE pattern from
   run/eval_task.py: ONE process per task, SYNC LIBERO env at obs_hw=256, inject
   perturbed `task_description` per episode, record success + reset_state_hash +
   poses. Mirror the summary/episodes output contract so run/aggregate.py +
   analyze/* work unchanged. models.yaml already has the openvla alias/flags.
6. Run the REDUCED grid (§8): libero_goal, conditions [original,blank,wrong_task,
   para_object,para_action] (start with original,blank,wrong_task for RQ1.1/1.2),
   seeds [7,42], n_episodes 10 (or 2 for a first pass). Preflight first.
7. Repeat for `openvla_oft` (env vla-oft, §6.3 — forked transformers + dlimp) and
   confirm the FiLM checkpoint id before `openvla_oft_film` (§2: mismatched
   use_film/ckpt = garbage).
8. RQ1.3 table: CSS vs params (SmolVLA 0.45B → OFT 7.5B) — analyze/ + report.

## Infra notes
- Home per-user quota is tight; keep HF cache on scratch (HF_HOME) and clear
  ~/.cache/{huggingface,uv} if `disk quota exceeded` recurs. Filesystem itself has
  space (df: 112T free) — it's the per-user quota that bites.
- H200 nodes (d4053/d4054) work; the P100 node (c2189) is sm_60, unsupported by
  torch — always `run/preflight.py` first. Never SIGKILL render procs (Xid faults).
