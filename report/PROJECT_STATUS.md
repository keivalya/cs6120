# VLA Language-Grounding Study — Project Status & Completion Guide

**Single source of truth.** Consolidates GATE 3/4/5 status. Read this first, then run
the commands in §6 on a fresh GPU node to complete the project.

Last updated: 2026-07-11. Head commit: `fcb6693`.

---

## 1. The question & design (recap)

Do VLA policies actually *use* the language instruction, or lean on visual shortcuts?
We hold the LIBERO-Goal scene fixed and perturb only the instruction.
- **RQ1 causal reliance** — blank/nonsense/wrong instructions on a fixed scene.
- **RQ2 meaning-preserving fragility** — paraphrases (same meaning), object- vs
  action-axis, PRIDE.

All causal claims are LIBERO-Goal only (shared scene → instruction is the sole task
cue). No fabricated numbers; every scene-fixed run proves identical reset-state hashes.

---

## 2. STATUS AT A GLANCE

| RQ | Scope | Status |
|----|-------|--------|
| RQ1.1 CSS(blank/nonsense) | 3 models | ✅ **DONE** |
| RQ1.2 Follow/Ignore/Fail + OAR | smolvla, openvla | ✅ done (OAR); ⚠️ Follow exact-classifier coded, needs `wrong_*` re-run |
| RQ1.2 | openvla_oft | ⛔ `wrong_object/action/repeated` not run (**A1**) |
| RQ1.3 CSS vs scale | 3 models (0.45B→7.5B) | ✅ **DONE** |
| RQ1 seeds | smolvla/oft seed-7 only | ⛔ seed-42 pending (**A3**) |
| RQ2.1–2.4 paraphrase | all models | ⛔ **not run** — all code turnkey (**B3**) |
| openvla_oft_film | — | 🚫 blocked: no FiLM checkpoint published on HF |

**Everything not "run" is coded, committed, and validated where CPU-testable. The only
thing left is GPU compute time on a fresh node** (§6). Current node d4054 render channel
is degraded (see §5).

---

## 3. RESULTS SO FAR (measured, in `results/` + `report/`)

### RQ1.1 + RQ1.3 — causal reliance vs scale (`report/rq1_scale.csv`, `rq1_scale.png`)

| model | params | TSR(orig) | CSS(blank) | CSS(nonsense) | OAR(wrong_task) | n / seeds |
|-------|--------|-----------|------------|---------------|-----------------|-----------|
| SmolVLA | 0.45B | 0.85 | **1.00** | **1.00** | **0.00** | 20 / 7 |
| OpenVLA | 7.0B | 0.70 | **1.00** | **1.00** | **0.00** | 40 / 7,42 (std 0) |
| OpenVLA-OFT | 7.5B | 1.00 | 0.85 | 0.95 | **0.00** | 20 / 7 |

**Headline:** language reliance is high and essentially scale-invariant across 16×;
wrong-task OAR=0 everywhere. **Nuance:** OFT (only model with wrist-cam + proprio) has a
small visual-shortcut capacity (CSS(blank)=0.85) vs 1.00 for the single-camera models.

### RQ1.2 — misleading/redundant instructions (OAR = did true task anyway), seed 7

| condition | SmolVLA | OpenVLA | OpenVLA-OFT |
|-----------|---------|---------|-------------|
| wrong_object (OAR) | 0.07 | 0.14 | ⛔ A1 |
| wrong_action (OAR) | 0.45 | 0.55 | ⛔ A1 |
| wrong_task (OAR) | 0.00 | 0.00 | 0.00 |
| repeated (TSR) | 0.10 | 0.60 | ⛔ A1 |

Object-noun ≫ action-verb binding at both scales; redundancy-robustness scales up.
(Follow/Fail split currently uses a displacement heuristic; **A2** upgrades it to the
exact goal-predicate method after the `wrong_*` re-run.)

### Failure locus — `analyze/locus.py` (SmolVLA)
Language-destruction failures are overwhelmingly **planning-level** (never approach the
right object) → instruction drives task *selection*.

---

## 4. WHAT'S BUILT (code map)

Envs (all built; locks in `envs/*.lock.txt`): `vla-smolvla` (py3.12, lerobot),
`vla-openvla` (py3.10, torch2.2/sdpa), `vla-oft` (py3.10, moojink forks +
tf-metadata 1.14/protobuf 3.20.3/wandb 0.16.6), `libero-para-metrics`.

| Component | File | Status |
|-----------|------|--------|
| Perturbation gen (RQ1) | `perturb/make_instructions.py` | ✅ |
| Paraphrase gen (RQ2) | `perturb/make_paraphrases.py` → `perturb/generated/libero_goal_paraphrases.jsonl` (4092) | ✅ validated |
| SmolVLA runner | `run/eval_task.py` | ✅ (+ goal-log, `--paraphrase_axis`) |
| OpenVLA runner | `run/eval_task_openvla.py` | ✅ (sdpa; + goal-log, paraphrase) |
| OFT runner | `run/eval_task_oft.py` | ✅ (L1/proprio/8-chunk; + goal-log, paraphrase) |
| Paraphrase loader | `run/para_util.py` | ✅ |
| Goal-predicate eval | `analyze/goal_eval.py` | ✅ validated |
| CSS + Follow/Ignore/Fail | `analyze/css.py` | ✅ goal-predicate trichotomy validated |
| TSR / aggregation | `analyze/tsr.py`, `run/aggregate.py` | ✅ |
| RQ1 CSVs/plots | `analyze/make_rq1_csv.py`, `make_rq1_scale.py`, `plot_rq1.py` | ✅ |
| PRIDE wrapper | `analyze/pride_wrap.py` | ✅ validated (PRIDE=66.7 vs hand-calc) |
| RQ2 CSVs/plots | `analyze/make_rq2_csv.py`, `plot_rq2.py` | ✅ (produce tables once data lands) |
| Failure locus | `analyze/locus.py` | ✅ (condition-agnostic; used for RQ2.4) |
| Preflight | `run/preflight.py` | ✅ |
| **Turnkey driver** | `run/run_fresh_node.sh [A\|B\|ALL]` | ✅ serialized, preflight-gated |

---

## 5. THE GPU BLOCKER (why runs are paused)

Running **two EGL render processes at once corrupts the GPU render channel** (a SmolVLA
grid + an OpenVLA smoke overlapped this session). Afterwards heavy CUDA+EGL (a VLA +
render) **SIGABRTs (exit 134)** on the first render even run alone; a *light* render-only
probe still passes (misleading — use the real runner to test). No user-space fix — a
**fresh allocation resets the channel**. Discipline: **one EGL process per GPU at a
time**; `run/run_fresh_node.sh` enforces this (fully sequential).

**SmolVLA caveat:** SmolVLA/lerobot has only ever rendered on **d4053-class** nodes; on
d4054 it aborts independent of the channel issue. Verify SmolVLA renders on the fresh
node (the driver preflights OpenVLA); if it still aborts, run SmolVLA's share on a
d4053-class node.

---

## 6. COMMANDS TO COMPLETE THE PROJECT (run on a FRESH GPU node)

```bash
cd /home/pandya.kei/CS6120

# 0) Global env (the driver sets these too; shown for manual runs)
export HF_HOME=/scratch/pandya.kei/hf_home HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 TOKENIZERS_PARALLELISM=false

# 1) PREFLIGHT — confirm the fresh channel is healthy (must exit 0)
/home/pandya.kei/.conda/envs/vla-openvla/bin/python run/preflight.py

# 2) ONE COMMAND for all remaining runs + analysis (serialized, ~several H200-hrs)
bash run/run_fresh_node.sh ALL
#   Phase A: goal-logged wrong_* re-runs (3 models, seeds 7+42) -> exact
#            Follow/Ignore/Fail; OFT wrong_object/action/repeated; seed-42 everywhere.
#   Phase B: paraphrase sweeps — SmolVLA full, OpenVLA+OFT --max_per_axis 15.
#   Then: aggregate + make_rq1_csv/make_rq1_scale/plot_rq1 + make_rq2_csv/plot_rq2.
#   NEVER run two copies at once (EGL). To split: `bash run/run_fresh_node.sh A`
#   then later `... B`. Progress log: /scratch/pandya.kei/fresh_node.log
```

If SmolVLA aborts on this node, run its share on a d4053-class node (same commands),
e.g. just the SmolVLA lines — full paraphrase axes + seed-42:
```bash
SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python
for tid in 0 1 2 3 4 5 6 7 8 9; do
  for ax in para_object para_action para_compositional; do
    $SM run/eval_task.py --task_id $tid --paraphrase_axis $ax --seed 7 --n_episodes 1 --obs_hw 256
  done
  $SM run/eval_task.py --task_id $tid --conditions original,blank,nonsense,repeated,wrong_object,wrong_action,wrong_task --seed 42 --n_episodes 2 --obs_hw 256
done
```

### 3) Finalize (Phase C — after runs land)
`run/run_fresh_node.sh` already runs `make_rq2_csv.py` (which now emits
`rq2_paraphrase.csv`, `rq2_axis.csv`, `rq2_operation.csv`, **and `rq2_locus.csv`**
— RQ2.4 planning/execution via `analyze/locus.py`) + `plot_rq2.py`. Spot-check:
```bash
OV=/home/pandya.kei/.conda/envs/vla-openvla/bin/python
$OV analyze/pride_wrap.py --model smolvla     # PRIDE in [0,100]; per-axis TSR <= orig
# validate the goal-predicate classifier on >=10 hand-labeled wrong_* eps:
$OV analyze/css.py --model openvla            # new Ignore should match old exact OAR
# validate locus threshold on hand labels (optional): analyze/locus.py --validate labels.json
```
Then update `report/report.md`: add **RQ2.1–2.4** sections (table+figure+grid caption
each); refresh **RQ1.2** with exact Follow/Ignore/Fail; **RQ1.3** with seed-42 mean±std.
Append every new run to `MANIFEST.json` (model, suite, condition/axis, seed, n, success,
checkpoint+repo SHAs). Keep `openvla_oft_film` documented as blocked (no FiLM ckpt).

---

## 7. VERIFICATION GATES (do these before trusting numbers)
1. **Preflight** passes on the fresh node (else request another — do not fabricate).
2. **Scene-fixed:** `run/aggregate.py` writes `scene_fixed_check.json` = pass (0
   mismatches) for every new causal/paraphrase run.
3. **Goal classifier (A2):** on ≥10 hand-labeled `wrong_*` episodes confirm
   `achieved_task_ids`; check new Ignore == old exact OAR (regression).
4. **PRIDE:** already validated vs LIBERO-Para (PRIDE=66.7). Assert PRIDE∈[0,100] and
   per-axis TSR ≤ original TSR on real data.
5. **Smoke before every grid** (1 task) — and **never** overlap two EGL processes.

## 8. DEFINITION OF DONE (CLAUDE.md §1)
Per RQ: a `report/*.csv` table + a `report/*.png` figure + a `report.md` paragraph with
the exact grid in the caption, plus `MANIFEST.json` covering every run. RQ1.1/1.3 already
meet this; RQ1.2 (all 3 models, exact Follow), RQ2.1–2.4 land after §6.

## 9. Out of scope / blocked
- `openvla_oft_film`: no FiLM-trained checkpoint published under `moojink` on HF
  (verified) — do NOT substitute the plain OFT checkpoint (§6.3: garbage). Documented.
- RQ1.4 horizon (libero_10/Long): optional; runners already take `--suite`. Causal claims
  stay LIBERO-Goal/Long only, never scene-confounded Spatial/Object.
- seed 123 / higher episode counts (full grid.yaml scale): beyond seed 42.
