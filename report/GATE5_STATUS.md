# GATE 5 status — RQ2 paraphrase + RQ1 rigor gaps (code turnkey; runs blocked)

## Code COMPLETE + validated (CPU), committed 27b1f2b..8908475
- **B1 paraphrase generator** `perturb/make_paraphrases.py` → `perturb/generated/
  libero_goal_paraphrases.jsonl` (4092 rows, 0 unmatched; axes obj/act/comp with
  SK/ST/operation). Stratified `--max_per_axis`. ✔ validated.
- **A2 goal-predicate Follow/Ignore/Fail** `analyze/goal_eval.py` (eval any Goal
  goal vs live shared-scene env; API in report/patches notes) + rewritten
  `analyze/css.py` trichotomy. ✔ validated exact 1/1/1 on synthetic. Runners log
  `achieved_task_ids` per episode (all 3, defensive).
- **B2 paraphrase eval mode** `--paraphrase_axis` in all 3 runners + shared
  `run/para_util.py`. py_compile clean, para_util CPU-tested.
- **B4 PRIDE** `analyze/pride_wrap.py` (wraps LIBERO-Para compute_model_pride via
  seed*/evalN.json adapter). ✔ validated PRIDE=66.7 vs hand-calc. `analyze/
  make_rq2_csv.py` → rq2_paraphrase/axis/operation.csv.
- **B6 figures** `analyze/plot_rq2.py` (axis bars, ΔTSR-vs-PD scatter, locus).
- **Turnkey driver** `run/run_fresh_node.sh [A|B|ALL]` — serialized ONE-EGL-proc-
  at-a-time sweep, preflight-gated, then aggregate+analyze.

## BLOCKED: needs a FRESH GPU allocation
d4054 render channel degraded by an accidental concurrent-EGL clash this session
(SmolVLA grid + OpenVLA smoke at once — see memory concurrent-egl-corrupts-channel).
Confirmed still degraded: a light render probe passes but the real runner (7B
inference + render) SIGABRTs (exit 134) on the first render, even run alone. No
user-space fix; a fresh node resets the channel.

**On a fresh node, one command:** `bash run/run_fresh_node.sh ALL`
It runs (ONE EGL process at a time — never parallelize):
1. Phase A: goal-logged `wrong_*` re-runs (all 3 models, seeds 7+42) → exact
   Follow/Ignore/Fail; OFT `wrong_object/action/repeated`; seed-42 everywhere.
2. Phase B: paraphrase sweeps — SmolVLA full, OpenVLA+OFT `--max_per_axis 15`.
3. aggregate → make_rq1_csv/make_rq1_scale/plot_rq1 → make_rq2_csv/plot_rq2.

Then finalize (Phase C): MANIFEST append, report.md RQ2.1–2.4 sections + refreshed
RQ1.2 (exact Follow) / RQ1.3 (seed-42 std), RQ2.4 locus (analyze/locus.py over the
para axes → rq2_locus.csv). `openvla_oft_film` stays blocked (no FiLM ckpt on HF).

## SmolVLA render caveat
SmolVLA/lerobot has only ever rendered on d4053; on d4054 its path SIGABRTs even
apart from the channel issue. Verify SmolVLA renders on the fresh node (the driver
preflights OpenVLA, not lerobot); if SmolVLA still aborts, its RQ2-full + seed-42
need a node where lerobot renders (d4053-class).
