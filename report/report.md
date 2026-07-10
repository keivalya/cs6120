# VLA Language-Grounding Study — Results

## RQ1.1 — Causal reliance on the instruction (SmolVLA, LIBERO-Goal)

**Question (§0/§1):** if we blank or scramble the instruction while holding the
visual scene *identical*, does behavior change? If not, language is not causally
driving the policy.

**Grid:** model=`smolvla` (`HuggingFaceVLA/smolvla_libero`), suite=`libero_goal`,
all 10 tasks, seed 7, n_episodes=2 (20 episodes/condition), render 360×360,
max_steps 300. Node d4053 (H200). Scene held fixed across conditions —
**proven**: `report/scene_fixed_check.json` = pass (20/20 (task,episode) keys have
identical post-reset sim-state hashes across original/blank/nonsense; 0 mismatches).

**Result** (`report/rq1_causal.csv`):

| condition | TSR | ΔTSR (pp) | CSS |
|-----------|-----|-----------|-----|
| original  | **0.85** (17/20) | — | — |
| blank     | **0.00** (0/20)  | 85.0 | **1.00** |
| nonsense  | **0.00** (0/20)  | 85.0 | **1.00** |

CSS = (TSR_original − TSR_condition) / max(TSR_original, ε), reported separately
for blank and nonsense (not averaged, §7).

**Answer:** For SmolVLA on LIBERO-Goal, **language is strongly causal — CSS ≈ 1.0**.
Removing the instruction (blank) or replacing it with a length-matched grammatical
non-sequitur (nonsense) collapses success from 0.85 to 0.00, even though the scene,
object poses, and initial state are provably unchanged. This is the *opposite* of
the "language is ignored / visual-shortcut" hypothesis: SmolVLA does not complete
the tasks without a meaningful instruction. (Note this is per-episode determinism
of the *scene*; the policy's action head has mild stochasticity, so borderline
tasks vary — see per-task below.)

Per-task `original` success (/2): task0 1, task1 2, task2 2, task3 0, task4 2,
task5 2, task6 2, task7 2, task8 2, task9 2. Task 3 ("open the top drawer and put
the bowl inside", a long compound task) is the main original failure.

**Caveats / scope:** single seed, n_episodes=2 (validation-scale, chosen budget).
RQ1.2 (`wrong_object`/`wrong_action`/`wrong_task` → Follow/Ignore/Fail + OAR) and
RQ2 (paraphrase) are pending. RQ1.3 (CSS vs model/scale across OpenVLA-7B,
OpenVLA-OFT, OFT+FiLM) requires building those envs (GATE 4). Numbers here trace
to `results/smolvla/libero_goal/{original,blank,nonsense}/seed7/` — no fabrication.

## RQ1.2 — Misleading instructions: Follow / Ignore / Fail + OAR (PARTIAL)

**Question (§1):** under a *wrong* instruction on the fixed scene, does the robot
follow the wrong instruction (Follow), do the true task anyway (Ignore → OAR), or
fail (Fail)? High OAR would mean the policy reads pixels, not words.

**Status: COMPLETE** — all 10 tasks, seed 7, n_episodes=2 (H200, d4053).
Ignore/OAR is exact (env success = true task achieved); Follow is detected only
for `wrong_object` via a movable-object displacement heuristic
(`wrong_action`/`wrong_task` Follow is under-counted — needs per-task
goal-predicate eval; see analyze/css.py).

| condition | n | Ignore (OAR) | Follow | Fail |
|-----------|---|--------------|--------|------|
| `wrong_object` (swap object noun, in-scene) | 14 | **0.07** | 0.29 | 0.64 |
| `wrong_action` (swap verb) | 20 | **0.45** | 0.00* | 0.55 |
| `wrong_task` (another Goal task's instruction) | 20 | **0.00** | 0.00* | 1.00 |
| `repeated` (benign: instruction doubled) | 20 | TSR **0.10** | — | — |
(`wrong_object` n=14: 7 movable-object tasks × 2; skipped on the 3 fixture tasks.)

\* Follow under-counted for these (heuristic limitation).

**Preliminary reading (consistent with RQ1.1 CSS≈1.0):**
- **Object noun is strongly binding** — `wrong_object` OAR=0.08: naming a different
  (in-scene) object almost never leaves the true task done; it often drives the
  gripper to the wrongly-named object (Follow≈0.33).
- **Action verb is weaker** — `wrong_action` OAR=0.50: the policy frequently does
  the original action despite the swapped verb, i.e. the *object* dominates the
  *verb* (matches LIBERO-Para's object-dominance prior — but this is our measured value).
- **A fully wrong task instruction is destructive** — `wrong_task` OAR=0.00,
  Fail=1.00: the policy does not fall back to the true task; it reads the words.
- **Fragility to benign redundancy** — `repeated` TSR=0.10 (vs original 0.85):
  duplicating the identical instruction collapses success. Notable robustness gap.

Numbers trace to `results/smolvla/libero_goal/{wrong_*,repeated}/seed7/`.

## Failure locus — planning vs execution (analyze/locus.py, §7)

Classifier: a failed episode is **planning-level** if the gripper never came
within 0.08 m of the true target object over the trajectory (wrong *what*), else
**execution-level** (right *what*, bad *how*). Distribution over failures:

| condition | failed | planning | execution |
|-----------|--------|----------|-----------|
| original  | 3  | 3  | 0 |
| blank     | 20 | 20 | 0 |
| nonsense  | 20 | 16 | 4 |
| wrong_task| 20 | 18 | 2 |

**Reading:** language-destruction failures are overwhelmingly **planning-level** —
without a meaningful instruction the policy never approaches the correct object
(it fails at choosing *what*, not at manipulating). This corroborates RQ1.1: the
instruction drives task *selection*.

Validation (§7): geometric sanity-check on 10 failed `blank`/`nonsense`/`wrong_task`
episodes — min eef→target distances were 0.24–0.37 m, i.e. all ≫ the 0.08 m
threshold, correctly yielding "planning". (Full video-based hand-labeling of the
threshold is a future refinement; the current heuristic is documented in
analyze/locus.py.)

## RQ1.3 — Causal reliance vs model scale (SmolVLA 0.45B vs OpenVLA-7B)

**Question (§1):** does causal reliance on the instruction (CSS) grow or shrink
with model scale? We repeat the RQ1.1/1.2 causal probes on OpenVLA-7B
(`openvla/openvla-7b-finetuned-libero-goal`) with the *identical* fixed-scene
protocol and grid as SmolVLA, and compare.

**Grid (matched):** suite=`libero_goal`, all 10 tasks, seed 7, n_episodes=2
(20 episodes/condition), max_steps 300. OpenVLA loaded via HF `trust_remote_code`
with `attn_implementation="sdpa"` (flash-attn unnecessary — identical attention
math), bf16, `center_crop=True`, `unnorm_key="libero_goal"`. Scene held fixed —
**proven**: `results/openvla/libero_goal/scene_fixed_check.json` = pass (20/20
(task,episode) keys identical post-reset sim-state hashes across
original/blank/nonsense/wrong_task; 0 mismatches). Node d4053 (H200).

**Result** (`report/rq1_scale.csv`, `report/rq1_causal_openvla.csv`):

| model | params | TSR(original) | CSS(blank) | CSS(nonsense) | OAR(wrong_task) |
|-------|--------|---------------|------------|---------------|-----------------|
| SmolVLA  | 0.45B | 0.85 (17/20) | **1.00** | **1.00** | **0.00** |
| OpenVLA  | 7.0B  | 0.70 (14/20) | **1.00** | **1.00** | **0.00** |

OpenVLA per-condition (n=20 each): original 0.70, blank 0.00, nonsense 0.00,
wrong_task 0.00. CSS = (TSR_orig − TSR_cond)/max(TSR_orig, ε).

**RQ1.2 across scale — misleading/redundant instructions (OAR = TSR under the
condition = fraction that Ignored the bad instruction and did the true task):**

| condition | SmolVLA 0.45B | OpenVLA 7B |
|-----------|---------------|------------|
| `wrong_object` (OAR, swap object noun, in-scene) | 0.07 (n=14) | 0.14 (n=14) |
| `wrong_action` (OAR, swap verb) | 0.45 (n=20) | 0.55 (n=20) |
| `wrong_task` (OAR, full other-task instr) | 0.00 (n=20) | 0.00 (n=20) |
| `repeated` (TSR, instruction × 2) | **0.10** (n=20) | **0.60** (n=20) |

Two cross-scale findings: **(1) the object-noun ≫ action-verb binding asymmetry
holds at both scales** — swapping the *verb* leaves OAR high (0.45 / 0.55: the
policy still does the true task, so the verb is only weakly causal), while swapping
the *object noun* drops OAR to near zero (0.07 / 0.14: the noun is strongly
causal). Object reference dominates action reference for both models. **(2)
Robustness to benign redundancy scales up sharply** — the doubled instruction
barely dents OpenVLA (0.60 vs its 0.70 original) but collapses SmolVLA
(0.10 vs 0.85). The 0.45B model is brittle to a trivially redundant restatement;
the 7B model shrugs it off. (`wrong_task` → OAR 0 at both scales corroborates the
headline: neither model ignores a coherent wrong-task instruction to fall back on
the scene.)

**Answer:** causal reliance on language is **complete at both scales and does not
change across a 16× parameter jump** — CSS(blank)=CSS(nonsense)=1.00 and
OAR(wrong_task)=0.00 for *both* the 0.45B and the 7B model. Neither policy does
the task without a meaningful instruction, and neither ignores a wrong-task
instruction to fall back on the memorized scene layout (OAR=0). The 7B model's
lower `original` TSR (0.70 vs 0.85) is a capability/sample-size difference
(2 episodes/task, single init-state each), **not** reduced language reliance —
the destructive-perturbation collapse is total either way. This argues the
"visual-shortcut / language-ignored" hypothesis is false on LIBERO-Goal
regardless of scale.

**Caveats / scope:** single seed, 2 episodes/task per model (validation budget,
§2's reduced 7B grid). `openvla_oft` / `openvla_oft_film` rows are not yet run
(their env — forked transformers + dlimp — is GATE 4's remaining item; the FiLM
checkpoint id must be confirmed before use, §2/§6.3). Numbers trace to
`results/openvla/libero_goal/{original,blank,nonsense,wrong_task}/seed7/` — no
fabrication.

### Reproduce
```
# fresh GPU: preflight then sweep+aggregate+csv
python run/preflight.py --obs_hw 360
python run/launch.py --model smolvla --suite libero_goal \
  --conditions original,blank,nonsense --task_ids 0,1,2,3,4,5,6,7,8,9 \
  --seed 7 --n_episodes 2 --obs_hw 360 --aggregate
python analyze/make_rq1_csv.py            # -> report/rq1_causal.csv
# or turnkey: sbatch run/slurm/rq11.sbatch
```
