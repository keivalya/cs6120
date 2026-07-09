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

**Status: PARTIAL** — tasks 0–7 complete (+task 8 partial), seed 7, n_episodes=2.
Tasks 8–9 pending a compatible GPU (the last allocated node has Tesla P100 /
sm_60, unsupported by torch 2.10). Ignore/OAR is exact (env success = true task
achieved); Follow is detected only for `wrong_object` via a movable-object
displacement heuristic (`wrong_action`/`wrong_task` Follow is under-counted —
needs per-task goal-predicate eval; see analyze/css.py).

| condition | n | Ignore (OAR) | Follow | Fail |
|-----------|---|--------------|--------|------|
| `wrong_object` (swap object noun, in-scene) | 12 | **0.08** | 0.33 | 0.58 |
| `wrong_action` (swap verb) | 18 | **0.50** | 0.00* | 0.50 |
| `wrong_task` (another Goal task's instruction) | 17 | **0.00** | 0.00* | 1.00 |
| `repeated` (benign: instruction doubled) | 16 | TSR **0.06** | — | — |

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
- **Fragility to benign redundancy** — `repeated` TSR=0.06 (vs original 0.85):
  duplicating the identical instruction collapses success. Notable robustness gap.

Numbers trace to `results/smolvla/libero_goal/{wrong_*,repeated}/seed7/`. Will be
refreshed to the full 10-task grid once an sm_70+ GPU is available (idempotent
resume finishes tasks 8–9).

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
