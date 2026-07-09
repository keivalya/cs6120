# CLAUDE.md — Autonomous VLA Language-Grounding Study

> **Read this file top to bottom before running anything.** You (Claude) are the
> research engineer executing this project end-to-end and mostly unattended. Your
> job is to set up environments, run the eval grid, compute metrics, debug
> failures, and produce honest results tables/plots that answer the research
> questions below. **Never fabricate, interpolate, or "estimate" a number that a
> run did not actually produce.** A missing cell in a results table is fine; a
> made-up one is a project-ending failure.

---

## 0. TL;DR of what we are testing

Do VLA models actually *use* the language instruction, or do they lean on visual
shortcuts and memorized task layouts? We hold the visual scene fixed and perturb
only the instruction. Two questions:

- **RQ1 — Causal reliance.** If we blank / scramble / contradict the instruction
  while the scene is unchanged, does behavior change? If it doesn't, language is
  not causally driving the policy.
- **RQ2 — Meaning-preserving fragility.** If we paraphrase the instruction
  (same meaning), does success collapse? And is the damage driven by the *object*
  noun or the *action* verb?

The whole design rests on **LIBERO-Goal**: every task in that suite shares an
identical initial scene, so the instruction is the *only* cue that disambiguates
the task. That is why LIBERO-Para builds on Goal, and it's why our causal probes
are only valid on Goal (and, for horizon analysis, on LIBERO-10/Long as a
stretch). Do **not** run causal probes on Spatial/Object suites and claim causal
conclusions — there the scene co-varies with the task and the probe is confounded.

---

## 1. Research questions → concrete experiments (the contract)

| RQ | What to run | Primary metric |
|----|-------------|----------------|
| RQ1.1 | `original`, `blank`, `nonsense` instructions, scene fixed | ΔTSR, Causal Sensitivity Score (CSS) |
| RQ1.2 | `wrong_object`, `wrong_action`, `wrong_task` (swap toward another valid Goal task) | Follow / Ignore / Fail trichotomy + Original-Action-Rate (OAR) |
| RQ1.3 | Repeat RQ1.1–1.2 across SmolVLA-450M, OpenVLA-7B, OpenVLA-OFT (+FiLM) | CSS by model/scale |
| RQ1.4 (opt) | Repeat on Spatial / Object / Goal / Long to look at horizon | CSS vs horizon |
| RQ2.1 | LIBERO-Para full paraphrase suite, all models | TSR drop (pp), PRIDE |
| RQ2.2 | Object-axis vs action-axis paraphrases separately | per-axis ΔTSR |
| RQ2.3 | Per-operation breakdown (synonym, hypernym, reorder, register, distractor) | ΔTSR vs paraphrase distance (PD) |
| RQ2.4 | For every failure, classify planning-level vs execution-level | failure-locus distribution |

**Definition of done:** a single `results/` tree + `report/` with, for each RQ, a
table and a figure, plus a `MANIFEST.json` listing every run (model, suite,
condition, seed, n_episodes, git SHA of each repo, timestamp, success count).

---

## 2. Models under test (and their reality checks)

| Alias in this repo | HF / repo | Params | Notes |
|--------------------|-----------|--------|-------|
| `smolvla` | `lerobot/smolvla_base` (base) → finetune, or `HuggingFaceVLA/smolvla_libero` | ~0.45B | **Primary workhorse.** Cheap, fits ~8–16 GB. Run the *full* grid here. |
| `openvla` | `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}` | 7B | Stretch. ~16 GB inference, needs flash-attn. |
| `openvla_oft` | `moojink/openvla-7b-oft-finetuned-libero-{spatial,object,goal,10}` | 7.5B | Parallel decoding, L1 head. `use_film=False` by default. |
| `openvla_oft_film` | OFT checkpoint trained **with FiLM** — locate in the OFT repo/HF and verify before use | 7.5B | This is the "FiLM-conditioned OFT+" of RQ1.3. Do **not** assume the plain checkpoint is the FiLM one; confirm `use_film=True` matches the checkpoint it was trained with, or results are garbage. |

Budget rule (minimal-budget project): **SmolVLA gets the complete grid across all
conditions and seeds.** The 7B models get a reduced grid (Goal suite, fewer seeds,
fewer episodes) unless HPC time is plentiful. Log which grid each model actually ran.

There is a naming subtlety: the LIBERO-Para paper reports SmolVLA in the ~0.6B
class and OFT at 7.5B; keep our alias→checkpoint mapping in `configs/models.yaml`
and treat the checkpoint hash as ground truth, not the nickname.

---

## 3. Data & benchmark

- **Base benchmark:** LIBERO — `github.com/Lifelong-Robot-Learning/LIBERO`.
- **Paraphrase suite (RQ2):** LIBERO-Para —
  `github.com/cau-hai-lab/LIBERO-Para`, dataset `HAI-Lab/LIBERO-Para` on HF.
  4,000+ paraphrases, 10 scenarios, two axes (object-referring, action-referring),
  built on LIBERO-Goal. Held out for eval only.
- **LeRobot LIBERO port (for SmolVLA):** dataset `HuggingFaceVLA/libero`,
  eval via `lerobot-eval`.
- **Optional robustness cross-check:** LIBERO-Plus (`--env.type=libero_plus` in
  LeRobot) — visual perturbations, useful to contrast "language barely moves the
  policy but vision does" (RQ1 motivation). **Warning:** LIBERO-Plus installs
  under the *same* `libero` package name as base LIBERO — never install both in
  one env (see §6).

**Hard rule from the proposal: use existing datasets only.** Do not author a new
trajectory dataset. We only synthesize *instruction strings* (the perturbations),
never new demonstrations.

---

## 4. Repository layout to build

Clone LIBERO-Para as the backbone; it already implements the exact loop we need
("pre-create 10 LIBERO envs, swap in paraphrased instructions, query the model")
and ships per-model conda guides under `eval_guides/` and scripts under
`eval_scripts/examples/`. Extend it, don't reinvent it.

```
vla-lang-grounding/
├── CLAUDE.md                      # this file
├── LIBERO-Para/                   # cloned backbone (metrics, eval loop, PRIDE)
├── envs/                          # our conda env yamls, one per model (see §6)
├── configs/
│   ├── models.yaml                # alias -> checkpoint + eval flags
│   ├── grid.yaml                  # models × suites × conditions × seeds × n_episodes
│   └── perturbations.yaml         # causal-probe definitions (RQ1)
├── perturb/
│   └── make_instructions.py       # builds blank/nonsense/wrong_* strings (RQ1)
├── run/
│   ├── run_one.py                 # single (model,suite,condition,seed) eval -> log
│   ├── launch.py                  # reads grid.yaml, skips completed, dispatches
│   └── slurm/                     # sbatch templates (HPC)
├── analyze/
│   ├── tsr.py                     # TSR + ΔTSR
│   ├── css.py                     # Causal Sensitivity Score, Follow/Ignore/Fail, OAR
│   ├── locus.py                   # planning vs execution failure classifier
│   └── pride_wrap.py              # calls LIBERO-Para/metrics/analyze_results.py
├── results/
│   └── <model>/<suite>/<condition>/seed<k>/  # raw per-episode logs + summary.json
├── report/                        # tables (.csv) + figures (.png) + report.md
└── MANIFEST.json                  # every run, appended atomically
```

---

## 5. The causal-probe generator (RQ1) — `perturb/make_instructions.py`

For each LIBERO-Goal task with original instruction `I` naming object `O` and
action `A`, generate these fixed-scene conditions:

- `original` — `I` unchanged (control).
- `blank` — empty string `""`.
- `nonsense` — grammatical-but-meaningless token salad of matched length
  (e.g. shuffled out-of-domain nouns/verbs); keep length comparable to `I` so
  degradation isn't just a sequence-length artifact.
- `wrong_object` — replace `O` with a *different object that exists in the scene*
  (so the misleading instruction is physically executable, not impossible).
- `wrong_action` — replace `A` with a different valid verb for the same object.
- `wrong_task` — substitute the full instruction of *another* LIBERO-Goal task
  that shares the identical scene ("put the apple in the sink" while the true task
  is the stove). This is the cleanest RQ1.2 probe because both tasks are grounded
  in the same pixels.
- `repeated` — `I` concatenated with itself ("turn on the stove turn on the
  stove") to probe robustness to benign redundancy.

Rules:
- Derive `O`/`A` from LIBERO task metadata / `bddl` files, not by parsing English
  loosely. Cross-check against `LIBERO-Para/metrics/libero_para_metadata.csv`,
  which already tags object vs action spans.
- Every generated string + its provenance goes in
  `perturb/generated/<suite>.jsonl` so runs are reproducible and auditable.
- Do **not** touch the simulator scene, object poses, or initial state. Only the
  instruction string changes. Assert this in code (hash the env reset state and
  confirm it's identical across conditions for the same task+seed).

---

## 6. Environments — one conda env per model, NO shared deps

These models have genuinely conflicting dependency stacks (OFT needs a *forked*
`transformers`; OpenVLA pins its own; LeRobot tracks mainline recent
`transformers`; LIBERO-Plus and base LIBERO collide on the `libero` package
name). So isolation is mandatory, not cosmetic. Create each env, verify it in
isolation, and record the resolved versions to `envs/<name>.lock.txt`.

Set once, globally, for every headless run:
```bash
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$SCRATCH/hf_home          # big cache off your home quota (HPC)
export HF_HUB_CACHE=$HF_HOME/hub
```

### 6.1 `vla-smolvla` (primary)
```bash
conda create -n vla-smolvla python=3.10 -y
conda activate vla-smolvla
git clone https://github.com/huggingface/lerobot.git && cd lerobot
pip install -e ".[libero]"                # installs LIBERO + smolvla deps
# sanity: base checkpoint loads and LIBERO env spins up
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; \
           SmolVLAPolicy.from_pretrained('lerobot/smolvla_base'); print('smolvla OK')"
```
Eval entrypoint (native LeRobot):
```bash
lerobot-eval --policy.path=<smolvla_libero_ckpt> \
  --env.type=libero --env.task=libero_goal \
  --eval.batch_size=1 --eval.n_episodes=10
```
For our perturbation conditions, the instruction is injected per-episode via the
LIBERO-Para eval loop (swap the task language before `env.reset`), not via the CLI.

### 6.2 `vla-openvla`
```bash
conda create -n vla-openvla python=3.10 -y
conda activate vla-openvla
git clone https://github.com/openvla/openvla.git && cd openvla
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation   # must match torch/CUDA
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```
Eval: `experiments/robot/libero/run_libero_eval.py --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-goal --task_suite_name libero_goal`.

### 6.3 `vla-oft` (OpenVLA-OFT, incl. FiLM variant)
```bash
conda create -n vla-oft python=3.10 -y
conda activate vla-oft
git clone https://github.com/moojink/openvla-oft.git && cd openvla-oft
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
# OFT REQUIRES the forked transformers + dlimp — do not use mainline here:
pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla.git"
pip install "flash-attn==2.5.5" --no-build-isolation
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
python -c "import transformers, draccus, diffusers; print('oft imports OK')"
```
Eval config knobs that matter (from the OFT checkpoint card):
`use_l1_regression=True, use_diffusion=False, num_images_in_input=2,
use_proprio=True, center_crop=True, unnorm_key="libero_goal_no_noops"`.
For the FiLM variant set `use_film=True` **and** point at the FiLM-trained
checkpoint; mismatched `use_film`/checkpoint = silent garbage.

### 6.4 `libero-para-metrics` (analysis only, no GPU)
```bash
conda create -n libero-para-metrics python=3.10 -y
conda activate libero-para-metrics
cd LIBERO-Para && pip install -r metrics/requirements.txt
python -m spacy download en_core_web_sm
```

> **Never `pip install` LIBERO-Plus into any of the above** — it shadows the
> `libero` package. If you need LIBERO-Plus, make a 5th env `vla-smolvla-plus`.

---

## 7. Metrics — `analyze/`

Compute all metrics **from raw per-episode logs**, never from screen output.

- **TSR** (per model × suite × condition): successes / episodes. Report mean ±
  std over seeds.
- **ΔTSR**: `TSR(original) − TSR(condition)`, in percentage points.
- **Causal Sensitivity Score (CSS)** for RQ1: normalized behavior change under a
  destructive perturbation.
  `CSS = (TSR_original − TSR_condition) / max(TSR_original, ε)` for
  `condition ∈ {blank, nonsense}`. CSS≈0 ⇒ language ignored; CSS≈1 ⇒ strongly
  causal. Report per condition; do not average blank and nonsense into one number.
- **Follow / Ignore / Fail (RQ1.2)** for `wrong_*` conditions — for each episode
  classify the *achieved* task:
  - **Follow** = the robot completed the (wrong) instructed task.
  - **Ignore** = it completed the original true task (→ **Original-Action-Rate**,
    OAR = fraction Ignore). High OAR under a wrong instruction is the strongest
    evidence the policy is reading pixels, not words.
  - **Fail** = neither. Determine "which task was achieved" from LIBERO's own goal
    predicates (each Goal task ships a `bddl` goal spec — evaluate *all* candidate
    goal predicates for the shared scene against the final state).
- **PRIDE (RQ2)**: use LIBERO-Para's metric as-is via
  `metrics/analyze_results.py --model_path results/<model>`; it computes
  Paraphrase Distance (PD) from keyword (SK) and structural (ST) similarity and
  rewards succeeding on harder paraphrases. Wrap, don't reimplement.
- **Failure-locus (RQ2.4 / RQ1)** — `analyze/locus.py`: for each failed episode,
  from the trajectory + object poses decide:
  - **Planning-level**: gripper never approached / contacted the correct target
    object (went to the wrong object or nowhere). → wrong *what*.
  - **Execution-level**: gripper reached & engaged the correct object but the
    manipulation failed (dropped, mis-grasped, wrong placement). → right *what*,
    bad *how*.
  Use a distance-to-target + contact/grasp-state heuristic; document thresholds in
  the file and sanity-check on 10 hand-labeled episodes before trusting it at scale.

---

## 8. The autonomous run loop — `run/launch.py`

Behavior contract:

1. Read `configs/grid.yaml`. Each grid cell = `(model, suite, condition, seed,
   n_episodes)`.
2. For each cell, compute a deterministic `run_id` = hash of the cell. If
   `results/.../seed<k>/summary.json` already exists **and** its recorded
   `n_episodes` ≥ requested, **skip** (idempotent / resumable).
3. Activate the correct conda env for that model (`envs/<model>.env`). Never run
   two models in one env.
4. Run `run/run_one.py`, which:
   - fixes all seeds (`random`, `numpy`, `torch`, env seed) from the cell,
   - asserts the env reset-state hash is identical to the `original` condition's
     for the same task+seed (proves scene held fixed),
   - streams per-episode records (success bool, achieved-goal predicates,
     trajectory summary, instruction string, wall-clock) to disk as it goes,
   - writes `summary.json` atomically at the end.
5. Append one line to `MANIFEST.json` (atomic write + flock) recording model,
   checkpoint hash, repo git SHA, env lock hash, and counts.
6. On exception: catch, write `error.json` with full traceback + the resolved
   command, mark the cell FAILED, and **continue to the next cell**. One bad cell
   never halts the sweep.
7. After the sweep, run `analyze/*` to regenerate `report/`.

Determinism: same seed ⇒ same episode init. Report every number as mean over
seeds with n_episodes and seed list printed in the table caption.

Suggested minimal grid (SmolVLA full, 7B reduced):
```yaml
seeds: [7, 42, 123]
smolvla:
  suites: [libero_goal]           # + libero_10 for RQ1.4
  conditions: [original, blank, nonsense, wrong_object, wrong_action, wrong_task, repeated, para_object, para_action, para_compositional]
  n_episodes: 20
openvla: { suites: [libero_goal], conditions: [original, blank, wrong_task, para_object, para_action], n_episodes: 10, seeds: [7,42] }
openvla_oft: { ...same reduced... }
openvla_oft_film: { ...same reduced... }
```

---

## 9. Debugging playbook (fix these yourself before asking)

- **Blank window / `GLFW`/`EGL` error on HPC** → you forgot `MUJOCO_GL=egl`.
  Multi-GPU node: set `MUJOCO_EGL_DEVICE_ID` to the visible GPU.
- **`flash-attn` build fails** → CUDA/torch mismatch. Install a prebuilt wheel
  matching your exact torch+CUDA, use `--no-build-isolation`, and make sure
  `nvcc --version` matches the torch CUDA tag. On a login node with no GPU, build
  on a compute node.
- **`transformers` import/attr errors in the OFT env** → you installed mainline
  transformers over the fork. Reinstall the moojink fork; OFT depends on it.
- **`ValueError: unnorm_key ... not found`** → pass the suite-specific key
  (`libero_goal_no_noops`, etc.); it must match the checkpoint's training suite.
- **Two `libero` versions / weird env-name errors** → base LIBERO and LIBERO-Plus
  both claim the `libero` package. Separate envs. Check `pip show libero`.
- **OOM on 7B** → drop `eval.batch_size`/`n_episodes`, keep bf16, try
  `load_in_4bit=True` for OpenVLA. SmolVLA should never OOM on ≥8 GB.
- **Querying GPU memory in this stack** → PyTorch 2.10+cu128 uses
  `torch.cuda.get_device_properties(0).total_memory` (there is **no** `total_mem`).
- **Gated checkpoint 401** → `huggingface-cli login`; accept the model license on
  the HF page first.
- **SmolVLA can't reproduce paper numbers** → known; SmolVLA is pretrained on a
  different embodiment (SO-101), so LIBERO usually needs the LIBERO-finetuned
  checkpoint, not `smolvla_base` zero-shot. Use `HuggingFaceVLA/smolvla_libero`
  (or finetune) and note it.
- **Observation key mismatch** → LeRobot normalization stats are keyed by
  observation name; rename obs keys to what the policy expects or the norm layer
  silently mangles inputs.

If a fix requires changing a cloned repo, patch a copy under `eval_scripts/` (never
edit read-only clones in place) and record the diff in `report/patches/`.

---

## 10. HPC (Northeastern cluster) notes

- Run evals on compute nodes via `sbatch`, never on login nodes. Templates in
  `run/slurm/`. Request 1 GPU (SmolVLA: ~16 GB is plenty; 7B: ≥24 GB comfortable).
- Put `HF_HOME` and all datasets/checkpoints on `$SCRATCH`, not home (quota).
- One SLURM array task per grid cell keeps failures isolated and the sweep
  resumable — a killed job just re-runs its own cells (they're idempotent).
- Log node/GPU model into `MANIFEST.json` for reproducibility.

---

## 11. Reporting — `report/`

Produce, and keep regenerable from `results/`:
- `rq1_causal.csv` + bar chart: CSS(blank), CSS(nonsense) per model; Follow/Ignore/Fail
  stacked bars for `wrong_*`; OAR table. Headline: is CSS≈0 (language ignored)?
- `rq1_scale.csv`: CSS vs params (SmolVLA 0.45B → OFT 7.5B). Does causal reliance
  grow with scale?
- `rq2_paraphrase.csv` + plot: ΔTSR and PRIDE, object-axis vs action-axis, and
  ΔTSR vs PD scatter. Headline number: pp drop under paraphrase (expect large,
  object-dominated per prior work — but report *our* measured values).
- `rq2_locus.csv`: planning vs execution split of failures.
- `report.md`: one paragraph per RQ stating the measured answer, with the table
  and figure inline and the exact grid (seeds, n_episodes) in each caption.

Frame findings against prior work (LIBERO-Para, "Is OpenVLA Truly Robust?",
LangGap) as *related*, but every claim we make must trace to a row in our own
`results/`. Cite; don't borrow their numbers as ours.

---

## 12. Guardrails (non-negotiable)

- No fabricated results. Missing > invented.
- Scene must be provably fixed across conditions (assert the reset-state hash).
- Causal claims only on LIBERO-Goal (and Long), never on scene-confounded suites.
- Every number reproducible: seed, checkpoint hash, repo SHA, env lock recorded.
- Isolated conda env per model, always.
- Stop and summarize for the human if: a model's `original` TSR is near zero
  (setup is broken, not a finding), or an env refuses to build after two distinct
  fix attempts, or GPU budget will be exceeded by the requested grid. Report the
  blocker and the partial results; don't silently drop scope.