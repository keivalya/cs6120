#!/bin/bash
# run/run_fresh_node.sh — turnkey RQ2 + RQ1-gap sweep for a FRESH GPU allocation.
#
# WHY FRESH: running two EGL render processes at once corrupts the GPU render
# channel (see memory concurrent-egl-corrupts-channel + gate3-egl-fork-blocker);
# heavy CUDA+EGL then SIGABRTs (exit 134) even run alone. A fresh node resets it.
# THIS SCRIPT RUNS STRICTLY ONE EGL PROCESS AT A TIME — never parallelize it.
#
# Idempotent-ish: each per-task run overwrites its own task<tid>.jsonl; safe to re-run.
# Usage:  bash run/run_fresh_node.sh [A|B|ALL]   (default ALL)
#   A = RQ1 gaps (goal-logged wrong_* re-runs, OFT RQ1.2, seed 42)
#   B = RQ2 paraphrase sweeps (SmolVLA full, 7B sampled)
set -u
PHASE="${1:-ALL}"
CS=/home/pandya.kei/CS6120
cd "$CS"
export HF_HOME=/scratch/pandya.kei/hf_home HF_HUB_CACHE=/scratch/pandya.kei/hf_home/hub
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python
OV=/home/pandya.kei/.conda/envs/vla-openvla/bin/python
OFT=/home/pandya.kei/.conda/envs/vla-oft/bin/python
TASKS="0 1 2 3 4 5 6 7 8 9"
SEEDS_CORE="7 42"          # add seed 42 everywhere (§7 mean±std)
LOG=/scratch/pandya.kei/fresh_node.log
echo "=== run_fresh_node PHASE=$PHASE node=$(hostname|cut -d. -f1) $(date) ===" | tee "$LOG"

# preflight: heavy CUDA+EGL must survive, else the channel is degraded -> STOP.
$OV run/preflight.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then echo "PREFLIGHT FAILED — request a fresh node (channel degraded)." | tee -a "$LOG"; exit 1; fi

# run one (model, task) eval process; $1=python $2=runner $3=extra-env $4..=args
runner () { local PY="$1" R="$2" EX="$3"; shift 3
  for tid in $TASKS; do
    env $EX "$PY" "$R" --task_id "$tid" "$@" 2>&1 \
      | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort|paraphrase mode" | grep -avE "%\|" | tee -a "$LOG"
  done
}
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"

if [ "$PHASE" = "A" ] || [ "$PHASE" = "ALL" ]; then
  echo "### PHASE A — RQ1 gaps (goal-logged) ###" | tee -a "$LOG"
  for s in $SEEDS_CORE; do
    # wrong_* re-runs with achieved_task_ids for EXACT Follow/Ignore/Fail (all 3 models)
    runner "$OV"  run/eval_task_openvla.py "$OVENV" --conditions wrong_object,wrong_action,wrong_task --seed $s --n_episodes 2
    runner "$OFT" run/eval_task_oft.py     "$OVENV" --conditions wrong_object,wrong_action,wrong_task,repeated --seed $s --n_episodes 2
    # smolvla: seed 42 non-wrong (seed 7 already present) + wrong_* both seeds (goal-logged)
    runner "$SM"  run/eval_task.py         "" --conditions original,blank,nonsense,repeated,wrong_object,wrong_action,wrong_task --seed $s --n_episodes 2 --obs_hw 256
    # oft: seed 42 core causal (seed 7 present) so mean±std
    [ "$s" = "42" ] && runner "$OFT" run/eval_task_oft.py "$OVENV" --conditions original,blank,nonsense --seed $s --n_episodes 2
  done
fi

if [ "$PHASE" = "B" ] || [ "$PHASE" = "ALL" ]; then
  echo "### PHASE B — RQ2 paraphrase ###" | tee -a "$LOG"
  for ax in para_object para_action para_compositional; do
    # SmolVLA: full paraphrase set (cheap), 1 episode/paraphrase
    runner "$SM"  run/eval_task.py         "" --paraphrase_axis $ax --seed 7 --n_episodes 1 --obs_hw 256
    # 7B: stratified sample (~15/task/axis), 1 episode/paraphrase
    runner "$OV"  run/eval_task_openvla.py "$OVENV" --paraphrase_axis $ax --max_per_axis 15 --seed 7 --n_episodes 1
    runner "$OFT" run/eval_task_oft.py     "$OVENV" --paraphrase_axis $ax --max_per_axis 15 --seed 7 --n_episodes 1
  done
fi

echo "### AGGREGATE + ANALYZE ###" | tee -a "$LOG"
for m in smolvla openvla openvla_oft; do
  $OV run/aggregate.py --model $m --suite libero_goal >> "$LOG" 2>&1
  $OV analyze/make_rq1_csv.py --model $m --suite libero_goal >> "$LOG" 2>&1
done
$OV analyze/make_rq1_scale.py >> "$LOG" 2>&1
$OV analyze/plot_rq1.py       >> "$LOG" 2>&1
$OV analyze/make_rq2_csv.py --suite libero_goal >> "$LOG" 2>&1
$OV analyze/plot_rq2.py       >> "$LOG" 2>&1
echo "=== DONE $(date) — see report/rq1_*.csv, rq2_*.csv, *.png ===" | tee -a "$LOG"
