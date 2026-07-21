#!/bin/bash
#SBATCH --job-name=vla-oft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=d4054
#SBATCH --output=vla-oft-%j.out
#SBATCH --error=vla-oft-%j.out

set -euo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"

SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python
OFT=/home/pandya.kei/.conda/envs/vla-oft/bin/python
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"
TASKS="0 1 2 3 4 5 6 7 8 9"
SEEDS_CORE="7 42"
LOG=/scratch/pandya.kei/oft_run.log

echo "=== OFT Sweep Start: $(date) ===" | tee "$LOG"

# 1) Preflight using SmolVLA to verify render health
$SM run/preflight.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "PREFLIGHT FAILED — GPU channel degraded. Aborting." | tee -a "$LOG"
  exit 1
fi

runner () {
  for tid in $TASKS; do
    env $OVENV "$OFT" run/eval_task_oft.py --task_id "$tid" "$@" 2>&1 \
      | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort|paraphrase mode" | grep -avE "%\|" | tee -a "$LOG"
  done
}

echo "### Phase A — OFT RQ1 gaps ###" | tee -a "$LOG"
for s in $SEEDS_CORE; do
  runner --conditions wrong_object,wrong_action,wrong_task,repeated --seed $s --n_episodes 2
  if [ "$s" = "42" ]; then
    runner --conditions original,blank,nonsense --seed $s --n_episodes 2
  fi
done

echo "### Phase B — OFT RQ2 paraphrase ###" | tee -a "$LOG"
for ax in para_object para_action para_compositional; do
  runner --paraphrase_axis $ax --max_per_axis 15 --seed 7 --n_episodes 1
done

echo "=== OFT Sweep Done: $(date) ===" | tee -a "$LOG"
