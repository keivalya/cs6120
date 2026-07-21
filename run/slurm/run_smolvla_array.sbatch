#!/bin/bash
#SBATCH --job-name=smolvla-arr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=d4054
#SBATCH --array=2-9
#SBATCH --output=vla-smolvla-arr-%A_%a.out
#SBATCH --error=vla-smolvla-arr-%A_%a.out

set -euo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python
TID=$SLURM_ARRAY_TASK_ID
SEEDS_CORE="7 42"
LOG=/scratch/pandya.kei/smolvla_run_task_${TID}.log

echo "=== SmolVLA Array Task ${TID} Start: $(date) ===" | tee "$LOG"

# 1) Preflight
$SM run/preflight.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "PREFLIGHT FAILED — GPU channel degraded or wrong node class. Aborting." | tee -a "$LOG"
  exit 1
fi

# 2) Phase A — SmolVLA RQ1 gaps
for s in $SEEDS_CORE; do
  "$SM" run/eval_task.py --task_id "$TID" --conditions original,blank,nonsense,repeated,wrong_object,wrong_action,wrong_task --seed $s --n_episodes 2 --obs_hw 256 2>&1 \
    | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort|paraphrase mode" | grep -avE "%\|" | tee -a "$LOG"
done

# 3) Phase B — SmolVLA RQ2 paraphrase
for ax in para_object para_action para_compositional; do
  "$SM" run/eval_task.py --task_id "$TID" --paraphrase_axis $ax --seed 7 --n_episodes 1 --obs_hw 256 2>&1 \
    | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort|paraphrase mode" | grep -avE "%\|" | tee -a "$LOG"
done

echo "=== SmolVLA Array Task ${TID} Done: $(date) ===" | tee -a "$LOG"
