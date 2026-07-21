#!/bin/bash
#SBATCH --job-name=smol-comp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=d4054
#SBATCH --array=1,2,3,5,7,8,9
#SBATCH --output=vla-smolvla-comp-%A_%a.out
#SBATCH --error=vla-smolvla-comp-%A_%a.out

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
LOG=/scratch/pandya.kei/smolvla_comp_task_${TID}.log

echo "=== SmolVLA Compositional Task ${TID} Start: $(date) ===" | tee "$LOG"

"$SM" run/eval_task.py --task_id "$TID" --paraphrase_axis para_compositional --seed 7 --n_episodes 1 --obs_hw 256 2>&1 \
  | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort|paraphrase mode|ep_global" | grep -avE "%\|" | tee -a "$LOG"

echo "=== SmolVLA Compositional Task ${TID} Done: $(date) ===" | tee -a "$LOG"
