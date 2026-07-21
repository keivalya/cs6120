#!/bin/bash
#SBATCH --job-name=vla-smolvla
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=d4054
#SBATCH --output=vla-smolvla-%j.out
#SBATCH --error=vla-smolvla-%j.out

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
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
SEEDS_CORE="7 42"
LOG="/scratch/pandya.kei/smolvla_run_${SLURM_JOB_ID:-local}.log"

echo "=== SmolVLA Sweep Start: $(date) ===" | tee "$LOG"

# 1) Preflight
$SM run/preflight.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "PREFLIGHT FAILED — GPU channel degraded or wrong node class. Aborting." | tee -a "$LOG"
  exit 1
fi

runner () {
  for tid in $TASKS; do
    "$SM" run/eval_task.py --task_id "$tid" "$@" 2>&1 \
      | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort|paraphrase mode" | grep -avE "%\|" | tee -a "$LOG"
  done
}

echo "### Phase A — SmolVLA RQ1 gaps ###" | tee -a "$LOG"
for s in $SEEDS_CORE; do
  runner --conditions original,blank,nonsense,repeated,wrong_object,wrong_action,wrong_task --seed $s --n_episodes 2 --obs_hw 256
done

echo "### Phase B — SmolVLA RQ2 paraphrase ###" | tee -a "$LOG"
for ax in para_object para_action para_compositional; do
  runner --paraphrase_axis $ax --seed 7 --n_episodes 1 --obs_hw 256
done

echo "=== SmolVLA Sweep Done: $(date) ===" | tee -a "$LOG"
