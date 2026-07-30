#!/bin/bash
# run/slurm/p2_rq1_smolvla.sbatch — P2 statistical power, SmolVLA RQ1 causal grid.
#
# NON-ARRAY design (to respect a tight QOSMaxSubmitJobPerUser limit): this is ONE
# job that loops seeds x tasks INTERNALLY, running strictly one EGL process at a
# time (the proven run_fresh_node.sh / P1b pattern). Scales RQ1 from 2 eps/task to
# the grid.yaml design (20/task) across seeds 7,42, all 7 causal conditions.
#
# Resumable: each per-task runner tops up task<tid>.jsonl to --n_episodes and skips
# finished ones, so if this job hits the wall-time limit just re-submit it.
#
# Optionally restrict to one seed to shorten wall time / parallelize across jobs:
#   sbatch --export=ALL,SEED_ONLY=7  run/slurm/p2_rq1_smolvla.sbatch
#   sbatch --export=ALL,SEED_ONLY=42 run/slurm/p2_rq1_smolvla.sbatch
#
#SBATCH --job-name=p2-rq1-sm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --exclude=d4054
#SBATCH --output=p2-rq1-sm-%j.out
#SBATCH --error=p2-rq1-sm-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"

SM=${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}/vla-smolvla/bin/python

# --- Tunables -----------------------------------------------------------------
N_EPISODES="${N_EPISODES:-20}"
CONDITIONS="${CONDITIONS:-original,blank,nonsense,wrong_object,wrong_action,wrong_task,repeated}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
if [ -n "${SEED_ONLY:-}" ]; then SEEDS="$SEED_ONLY"; else SEEDS="7 42"; fi
# ------------------------------------------------------------------------------

LOG=$SCRATCH/p2_rq1_smolvla_${SLURM_JOB_ID:-manual}.log
echo "=== P2 SmolVLA RQ1 | seeds=[$SEEDS] tasks=[$TASKS] n_ep=$N_EPISODES | $(hostname|cut -d. -f1) $(date) ===" | tee "$LOG"

# Preflight ONCE (heavy CUDA+EGL must survive or the render channel is degraded).
$SM run/preflight.py >> "$LOG" 2>&1 || {
  echo "PREFLIGHT FAILED on $(hostname|cut -d. -f1) — real error below (do NOT assume a bad node):" | tee -a "$LOG"
  tail -30 "$LOG"
  exit 1
}

for SEED in $SEEDS; do
  for TID in $TASKS; do
    echo "### smolvla seed=$SEED task=$TID ###" | tee -a "$LOG"
    $SM run/eval_task.py --model smolvla --suite libero_goal \
        --task_id "$TID" --conditions "$CONDITIONS" \
        --seed "$SEED" --n_episodes "$N_EPISODES" --obs_hw 256 2>&1 \
      | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort" | grep -avE "%\|" | tee -a "$LOG"
  done
done

echo "=== P2 SmolVLA RQ1 DONE $(date) ===" | tee -a "$LOG"
