#!/bin/bash
#SBATCH --job-name=vla-rq11
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out
#
# GATE 3 RQ1.1 causal sweep — SmolVLA on LIBERO-Goal (CLAUDE.md §8, §10).
# Turnkey + resumable: idempotent per-task skip means a re-queued job continues.
# One process per task (single EGL context) avoids the multi-context render abort.

set -euo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

source "$(conda info --base)/etc/profile.d/conda.sh"
RUN="conda run -n vla-smolvla --no-capture-output python"

# 1) fail-fast GPU render health check — bail if the GPU channel is degraded
if ! $RUN core/run/preflight.py --obs_hw 360; then
  echo "PREFLIGHT FAILED — GPU render unhealthy; request a fresh node. Aborting." >&2
  exit 1
fi

# 2) RQ1.1 sweep (10 tasks x {original,blank,nonsense}, seed 7, n_episodes=2) + aggregate
$RUN core/run/launch.py --model smolvla --suite libero_goal \
  --conditions original,blank,nonsense \
  --task_ids 0,1,2,3,4,5,6,7,8,9 --seed 7 --n_episodes 2 --obs_hw 360 --aggregate

# 3) metrics + report
$RUN core/analyze/make_rq1_csv.py --model smolvla --suite libero_goal
echo "DONE — see paper/rq1_causal.csv and paper/scene_fixed_check.json"
