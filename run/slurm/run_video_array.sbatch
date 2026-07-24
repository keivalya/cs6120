#!/bin/bash
#SBATCH --job-name=vla-video
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --exclude=d4054
#SBATCH --array=0-9
#SBATCH --output=vla-video-%A_%a.out
#SBATCH --error=vla-video-%A_%a.out

set -euo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python
TID=$SLURM_ARRAY_TASK_ID

echo "=== Generating Videos for Task ${TID}: $(date) ==="
"$SM" run/generate_videos.py --task_id "$TID" --num_videos 5 "$@"
echo "=== Done Task ${TID}: $(date) ==="
