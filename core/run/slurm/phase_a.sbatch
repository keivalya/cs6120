#!/bin/bash
#SBATCH --job-name=vla-phase-a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --exclude=d4054
#SBATCH --output=logs/slurm/vla-phase-a-%j.out
#SBATCH --error=logs/slurm/vla-phase-a-%j.out

set -euo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false

bash core/run/run_fresh_node.sh A
