#!/bin/bash
#SBATCH --job-name=vla-openvla-repair
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --exclude=d4054
#SBATCH --output=logs/slurm/vla-openvla-repair-%j.out
#SBATCH --error=logs/slurm/vla-openvla-repair-%j.out

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

OV=/home/pandya.kei/.conda/envs/vla-openvla/bin/python
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"

echo "=== OpenVLA Repair Start: $(date) ==="
env $OVENV "$OV" core/run/eval_task_openvla.py --task_id 1 --conditions original --seed 7 --n_episodes 2
echo "=== OpenVLA Repair Done: $(date) ==="
