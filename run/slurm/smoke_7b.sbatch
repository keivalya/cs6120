#!/bin/bash
# run/slurm/smoke_7b.sbatch — GPU validation gate for the two rebuilt 7B envs.
#
# Runs on gpu-short, whose QOS (2 jobs / 2 GPUs) is INDEPENDENT of the gpu QOS, so
# this costs none of the 4 production slots. Typed GRES because both models load
# bfloat16 and sm_70 (v100) has no native bf16.
#
# Nothing 7B may go to production until this prints both SMOKE OK lines.
#
#SBATCH --job-name=7b-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=7b-smoke-%j.out
#SBATCH --error=7b-smoke-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$(echo "${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}" | cut -d, -f1)
export LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

E=${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}
echo "=== 7b smoke on $(hostname|cut -d. -f1) $(date) ==="
nvidia-smi -L

rc=0
echo "########## preflight (model-free render gate), per env ##########"
$E/vla-openvla/bin/python run/preflight_libero.py || { rc=1; echo "!!! vla-openvla preflight FAILED"; }
$E/vla-oft/bin/python     run/preflight_libero.py || { rc=1; echo "!!! vla-oft preflight FAILED"; }

echo "########## smoke_openvla ##########"
$E/vla-openvla/bin/python run/smoke_openvla.py || { rc=1; echo "!!! smoke_openvla FAILED"; }

echo "########## smoke_oft ##########"
$E/vla-oft/bin/python run/smoke_oft.py || { rc=1; echo "!!! smoke_oft FAILED"; }

echo "=== 7b smoke done rc=$rc $(date) ==="
exit $rc
