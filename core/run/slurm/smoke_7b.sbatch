#!/bin/bash
# core/run/slurm/smoke_7b.sbatch — GPU validation gate for the two rebuilt 7B envs.
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
# d4054 EXCLUDED. Measured 2026-07-31: every rollout process on d4054 SIGABRTs
# (core dump) a couple of episodes after "env ready", while the identical job
# succeeds on d4052 — openvla task0 ran 60 episodes clean on d4052 (job 8853238)
# and job 8853239 aborted on d4054; the OFT grid aborted on ALL TEN tasks there
# (job 8853355). A short smoke test PASSES on d4054, so the node looks healthy and
# only sustained EGL rendering exposes it. HANDOFF 1.5 said excluding d4054 was a
# no-op because it was DRAIN; it is schedulable again, so the exclude is live and
# load-bearing. Drop it only after a long rollout is shown to survive there.
#SBATCH --exclude=d4054
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/7b-smoke-%j.out
#SBATCH --error=logs/slurm/7b-smoke-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
# index into CUDA_VISIBLE_DEVICES — robosuite asserts this (see p2_rq1_*.sbatch)
export MUJOCO_EGL_DEVICE_ID=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
export LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla
export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

E=${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}
echo "=== 7b smoke on $(hostname|cut -d. -f1) $(date) ==="
nvidia-smi -L

rc=0
echo "########## preflight (model-free render gate), per env ##########"
$E/vla-openvla/bin/python core/run/preflight_libero.py || { rc=1; echo "!!! vla-openvla preflight FAILED"; }
$E/vla-oft/bin/python     core/run/preflight_libero.py || { rc=1; echo "!!! vla-oft preflight FAILED"; }

echo "########## smoke_openvla ##########"
$E/vla-openvla/bin/python core/run/smoke_openvla.py || { rc=1; echo "!!! smoke_openvla FAILED"; }

echo "########## smoke_oft ##########"
$E/vla-oft/bin/python core/run/smoke_oft.py || { rc=1; echo "!!! smoke_oft FAILED"; }

echo "=== 7b smoke done rc=$rc $(date) ==="
exit $rc
