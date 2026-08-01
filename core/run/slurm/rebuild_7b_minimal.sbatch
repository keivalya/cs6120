#!/bin/bash
# core/run/slurm/rebuild_7b_minimal.sbatch — rebuild BOTH 7B envs from scratch.
#
# CPU partition on purpose: this is conda create + pip, no CUDA at all, and the gpu
# QOS only gives 4 running jobs / 4 GPUs. Building here keeps every GPU slot on
# production evals. (conda create OOMs on the LOGIN node, hence a job at all.)
# Network + git over HTTPS verified available on `short` (c0593).
#
# Validate afterwards on a GPU node:  sbatch core/run/slurm/smoke_7b.sbatch
#
#SBATCH --job-name=7b-rebuild
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/7b-rebuild-%j.out
#SBATCH --error=logs/slurm/7b-rebuild-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate base

export PIP_NO_INPUT=1
# Keep the wheel cache off $HOME: the per-user quota fills from ~/.cache.
export PIP_CACHE_DIR=/scratch/pandya.kei/pip_cache

rc=0
echo "########## vla-openvla ##########"
bash core/run/rebuild_openvla_minimal.sh || { rc=1; echo "!!! vla-openvla build FAILED"; }

echo "########## vla-oft ##########"
bash core/run/rebuild_oft_minimal.sh || { rc=1; echo "!!! vla-oft build FAILED"; }

echo "=== 7b-rebuild done rc=$rc $(date) ==="
[ $rc -eq 0 ] && echo "NEXT: sbatch core/run/slurm/smoke_7b.sbatch (GPU validation)"
exit $rc
