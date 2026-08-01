#!/bin/bash
# core/run/slurm/rebuild_smolvla_minimal.sbatch — build the minimal SmolVLA env in a job.
# (Login node OOM-kills conda create, so this runs on a compute node w/ real RAM.)
# Needs network/git access on the compute node.
#
# Usage:  sbatch core/run/slurm/rebuild_smolvla_minimal.sbatch
#
#SBATCH --job-name=sm-rebuild
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=d4054
#SBATCH --output=logs/slurm/sm-rebuild-%j.out
#SBATCH --error=logs/slurm/sm-rebuild-%j.out

set -uo pipefail
cd /home/pandya.kei/CS6120

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true

echo "=== sm-rebuild (minimal) $(date) on $(hostname) | conda: $(command -v conda || echo MISSING) ==="
bash core/run/rebuild_smolvla_minimal.sh
echo "=== sm-rebuild DONE $(date) ==="
