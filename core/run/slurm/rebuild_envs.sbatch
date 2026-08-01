#!/bin/bash
# core/run/slurm/rebuild_envs.sbatch — rebuild the vla-* conda envs INSIDE an allocation.
#
# WHY A JOB: running the rebuild on the login node gets OOM-killed
# ("conda create ... Killed") because login nodes cap per-user memory. This
# submits the rebuild to a compute node with real RAM. It needs network/git
# access on the compute node (true on most clusters; if your compute nodes are
# air-gapped, tell your admins or build via a proxy).
#
# Usage:
#   sbatch core/run/slurm/rebuild_envs.sbatch                 # all three envs
#   sbatch core/run/slurm/rebuild_envs.sbatch vla-smolvla     # just one
# Optional persistent location (if you ever get one with quota):
#   sbatch --export=ALL,CONDA_ENVS=/some/persistent/vla-envs core/run/slurm/rebuild_envs.sbatch
#
#SBATCH --job-name=vla-rebuild
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=d4054
#SBATCH --output=logs/slurm/vla-rebuild-%j.out
#SBATCH --error=logs/slurm/vla-rebuild-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

# Make conda available inside the job (login-shell init does not run in batch).
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true

echo "=== vla-rebuild $(date) on $(hostname) ==="
echo "conda: $(command -v conda || echo MISSING)"
echo "targets: ${*:-all three}"

bash core/run/rebuild_envs.sh "$@"

echo "=== vla-rebuild DONE $(date) ==="
echo "Verify: <CONDA_ENVS>/vla-*/bin/python -c 'import torch;print(torch.__version__)'"
