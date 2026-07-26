#!/bin/bash
# run/rebuild_envs.sh — rebuild the vla-* conda envs from the pinned pip locks.
#
# WHY: the envs live under /scratch (CONDA_ENVS below), and /scratch is purged on
# a schedule, which gutted them (python stub remained, packages gone -> "No module
# named numpy"). The envs/*.lock.txt files are full pip freezes that pin even the
# git/source installs (LeRobot, OpenVLA, OpenVLA-OFT, LIBERO, dlimp, the moojink
# transformers fork), so the envs are fully reproducible.
#
# Run this ON A NODE with conda + build tools + network/git access (a login node
# or an interactive GPU alloc). It is NOT a batch job. Rebuilding is network-heavy
# (git clones + torch wheels) and takes a while per env.
#
# Usage:
#   bash run/rebuild_envs.sh                 # rebuild all three
#   bash run/rebuild_envs.sh vla-smolvla     # rebuild just one
#   CONDA_ENVS=/work/pandya.kei/vla-envs bash run/rebuild_envs.sh   # persistent location
#
# TIP: /scratch purges will keep eating these. Consider pointing CONDA_ENVS at a
# persistent filesystem (e.g. /work or /home if quota allows) and setting the same
# CONDA_ENVS when you submit the P2 sbatch jobs so the paths match.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONDA_ENVS="${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}"
mkdir -p "$CONDA_ENVS"

# env name -> python version (smolvla needs 3.12 for LeRobot 0.6.1; the OpenVLA
# family uses 3.10 per CLAUDE.md §6.1). Adjust here if your build differs.
declare -A PYVER=( [vla-smolvla]=3.12 [vla-openvla]=3.10 [vla-oft]=3.10 )

TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then TARGETS=(vla-smolvla vla-openvla vla-oft); fi

command -v conda >/dev/null || { echo "FATAL: conda not on PATH. module load anaconda / source conda first."; exit 1; }

for ENV in "${TARGETS[@]}"; do
  LOCK="$REPO/envs/${ENV}.lock.txt"
  PREFIX="$CONDA_ENVS/$ENV"
  PY="${PYVER[$ENV]:-3.10}"
  [ -f "$LOCK" ] || { echo "SKIP $ENV: no lock at $LOCK"; continue; }

  echo "=============================================================="
  echo "Rebuilding $ENV  (python=$PY)  ->  $PREFIX"
  echo "=============================================================="
  # Fresh prefix (remove any gutted skeleton first).
  rm -rf "$PREFIX"
  conda create -y -p "$PREFIX" "python=$PY" || { echo "FATAL: conda create failed for $ENV"; continue; }
  PIP="$PREFIX/bin/pip"

  # Build-time prerequisites (SmolVLA: cmake<4 for egl_probe, num2words for SmolVLM).
  "$PIP" install --upgrade pip wheel setuptools
  if [ "$ENV" = "vla-smolvla" ]; then
    "$PIP" install "cmake<4" num2words
  fi

  # The lock is a pip freeze incl. git+ / editable installs; -r reconstructs all.
  echo "[pip] installing from $LOCK (this clones git deps and can take a while)..."
  "$PIP" install -r "$LOCK" || { echo "WARN: some packages failed for $ENV — inspect above; env may be partially built."; }

  # Smoke test the core imports the runners need.
  "$PREFIX/bin/python" - <<'PY' || echo "WARN: smoke import failed — env not ready."
import numpy, torch
print("  OK: numpy", numpy.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY
done

echo
echo "Done. Verify each env, then resubmit the P2 jobs. If you built under a"
echo "non-default CONDA_ENVS, pass the SAME CONDA_ENVS to sbatch, e.g.:"
echo "  sbatch --export=ALL,CONDA_ENVS=$CONDA_ENVS,SEED_ONLY=7 run/slurm/p2_rq1_smolvla.sbatch"
