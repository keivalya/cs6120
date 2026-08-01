#!/bin/bash
# core/run/slurm/p2_aggregate.sbatch — P2 aggregation + CSV/plot refresh (RQ1).
#
# Run this ONCE, only AFTER all p2_rq1_{smolvla,openvla,oft} array elements have
# finished. It re-aggregates the (now higher-n) per-task jsonl into summary.json,
# appends provenance to MANIFEST.json, and regenerates the RQ1 CSVs and plots.
# rq1_causal.csv already carries TSR_std_over_seeds, so seed CIs come for free.
#
# Pure file aggregation — no EGL, no rollouts, NO GPU. It runs on the CPU
# partition on purpose: the gpu QOS only allows 4 running jobs / 4 GPUs, and this
# job used to reserve one of them for nothing.
#
# It also used to run everything with vla-openvla/bin/python, which is an EMPTY
# env (no numpy/matplotlib/yaml) — every step failed into $LOG while the job
# still exited COMPLETED, which is why the earlier p2-agg jobs produced nothing.
# vla-smolvla has numpy/matplotlib/scipy/yaml/seaborn.
#
# Submit:   sbatch core/run/slurm/p2_aggregate.sbatch
#
#SBATCH --job-name=p2-agg
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/p2-agg-%j.out
#SBATCH --error=logs/slurm/p2-agg-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false

OV=${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}/vla-smolvla/bin/python
LOG=$SCRATCH/p2_aggregate.log

echo "=== P2 aggregate $(date) ===" | tee "$LOG"

# Fail loudly instead of exiting COMPLETED with an empty result.
$OV -c "import numpy, matplotlib, yaml, seaborn" \
  || { echo "FATAL: analysis env is missing deps (numpy/matplotlib/yaml/seaborn)" | tee -a "$LOG"; exit 1; }
rc=0

step() {  # step <label> <cmd...> — record failures instead of swallowing them
  echo "### $1 ###" | tee -a "$LOG"
  shift
  "$@" >> "$LOG" 2>&1 || { rc=1; echo "  FAILED: $* (see $LOG)" | tee -a "$LOG"; }
}

for m in smolvla openvla openvla_oft; do
  step "aggregate: $m"    $OV core/run/aggregate.py        --model "$m" --suite libero_goal
  step "rq1 csv: $m"      $OV core/analyze/make_rq1_csv.py --model "$m" --suite libero_goal
done

step "rq1 scale"    $OV core/analyze/make_rq1_scale.py
step "rq1 plots"    $OV core/analyze/plot_rq1.py
step "rq3 kinematic divergence" $OV core/analyze/kinematic_divergence.py

echo "=== P2 aggregate DONE $(date) — check paper/rq1_causal.csv, rq1_scale.csv, rq1_*.png ===" | tee -a "$LOG"
echo "Sanity: episode counts should now be ~10 per task per seed (all 3 models), seeds 7;42." | tee -a "$LOG"
exit $rc
