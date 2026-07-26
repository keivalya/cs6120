#!/bin/bash
# run/slurm/p2_aggregate.sbatch — P2 aggregation + CSV/plot refresh (RQ1).
#
# Run this ONCE, only AFTER all p2_rq1_{smolvla,openvla,oft} array elements have
# finished. It re-aggregates the (now higher-n) per-task jsonl into summary.json,
# appends provenance to MANIFEST.json, and regenerates the RQ1 CSVs and plots.
# rq1_causal.csv already carries TSR_std_over_seeds, so seed CIs come for free.
#
# No EGL/rollouts here (pure file aggregation), but we keep the gpu env pythons
# for dependency parity; GPU is not actually required.
#
# Submit:   sbatch run/slurm/p2_aggregate.sbatch
#
#SBATCH --job-name=p2-agg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=d4054
#SBATCH --output=p2-agg-%j.out
#SBATCH --error=p2-agg-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false

OV=/home/pandya.kei/.conda/envs/vla-openvla/bin/python
LOG=$SCRATCH/p2_aggregate.log

echo "=== P2 aggregate $(date) ===" | tee "$LOG"

for m in smolvla openvla openvla_oft; do
  echo "### aggregate + rq1 csv: $m ###" | tee -a "$LOG"
  $OV run/aggregate.py     --model "$m" --suite libero_goal >> "$LOG" 2>&1
  $OV analyze/make_rq1_csv.py --model "$m" --suite libero_goal >> "$LOG" 2>&1
done

$OV analyze/make_rq1_scale.py >> "$LOG" 2>&1
$OV analyze/plot_rq1.py       >> "$LOG" 2>&1

echo "=== P2 aggregate DONE $(date) — check report/rq1_causal.csv, rq1_scale.csv, rq1_*.png ===" | tee -a "$LOG"
echo "Sanity: episode counts should now be ~20 (SmolVLA) / ~10 (7B) per task, seeds 7;42." | tee -a "$LOG"
