#!/bin/bash
#SBATCH --job-name=vla-aggregate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/vla-aggregate-%j.out
#SBATCH --error=logs/slurm/vla-aggregate-%j.out

set -euo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

OV=/home/pandya.kei/.conda/envs/vla-openvla/bin/python

echo "=== Aggregating results ==="
$OV core/run/aggregate.py --model smolvla --suite libero_goal
$OV core/run/aggregate.py --model openvla --suite libero_goal
$OV core/run/aggregate.py --model openvla_oft --suite libero_goal

echo "=== Generating CSVs and Plots ==="
$OV core/analyze/make_rq1_csv.py --model smolvla --suite libero_goal
$OV core/analyze/make_rq1_csv.py --model openvla --suite libero_goal
$OV core/analyze/make_rq1_csv.py --model openvla_oft --suite libero_goal
$OV core/analyze/make_rq1_scale.py
$OV core/analyze/plot_rq1.py
SM_PY=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python
$SM_PY core/analyze/make_rq2_csv.py --suite libero_goal
$SM_PY core/analyze/plot_rq2.py

echo "=== DONE ==="
