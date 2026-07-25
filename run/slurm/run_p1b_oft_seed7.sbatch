#!/bin/bash
#SBATCH --job-name=vla-oft-p1b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=d4054
#SBATCH --output=vla-oft-p1b-%j.out
#SBATCH --error=vla-oft-p1b-%j.out

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

SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python
OFT=/home/pandya.kei/.conda/envs/vla-oft/bin/python
OV=/home/pandya.kei/.conda/envs/vla-openvla/bin/python
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"
TASKS="0 1 2 3 4 5 6 7 8 9"
LOG=/scratch/pandya.kei/oft_p1b_run.log

echo "=== OFT P1b Fix Start: $(date) ===" | tee "$LOG"

# 1) Preflight
$SM run/preflight.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "PREFLIGHT FAILED — GPU channel degraded. Aborting." | tee -a "$LOG"
  exit 1
fi

# 2) Delete mislabeled original/seed7 directory
echo "Removing old mislabeled OFT original/seed7 results..." | tee -a "$LOG"
rm -rf results/openvla_oft/libero_goal/original/seed7

# 3) Re-run OpenVLA-OFT original seed 7
echo "### Re-running OpenVLA-OFT libero_goal original seed 7 ###" | tee -a "$LOG"
for tid in $TASKS; do
  env $OVENV "$OFT" run/eval_task_oft.py --task_id "$tid" --conditions original --seed 7 --n_episodes 2 2>&1 \
    | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort" | grep -avE "%\|" | tee -a "$LOG"
done

# 4) Re-aggregate and refresh CSVs
echo "### Re-aggregating and refreshing CSVs ###" | tee -a "$LOG"
$OV run/aggregate.py --model openvla_oft --suite libero_goal >> "$LOG" 2>&1
$OV analyze/make_rq1_csv.py --model openvla_oft --suite libero_goal >> "$LOG" 2>&1
$OV analyze/make_rq1_scale.py >> "$LOG" 2>&1
$OV analyze/plot_rq1.py >> "$LOG" 2>&1

echo "=== OFT P1b Fix Done: $(date) ===" | tee -a "$LOG"
