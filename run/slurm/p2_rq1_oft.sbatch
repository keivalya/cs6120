#!/bin/bash
# run/slurm/p2_rq1_oft.sbatch — P2 statistical power, OpenVLA-OFT-7.5B RQ1 causal grid.
#
# Same job-array model as the other P2 scripts (one process per seed x task on its
# own GPU). Scales RQ1 to 10 episodes/task across seeds 7,42. Also backfills the
# OFT seed42 original checkpoint_hash (null today) since eval_task_oft.py now
# records it. Resumable: re-submit after a timeout to continue.
#
# Submit:   sbatch run/slurm/p2_rq1_oft.sbatch
# Aggregate (after ALL arrays finish):  sbatch run/slurm/p2_aggregate.sbatch
#
#SBATCH --job-name=p2-rq1-oft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=d4054
#SBATCH --array=0-19
#SBATCH --output=p2-rq1-oft-%A_%a.out
#SBATCH --error=p2-rq1-oft-%A_%a.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export TOKENIZERS_PARALLELISM=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"

OFT=/home/pandya.kei/.conda/envs/vla-oft/bin/python
SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python   # preflight only
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"

# --- Tunables -----------------------------------------------------------------
SEEDS=(7 42)
N_EPISODES="${N_EPISODES:-10}"
CONDITIONS="${CONDITIONS:-original,blank,nonsense,wrong_object,wrong_action,wrong_task,repeated}"
# ------------------------------------------------------------------------------

IDX=${SLURM_ARRAY_TASK_ID:?must run as a job array}
SEED=${SEEDS[$((IDX / 10))]}
TID=$((IDX % 10))
LOG=$SCRATCH/p2_rq1_oft_s${SEED}_t${TID}.log

echo "=== P2 OFT RQ1 | seed=$SEED task=$TID n_ep=$N_EPISODES | $(hostname|cut -d. -f1) $(date) ===" | tee "$LOG"

$SM run/preflight.py >> "$LOG" 2>&1 || { echo "PREFLIGHT FAILED — channel degraded, need fresh node." | tee -a "$LOG"; exit 1; }

env $OVENV "$OFT" run/eval_task_oft.py --model openvla_oft --suite libero_goal \
    --task_id "$TID" --conditions "$CONDITIONS" \
    --seed "$SEED" --n_episodes "$N_EPISODES" 2>&1 \
  | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort" | grep -avE "%\|" | tee -a "$LOG"

echo "=== done seed=$SEED task=$TID $(date) ===" | tee -a "$LOG"
