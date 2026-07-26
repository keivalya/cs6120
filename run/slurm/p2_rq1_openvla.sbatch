#!/bin/bash
# run/slurm/p2_rq1_openvla.sbatch — P2 statistical power, OpenVLA-7B RQ1 causal grid.
#
# NON-ARRAY design (respects a tight QOSMaxSubmitJobPerUser limit): ONE job that
# loops seeds x tasks internally, one EGL process at a time. Scales RQ1 to 10
# episodes/task (grid.yaml 7B budget) across seeds 7,42. 7B is much slower than
# SmolVLA, so if this hits the wall-time limit just re-submit (runners resume).
#
# To halve wall time / parallelize, submit one seed per job:
#   sbatch --export=ALL,SEED_ONLY=7  run/slurm/p2_rq1_openvla.sbatch
#   sbatch --export=ALL,SEED_ONLY=42 run/slurm/p2_rq1_openvla.sbatch
#
#SBATCH --job-name=p2-rq1-ov
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=d4054
#SBATCH --output=p2-rq1-ov-%j.out
#SBATCH --error=p2-rq1-ov-%j.out

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

OV=/home/pandya.kei/.conda/envs/vla-openvla/bin/python
SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python   # preflight only
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"

# --- Tunables -----------------------------------------------------------------
N_EPISODES="${N_EPISODES:-10}"
CONDITIONS="${CONDITIONS:-original,blank,nonsense,wrong_object,wrong_action,wrong_task,repeated}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
if [ -n "${SEED_ONLY:-}" ]; then SEEDS="$SEED_ONLY"; else SEEDS="7 42"; fi
# ------------------------------------------------------------------------------

LOG=$SCRATCH/p2_rq1_openvla_${SLURM_JOB_ID:-manual}.log
echo "=== P2 OpenVLA RQ1 | seeds=[$SEEDS] tasks=[$TASKS] n_ep=$N_EPISODES | $(hostname|cut -d. -f1) $(date) ===" | tee "$LOG"

$SM run/preflight.py >> "$LOG" 2>&1 || { echo "PREFLIGHT FAILED — channel degraded, need a fresh node." | tee -a "$LOG"; exit 1; }

for SEED in $SEEDS; do
  for TID in $TASKS; do
    echo "### openvla seed=$SEED task=$TID ###" | tee -a "$LOG"
    env $OVENV "$OV" run/eval_task_openvla.py --model openvla --suite libero_goal \
        --task_id "$TID" --conditions "$CONDITIONS" \
        --seed "$SEED" --n_episodes "$N_EPISODES" 2>&1 \
      | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort" | grep -avE "%\|" | tee -a "$LOG"
  done
done

echo "=== P2 OpenVLA RQ1 DONE $(date) ===" | tee -a "$LOG"
