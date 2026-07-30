#!/bin/bash
# run/slurm/p2_rq1_oft.sbatch — P2 statistical power, OpenVLA-OFT-7.5B RQ1 causal grid.
#
# NON-ARRAY design (respects a tight QOSMaxSubmitJobPerUser limit): ONE job that
# loops seeds x tasks internally, one EGL process at a time. Scales RQ1 to 10
# episodes/task across seeds 7,42. Also backfills the OFT seed42 original
# checkpoint_hash (null today) since eval_task_oft.py now records it.
# Resumable: re-submit if it hits the wall-time limit.
#
# To halve wall time / parallelize, submit one seed per job:
#   sbatch --export=ALL,SEED_ONLY=7  run/slurm/p2_rq1_oft.sbatch
#   sbatch --export=ALL,SEED_ONLY=42 run/slurm/p2_rq1_oft.sbatch
#
#SBATCH --job-name=p2-rq1-oft
#SBATCH --partition=gpu
# bf16 REQUIRED: these load torch_dtype=bfloat16, and on sm_70 (v100) sdpa
# falls back to the math backend with emulated bf16 — several x slower, which
# is how an 8h slice silently overruns. h200 (d4055) or a100 (d1026) only.
# Override per-submission with: sbatch --gres=gpu:a100:1 ...
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=p2-rq1-oft-%j.out
#SBATCH --error=p2-rq1-oft-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
# EGL enumerates PHYSICAL devices and ignores CUDA_VISIBLE_DEVICES, so a
# hardcoded 0 makes every concurrent job on a node render on physical GPU 0 while
# computing on its own GPU. That is the documented NVRM Xid 31 / degraded-channel
# trigger (report/GATE3_STATUS.md) and it only bites once >1 of these run per
# node — which is exactly what filling 4+ GPU slots does. Derive it instead.
export MUJOCO_EGL_DEVICE_ID=$(echo "${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}" | cut -d, -f1)
export TOKENIZERS_PARALLELISM=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"

OFT=${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}/vla-oft/bin/python
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"

# --- Tunables -----------------------------------------------------------------
N_EPISODES="${N_EPISODES:-10}"
CONDITIONS="${CONDITIONS:-original,blank,nonsense,wrong_object,wrong_action,wrong_task,repeated}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
if [ -n "${SEED_ONLY:-}" ]; then SEEDS="$SEED_ONLY"; else SEEDS="7 42"; fi
# ------------------------------------------------------------------------------

LOG=$SCRATCH/p2_rq1_oft_${SLURM_JOB_ID:-manual}.log
echo "=== P2 OFT RQ1 | seeds=[$SEEDS] tasks=[$TASKS] n_ep=$N_EPISODES | $(hostname|cut -d. -f1) $(date) ===" | tee "$LOG"

# Gate on THIS env, not on smolvla: run/preflight.py loads SmolVLA, so it could
# neither validate this stack nor fail for a reason relevant to it.
env $OVENV $OFT run/preflight_libero.py >> "$LOG" 2>&1 || {
  echo "PREFLIGHT FAILED on $(hostname|cut -d. -f1) — real error below (do NOT assume a bad node):" | tee -a "$LOG"
  tail -30 "$LOG"
  exit 1
}

for SEED in $SEEDS; do
  for TID in $TASKS; do
    echo "### oft seed=$SEED task=$TID ###" | tee -a "$LOG"
    env $OVENV "$OFT" run/eval_task_oft.py --model openvla_oft --suite libero_goal \
        --task_id "$TID" --conditions "$CONDITIONS" \
        --seed "$SEED" --n_episodes "$N_EPISODES" 2>&1 \
      | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort" | grep -avE "%\|" | tee -a "$LOG"
  done
done

echo "=== P2 OFT RQ1 DONE $(date) ===" | tee -a "$LOG"
