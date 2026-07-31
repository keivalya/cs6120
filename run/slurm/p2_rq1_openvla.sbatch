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
# bf16 REQUIRED: these load torch_dtype=bfloat16, and on sm_70 (v100) sdpa
# falls back to the math backend with emulated bf16 — several x slower, which
# is how an 8h slice silently overruns. h200 (d4055) or a100 (d1026) only.
# Override per-submission with: sbatch --gres=gpu:a100:1 ...
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=p2-rq1-ov-%j.out
#SBATCH --error=p2-rq1-ov-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
# Must be an index INTO CUDA_VISIBLE_DEVICES, not a physical device id:
# robosuite/utils/binding_utils.py:31-35 asserts MUJOCO_EGL_DEVICE_ID is one of the
# ids in CUDA_VISIBLE_DEVICES and aborts the import otherwise.
# Measured on this cluster (job 8827909): a 1-GPU job gets CUDA_VISIBLE_DEVICES=0
# while SLURM_JOB_GPUS=2 (the physical index), nvidia-smi -L lists exactly one
# device, and egl_probe.get_available_devices() returns [0]. So SLURM's device
# cgroup already isolates the GPU: 0 IS the allocated device, and concurrent jobs
# on one node do NOT collide on physical GPU 0. Deriving this from SLURM_JOB_GPUS
# instead looks safer but breaks the assertion outright (job 8827887).
export MUJOCO_EGL_DEVICE_ID=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
export TOKENIZERS_PARALLELISM=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"

OV=${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}/vla-openvla/bin/python
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"

# --- Tunables -----------------------------------------------------------------
N_EPISODES="${N_EPISODES:-10}"
CONDITIONS="${CONDITIONS:-original,blank,nonsense,wrong_object,wrong_action,wrong_task,repeated}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
if [ -n "${SEED_ONLY:-}" ]; then SEEDS="$SEED_ONLY"; else SEEDS="7 42"; fi
# ------------------------------------------------------------------------------

LOG=$SCRATCH/p2_rq1_openvla_${SLURM_JOB_ID:-manual}.log
echo "=== P2 OpenVLA RQ1 | seeds=[$SEEDS] tasks=[$TASKS] n_ep=$N_EPISODES | $(hostname|cut -d. -f1) $(date) ===" | tee "$LOG"

# Gate on THIS env, not on smolvla: run/preflight.py loads SmolVLA, so it could
# neither validate this stack nor fail for a reason relevant to it.
env $OVENV $OV run/preflight_libero.py >> "$LOG" 2>&1 || {
  echo "PREFLIGHT FAILED on $(hostname|cut -d. -f1) — real error below (do NOT assume a bad node):" | tee -a "$LOG"
  tail -30 "$LOG"
  exit 1
}

for SEED in $SEEDS; do
  for TID in $TASKS; do
    echo "### openvla seed=$SEED task=$TID ###" | tee -a "$LOG"
    env $OVENV "$OV" run/eval_task_openvla.py --model openvla --suite libero_goal \
        --task_id "$TID" --conditions "$CONDITIONS" \
        --seed "$SEED" --n_episodes "$N_EPISODES" 2>&1 \
      | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort|scene_atomic" | grep -avE "%\|" | tee -a "$LOG"
  done
done

echo "=== P2 OpenVLA RQ1 DONE $(date) ===" | tee -a "$LOG"
