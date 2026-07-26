#!/bin/bash
# run/slurm/p2_rq1_smolvla.sbatch — P2 statistical power, SmolVLA RQ1 causal grid.
#
# Scales RQ1 from the old 2 episodes/task to the grid.yaml design (20/task) across
# seeds 7 and 42, all 7 causal conditions, all 10 LIBERO-Goal tasks.
#
# PARALLELISM MODEL: this is a SLURM JOB ARRAY. Each array element is ONE eval
# process (one seed, one task) on its OWN GPU/node. That respects the hard rule
# "one EGL process at a time PER node/process" — separate array elements on
# separate GPUs do NOT share an EGL channel, so this is safe and fast.
#
# Array index -> (seed, task):  idx in [0..19]; seed = SEEDS[idx/10]; task = idx%10.
# Runner is idempotent/resumable: it tops up each task<tid>.jsonl to --n_episodes
# and skips finished ones, so re-submitting after a timeout just continues.
#
# Submit:
#   sbatch run/slurm/p2_rq1_smolvla.sbatch
# Then, only AFTER all elements finish:
#   sbatch run/slurm/p2_aggregate.sbatch
#
#SBATCH --job-name=p2-rq1-sm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --exclude=d4054
#SBATCH --array=0-19
#SBATCH --output=p2-rq1-sm-%A_%a.out
#SBATCH --error=p2-rq1-sm-%A_%a.out

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

SM=/home/pandya.kei/.conda/envs/vla-smolvla/bin/python

# --- Tunables -----------------------------------------------------------------
SEEDS=(7 42)
N_EPISODES="${N_EPISODES:-20}"
CONDITIONS="${CONDITIONS:-original,blank,nonsense,wrong_object,wrong_action,wrong_task,repeated}"
# ------------------------------------------------------------------------------

IDX=${SLURM_ARRAY_TASK_ID:?must run as a job array}
SEED=${SEEDS[$((IDX / 10))]}
TID=$((IDX % 10))
LOG=$SCRATCH/p2_rq1_smolvla_s${SEED}_t${TID}.log

echo "=== P2 SmolVLA RQ1 | seed=$SEED task=$TID n_ep=$N_EPISODES | $(hostname|cut -d. -f1) $(date) ===" | tee "$LOG"

# Preflight ONCE per array element (heavy CUDA+EGL must survive or channel is degraded).
$SM run/preflight.py >> "$LOG" 2>&1 || { echo "PREFLIGHT FAILED — channel degraded, need fresh node." | tee -a "$LOG"; exit 1; }

$SM run/eval_task.py --model smolvla --suite libero_goal \
    --task_id "$TID" --conditions "$CONDITIONS" \
    --seed "$SEED" --n_episodes "$N_EPISODES" --obs_hw 256 2>&1 \
  | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort" | grep -avE "%\|" | tee -a "$LOG"

echo "=== done seed=$SEED task=$TID $(date) ===" | tee -a "$LOG"
