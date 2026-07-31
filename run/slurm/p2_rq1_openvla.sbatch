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
# d4054 EXCLUDED. Measured 2026-07-31: every rollout process on d4054 SIGABRTs
# (core dump) a couple of episodes after "env ready", while the identical job
# succeeds on d4052 — openvla task0 ran 60 episodes clean on d4052 (job 8853238)
# and job 8853239 aborted on d4054; the OFT grid aborted on ALL TEN tasks there
# (job 8853355). A short smoke test PASSES on d4054, so the node looks healthy and
# only sustained EGL rendering exposes it. HANDOFF 1.5 said excluding d4054 was a
# no-op because it was DRAIN; it is schedulable again, so the exclude is live and
# load-bearing. Drop it only after a long rollout is shown to survive there.
#SBATCH --exclude=d4054
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
# NINE conditions: the 7 causal ones plus the nouns_masked/verb_dropped word-class
# ablation. HANDOFF 1.7 deferred this for openvla on a ~18 GPU-h estimate drawn
# from 160 s/episode; the measured rate is ~46 s/episode on h200 (job 8853238 ran
# task0's whole grid, 60 episodes, in 46.5 min), so the 2 extra conditions cost
# ~7 GPU-h, not 18. Folding them in now is cheaper than the alternative, because
# conditions that get COMPARED must come from one process (1.6/1.7) — adding the
# ablation later would discard and re-run this entire grid. Decided 2026-07-31.
# smolvla stays at 7: there the rate really is ~161 s/episode.
#
# Do NOT pass this via `sbatch --export=ALL,CONDITIONS=a,b,c`: SLURM splits that
# list on commas. Use `CONDITIONS=a,b,c sbatch --export=ALL ...` instead.
CONDITIONS="${CONDITIONS:-original,blank,nonsense,wrong_object,wrong_action,wrong_task,repeated,nouns_masked,verb_dropped}"
TASKS="${TASKS:-0 1 2 3 4 5 6 7 8 9}"
if [ -n "${SEED_ONLY:-}" ]; then SEEDS="$SEED_ONLY"; else SEEDS="7 42"; fi
# ------------------------------------------------------------------------------

LOG=$SCRATCH/p2_rq1_openvla_${SLURM_JOB_ID:-manual}.log
# CONDITIONS echoed: it is overridable and decides how big the grid actually is.
echo "=== P2 OpenVLA RQ1 | seeds=[$SEEDS] tasks=[$TASKS] n_ep=$N_EPISODES conds=[$CONDITIONS] | $(hostname|cut -d. -f1) $(date) ===" | tee "$LOG"

# Gate on THIS env, not on smolvla: run/preflight.py loads SmolVLA, so it could
# neither validate this stack nor fail for a reason relevant to it.
env $OVENV $OV run/preflight_libero.py >> "$LOG" 2>&1 || {
  echo "PREFLIGHT FAILED on $(hostname|cut -d. -f1) — real error below (do NOT assume a bad node):" | tee -a "$LOG"
  tail -30 "$LOG"
  exit 1
}

rc=0
for SEED in $SEEDS; do
  for TID in $TASKS; do
    echo "### openvla seed=$SEED task=$TID ###" | tee -a "$LOG"
    env $OVENV "$OV" run/eval_task_openvla.py --model openvla --suite libero_goal \
        --task_id "$TID" --conditions "$CONDITIONS" \
        --seed "$SEED" --n_episodes "$N_EPISODES" 2>&1 \
      | grep -aE "ready|success=|DONE|skip|Error|Traceback|abort|scene_atomic" | grep -avE "%\|" | tee -a "$LOG"
    # PIPESTATUS[0] is the RUNNER's status, not tee's. Without this a runner that
    # SIGABRTs (job 8853239 core-dumped right after "env ready") still let the
    # script fall through to its final echo and exit 0, so SLURM recorded
    # COMPLETED for a task that produced nothing. Same silent-success class as the
    # p2-agg jobs that "COMPLETED" against an empty env.
    st=${PIPESTATUS[0]}
    if [ "$st" -ne 0 ]; then
      rc=1
      echo "!!! RUNNER FAILED (exit $st) seed=$SEED task=$TID — cell left INCOMPLETE" | tee -a "$LOG"
    fi
  done
done

echo "=== P2 OpenVLA RQ1 DONE $(date) ===" | tee -a "$LOG"
exit $rc
