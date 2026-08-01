#!/bin/bash
# run/slurm/debug_abort.sbatch — capture the REAL error behind the exit-134 aborts.
#
# WHY THIS EXISTS. Every 7B rollout job that dies does so with
#   Aborted (core dumped) ... !!! RUNNER FAILED (exit 134)
# and nothing else, because the production sbatch pipes the runner through
#   2>&1 | grep -aE "ready|success=|DONE|..." | tee
# A C-level abort prints its message and dies without a Python traceback, and that
# message does not match the grep, so the only evidence is discarded at the moment
# it is produced. This project has already lost weeks to exactly that: a blanket
# "PREFLIGHT FAILED — channel degraded" hid a FileNotFoundError for five stale paths.
# Do not add another node to --exclude until this job has printed the actual error.
#
# It reproduces on OFT because OFT is the fast one: ~7 s/episode, and job 8867194
# aborted after 3 clean episodes, so a reproduction costs about a minute.
#
# Writes to a THROWAWAY results root so a debug run can never contaminate the grid
# or trip scene_atomic on a real cell.
#
#   sbatch run/slurm/debug_abort.sbatch                    # wherever it lands
#   sbatch -w d4055 run/slurm/debug_abort.sbatch           # a node that aborted
#   sbatch -w d4052 run/slurm/debug_abort.sbatch           # the one that worked
#
#SBATCH --job-name=dbg-abort
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=dbg-abort-%j.out
#SBATCH --error=dbg-abort-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"

export SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
export TOKENIZERS_PARALLELISM=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"
# Make the abort as legible as it can be made:
export PYTHONFAULTHANDLER=1        # dumps a Python stack even on a fatal signal
export MESA_DEBUG=1
export EGL_LOG_LEVEL=debug
export MUJOCO_LOG_PATH=$SCRATCH/mujoco_${SLURM_JOB_ID}.log   # MuJoCo writes its own fatals here
ulimit -c unlimited
# Land cores somewhere we can actually read them; the node default may be a pipe
# to systemd-coredump, which we have no permission to query (job 8873595).
CORED=$SCRATCH/cores_${SLURM_JOB_ID}
mkdir -p "$CORED"
cd "$CORED" 2>/dev/null && echo "core pattern: $(cat /proc/sys/kernel/core_pattern)" && cd "$REPO"

OFT=${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}/vla-oft/bin/python
# MUST match p2_rq1_oft.sbatch line 61 EXACTLY. Job 8873595 used ~/.libero_oft,
# which has never existed — the OFT env reads the openvla config dir — so the
# runner died on the line-63 assert in 5 s and never reached the abort. A debug
# harness that diverges from the job it is debugging measures nothing.
OFTENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"
RAW=$SCRATCH/debug_abort_${SLURM_JOB_ID}.raw.log
THROWAWAY=$SCRATCH/debug_results_${SLURM_JOB_ID}

echo "=== node $(hostname|cut -d. -f1) $(date) ==="
echo "--- GPU / driver ---"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,ecc.errors.uncorrected.volatile.total --format=csv
echo "--- who else is on this node ---"
squeue -w "$(hostname|cut -d. -f1)" -h -o "%i %u %b" | wc -l
echo "--- memory ---"
free -g | head -2
echo
# ANSWERED 2026-07-31 23:40 (job 8874110): TASK, NOT NODE. OFT seed7/task5 aborted
# on d4052 — the node three "bad node" claims were measured against — at episode 2
# of `original`, the same place it aborts on d4055 (8867194 died at episode 3).
# faulthandler put the crash in robosuite binding_utils.py:171 read_pixels, i.e.
# mjr_readPixels, so the remaining question is what the driver/MuJoCo is
# complaining about there. Only gdb can say: the abort prints nothing, and the
# core goes to systemd-coredump, which we cannot read (job 8874110).
DBG_TASK="${DBG_TASK:-5}"
echo "=== running OFT task$DBG_TASK seed7, RAW output, no grep ==="
echo "raw log: $RAW"

# NOTHING is filtered here. That is the entire point of this job.
# Run UNDER gdb so the native stack is captured in-process; a core file never
# survives to disk on these nodes.
if [ "${DBG_GDB:-1}" = 1 ] && command -v gdb >/dev/null; then
  env $OFTENV gdb -batch -q \
      -ex "set confirm off" -ex "handle SIGABRT stop print" -ex run \
      -ex "thread apply all bt full" -ex "info sharedlibrary" \
      --args "$OFT" run/eval_task_oft.py --model openvla_oft --suite libero_goal \
      --task_id "$DBG_TASK" --conditions original,blank,nonsense --seed 7 --n_episodes 10 \
      --results_root "$THROWAWAY" > "$RAW" 2>&1
else
  env $OFTENV "$OFT" run/eval_task_oft.py --model openvla_oft --suite libero_goal \
      --task_id "$DBG_TASK" --conditions original,blank,nonsense --seed 7 --n_episodes 10 \
      --results_root "$THROWAWAY" > "$RAW" 2>&1
fi
rc=$?
echo "=== runner exit code: $rc ==="

echo "=== last 150 lines of the RAW log (the error, and gdb's stack if it ran) ==="
tail -150 "$RAW"

echo
echo "=== MuJoCo's own fatal log ==="
cat "$MUJOCO_LOG_PATH" 2>/dev/null || echo "(none written)"

echo
echo "=== core file + native backtrace ==="
# The Python traceback is useless for a C-level abort; the C stack is the evidence.
CORE=$(ls -t "$CORED"/core* "$REPO"/core* /tmp/core* 2>/dev/null | head -1)
if [ -n "$CORE" ]; then
  echo "core: $CORE ($(du -h "$CORE" | cut -f1))"
  if command -v gdb >/dev/null; then
    gdb -batch -q -ex "thread apply all bt" "$OFT" "$CORE" 2>&1 | tail -80
  else
    echo "no gdb on this node; core kept at $CORE"
  fi
else
  echo "no core file found (pattern above says where the kernel sent it)"
fi

echo
echo "=== systemd coredump record, if the kernel kept one ==="
coredumpctl list --no-pager 2>&1 | tail -5
coredumpctl info --no-pager 2>&1 | tail -40

echo
echo "=== full raw log kept at $RAW ==="
rm -rf "$THROWAWAY"
