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
# NODE vs TASK IS CONFOUNDED in every run so far: OFT tasks 0-4,6-9 have only ever
# run on d4052 (all fine) and task5 has only ever run on d4055 (aborted). Submit
# BOTH of these to separate the two hypotheses:
#   DBG_TASK=5 sbatch -w d4052 ...   aborts => task-specific, not the node
#   DBG_TASK=0 sbatch -w d4055 ...   aborts => node-specific, not the task
DBG_TASK="${DBG_TASK:-5}"
echo "=== running OFT task$DBG_TASK seed7, RAW output, no grep ==="
echo "raw log: $RAW"

# NOTHING is filtered here. That is the entire point of this job.
env $OFTENV "$OFT" run/eval_task_oft.py --model openvla_oft --suite libero_goal \
    --task_id "$DBG_TASK" --conditions original,blank,nonsense --seed 7 --n_episodes 10 \
    --results_root "$THROWAWAY" > "$RAW" 2>&1
rc=$?
echo "=== runner exit code: $rc ==="

echo "=== last 60 lines of the RAW log (the error, if it printed one) ==="
tail -60 "$RAW"

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
