#!/bin/bash
# core/run/slurm/render_qualitative.sbatch — the videos the qualitative figure needs.
#
# paper/qualitative_grid.png was CUT on 2026-08-01 because two of its four rows were
# synthetic: make_qualitative_grid.py matches a rollout video per condition and, when
# none exists, used to draw coloured rectangles. There has never been a video for
# wrong_object or para_compositional. This renders them, on the ONE task where the
# compliance case is decisive — task4's object swap yields, verbatim, task2's
# instruction, and every model performs task2 and never task4. That row is the most
# convincing image the paper can carry, and it has to be a real rollout.
#
#   sbatch core/run/slurm/render_qualitative.sbatch
#   then:  python core/analyze/make_qualitative_grid.py   (raises if any row is missing)
#
# Cheap: 4 conditions x 3 episodes on one task, not a grid. Submit only once the P2
# grid has drained — it is not on the critical path and the grid is.
#SBATCH --job-name=vla-qual
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/vla-qual-%j.out
#SBATCH --error=logs/slurm/vla-qual-%j.out

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO" || exit 1

SCRATCH=/scratch/pandya.kei
export HF_HOME=$SCRATCH/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
export TOKENIZERS_PARALLELISM=false
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"

OFT=${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}/vla-oft/bin/python
OVENV="LIBERO_CONFIG_PATH=/home/pandya.kei/.libero_openvla HF_HUB_OFFLINE=1"

MODEL=${MODEL:-openvla_oft}
TASK=${TASK:-4}          # the decisive compliance case: task4's swap names task2
N=${N:-3}

# Log the GPU: the exit-134 EGL abort belongs to individual devices, not nodes
# (d4055 IDX 3 aborts every allocation; d4055 IDX 0 ran four hours clean).
echo "[gpu] $(nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader)"

rc=0
for COND in original wrong_object wrong_action para_compositional; do
  echo "=== rendering $MODEL task$TASK $COND ==="
  env $OVENV "$OFT" core/run/generate_videos.py --model "$MODEL" --task_id "$TASK" \
      --condition "$COND" --num_videos "$N" --output_dir data/videos || rc=1
done

echo "=== DONE rc=$rc $(date) ==="
echo "next: python core/analyze/make_qualitative_grid.py"
exit $rc
