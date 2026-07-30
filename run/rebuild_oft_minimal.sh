#!/bin/bash
# run/rebuild_oft_minimal.sh — rebuild the vla-oft env (OpenVLA-OFT-7.5B RQ1.3).
#
# See run/rebuild_openvla_minimal.sh for why we don't replay envs/vla-oft.lock.txt.
#
# WHAT MAKES THIS ENV DIFFERENT (and why the step ORDER below is load-bearing):
# run/eval_task_oft.py reuses the OFT repo's experiments.robot.* helpers, so
# `import prismatic` must work, and prismatic.vla eagerly imports dlimp ->
# tensorflow-datasets -> tensorflow-metadata. The latest tensorflow-metadata's
# generated anomalies_pb2.py needs protobuf>=5.26, which blows up under tf 2.15.
# tensorflow-metadata==1.14.0 is the last release whose _pb2 files work with
# protobuf 3.20.3. That is the ENTIRE reason this env lives in a 3.20.3 world
# while vla-openvla stays on 4.25.9 — do not copy pins between the two scripts.
#
# Ordering constraints, each learned the hard way (see envs/oft_*.log):
#   1. tensorflow-datasets has an unpinned tensorflow-metadata requirement, so it
#      pulls the latest. Install it BEFORE pinning metadata/protobuf, then pin
#      them in one command so the downgrade isn't immediately re-upgraded.
#   2. dlimp's setup requires unpinned tensorflow/tensorflow-datasets and will
#      undo step 1 -> install it --no-deps.
#   3. wandb 0.28's wandb_internal_pb2 needs protobuf>=5, so it must come after
#      the pin and re-assert 3.20.3. peft/diffusers can drag wandb in.
#   4. the moojink transformers fork's install_requires moves tokenizers/numpy/
#      protobuf -> install it --no-deps, then its runtime deps pinned.
#
# Needs network + git over HTTPS (two git+https pins). No GPU for the build.
#   sbatch run/slurm/rebuild_7b_minimal.sbatch
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONDA_ENVS="${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}"
PREFIX="$CONDA_ENVS/vla-oft"
mkdir -p "$CONDA_ENVS"

command -v conda >/dev/null || { echo "FATAL: conda not on PATH (source conda first)."; exit 1; }

echo "=== minimal vla-oft -> $PREFIX (python 3.10) $(date) ==="
rm -rf "$PREFIX"
conda create -y -p "$PREFIX" python=3.10 || { echo "FATAL: conda create failed"; exit 1; }
PIP="$PREFIX/bin/pip"
export PATH="$PREFIX/bin:$PATH"

C="$PREFIX/constraints.txt"
cat > "$C" <<'EOF'
numpy==1.26.4
torch==2.2.0
torchvision==0.17.0
tokenizers==0.19.1
tensorflow==2.15.0
tensorflow-datasets==4.9.3
tensorflow-metadata==1.14.0
protobuf==3.20.3
wandb==0.16.6
peft==0.11.1
diffusers==0.30.3
mujoco==2.3.2
robosuite==1.4.1
opencv-python==4.11.0.86
EOF

"$PIP" install --upgrade pip wheel setuptools || exit 1
"$PIP" install "numpy==1.26.4" || exit 1
"$PIP" install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0 torchvision==0.17.0 || exit 1

# (1) tfds first, then pin metadata+protobuf together.
"$PIP" install -c "$C" tensorflow==2.15.0 tensorflow-datasets==4.9.3 || exit 1
"$PIP" install -c "$C" tensorflow-metadata==1.14.0 protobuf==3.20.3 || exit 1

# (2) dlimp --no-deps or it undoes the above.
"$PIP" install -c "$C" --no-deps \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46" || exit 1

# (3) wandb pinned old, re-asserting protobuf in the same command.
"$PIP" install -c "$C" wandb==0.16.6 protobuf==3.20.3 || exit 1

# (4) the transformers fork bare, then its runtime deps pinned.
"$PIP" install -c "$C" --no-deps \
    "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab" || exit 1
"$PIP" install -c "$C" tokenizers==0.19.1 timm==0.9.10 peft==0.11.1 \
    diffusers==0.30.3 accelerate huggingface_hub safetensors einops \
    sentencepiece draccus rich regex filelock || exit 1

"$PIP" install -c "$C" mujoco==2.3.2 robosuite==1.4.1 bddl easydict h5py \
    imageio imageio-ffmpeg opencv-python==4.11.0.86 termcolor cloudpickle \
    "gym==0.26.2" numba scipy PyYAML pillow trimesh PyOpenGL glfw tqdm || exit 1

"$PIP" install -e "$REPO/openvla-oft/LIBERO" --config-settings editable_mode=compat --no-deps || exit 1

# The lazy prismatic/__init__.py patch is what keeps `import prismatic` from
# pulling the training stack. It is currently applied in the checkout; re-apply if
# the checkout was ever reset.
if ! grep -q "PATCHED" "$REPO/openvla-oft/prismatic/__init__.py" 2>/dev/null; then
  echo "[patch] re-applying report/patches/oft_prismatic_init_lazy.diff"
  patch -p0 -d "$REPO" < "$REPO/report/patches/oft_prismatic_init_lazy.diff" || {
    echo "FATAL: prismatic lazy patch failed to apply"; exit 1; }
fi

echo "[gate] checking the pins that matter..."
"$PREFIX/bin/python" - <<'PY' || exit 1
import sys
from importlib.metadata import version
want = {"numpy": "1.26.4", "protobuf": "3.20.3", "tensorflow-metadata": "1.14.0",
        "wandb": "0.16.6", "peft": "0.11.1", "diffusers": "0.30.3",
        "mujoco": "2.3.2", "robosuite": "1.4.1", "torch": "2.2.0"}
bad = []
for pkg, exp in want.items():
    got = version(pkg)
    if not got.startswith(exp):
        bad.append(f"  {pkg}: want {exp}, got {got}")
if bad:
    print("FATAL: version drift in vla-oft:\n" + "\n".join(bad), file=sys.stderr)
    sys.exit(1)
print("  pins OK:", ", ".join(f"{k}={version(k)}" for k in sorted(want)))
PY

echo "=== vla-oft BUILT at $PREFIX $(date) ==="
echo "Validate on a GPU node:  $PREFIX/bin/python run/smoke_oft.py"
