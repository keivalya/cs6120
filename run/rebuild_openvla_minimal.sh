#!/bin/bash
# run/rebuild_openvla_minimal.sh — rebuild the vla-openvla env (OpenVLA-7B RQ1.3).
#
# WHY NOT REPLAY envs/vla-openvla.lock.txt: that freeze cannot round-trip. It pins
# torch==2.2.0+cu121 with no torch index (PyPI has no +cu121 local versions), and
# tensorflow-addons==0.23.0, which requires Python <3.10 and so can never install
# on the pinned 3.10 interpreter — job 8726842 died on exactly those two. It also
# pins ~100 transitive packages frozen at their 2026-07-10 latest, every one a
# candidate for the next ResolutionImpossible at ~15 min per batch round-trip, and
# it pins packages this runner deliberately never imports (tensorflow-addons has
# zero grep hits in openvla/; the whole dlimp/tensorflow-datasets/prismatic chain
# is what run/eval_task_openvla.py:115-125 exists to route around). So: install
# the top-level packages the runner actually imports, hold the versions that
# matter with a constraints file, and gate on a smoke test.
#
# CONTRAST WITH vla-oft: this env keeps protobuf 4.25.x. Do NOT copy the 3.20.3
# pin from run/rebuild_oft_minimal.sh — that downgrade exists only because OFT
# must import dlimp. Here there is no dlimp, so 4.25 is correct and simpler.
#
# Needs network + PyPI + download.pytorch.org. No GPU (validate separately with
# run/smoke_openvla.py on a GPU node). conda create OOMs on the login node, so run
# it in an allocation:  sbatch run/slurm/rebuild_7b_minimal.sbatch
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONDA_ENVS="${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}"
PREFIX="$CONDA_ENVS/vla-openvla"
mkdir -p "$CONDA_ENVS"

command -v conda >/dev/null || { echo "FATAL: conda not on PATH (source conda first)."; exit 1; }

echo "=== minimal vla-openvla -> $PREFIX (python 3.10) $(date) ==="
rm -rf "$PREFIX"
conda create -y -p "$PREFIX" python=3.10 || { echo "FATAL: conda create failed"; exit 1; }
PIP="$PREFIX/bin/pip"
export PATH="$PREFIX/bin:$PATH"

# The versions that actually determine behaviour, from envs/vla-openvla.lock.txt.
C="$PREFIX/constraints.txt"
cat > "$C" <<'EOF'
numpy==1.26.4
torch==2.2.0
torchvision==0.17.0
transformers==4.40.1
tokenizers==0.19.1
tensorflow==2.15.0
protobuf==4.25.9
mujoco==2.3.2
robosuite==1.4.1
opencv-python==4.11.0.86
matplotlib==3.9.4
EOF

"$PIP" install --upgrade pip wheel setuptools || exit 1

# numpy FIRST: tf 2.15 needs numpy<2, and mujoco 2.3.2 / robosuite 1.4.1 were
# built against 1.x. Letting a later step pull numpy 2.x breaks both.
"$PIP" install "numpy==1.26.4" || exit 1

# --index-url, NOT --extra-index-url: with an extra index pip may still prefer the
# default-CUDA PyPI wheel for 2.2.0, and the +cu121 local version never resolves.
"$PIP" install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0 torchvision==0.17.0 || exit 1

# TF before the HF stack so protobuf settles at 4.25 first.
"$PIP" install -c "$C" tensorflow==2.15.0 protobuf==4.25.9 || exit 1

"$PIP" install -c "$C" transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10 \
    accelerate huggingface_hub safetensors einops sentencepiece || exit 1

"$PIP" install -c "$C" mujoco==2.3.2 robosuite==1.4.1 bddl easydict h5py \
    imageio imageio-ffmpeg opencv-python==4.11.0.86 termcolor cloudpickle \
    "gym==0.26.2" numba scipy PyYAML pillow trimesh PyOpenGL glfw tqdm \
    matplotlib==3.9.4 || exit 1
# matplotlib is NOT optional and NOT a plotting convenience here: LIBERO's
# libero/libero/envs/env_wrapper.py does `import matplotlib.cm` at module scope, so
# OffScreenRenderEnv cannot even be imported without it. Omitting it made every
# preflight and smoke run fail with ModuleNotFoundError (job 8828378).

# LIBERO editable, from the checkout ~/.libero_openvla/config.yaml already points
# at. setup.py has install_requires=[], so --no-deps is belt and braces.
"$PIP" install -e "$REPO/openvla/LIBERO" --config-settings editable_mode=compat --no-deps || exit 1

# HARD gate. rebuild_smolvla_minimal.sh only warns on drift; a silently-drifted
# protobuf or numpy here yields an env that imports fine and then dies hours into
# a production job, so fail the build instead.
echo "[gate] checking the pins that matter..."
"$PREFIX/bin/python" - <<'PY' || exit 1
import sys
from importlib.metadata import version
want = {"numpy": "1.26.4", "protobuf": "4.25.9", "transformers": "4.40.1",
        "mujoco": "2.3.2", "robosuite": "1.4.1", "torch": "2.2.0",
        "matplotlib": "3.9.4"}
bad = []
for pkg, exp in want.items():
    got = version(pkg)
    if not got.startswith(exp):
        bad.append(f"  {pkg}: want {exp}, got {got}")
if bad:
    print("FATAL: version drift in vla-openvla:\n" + "\n".join(bad), file=sys.stderr)
    sys.exit(1)
print("  pins OK:", ", ".join(f"{k}={version(k)}" for k in sorted(want)))
PY

echo "=== vla-openvla BUILT at $PREFIX $(date) ==="
echo "Validate on a GPU node:  $PREFIX/bin/python run/smoke_openvla.py"
