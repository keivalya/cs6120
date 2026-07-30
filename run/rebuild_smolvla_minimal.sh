#!/bin/bash
# run/rebuild_smolvla_minimal.sh — minimal fresh vla-smolvla env (top-level deps).
#
# Instead of replaying the full 176-line pinned freeze, this installs only the
# top-level packages the SmolVLA runner needs and lets pip resolve the rest:
#   - LeRobot at its pinned commit (provides lerobot.policies.smolvla and
#     lerobot.envs.libero) with the [smolvla] extra
#   - the LIBERO benchmark env deps the runner imports (libero.libero via
#     hf_libero, plus robosuite + mujoco)
#   - SmolVLA build/runtime bits: cmake<4 (egl_probe build), num2words (SmolVLM)
#
# Faster and more forgiving than the full freeze; accepts a small risk that
# looser transitive versions differ from the original run. Verified by a smoke
# test that imports exactly what run/eval_task.py imports.
#
# MUST run inside an allocation (login node OOM-kills conda create). Use the
# wrapper: sbatch run/slurm/rebuild_smolvla_minimal.sbatch
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
CONDA_ENVS="${CONDA_ENVS:-/scratch/pandya.kei/conda-envs/.conda/envs}"
PREFIX="$CONDA_ENVS/vla-smolvla"
LEROBOT_COMMIT="e40b58a8dfa9e7b86918c374791599d070518d11"   # from envs/vla-smolvla.lock.txt
mkdir -p "$CONDA_ENVS"

command -v conda >/dev/null || { echo "FATAL: conda not on PATH (module load anaconda / source conda first)."; exit 1; }

echo "=== minimal vla-smolvla -> $PREFIX (python 3.12) ==="
rm -rf "$PREFIX"
conda create -y -p "$PREFIX" python=3.12 || { echo "FATAL: conda create failed"; exit 1; }
PIP="$PREFIX/bin/pip"
export PATH="$PREFIX/bin:$PATH"

# Hold the packages whose version the measured runs depended on
# (envs/vla-smolvla.lock.txt) so a loose resolve can't silently swap out the
# torch/transformers stack, the gym API, or bump opencv across a major version.
CONSTRAINTS="$PREFIX/constraints.txt"
cat > "$CONSTRAINTS" <<'EOF'
torch==2.10.0
torchvision==0.25.0
transformers==5.1.0
numpy==2.2.6
gymnasium==1.2.3
opencv-python==4.13.0.92
opencv-python-headless==4.10.0.84
EOF

"$PIP" install --upgrade pip wheel setuptools
"$PIP" install "cmake<4" num2words

# egl_probe/hf-egl-probe shell out to a bare `cmake` from their setup.py. The pip
# cmake wheel's bin/cmake is a Python console script (`from cmake import cmake`),
# and under pip's build isolation that import fails -> exit 1 -> "CMake must be
# installed". Put the wheel's REAL cmake binary (site-packages/cmake/data/bin)
# first on PATH instead; it has no Python dependency.
CMAKE_REAL_BIN=$("$PREFIX/bin/python" -c "import cmake,os;print(os.path.join(os.path.dirname(cmake.__file__),'data','bin'))")
export PATH="$CMAKE_REAL_BIN:$PATH"
command -v cmake >/dev/null && cmake --version || { echo "FATAL: no working cmake at $CMAKE_REAL_BIN"; exit 1; }

echo "[pip] LeRobot @ $LEROBOT_COMMIT (with [smolvla] extra)..."
"$PIP" install -c "$CONSTRAINTS" "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}" \
  || "$PIP" install -c "$CONSTRAINTS" "lerobot @ git+https://github.com/huggingface/lerobot.git@${LEROBOT_COMMIT}"

echo "[pip] LIBERO env deps (hf_libero provides the 'libero' module)..."
"$PIP" install -c "$CONSTRAINTS" "hf_libero==0.1.4" "robosuite==1.4.0" "mujoco==3.4.0"

echo "[smoke] importing exactly what run/eval_task.py needs..."
"$PREFIX/bin/python" - <<'PY'
import torch, numpy
from lerobot.envs import make_env
from lerobot.envs.configs import LiberoEnv as _C
from lerobot.policies import make_policy
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark, get_libero_path
print("SMOKE OK: torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY
rc=$?
if [ $rc -ne 0 ]; then
  echo "SMOKE FAILED (rc=$rc) — the env is missing something the runner imports."
  echo "Send me the traceback above; likely a lerobot extra or an env dep to add."
  exit $rc
fi
echo "=== minimal vla-smolvla READY at $PREFIX ==="
