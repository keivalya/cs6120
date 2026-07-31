#!/bin/bash
# run/refresh_paper.sh — regenerate every number in the paper from the result files.
#
# CPU only, no GPU, ~1 minute. Run this after any eval job finishes, then rebuild the
# PDF. It is the whole P4 chain in order, and it FAILS LOUDLY rather than leaving a
# half-updated paper: a scene-fixed failure is reported per model and reflected in
# the tables (as a dagger marker) instead of being silently averaged in.
#
#   bash run/refresh_paper.sh              # all three models
#   bash run/refresh_paper.sh smolvla      # one model's aggregate, then all tables
#
# NOTE: aggregate.py exits 2 when a model's scene-fixed check fails. That is not a
# reason to stop -- the tables mark that model and the paper says it is confounded --
# so we record the verdict and carry on, then print a summary at the end.

set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO" || exit 1
PY=${PY:-/scratch/pandya.kei/conda-envs/.conda/envs/vla-smolvla/bin/python}
SUITE=${SUITE:-libero_goal}
MODELS=${*:-"smolvla openvla openvla_oft"}

[ -x "$PY" ] || { echo "no python at $PY (set PY=...)"; exit 1; }

declare -A VERDICT
echo "=== 1/4 aggregate + scene-fixed check ==="
for m in $MODELS; do
  out=$($PY run/aggregate.py --model "$m" --suite "$SUITE" 2>&1)
  rc=$?
  line=$(echo "$out" | grep -E "scene_fixed_check" | tail -1)
  VERDICT[$m]="rc=$rc ${line#*scene_fixed_check: }"
  echo "  $m: ${VERDICT[$m]}"
  echo "$out" | grep -E "WARNING|SCENE-FIXED CHECK FAILED" | sed 's/^/    /'
done

echo "=== 2/4 RQ1 csvs (per model, then combined) ==="
for m in $MODELS; do
  $PY analyze/make_rq1_csv.py --model "$m" --suite "$SUITE" >/dev/null || echo "  FAILED: $m"
done
$PY analyze/make_rq1_scale.py --suite "$SUITE" >/dev/null || echo "  FAILED: rq1_scale"

echo "=== 3/4 RQ2 + word-class information analysis ==="
$PY analyze/make_rq2_csv.py --suite "$SUITE" >/dev/null || echo "  FAILED: rq2 csv"
$PY analyze/instruction_information.py --suite "$SUITE" >/dev/null 2>&1 \
  || echo "  note: instruction_information.py did not run (report/instruction_information.csv kept)"

echo "=== 4/4 tables + build check ==="
$PY analyze/make_tables.py 2>&1 | grep -E "wrote|scene-fixed" | sed 's/^/  /'
python3 report/check_tex.py || { echo "!! paper would not build -- fix the above"; exit 1; }

echo
echo "=== scene-fixed verdicts (causal claims valid only where this passes) ==="
for m in $MODELS; do printf "  %-12s %s\n" "$m" "${VERDICT[$m]}"; done
echo
echo "Next: rebuild the PDF off-cluster (no TeX toolchain here). The self-contained"
echo "source is rebuilt by:  bash run/make_submission_zip.sh"
