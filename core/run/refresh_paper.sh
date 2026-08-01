#!/bin/bash
# core/run/refresh_paper.sh — regenerate every number in the paper from the result files.
#
# CPU only, no GPU, ~1 minute. Run this after any eval job finishes, then rebuild the
# PDF. It is the whole P4 chain in order, and it FAILS LOUDLY rather than leaving a
# half-updated paper: a scene-fixed failure is reported per model and reflected in
# the tables (as a dagger marker) instead of being silently averaged in.
#
#   bash core/run/refresh_paper.sh              # all three models
#   bash core/run/refresh_paper.sh smolvla      # one model's aggregate, then all tables
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
echo "=== 1/5 aggregate + scene-fixed check ==="
for m in $MODELS; do
  out=$($PY core/run/aggregate.py --model "$m" --suite "$SUITE" 2>&1)
  rc=$?
  line=$(echo "$out" | grep -E "scene_fixed_check" | tail -1)
  VERDICT[$m]="rc=$rc ${line#*scene_fixed_check: }"
  echo "  $m: ${VERDICT[$m]}"
  echo "$out" | grep -E "WARNING|SCENE-FIXED CHECK FAILED" | sed 's/^/    /'
done

echo "=== 2/5 RQ1 csvs (per model, then combined) ==="
for m in $MODELS; do
  $PY core/analyze/make_rq1_csv.py --model "$m" --suite "$SUITE" >/dev/null || echo "  FAILED: $m"
done
$PY core/analyze/make_rq1_scale.py --suite "$SUITE" >/dev/null || echo "  FAILED: rq1_scale"

echo "=== 3/5 RQ2 + word-class information analysis ==="
$PY core/analyze/make_rq2_csv.py --suite "$SUITE" >/dev/null || echo "  FAILED: rq2 csv"
$PY core/analyze/make_ablation_csv.py --suite "$SUITE" >/dev/null 2>&1 || echo "  note: ablation csv not built"
# NO --suite here: this script does not take one, and passing it made argparse exit
# 2 on every run. The failure was swallowed by `2>&1 >/dev/null || echo note`, so the
# word-class table silently kept 2-episode-era numbers for weeks — it still said
# OpenVLA-OFT was unmoved by antonyms at n=12 when the real grid says 55/60 at n=60.
# A generated table that quietly keeps its previous contents is worse than no table.
$PY core/analyze/instruction_information.py \
  || { echo "!! instruction_information.py FAILED — paper/instruction_information.csv is STALE"; exit 1; }

# RQ3 AND THE FIGURES BELONG HERE. They were left out, so while the tables tracked
# the current grid the figures sat at their 2026-07-24 build for a week and
# rq3_divergence.csv with them — and one of those figures turned out to be partly
# synthetic (see git 9c517ab). A refresh that regenerates only what is cheap is how
# a paper ends up internally inconsistent.
echo "=== 4/5 RQ3 trajectories + figures ==="
$PY core/analyze/kinematic_divergence.py >/dev/null 2>&1 || echo "  FAILED: rq3"
for f in plot_rq1 plot_rq2 plot_concepts; do
  $PY core/analyze/$f.py >/dev/null 2>&1 || echo "  FAILED: $f"
done
# plot_rq4 and make_qualitative_grid are deliberately NOT here: RQ4 is not in the
# paper, and the qualitative grid must not be rebuilt until real rollout videos
# exist for every condition it claims to show.

echo "=== 5/5 tables + build check ==="
$PY core/analyze/make_tables.py 2>&1 | grep -E "wrote|scene-fixed" | sed 's/^/  /'
python3 paper/check_tex.py || { echo "!! paper would not build -- fix the above"; exit 1; }

echo
echo "=== scene-fixed verdicts (causal claims valid only where this passes) ==="
for m in $MODELS; do printf "  %-12s %s\n" "$m" "${VERDICT[$m]}"; done
echo
echo "Next: rebuild the PDF off-cluster (no TeX toolchain here). The self-contained"
echo "source is rebuilt by:  bash core/run/make_submission_zip.sh"
