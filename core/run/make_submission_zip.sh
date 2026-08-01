#!/bin/bash
# core/run/make_submission_zip.sh — build a self-contained LaTeX source bundle.
#
# There is no TeX toolchain anywhere on this cluster (pdflatex/xelatex/lualatex/
# tectonic/latexmk/bibtex/biber all absent, checked 2026-07-31), so the PDF must be
# built elsewhere. This collects exactly what the build needs -- the .tex, the bib,
# the vendored acl.sty/acl_natbib.bst, every generated table, and only the figures
# actually referenced by \includegraphics -- and validates it before zipping.
#
#   bash core/run/make_submission_zip.sh
#   -> paper_submission.zip   (drag into Overleaf, or `unzip` and run latexmk)
#
# Build order once unpacked:  pdflatex paper_acl && bibtex paper_acl && pdflatex paper_acl && pdflatex paper_acl
# Switch \usepackage[review]{acl} to [final] for camera-ready.

set -euo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO"
OUT="$REPO/paper_submission.zip"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/tables"
cp paper/paper_acl.tex paper/references.bib paper/acl.sty paper/acl_natbib.bst "$STAGE"/
cp paper/tables/*.tex "$STAGE"/tables/

# Only the figures the paper actually includes -- paper/ holds stale ones too
# (e.g. rq4_attention.png, from the withdrawn attention analysis).
for f in $(grep -o '\\includegraphics\[[^]]*\]{[^}]*}' paper/paper_acl.tex \
           | sed 's/.*{//;s/}//'); do
  if [ -f "paper/$f" ]; then cp "paper/$f" "$STAGE"/; else echo "MISSING FIGURE: $f"; exit 1; fi
done

# Validate the staged copy, not the working tree: this is what actually ships.
cp paper/check_tex.py "$STAGE"/
python3 "$STAGE"/check_tex.py || { echo "!! staged source would not build"; exit 1; }
rm "$STAGE"/check_tex.py

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" .)
echo
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files)"
