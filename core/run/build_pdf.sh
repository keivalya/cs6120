#!/bin/bash
# core/run/build_pdf.sh — build paper/paper_acl.pdf ON THIS CLUSTER.
#
# The standing belief in docs/HANDOFF.txt was that no TeX toolchain exists here and
# the PDF had to be built off-cluster. That was true of the module system and of
# pdflatex/xelatex/latexmk, but not of tectonic: it is a single static binary that
# fetches the TeX packages it needs on first run, and the login nodes have outbound
# HTTPS. So the paper is buildable here after all, which matters because
# paper/check_tex.py is a LINTER -- it reported "no build-breaking problems found"
# on a source that aborted in TeX with
#     Argument of \caption@ydblarg has an extra }
# from an \input inside a \caption. Only a real compile catches that class of bug.
#
#   bash core/run/build_pdf.sh              # build, report page counts
#   bash core/run/build_pdf.sh --strict     # also fail on any Overfull box
#
# ACL long papers allow 8 pages of content; Limitations, references and appendices
# do not count. This script reports where the body ends so the limit is checked
# rather than assumed.
set -uo pipefail
REPO=/home/pandya.kei/CS6120
cd "$REPO" || exit 1

TECTONIC=${TECTONIC:-/scratch/pandya.kei/bin/tectonic}
VER=0.17.0
if [ ! -x "$TECTONIC" ]; then
  echo "=== tectonic not found, installing to $TECTONIC ==="
  mkdir -p "$(dirname "$TECTONIC")" || exit 1
  URL="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40${VER}/tectonic-${VER}-x86_64-unknown-linux-musl.tar.gz"
  TMP=$(mktemp -d)
  curl -sL -o "$TMP/t.tar.gz" "$URL" && tar xzf "$TMP/t.tar.gz" -C "$TMP" \
    && mv "$TMP/tectonic" "$TECTONIC" || { echo "install failed"; exit 1; }
  rm -rf "$TMP"
fi

# Build from the SUBMISSION BUNDLE, not from paper/, so what is checked is exactly
# what ships. make_submission_zip.sh copies only the figures \includegraphics names.
bash core/run/make_submission_zip.sh >/dev/null || { echo "bundle failed"; exit 1; }
BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
unzip -q "$REPO/paper_submission.zip" -d "$BUILD" || exit 1

# A label right before Limitations is how we find the last page of countable body.
python3 - "$BUILD/paper_acl.tex" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
s = s.replace(r"\section*{Limitations}",
              "\\label{ENDOFBODY}\n" + r"\section*{Limitations}", 1)
s = s.replace(r"\end{document}", "\\label{ENDOFDOC}\n" + r"\end{document}", 1)
p.write_text(s)
PY

echo "=== compiling ==="
LOG="$BUILD/out.txt"
( cd "$BUILD" && "$TECTONIC" -X compile paper_acl.tex --keep-intermediates ) >"$LOG" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "!! COMPILE FAILED"
  grep -aE "^error|^!" "$LOG" | head -20
  exit 1
fi

grep -aoE "Overfull \\\\hbox \([0-9.]+pt too wide\)" "$LOG" | sort -u > "$BUILD/over.txt"
NOVER=$(wc -l < "$BUILD/over.txt")
UNDEF=$(grep -acE "Reference .* undefined|Citation .* undefined" "$BUILD/paper_acl.log" 2>/dev/null || echo 0)

# The page is the SECOND braced field of \newlabel, and the match ends on "}",
# so anchoring the digits to end-of-line finds nothing -- capture the group instead.
BODY=$(grep -a "ENDOFBODY" "$BUILD/paper_acl.aux" 2>/dev/null \
       | sed -E 's/.*\{ENDOFBODY\}\{\{[^}]*\}\{([0-9]+)\}.*/\1/' | head -1)
# Total comes from a label at \end{document}, not from the PDF: tectonic writes
# object streams, so /Count is compressed and not greppable.
TOTAL=$(grep -a "ENDOFDOC" "$BUILD/paper_acl.aux" 2>/dev/null \
        | sed -E 's/.*\{ENDOFDOC\}\{\{[^}]*\}\{([0-9]+)\}.*/\1/' | head -1)

cp "$BUILD/paper_acl.pdf" "$REPO/paper/paper_acl.pdf"
echo
echo "  body ends on page : ${BODY:-?}   (ACL long-paper limit is 8)"
echo "  last text page    : ${TOTAL:-?}   (trailing float pages may follow)"
echo "  overfull boxes    : $NOVER"
echo "  undefined refs    : $UNDEF"
echo "  wrote             : paper/paper_acl.pdf"

FAIL=0
[ -n "$BODY" ] && [ "$BODY" -gt 8 ] && { echo "!! OVER THE PAGE LIMIT by $((BODY-8))"; FAIL=1; }
[ "$UNDEF" != "0" ] && { echo "!! undefined references or citations"; FAIL=1; }
if [ "${1:-}" = "--strict" ] && [ "$NOVER" -gt 0 ]; then
  echo "!! overfull boxes (--strict):"; sed 's/^/    /' "$BUILD/over.txt"; FAIL=1
fi
exit $FAIL
