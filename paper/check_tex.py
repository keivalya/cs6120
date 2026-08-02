#!/usr/bin/env python3
"""report/check_tex.py — catch the errors a LaTeX build would, without a LaTeX build.

There is no TeX toolchain on this cluster (checked: pdflatex/xelatex/lualatex/
tectonic/latexmk/bibtex/biber all absent), so the paper is compiled elsewhere. This
script covers the failure modes that would otherwise only surface there:

  * \\ref / \\label mismatches (undefined references, "??" in the PDF)
  * \\cite keys with no matching bib entry, and bib entries nobody cites
  * \\input / \\includegraphics targets that do not exist on disk
  * @inproceedings without booktitle, @article without journal -- BibTeX drops the
    field silently and emits a citation with no venue
  * unbalanced \\begin/\\end environments
  * stray non-ASCII that a T1/inputenc setup may not render

Exit 0 = clean, 1 = problems found. Run before shipping the tarball.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "acl2023.tex"
BIB_FILES = [HERE / "custom.bib", HERE / "references.bib"]

problems: list[str] = []
notes: list[str] = []



def strip_comments(text: str) -> str:
    # Drop % comments but keep \%
    return re.sub(r"(?<!\\)%.*", "", text)


def main() -> int:
    if not TEX.exists():
        print(f"FATAL: {TEX} not found")
        return 1
    raw = TEX.read_text()
    src = strip_comments(raw)

    # --- inputs, and inline them so refs inside generated tables are seen ---------
    inputs = re.findall(r"\\input\{([^}]+)\}", src)
    for rel in inputs:
        cand = [HERE / rel, HERE / f"{rel}.tex"]
        hit = next((c for c in cand if c.exists()), None)
        if hit is None:
            problems.append(f"\\input{{{rel}}} -> no such file (tried {', '.join(str(c.relative_to(HERE)) for c in cand)})")
        else:
            src += "\n" + strip_comments(hit.read_text())

    # --- graphics ----------------------------------------------------------------
    for rel in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
        cand = [HERE / rel, HERE / "figures" / rel]
        if not any(c.exists() for c in cand):
            problems.append(f"\\includegraphics{{{rel}}} -> missing file")

    # --- labels vs refs ----------------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    refs: set[str] = set()
    for cmd in ("ref", "autoref", "eqref", "pageref", "Cref", "cref"):
        refs |= set(re.findall(rf"\\{cmd}\{{([^}}]+)\}}", src))
    for r in sorted(refs - labels):
        problems.append(f"\\ref{{{r}}} has no \\label")
    for l in sorted(labels - refs):
        notes.append(f"\\label{{{l}}} is never referenced")

    # --- citations vs bib --------------------------------------------------------
    bib_keys, entries = set(), []
    for bib_file in BIB_FILES:
        if bib_file.exists():
            btxt = bib_file.read_text()
            for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", btxt):
                bib_keys.add(m.group(2))
                entries.append((m.group(1).lower(), m.group(2), m.start()))
            bodies = re.split(r"\n(?=@)", btxt)
            for body in bodies:
                m = re.match(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", body.strip())
                if not m:
                    continue
                etype, key = m.group(1).lower(), m.group(2)
                has = lambda f: re.search(rf"\b{f}\s*=", body) is not None
                if etype == "inproceedings" and not has("booktitle"):
                    problems.append(f"bib {key}: @inproceedings without booktitle "
                                    f"(BibTeX will emit no venue)")
                if etype == "inproceedings" and has("journal"):
                    problems.append(f"bib {key}: @inproceedings with a `journal` field "
                                    f"(silently ignored -- use booktitle)")
                if etype == "article" and not has("journal"):
                    problems.append(f"bib {key}: @article without journal")
                if etype == "article" and has("booktitle"):
                    problems.append(f"bib {key}: @article with a `booktitle` field "
                                    f"(silently ignored -- use @inproceedings)")


    cited: set[str] = set()
    for m in re.finditer(r"\\(?:cite|citep|citet|citealp|citeauthor|citeyear)"
                         r"(?:\[[^\]]*\])*\{([^}]+)\}", src):
        cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}
    for k in sorted(cited):
        if k not in bib_keys:
            problems.append(f"\\cite{{{k}}} has no entry in bib files")
    for k in sorted(bib_keys - cited):
        notes.append(f"bib entry {k} is never cited")

    # --- environments ------------------------------------------------------------
    begins = re.findall(r"\\begin\{([^}]+)\}", src)
    ends = re.findall(r"\\end\{([^}]+)\}", src)
    for env in sorted(set(begins) | set(ends)):
        b, e = begins.count(env), ends.count(env)
        if b != e:
            problems.append(f"environment `{env}`: {b} \\begin vs {e} \\end")

    # --- style files the preamble needs -----------------------------------------
    for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", strip_comments(raw)):
        for pkg in (p.strip() for p in m.group(1).split(",")):
            local = HERE / f"{pkg}.sty"
            if pkg in {"acl", "acl2023"} and not local.exists():
                problems.append(f"\\usepackage{{{pkg}}} but {pkg}.sty is not in report/")
    bst = re.search(r"\\bibliographystyle\{([^}]+)\}", src)
    if bst and not (HERE / f"{bst.group(1)}.bst").exists():
        problems.append(f"\\bibliographystyle{{{bst.group(1)}}} but "
                        f"{bst.group(1)}.bst is not in report/")

    # --- non-ASCII ---------------------------------------------------------------
    for i, line in enumerate(raw.splitlines(), 1):
        bad = [c for c in line if ord(c) > 127]
        if bad:
            notes.append(f"line {i}: non-ASCII {sorted(set(bad))!r}")

    # --- report ------------------------------------------------------------------
    print(f"paper_acl.tex: {len(raw.splitlines())} lines, {len(labels)} labels, "
          f"{len(cited)} cite keys, {len(inputs)} \\input, {len(bib_keys)} bib entries")
    if notes:
        print(f"\n-- {len(notes)} note(s) (not fatal) --")
        for n in notes:
            print(f"   . {n}")
    if problems:
        print(f"\n!! {len(problems)} PROBLEM(S) !!")
        for p in problems:
            print(f"   x {p}")
        return 1
    print("\nOK: no build-breaking problems found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
