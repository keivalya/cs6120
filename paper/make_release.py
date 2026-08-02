#!/usr/bin/env python3
"""report/make_release.py — Open-Source Release Archive Builder.

Bundles project code, quantitative CSVs, generated plots, manifest files,
the manuscript PDF (paper.pdf), presentation slides (slides.pdf), and the interactive
dashboard (dashboard.html) into a compressed release tarball.

Outputs:
  - CS6120_VLA_Language_Grounding_Release.tar.gz
"""
import tarfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = REPO_ROOT / "CS6120_VLA_Language_Grounding_Release.tar.gz"

FILES_TO_INCLUDE = [
    "README.md",
    "MANIFEST.json",
    "core/analyze/",
    "core/run/",
    "paper/acl2023.pdf",
    "paper/acl2023.tex",
    "paper/acl2023.sty",
    "paper/acl_natbib.bst",
    "paper/anthology.bib",
    "paper/custom.bib",
    "paper/dashboard.html",
    "paper/figures/fig_information.png",
    "paper/figures/fig_removal.png",
    "paper/figures/fig_verbsplit.png",
    "paper/figures/rq1_scale.png",
    "paper/figures/rq1_causal_bars.png",
    "paper/figures/rq2_axis.png",
    "paper/figures/rq2_pd_scatter.png",
    "paper/figures/rq2_locus.png",
    "paper/figures/rq3_divergence.png",
    "paper/data/rq1_scale.csv",
    "paper/data/rq1_causal.csv",
    "paper/data/rq2_axis.csv",
    "paper/data/rq2_paraphrase.csv",
    "paper/data/rq2_operation.csv",
    "paper/data/rq2_locus.csv",
    "paper/data/rq3_divergence.csv",
    "paper/data/scene_fixed_check.json",
]

def main():
    print(f"[make_release] Creating archive {ARCHIVE_NAME}...", flush=True)
    count = 0
    with tarfile.open(ARCHIVE_NAME, "w:gz") as tar:
        for rel_path in FILES_TO_INCLUDE:
            abs_path = REPO_ROOT / rel_path
            if abs_path.exists():
                tar.add(abs_path, arcname=rel_path)
                count += 1
                print(f"  + added {rel_path}")
            else:
                print(f"  - warning: {rel_path} not found")

    print(f"wrote {ARCHIVE_NAME} ({count} items bundled)")

if __name__ == "__main__":
    main()
