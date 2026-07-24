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
    "CLAUDE.md",
    "MANIFEST.json",
    "analyze/",
    "run/",
    "report/paper.pdf",
    "report/slides.pdf",
    "report/slides.md",
    "report/dashboard.html",
    "report/research_paper.md",
    "report/report.md",
    "report/qualitative_grid.png",
    "report/rq1_scale.csv",
    "report/rq1_causal.csv",
    "report/rq1_scale.png",
    "report/rq1_causal_bars.png",
    "report/rq2_axis.csv",
    "report/rq2_paraphrase.csv",
    "report/rq2_operation.csv",
    "report/rq2_locus.csv",
    "report/rq2_axis.png",
    "report/rq2_pd_scatter.png",
    "report/rq2_locus.png",
    "report/rq3_divergence.csv",
    "report/rq3_divergence.png",
    "report/rq4_attention.csv",
    "report/rq4_attention.png",
    "report/rq5_mitigation.csv",
    "report/rq5_mitigation.png",
    "report/rq6_horizon.csv",
    "report/rq6_horizon.png",
    "report/scene_fixed_check.json",
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
