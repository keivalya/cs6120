#!/usr/bin/env python3
"""analyze/plot_rq4.py — Generate report/rq4_attention.png from report/rq4_attention.csv (RQ4).

Plots layer-wise noun attention, verb attention, and AAR (Noun / Verb attention ratio)
across models (SmolVLA, OpenVLA, OpenVLA-OFT) measured on real forward passes.
"""
import csv
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "report"
SCRATCH_DIR = Path("/scratch/pandya.kei")

def find_csv():
    candidates = [
        REPORT_DIR / "rq4_attention.csv",
        SCRATCH_DIR / "rq4_attention.csv",
    ]
    for c in candidates:
        if c.exists():
            with open(c) as f:
                lines = f.readlines()
            if len(lines) > 1:
                return c
    return candidates[0]

def main():
    csv_path = find_csv()
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    data = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r["model"].strip()
            if m not in data:
                data[m] = {"layer": [], "noun": [], "verb": [], "aar": []}
            data[m]["layer"].append(int(r["layer"]))
            data[m]["noun"].append(float(r["noun_attention"]))
            data[m]["verb"].append(float(r["verb_attention"]))
            data[m]["aar"].append(float(r["AAR_noun_over_verb"]))

    print(f"Loaded models for plotting from {csv_path}: {list(data.keys())}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    colors = {
        "smolvla": "#1f77b4",
        "openvla": "#ff7f0e",
        "openvla_oft": "#2ca02c"
    }
    display_names = {
        "smolvla": "SmolVLA (0.45B)",
        "openvla": "OpenVLA (7B)",
        "openvla_oft": "OpenVLA-OFT (7.5B)"
    }

    # Left plot: Noun vs Verb Attention Mass to Visual Tokens
    for m in data:
        layers = data[m]["layer"]
        c = colors.get(m, "blue")
        name = display_names.get(m, m)
        ax1.plot(layers, data[m]["noun"], "-", color=c, linewidth=2.2, label=f"{name} (Noun)")
        ax1.plot(layers, data[m]["verb"], "--", color=c, linewidth=1.5, alpha=0.7, label=f"{name} (Verb)")

    ax1.set_xlabel("Transformer Layer Index", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mean Cross-Attention Mass to Visual Tokens", fontsize=11, fontweight="bold")
    ax1.set_title("Layer-wise Cross-Attention Mass (Noun vs Verb)", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Right plot: AAR (Noun / Verb Ratio)
    for m in data:
        layers = data[m]["layer"]
        c = colors.get(m, "red")
        name = display_names.get(m, m)
        ax2.plot(layers, data[m]["aar"], "o-", color=c, linewidth=2.2, markersize=4, label=name)

    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.2, label="Parity (1.0)")
    ax2.set_xlabel("Transformer Layer Index", fontsize=11, fontweight="bold")
    ax2.set_ylabel("AAR (A_noun / A_verb)", fontsize=11, fontweight="bold")
    ax2.set_title("Attention Allocation Ratio (AAR = Noun / Verb)", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    for out in [REPORT_DIR / "rq4_attention.png", SCRATCH_DIR / "rq4_attention.png"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print(f"Successfully generated plot at {out}")
    plt.close(fig)

if __name__ == "__main__":
    main()
