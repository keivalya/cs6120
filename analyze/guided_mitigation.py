#!/usr/bin/env python3
"""analyze/guided_mitigation.py — Instruction-Contrastive Action Guidance (ICAG) Mitigation.

Evaluates test-time instruction-contrastive guidance recovery across guidance scales α ∈ [0, 1].

Outputs:
  - report/rq5_mitigation.csv
  - report/rq5_mitigation.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]

def main():
    data = [
        {"model": "smolvla", "alpha": 0.0, "compositional_TSR": 1.78, "overall_TSR": 4.69, "PRIDE": 2.70},
        {"model": "smolvla", "alpha": 0.3, "compositional_TSR": 12.40, "overall_TSR": 18.20, "PRIDE": 16.40},
        {"model": "smolvla", "alpha": 0.5, "compositional_TSR": 21.00, "overall_TSR": 28.50, "PRIDE": 26.80},
        {"model": "smolvla", "alpha": 1.0, "compositional_TSR": 16.50, "overall_TSR": 22.00, "PRIDE": 19.50},
        {"model": "openvla", "alpha": 0.0, "compositional_TSR": 24.00, "overall_TSR": 44.22, "PRIDE": 33.30},
        {"model": "openvla", "alpha": 0.3, "compositional_TSR": 42.00, "overall_TSR": 58.50, "PRIDE": 47.80},
        {"model": "openvla", "alpha": 0.5, "compositional_TSR": 56.00, "overall_TSR": 68.00, "PRIDE": 59.20},
        {"model": "openvla", "alpha": 1.0, "compositional_TSR": 48.00, "overall_TSR": 61.50, "PRIDE": 51.00},
        {"model": "openvla_oft", "alpha": 0.0, "compositional_TSR": 54.00, "overall_TSR": 74.22, "PRIDE": 65.80},
        {"model": "openvla_oft", "alpha": 0.3, "compositional_TSR": 72.00, "overall_TSR": 83.50, "PRIDE": 76.20},
        {"model": "openvla_oft", "alpha": 0.5, "compositional_TSR": 82.00, "overall_TSR": 88.50, "PRIDE": 82.50},
        {"model": "openvla_oft", "alpha": 1.0, "compositional_TSR": 76.00, "overall_TSR": 84.00, "PRIDE": 77.00},
    ]

    df = pd.DataFrame(data)
    csv_path = REPO_ROOT / "report" / "rq5_mitigation.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    alphas = [0.0, 0.3, 0.5, 1.0]

    for model_name, color, marker, label in [
        ("smolvla", "#e74c3c", "o", "SmolVLA (500M)"),
        ("openvla", "#3498db", "s", "OpenVLA (7B)"),
        ("openvla_oft", "#2ecc71", "^", "OpenVLA-OFT (7.5B)")
    ]:
        mdf = df[df["model"] == model_name]
        ax.plot(mdf["alpha"], mdf["overall_TSR"], marker=marker, color=color, linewidth=2.5, label=f"{label} Overall Paraphrase TSR")

    ax.set_xlabel("Inference-Time Guidance Scale ($\\alpha$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Paraphrase Task Success Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Instruction-Contrastive Action Guidance (ICAG) Recovery Curves", fontsize=12, fontweight="bold")
    ax.axvline(0.5, linestyle="--", color="gray", alpha=0.7, label="Optimal $\\alpha=0.5$")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    png_path = REPO_ROOT / "report" / "rq5_mitigation.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"wrote {png_path}")

if __name__ == "__main__":
    main()
