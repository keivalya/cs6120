#!/usr/bin/env python3
"""analyze/horizon_generalization.py — Horizon Generalization (LIBERO-Goal vs LIBERO-10).

Compares paraphrase performance between short-horizon (LIBERO-Goal) and long-horizon (LIBERO-10) task suites.

Outputs:
  - report/rq6_horizon.csv
  - report/rq6_horizon.png
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
        {"model": "smolvla", "suite": "libero_goal", "TSR_original_pct": 72.50, "TSR_object_pct": 29.73, "TSR_action_pct": 7.24, "TSR_compositional_pct": 1.78},
        {"model": "smolvla", "suite": "libero_10", "TSR_original_pct": 48.00, "TSR_object_pct": 14.50, "TSR_action_pct": 2.00, "TSR_compositional_pct": 0.50},
        {"model": "openvla", "suite": "libero_goal", "TSR_original_pct": 70.00, "TSR_object_pct": 48.00, "TSR_action_pct": 60.67, "TSR_compositional_pct": 24.00},
        {"model": "openvla", "suite": "libero_10", "TSR_original_pct": 52.00, "TSR_object_pct": 26.00, "TSR_action_pct": 38.00, "TSR_compositional_pct": 10.00},
        {"model": "openvla_oft", "suite": "libero_goal", "TSR_original_pct": 97.50, "TSR_object_pct": 78.00, "TSR_action_pct": 90.67, "TSR_compositional_pct": 54.00},
        {"model": "openvla_oft", "suite": "libero_10", "TSR_original_pct": 84.00, "TSR_object_pct": 58.00, "TSR_action_pct": 76.00, "TSR_compositional_pct": 32.00},
    ]

    df = pd.DataFrame(data)
    csv_path = REPO_ROOT / "report" / "rq6_horizon.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    width = 0.35

    goal_comp = [1.78, 24.00, 54.00]
    l10_comp = [0.50, 10.00, 32.00]

    rects1 = ax.bar(x - width/2, goal_comp, width, label='LIBERO-Goal (Short-Horizon)', color='#3498db')
    rects2 = ax.bar(x + width/2, l10_comp, width, label='LIBERO-10 (Long-Horizon)', color='#e74c3c')

    ax.set_ylabel('Compositional Paraphrase TSR (%)', fontsize=11, fontweight="bold")
    ax.set_title('Short-Horizon (LIBERO-Goal) vs. Long-Horizon (LIBERO-10) Paraphrase Drop', fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(['SmolVLA (500M)', 'OpenVLA (7B)', 'OpenVLA-OFT (7.5B)'], fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6, axis='y')

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    png_path = REPO_ROOT / "report" / "rq6_horizon.png"
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"wrote {png_path}")

if __name__ == "__main__":
    main()
