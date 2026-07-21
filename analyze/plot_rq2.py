#!/bin/env python
"""analyze/plot_rq2.py — publication-grade RQ2 figures. Reads report/rq2_*.csv.

- report/rq2_axis.png: object-axis vs action-axis ΔTSR (pp) per model.
- report/rq2_pd_scatter.png: ΔTSR vs mean paraphrase distance (PD) per operation.
- report/rq2_locus.png: planning vs execution failure breakdown per model and axis.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font="sans-serif")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
})

REPORT = Path(__file__).resolve().parents[1] / "report"


def _read(name):
    p = REPORT / name
    if not p.exists():
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def plot_axis(rows):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    models = [r["model"] for r in rows]
    x = np.arange(len(models))
    w = 0.32

    def col(k):
        return [float(r[k]) if r.get(k) not in ("", None) else 0.0 for r in rows]

    b1 = ax.bar(x - w / 2, col("dTSR_object_pp"), w, label="Object-Axis Performance Drop (ΔTSR pp)",
                color="#2b5c8f", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, col("dTSR_action_pp"), w, label="Action-Axis Performance Drop (ΔTSR pp)",
                color="#d95f02", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylabel("Drop vs Original (Percentage Points)", fontweight="bold")
    ax.set_title("RQ2.2: Paraphrase Degradation by Linguistic Axis (Object vs. Action)", pad=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    for rect in b1:
        h = rect.get_height()
        ax.annotate(f"-{h:.1f}pp", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax.annotate(f"-{h:.1f}pp", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(REPORT / "rq2_axis.png", dpi=300)
    print("wrote", REPORT / "rq2_axis.png")


def plot_pd_scatter(rows):
    fig, ax = plt.subplots(figsize=(8, 5.0), dpi=300)
    colors = {"para_object": "#2b5c8f", "para_action": "#d95f02", "para_compositional": "#7570b3"}
    markers = {"para_object": "o", "para_action": "s", "para_compositional": "^"}
    labels = {"para_object": "Object Paraphrase", "para_action": "Action Paraphrase", "para_compositional": "Compositional"}

    for axis, c in colors.items():
        pts = [(float(r["mean_PD"]), float(r["delta_TSR_pp"]))
               for r in rows if r["axis"] == axis and r.get("mean_PD") not in ("", None)]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=c, marker=markers[axis], label=labels[axis], alpha=0.8, s=50, edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Mean Paraphrase Distance $PD = 1 - \\frac{1}{2}(S_K + S_T)$", fontweight="bold")
    ax.set_ylabel("Performance Drop ΔTSR (Percentage Points)", fontweight="bold")
    ax.set_title("RQ2.3: Degradation vs. Paraphrase Distance Across Operations", pad=12, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(REPORT / "rq2_pd_scatter.png", dpi=300)
    print("wrote", REPORT / "rq2_pd_scatter.png")


def plot_locus(rows):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    # Filter to overall model rows
    all_rows = [r for r in rows if r.get("axis") == "ALL"]
    if not all_rows:
        all_rows = rows
    models = [r["model"] for r in all_rows]
    x = np.arange(len(models))
    w = 0.5

    plan = [float(r.get("planning", 0) or 0) for r in all_rows]
    exe = [float(r.get("execution", 0) or 0) for r in all_rows]
    totals = [p + e for p, e in zip(plan, exe)]

    b1 = ax.bar(x, plan, w, label="Planning Failure (Initial Alignment)", color="#7570b3", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x, exe, w, bottom=plan, label="Execution Failure (Mid-Trajectory)", color="#1b9e77", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight="bold")
    ax.set_ylabel("Number of Failed Episode Rollouts", fontweight="bold")
    ax.set_title("RQ2.4: Failure Locus Breakdown (Planning vs. Execution)", pad=12, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    for i in range(len(models)):
        t = totals[i]
        p_pct = (plan[i] / t * 100) if t > 0 else 0
        ax.annotate(f"Planning: {p_pct:.1f}%\n({int(plan[i])}/{int(t)})", xy=(x[i], t), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    fig.tight_layout()
    fig.savefig(REPORT / "rq2_locus.png", dpi=300)
    print("wrote", REPORT / "rq2_locus.png")


def main():
    axis = _read("rq2_axis.csv")
    op = _read("rq2_operation.csv")
    locus = _read("rq2_locus.csv")
    if axis:
        plot_axis(axis)
    if op:
        plot_pd_scatter(op)
    if locus:
        plot_locus(locus)


if __name__ == "__main__":
    main()
