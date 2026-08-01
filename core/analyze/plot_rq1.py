#!/bin/env python
"""core/analyze/plot_rq1.py — publication-grade figures for RQ1.

- paper/rq1_scale.png: CSS(blank), CSS(nonsense), OAR(wrong_task) per model.
- paper/rq1_causal_bars.png: per-model Task Success Rate (TSR) across perturbation conditions.
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

REPORT = Path(__file__).resolve().parents[2] / "paper"


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


scale = read_csv(REPORT / "rq1_scale.csv")
labels = [f"{r['model']}\n({r['params_B']}B)" for r in scale]

# ---- Figure 1: CSS + OAR vs scale ---------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
x = np.arange(len(scale))
w = 0.22
css_b = [float(r["CSS_blank"] or 0) for r in scale]
css_n = [float(r["CSS_nonsense"] or 0) for r in scale]
oar = [float(r["OAR_wrong_task"] or 0) for r in scale]

b1 = ax.bar(x - w, css_b, w, label="CSS (Blank Instruction)", color="#2b5c8f", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x, css_n, w, label="CSS (Nonsense Instruction)", color="#4682b4", edgecolor="black", linewidth=0.5)
b3 = ax.bar(x + w, oar, w, label="OAR (Wrong Task)", color="#d95f02", edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontweight="bold")
ax.set_ylabel("Metric Score", fontweight="bold")
ax.set_ylim(0, 1.15)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
ax.set_title("RQ1.3: Causal Language Reliance vs. Model Scale & Fine-Tuning", pad=12, fontweight="bold")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

for rect in b1:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for rect in b2:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
for rect in b3:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

fig.tight_layout()
fig.savefig(REPORT / "rq1_scale.png", dpi=300)
print("wrote", REPORT / "rq1_scale.png")

# ---- Figure 2: per-model TSR by condition -------------------------------- #
order = ["original", "blank", "nonsense", "wrong_task", "wrong_object", "wrong_action", "repeated"]
models = [r["model"] for r in scale]
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
nm = len(models)
bw = 0.8 / nm
colors = ["#2b5c8f", "#7570b3", "#1b9e77"][:nm]

for mi, model in enumerate(models):
    p = REPORT / f"rq1_causal_{model}.csv"
    if not p.exists():
        continue
    rows = {r["condition"]: r for r in read_csv(p)}
    conds = [c for c in order if c in rows]
    xs = np.arange(len(conds))
    tsr = [float(rows[c]["TSR"] or 0) for c in conds]
    bars = ax.bar(xs + mi * bw, tsr, bw, label=f"{model} ({scale[mi]['params_B']}B)",
                  color=colors[mi], edgecolor="black", linewidth=0.5)
    for rect in bars:
        h = rect.get_height()
        if h > 0.01:
            ax.annotate(f"{h*100:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(np.arange(len(order)) + bw * (nm - 1) / 2)
ax.set_xticklabels([c.replace("_", "\n") for c in order], fontweight="bold")
ax.set_ylabel("Task Success Rate (TSR)", fontweight="bold")
ax.set_ylim(0, 1.12)
ax.set_title("RQ1: Task Success Rate Across Language Perturbation Conditions", pad=12, fontweight="bold")
ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
fig.tight_layout()
fig.savefig(REPORT / "rq1_causal_bars.png", dpi=300)
print("wrote", REPORT / "rq1_causal_bars.png")
