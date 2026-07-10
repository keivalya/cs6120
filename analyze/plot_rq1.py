#!/usr/bin/env python
"""analyze/plot_rq1.py — figures for RQ1 (§11).

- report/rq1_scale.png: CSS(blank), CSS(nonsense), OAR(wrong_task) per model,
  ordered by params — the RQ1.3 "does reliance change with scale?" figure.
- report/rq1_causal_bars.png: per-model TSR by condition (original vs destructive
  perturbations), showing the causal collapse.

Reads report/rq1_scale.csv + report/rq1_causal_<model>.csv. No fabrication;
plots only models that have CSVs.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1] / "report"


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


scale = read_csv(REPORT / "rq1_scale.csv")
labels = [f"{r['model']}\n({r['params_B']}B)" for r in scale]

# ---- Figure 1: CSS + OAR vs scale ---------------------------------------- #
fig, ax = plt.subplots(figsize=(7, 4.2))
x = np.arange(len(scale))
w = 0.25
css_b = [float(r["CSS_blank"] or 0) for r in scale]
css_n = [float(r["CSS_nonsense"] or 0) for r in scale]
oar = [float(r["OAR_wrong_task"] or 0) for r in scale]
ax.bar(x - w, css_b, w, label="CSS(blank)", color="#2166ac")
ax.bar(x, css_n, w, label="CSS(nonsense)", color="#4393c3")
ax.bar(x + w, oar, w, label="OAR(wrong_task)", color="#d6604d")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("score")
ax.set_ylim(0, 1.08)
ax.axhline(1.0, color="grey", ls=":", lw=0.8)
ax.set_title("RQ1.3 — causal language reliance vs model scale\n"
             "CSS≈1 ⇒ instruction is causal;  OAR≈0 ⇒ no visual-shortcut fallback")
ax.legend(loc="center right", fontsize=8)
for i, v in enumerate(css_b):
    ax.text(i - w, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
for i, v in enumerate(css_n):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
for i, v in enumerate(oar):
    ax.text(i + w, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
fig.tight_layout()
fig.savefig(REPORT / "rq1_scale.png", dpi=140)
print("wrote", REPORT / "rq1_scale.png")

# ---- Figure 2: per-model TSR by condition -------------------------------- #
order = ["original", "blank", "nonsense", "wrong_task", "wrong_object",
         "wrong_action", "repeated"]
models = [r["model"] for r in scale]
fig, ax = plt.subplots(figsize=(9, 4.2))
nm = len(models)
bw = 0.8 / nm
colors = ["#1b7837", "#762a83", "#c2a5cf", "#e08214"][:nm]
for mi, model in enumerate(models):
    p = REPORT / f"rq1_causal_{model}.csv"
    if not p.exists():
        continue
    rows = {r["condition"]: r for r in read_csv(p)}
    conds = [c for c in order if c in rows]
    xs = np.arange(len(conds))
    tsr = [float(rows[c]["TSR"] or 0) for c in conds]
    ax.bar(xs + mi * bw, tsr, bw, label=f"{model} ({scale[mi]['params_B']}B)",
           color=colors[mi])
ax.set_xticks(np.arange(len(order)) + bw * (nm - 1) / 2)
ax.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("TSR")
ax.set_ylim(0, 1.0)
ax.set_title("RQ1 — task success rate by instruction condition (LIBERO-Goal, fixed scene)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(REPORT / "rq1_causal_bars.png", dpi=140)
print("wrote", REPORT / "rq1_causal_bars.png")
