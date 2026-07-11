#!/usr/bin/env python
"""analyze/plot_rq2.py — RQ2 figures (§11). Reads report/rq2_*.csv.

- report/rq2_axis.png: object-axis vs action-axis ΔTSR (pp) per model (RQ2.2).
- report/rq2_pd_scatter.png: ΔTSR vs mean paraphrase distance (PD) per operation,
  colored by axis (RQ2.3).
- report/rq2_locus.png: planning vs execution split of paraphrase failures (RQ2.4),
  if report/rq2_locus.csv exists.

No fabrication: only plots CSVs that exist.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT = Path(__file__).resolve().parents[1] / "report"


def _read(name):
    p = REPORT / name
    if not p.exists():
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def plot_axis(rows):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    models = [r["model"] for r in rows]
    x = np.arange(len(models)); w = 0.38

    def col(k):
        return [float(r[k]) if r.get(k) not in ("", None) else 0.0 for r in rows]

    ax.bar(x - w / 2, col("dTSR_object_pp"), w, label="object-axis ΔTSR", color="#2166ac")
    ax.bar(x + w / 2, col("dTSR_action_pp"), w, label="action-axis ΔTSR", color="#d6604d")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("ΔTSR (pp) vs original")
    ax.set_title("RQ2.2 — paraphrase fragility by axis (object vs action)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT / "rq2_axis.png", dpi=140)
    print("wrote", REPORT / "rq2_axis.png")


def plot_pd_scatter(rows):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"para_object": "#2166ac", "para_action": "#d6604d", "para_compositional": "#1b7837"}
    for axis, c in colors.items():
        pts = [(float(r["mean_PD"]), float(r["delta_TSR_pp"]))
               for r in rows if r["axis"] == axis and r.get("mean_PD") not in ("", None)]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, c=c, label=axis, alpha=0.7, s=30)
    ax.set_xlabel("mean paraphrase distance PD = 1 − ½(SK+ST)")
    ax.set_ylabel("ΔTSR (pp) vs original")
    ax.set_title("RQ2.3 — degradation vs paraphrase distance, per operation")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT / "rq2_pd_scatter.png", dpi=140)
    print("wrote", REPORT / "rq2_pd_scatter.png")


def plot_locus(rows):
    fig, ax = plt.subplots(figsize=(7, 4.0))
    models = [r["model"] for r in rows]
    x = np.arange(len(models))
    plan = [float(r.get("planning", 0) or 0) for r in rows]
    exe = [float(r.get("execution", 0) or 0) for r in rows]
    ax.bar(x, plan, 0.6, label="planning (wrong what)", color="#762a83")
    ax.bar(x, exe, 0.6, bottom=plan, label="execution (bad how)", color="#e08214")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("failed paraphrase episodes")
    ax.set_title("RQ2.4 — failure locus of paraphrase failures")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORT / "rq2_locus.png", dpi=140)
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
    if not any((axis, op, locus)):
        print("No RQ2 CSVs yet — run analyze/make_rq2_csv.py after RQ2 rollouts.")


if __name__ == "__main__":
    main()
