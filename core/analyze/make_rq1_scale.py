#!/usr/bin/env python
"""core/analyze/make_rq1_scale.py — RQ1.3 CSS-vs-scale table (§1, §11).

Concatenates the per-model paper/rq1_causal_<model>.csv files into a single
paper/rq1_causal.csv, and emits paper/rq1_scale.csv: CSS(blank), CSS(nonsense),
and OAR(wrong_task) per model, ordered by parameter count, to answer "does causal
reliance on language grow / shrink with scale?".

OAR(wrong_task) = TSR under the wrong_task instruction (env goal = TRUE task, so a
success == robot ignored the wrong instruction and did the original task). Pulled
from the per-model causal CSV's wrong_task TSR row.

Reads params from configs/models.yaml. No fabrication: a model missing a
per-model CSV is simply omitted (missing > invented, §12).
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "paper"

MODELS = ["smolvla", "openvla", "openvla_oft", "openvla_oft_film"]

with open(REPO_ROOT / "configs" / "models.yaml") as f:
    mcfg = yaml.safe_load(f)


def read_causal(model):
    p = REPORT / f"rq1_causal_{model}.csv"
    if not p.exists():
        return None
    rows = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            rows[r["condition"]] = r
    return rows


combined = []
scale = []
for model in MODELS:
    rows = read_causal(model)
    if not rows:
        continue
    combined.extend(rows.values())
    params = mcfg.get(model, {}).get("params")
    orig = rows.get("original", {})
    blank = rows.get("blank", {})
    nonsense = rows.get("nonsense", {})
    wrong_task = rows.get("wrong_task", {})
    scale.append({
        "model": model,
        "params": params,
        "params_B": ("" if params is None else round(float(params) / 1e9, 3)),
        "TSR_original": orig.get("TSR", ""),
        "CSS_blank": blank.get("CSS", ""),
        "CSS_nonsense": nonsense.get("CSS", ""),
        "OAR_wrong_task": wrong_task.get("TSR", ""),  # success under wrong instr = Ignore
        "n_per_condition": orig.get("n_total", ""),
        "seeds": orig.get("seeds", ""),
    })

# combined causal CSV (all models, one file) — regenerates the §11 rq1_causal.csv
if combined:
    fields = ["model", "suite", "condition", "TSR", "TSR_std_over_seeds",
              "delta_TSR_pp", "CSS", "n_total", "seeds"]
    with open(REPORT / "rq1_causal.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in combined:
            w.writerow({k: r.get(k, "") for k in fields})
    print("wrote", REPORT / "rq1_causal.csv", f"({len(combined)} rows)")

# scale CSV
scale.sort(key=lambda d: (float(d["params"]) if d["params"] else 0.0))
sfields = ["model", "params_B", "TSR_original", "CSS_blank", "CSS_nonsense",
           "OAR_wrong_task", "n_per_condition", "seeds"]
with open(REPORT / "rq1_scale.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sfields)
    w.writeheader()
    for r in scale:
        w.writerow({k: r[k] for k in sfields})
print("wrote", REPORT / "rq1_scale.csv")
print(open(REPORT / "rq1_scale.csv").read())
