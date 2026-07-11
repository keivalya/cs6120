#!/usr/bin/env python
"""analyze/make_rq2_csv.py — RQ2 paraphrase tables (§7,§11).

From the per-paraphrase results (results/<model>/<suite>/{para_object,para_action,
para_compositional}/seed*/task*.jsonl) + PRIDE (analyze/pride_wrap.py), emit:

- report/rq2_paraphrase.csv  (RQ2.1): per model × axis + overall — TSR, ΔTSR (pp)
  vs original, PRIDE, n. ΔTSR uses the model's original-condition TSR baseline.
- report/rq2_axis.csv        (RQ2.2): object-axis vs action-axis ΔTSR side by side.
- report/rq2_operation.csv   (RQ2.3): per operation (`mid`) — TSR, mean PD, n
  (PD = 1 - 0.5*(SK+ST), the alpha=0.5 LIBERO-Para distance) for the ΔTSR-vs-PD view.

No fabrication: models/axes with no results are skipped. Baseline original TSR is
read from results/<model>/<suite>/original (pooled over seeds).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "analyze"))
import pride_wrap as PW  # noqa: E402

AXES = ("para_object", "para_action", "para_compositional")
MODELS = ("smolvla", "openvla", "openvla_oft")


def _records(model, suite, cond, results_root):
    base = results_root / model / suite / cond
    if not base.exists():
        return []
    recs = []
    for seed_dir in sorted(base.glob("seed*")):
        for tf in sorted(seed_dir.glob("task*.jsonl")):
            recs += [json.loads(l) for l in tf.read_text().splitlines() if l.strip()]
    return recs


def _tsr(recs):
    return (sum(int(r["success"]) for r in recs) / len(recs)) if recs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    rr = Path(args.results_root)
    report = REPO_ROOT / "report"

    para_rows, axis_rows, op_rows = [], [], []
    for model in MODELS:
        orig_tsr = _tsr(_records(model, args.suite, "original", rr))
        if orig_tsr is None:
            continue
        axis_tsr = {}
        for axis in AXES:
            recs = _records(model, args.suite, axis, rr)
            if not recs:
                continue
            t = _tsr(recs)
            axis_tsr[axis] = t
            pride = PW.pride_for_model(model, args.suite, str(rr), axis=axis) or {}
            para_rows.append({
                "model": model, "axis": axis, "n": len(recs),
                "TSR_original": round(orig_tsr, 4), "TSR": round(t, 4),
                "delta_TSR_pp": round((orig_tsr - t) * 100, 2),
                "PRIDE": pride.get("pride", ""), "SK_score": pride.get("sk_score", ""),
                "ST_score": pride.get("st_score", ""),
            })
            # per-operation (RQ2.3)
            by_op = defaultdict(list)
            for r in recs:
                by_op[r.get("operation", "")].append(r)
            for op, ors in sorted(by_op.items()):
                pds = [1 - 0.5 * (r["keyword_similarity"] + r["structural_similarity"])
                       for r in ors if "keyword_similarity" in r]
                op_rows.append({
                    "model": model, "axis": axis, "operation": op, "n": len(ors),
                    "TSR": round(_tsr(ors), 4),
                    "delta_TSR_pp": round((orig_tsr - _tsr(ors)) * 100, 2),
                    "mean_PD": round(sum(pds) / len(pds), 4) if pds else "",
                })
        # overall (pooled paraphrases)
        allp = [r for axis in AXES for r in _records(model, args.suite, axis, rr)]
        if allp:
            pride_all = PW.pride_for_model(model, args.suite, str(rr)) or {}
            para_rows.append({
                "model": model, "axis": "ALL", "n": len(allp),
                "TSR_original": round(orig_tsr, 4), "TSR": round(_tsr(allp), 4),
                "delta_TSR_pp": round((orig_tsr - _tsr(allp)) * 100, 2),
                "PRIDE": pride_all.get("pride", ""), "SK_score": pride_all.get("sk_score", ""),
                "ST_score": pride_all.get("st_score", ""),
            })
        axis_rows.append({
            "model": model, "TSR_original": round(orig_tsr, 4),
            "TSR_object": round(axis_tsr["para_object"], 4) if "para_object" in axis_tsr else "",
            "TSR_action": round(axis_tsr["para_action"], 4) if "para_action" in axis_tsr else "",
            "dTSR_object_pp": round((orig_tsr - axis_tsr["para_object"]) * 100, 2) if "para_object" in axis_tsr else "",
            "dTSR_action_pp": round((orig_tsr - axis_tsr["para_action"]) * 100, 2) if "para_action" in axis_tsr else "",
        })

    def _write(path, rows, fields):
        if not rows:
            print(f"(no rows for {path.name} — RQ2 results absent)")
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote", path, f"({len(rows)} rows)")

    _write(report / "rq2_paraphrase.csv", para_rows,
           ["model", "axis", "n", "TSR_original", "TSR", "delta_TSR_pp", "PRIDE", "SK_score", "ST_score"])
    _write(report / "rq2_axis.csv", axis_rows,
           ["model", "TSR_original", "TSR_object", "TSR_action", "dTSR_object_pp", "dTSR_action_pp"])
    _write(report / "rq2_operation.csv", op_rows,
           ["model", "axis", "operation", "n", "TSR", "delta_TSR_pp", "mean_PD"])


if __name__ == "__main__":
    main()
