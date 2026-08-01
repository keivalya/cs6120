#!/usr/bin/env python
"""core/analyze/make_rq2_csv.py — RQ2 paraphrase tables (§7,§11).

From the per-paraphrase results (results/<model>/<suite>/{para_object,para_action,
para_compositional}/seed*/task*.jsonl) + PRIDE (core/analyze/pride_wrap.py), emit:

- paper/rq2_paraphrase.csv  (RQ2.1): per model × axis + overall — TSR, ΔTSR (pp)
  vs original, PRIDE, n. ΔTSR uses the model's original-condition TSR baseline, and
  is reported TWICE: against the full original pool, and against the scene-matched
  subset of it (see _scene_matched). The two differ by up to ~12 pp because the
  paraphrase and original cells sample different initial scenes. n_scenes is
  reported alongside n: smolvla's 2963 paraphrase episodes cover only 2 distinct
  scenes per task, so n overstates how much scene variation is behind the estimate.
- paper/rq2_axis.csv        (RQ2.2): object-axis vs action-axis ΔTSR side by side.
- paper/rq2_operation.csv   (RQ2.3): per operation (`mid`) — TSR, mean PD, n
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core" / "analyze"))
import pride_wrap as PW  # noqa: E402
import locus as LOCUS  # noqa: E402  (RQ2.4 planning vs execution)

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


def _scenes(recs):
    return {r.get("reset_state_hash") for r in recs if r.get("reset_state_hash")}


def _scene_matched(orig_recs, axis_recs):
    """The original episodes that ran from the SAME initial scenes as this axis.

    delta_TSR against the full original pool is confounded: the paraphrase cells and
    the original cell were produced by different processes and therefore sample
    different scenes (measured: smolvla paraphrases overlap the original pool on only
    2 scenes per task; both 7B models overlap on none). Since the scene is a large
    share of the variance in TSR, some of the reported "paraphrase cost" is really a
    scene difference. Restricting the baseline to the shared scenes removes that part
    of it. Where the overlap is empty or tiny the scene-matched delta is reported as
    blank / on a small n rather than silently substituted — it cannot be computed.
    """
    sc = _scenes(axis_recs)
    return [r for r in orig_recs if r.get("reset_state_hash") in sc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "data" / "results"))
    args = ap.parse_args()
    rr = Path(args.results_root)
    report = REPO_ROOT / "paper"

    para_rows, axis_rows, op_rows, locus_rows = [], [], [], []
    for model in MODELS:
        orig_recs = _records(model, args.suite, "original", rr)
        orig_tsr = _tsr(orig_recs)
        if orig_tsr is None:
            continue
        # RQ2.4 failure locus per axis (planning vs execution), pooled over axes too.
        pl = ex = un = nf = 0
        for axis in AXES:
            if not (rr / model / args.suite / axis).exists():
                continue
            d = LOCUS.locus_distribution(rr, model, args.suite, axis)
            locus_rows.append({"model": model, "axis": axis, "n_failed": d["n_failed"],
                               "planning": d["planning"], "execution": d["execution"],
                               "unclassified": d["unclassified"]})
            pl += d["planning"]; ex += d["execution"]; un += d["unclassified"]; nf += d["n_failed"]
        if nf:
            locus_rows.append({"model": model, "axis": "ALL", "n_failed": nf,
                               "planning": pl, "execution": ex, "unclassified": un})
        axis_tsr = {}
        for axis in AXES:
            recs = _records(model, args.suite, axis, rr)
            if not recs:
                continue
            t = _tsr(recs)
            axis_tsr[axis] = t
            pride = PW.pride_for_model(model, args.suite, str(rr), axis=axis) or {}
            sm = _scene_matched(orig_recs, recs)
            sm_tsr = _tsr(sm)
            para_rows.append({
                "model": model, "axis": axis, "n": len(recs),
                "n_scenes": len(_scenes(recs)),
                "TSR_original": round(orig_tsr, 4), "TSR": round(t, 4),
                "delta_TSR_pp": round((orig_tsr - t) * 100, 2),
                "n_original_scene_matched": len(sm),
                "TSR_original_scene_matched": round(sm_tsr, 4) if sm_tsr is not None else "",
                "delta_TSR_pp_scene_matched": round((sm_tsr - t) * 100, 2) if sm_tsr is not None else "",
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
            sm_all = _scene_matched(orig_recs, allp)
            sm_all_tsr = _tsr(sm_all)
            para_rows.append({
                "model": model, "axis": "ALL", "n": len(allp),
                "n_scenes": len(_scenes(allp)),
                "TSR_original": round(orig_tsr, 4), "TSR": round(_tsr(allp), 4),
                "delta_TSR_pp": round((orig_tsr - _tsr(allp)) * 100, 2),
                "n_original_scene_matched": len(sm_all),
                "TSR_original_scene_matched": round(sm_all_tsr, 4) if sm_all_tsr is not None else "",
                "delta_TSR_pp_scene_matched": (round((sm_all_tsr - _tsr(allp)) * 100, 2)
                                               if sm_all_tsr is not None else ""),
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
           ["model", "axis", "n", "n_scenes", "TSR_original", "TSR", "delta_TSR_pp",
            "n_original_scene_matched", "TSR_original_scene_matched",
            "delta_TSR_pp_scene_matched", "PRIDE", "SK_score", "ST_score"])
    _write(report / "rq2_axis.csv", axis_rows,
           ["model", "TSR_original", "TSR_object", "TSR_action", "dTSR_object_pp", "dTSR_action_pp"])
    _write(report / "rq2_operation.csv", op_rows,
           ["model", "axis", "operation", "n", "TSR", "delta_TSR_pp", "mean_PD"])
    _write(report / "rq2_locus.csv", locus_rows,
           ["model", "axis", "n_failed", "planning", "execution", "unclassified"])


if __name__ == "__main__":
    main()
