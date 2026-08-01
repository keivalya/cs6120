#!/usr/bin/env python
"""core/analyze/make_rq1_csv.py — write paper/rq1_causal.csv from raw logs (§7,§11).

Combines core/analyze/tsr.py (TSR/ΔTSR) and core/analyze/css.py (CSS, Follow/Ignore/Fail,
OAR) into a single per-condition CSV, with the exact grid recorded in a companion
caption file. Numbers come only from raw per-episode logs. Missing cells stay
blank — never invented (§12).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "core" / "analyze"))
import css as CSS  # noqa: E402
import tsr as TSR  # noqa: E402

CAUSAL_ORDER = ["original", "blank", "nonsense", "wrong_object", "wrong_action",
                "wrong_task", "repeated"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "data" / "results"))
    args = ap.parse_args()
    rr = Path(args.results_root)

    summ = TSR.load_summaries(rr, args.model, args.suite)
    rows = TSR.tsr_table(summ)
    dtsr = TSR.delta_tsr(rows)
    cssres = CSS.css(rr, args.model, args.suite)

    # RQ1 is the causal-probe table only. Paraphrase conditions (para_*) belong to
    # RQ2 and are reported from raw task*.jsonl in paper/rq2_paraphrase.csv; their
    # per-condition summary.json can lag the full episode set, so we never source
    # paraphrase numbers here (avoids the stale-summary discrepancy).
    present = [c for c in CAUSAL_ORDER if c in rows]
    # Per-model file (RQ1.3 runs multiple models); a combined rq1_causal.csv is
    # assembled by core/analyze/make_rq1_scale.py from these per-model CSVs.
    out = REPO_ROOT / "paper" / f"rq1_causal_{args.model}.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "suite", "condition", "TSR", "TSR_std_over_seeds",
                    "delta_TSR_pp", "CSS", "n_total", "seeds"])
        for cond in present:
            r = rows[cond]
            css_val = ""
            if cond in ("blank", "nonsense") and cssres.get(cond):
                css_val = "" if cssres[cond]["css"] is None else f"{cssres[cond]['css']:.4f}"
            w.writerow([
                args.model, args.suite, cond,
                "" if r["tsr_pooled"] is None else f"{r['tsr_pooled']:.4f}",
                f"{r['tsr_std_over_seeds']:.4f}",
                "" if dtsr[cond] is None else f"{dtsr[cond]:.2f}",
                css_val, r["n_total"], ";".join(map(str, r["seeds"])),
            ])
    # caption / grid provenance (§11)
    seeds = sorted({s for r in rows.values() for s in r["seeds"]})
    n_ep = max((r["n_total"] for r in rows.values()), default=0)
    (REPO_ROOT / "paper" / "rq1_causal_caption.txt").write_text(
        f"RQ1 causal, model={args.model}, suite={args.suite}. "
        f"Conditions={present}. Seeds={seeds}. Total episodes/condition vary; see n_total. "
        f"CSS=(TSR_orig-TSR_cond)/max(TSR_orig,eps) for blank/nonsense. "
        f"See paper/scene_fixed_check.json for the fixed-scene proof (§5/§12).\n"
    )
    print(f"wrote {out}")
    print(open(out).read())


if __name__ == "__main__":
    main()
