#!/usr/bin/env python
"""analyze/pride_wrap.py — PRIDE via LIBERO-Para's own metric (CLAUDE.md §7).

"Wrap, don't reimplement": we materialize our per-paraphrase results
(results/<model>/<suite>/<axis>/seed<k>/task*.jsonl) into the exact
`seed*/eval<N>.json` layout that `LIBERO-Para/metrics/analyze_results.py`
consumes, then call its `load_pride_csv` + `compute_model_pride` unchanged. PRIDE
and PD are therefore computed by the original authors' code, not ours:
  PD_i = 1 - (alpha*SK_i + (1-alpha)*ST_i);  PRIDE = sum(success*PD)/sum(PD)*100
matched by `new_instruction`==our `instruction` (both lowercased).

Public API:
  pride_for_model(model, suite, results_root, axis=None) -> dict
    (sr, n_episodes, n_success, pride, sk_score, st_score, pride_a*). axis=None
    pools all paraphrase axes; pass an axis to score object/action/comp separately.

Run directly for a quick report:
  python analyze/pride_wrap.py --model smolvla [--suite libero_goal]
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "LIBERO-Para" / "metrics"
PARA_AXES = ("para_object", "para_action", "para_compositional")


def _load_para_records(model, suite, results_root, axis=None):
    """Yield our per-paraphrase episode records for the given model (optionally one axis)."""
    root = Path(results_root) / model / suite
    axes = [axis] if axis else PARA_AXES
    for ax in axes:
        adir = root / ax
        if not adir.exists():
            continue
        for seed_dir in sorted(adir.glob("seed*")):
            seed = seed_dir.name
            for tf in sorted(seed_dir.glob("task*.jsonl")):
                for line in tf.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        r["_seed"] = seed
                        yield r


def _materialize(records, tmp_model_dir):
    """Write records into <tmp_model_dir>/seed<k>/eval<tid>.json (analyze_results layout)."""
    from collections import defaultdict
    by_seed_task = defaultdict(list)
    for r in records:
        by_seed_task[(r["_seed"], r["task_id"])].append(r)
    for (seed, tid), recs in by_seed_task.items():
        d = tmp_model_dir / seed
        d.mkdir(parents=True, exist_ok=True)
        episodes = [{
            "paraphrased_instruction": r["instruction"],
            "success": bool(r["success"]),
            "axis": r.get("axis"),
            "operation": r.get("operation"),
        } for r in recs]
        (d / f"eval{tid}.json").write_text(json.dumps({"eval_id": tid, "episodes": episodes}))


def pride_for_model(model, suite="libero_goal", results_root=None, axis=None,
                    pride_csv=None):
    """Compute PRIDE for a model (optionally one axis) via LIBERO-Para's code."""
    results_root = results_root or (REPO_ROOT / "results")
    pride_csv = str(pride_csv or (METRICS_DIR / "libero_para_metadata.csv"))
    sys.path.insert(0, str(METRICS_DIR))
    import analyze_results as AR  # noqa: E402

    records = list(_load_para_records(model, suite, results_root, axis))
    if not records:
        return None
    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td) / model
        _materialize(records, model_dir)
        lookup = AR.load_pride_csv(pride_csv)
        return AR.compute_model_pride(str(model_dir), lookup)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    print(f"=== PRIDE for {args.model} / {args.suite} ===")
    print("overall:", pride_for_model(args.model, args.suite, args.results_root))
    for ax in PARA_AXES:
        print(f"{ax}:", pride_for_model(args.model, args.suite, args.results_root, axis=ax))


if __name__ == "__main__":
    main()
