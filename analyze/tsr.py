#!/usr/bin/env python
"""analyze/tsr.py — TSR + ΔTSR from raw per-episode logs (CLAUDE.md §7).

Reads results/<model>/<suite>/<condition>/seed<k>/summary.json (aggregate) and/or
episodes.jsonl (raw). Computes TSR per (model,suite,condition), mean±std over
seeds, and ΔTSR = TSR(original) − TSR(condition) in percentage points.
Numbers come only from raw logs, never screen output (§7).
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_summaries(results_root: Path, model: str, suite: str):
    """Return {condition: {seed: (n_success, n_total)}}."""
    out = defaultdict(dict)
    base = results_root / model / suite
    if not base.exists():
        return out
    for cond_dir in sorted(base.iterdir()):
        if not cond_dir.is_dir():
            continue
        for seed_dir in sorted(cond_dir.glob("seed*")):
            s = seed_dir / "summary.json"
            if not s.exists():
                continue
            d = json.loads(s.read_text())
            if d.get("tsr") is None:
                continue
            seed = int(seed_dir.name.replace("seed", ""))
            out[cond_dir.name][seed] = (d["n_success"], d.get("n_total_episodes", d["n_episodes"]))
    return out


def tsr_table(summaries):
    """{condition: {'tsr_mean','tsr_std','n_total','seeds'}} pooled over seeds."""
    import statistics
    rows = {}
    for cond, per_seed in summaries.items():
        per_seed_tsr = [ns / nt for (ns, nt) in per_seed.values() if nt]
        n_total = sum(nt for (_, nt) in per_seed.values())
        n_succ = sum(ns for (ns, _) in per_seed.values())
        rows[cond] = {
            "tsr_pooled": (n_succ / n_total) if n_total else None,
            "tsr_mean_over_seeds": (sum(per_seed_tsr) / len(per_seed_tsr)) if per_seed_tsr else None,
            "tsr_std_over_seeds": (statistics.pstdev(per_seed_tsr) if len(per_seed_tsr) > 1 else 0.0),
            "n_total": n_total,
            "seeds": sorted(per_seed),
        }
    return rows


def delta_tsr(rows):
    """ΔTSR (pp) vs original for each condition."""
    base = rows.get("original", {}).get("tsr_pooled")
    out = {}
    for cond, r in rows.items():
        if base is None or r["tsr_pooled"] is None:
            out[cond] = None
        else:
            out[cond] = 100.0 * (base - r["tsr_pooled"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smolvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    summ = load_summaries(Path(args.results_root), args.model, args.suite)
    rows = tsr_table(summ)
    d = delta_tsr(rows)
    print(f"{'condition':16s} {'TSR':>7s} {'±std':>6s} {'ΔTSR(pp)':>9s} {'n':>5s}  seeds")
    for cond in sorted(rows):
        r = rows[cond]
        t = f"{r['tsr_pooled']:.3f}" if r["tsr_pooled"] is not None else "  -  "
        dd = f"{d[cond]:+.1f}" if d[cond] is not None else "  -  "
        print(f"{cond:16s} {t:>7s} {r['tsr_std_over_seeds']:>6.3f} {dd:>9s} {r['n_total']:>5d}  {r['seeds']}")
    return rows, d


if __name__ == "__main__":
    main()
