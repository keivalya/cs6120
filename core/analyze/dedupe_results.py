#!/usr/bin/env python
"""core/analyze/dedupe_results.py — drop duplicate episode records from results/*.jsonl.

WHY THIS EXISTS
The per-task runners resume by LINE COUNT. In paraphrase mode one file was resumed
against a shorter prefix than it actually held, so the tail was rolled out a second
time and appended: results/smolvla/libero_goal/para_compositional/seed7/task7.jsonl
carries 321 rows for 300 distinct (para_idx, episode) keys. Unfixed, PRIDE and the
RQ2 aggregates double-count those 21 episodes.

WHICH COPY WINS: the FIRST. The duplicate pairs are not redundant records — they
share an instruction but ran from DIFFERENT initial scenes (different
reset_state_hash, different eef_traj). In that one file rows 0-278 sit entirely on
two init-state hashes, and every duplicate's first row is on that same pair, so
keep-first leaves the file homogeneous with the run it belongs to. Keep-last would
splice a second init-state scheme into the tail of a single condition.

Note this is NOT the handoff's stated cause ("the paraphrase list changed between
runs"): all 21 duplicate pairs have byte-identical instructions, so the paraphrase
list was stable. Resume-by-line-count alone explains it.

Idempotent: a deduped tree reports 0 changes on a second run. Dry-run by default.

    python core/analyze/dedupe_results.py            # report only
    python core/analyze/dedupe_results.py --apply    # rewrite, keeping .bak alongside
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def dedupe_key(row: dict) -> tuple:
    """Paraphrase files are keyed by paraphrase index; causal files by condition."""
    if "para_idx" in row:
        return ("para_idx", row.get("para_idx"), row.get("episode"))
    return ("condition", row.get("condition"), row.get("episode"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="rewrite files (default: report only)")
    ap.add_argument("--results", default=str(ROOT / "data" / "results"))
    args = ap.parse_args()

    total_dropped = 0
    for path in sorted(pathlib.Path(args.results).rglob("task*.jsonl")):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        if not rows:
            continue

        seen: set = set()
        kept, dropped = [], []
        for row in rows:
            key = dedupe_key(row)
            (dropped if key in seen else kept).append(row)
            seen.add(key)
        if not dropped:
            continue

        total_dropped += len(dropped)
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        print(f"{len(dropped):>4} dup rows ({len(rows)} -> {len(kept)})  {rel}")

        # Surface the thing that makes keep-first a judgement call rather than a
        # no-op: if the dropped copies differ from the kept ones, say so.
        differing = sum(
            1 for d in dropped
            if any(k.get("reset_state_hash") != d.get("reset_state_hash")
                   for k in kept if dedupe_key(k) == dedupe_key(d))
        )
        if differing:
            print(f"       {differing} of them ran from a DIFFERENT init scene "
                  f"(distinct rollouts, not redundant records)")

        if args.apply:
            backup = path.with_suffix(".jsonl.bak")
            if not backup.exists():          # never clobber an earlier backup
                backup.write_bytes(path.read_bytes())
            with path.open("w") as fh:
                for row in kept:
                    fh.write(json.dumps(row) + "\n")

    if total_dropped == 0:
        print("no duplicate episode records found")
    elif not args.apply:
        print(f"\n{total_dropped} duplicate rows total — re-run with --apply to rewrite")
    else:
        print(f"\ndropped {total_dropped} duplicate rows (.bak written alongside)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
