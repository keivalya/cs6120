"""run/para_util.py — load per-task paraphrase items for RQ2 eval (shared by runners).

Reads perturb/generated/<suite>_paraphrases.jsonl (produced by
perturb/make_paraphrases.py) and returns the paraphrases for one (task, axis),
optionally capped per axis with the same operation-stratified round-robin as the
generator (so a capped 7B run spans the PD range). Each item carries the fields the
runners persist for PRIDE/locus: instruction, keyword_similarity, structural_similarity,
operation, para_idx.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _stratified(rows, k, rng):
    if k is None or len(rows) <= k:
        return rows
    groups = defaultdict(list)
    for r in rows:
        groups[r.get("operation", "")].append(r)
    for g in groups.values():
        rng.shuffle(g)
    out, order = [], sorted(groups)
    while len(out) < k and any(groups[g] for g in order):
        for g in order:
            if groups[g]:
                out.append(groups[g].pop())
                if len(out) >= k:
                    break
    return out


def load_task_paraphrases(suite, task_id, axis, max_per_axis=None, seed=0):
    """Return list of paraphrase dicts for (task_id, axis), capped + stratified."""
    path = REPO_ROOT / "perturb" / "generated" / f"{suite}_paraphrases.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if int(r["task_id"]) == int(task_id) and r["axis"] == axis:
            rows.append(r)
    rng = random.Random(f"{suite}:{task_id}:{axis}:{seed}")
    sampled = _stratified(rows, max_per_axis, rng)
    # deterministic order for reproducible eval + resumability
    sampled.sort(key=lambda r: r.get("para_idx", 0))
    return sampled
