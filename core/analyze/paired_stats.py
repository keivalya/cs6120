#!/usr/bin/env python3
"""core/analyze/paired_stats.py — the test the scene-fixed design actually earns.

Every perturbed rollout is run from the SAME initial simulator state as its
`original` counterpart, keyed by (task_id, episode, seed) and verified by
reset_state_hash. That makes each comparison a matched pair, not two independent
samples, and paired data has a paired test: McNemar on the discordant pairs.

This matters both ways. It is more powerful — an unpaired interval over two
proportions throws away the pairing and widens the interval for no reason — and it
is more honest: it reports how many episodes actually FLIPPED, which is the
quantity a reader cares about when we claim a policy "ignores the verb".

Pure stdlib on purpose. These numbers end up in the paper, and a dependency that
resolves differently on another machine is a reproducibility problem.
"""
from __future__ import annotations

import math
import random


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value.

    b = pairs the baseline won (original succeeded, perturbed failed)
    c = pairs the perturbation won (original failed, perturbed succeeded)

    Conditional on n=b+c discordant pairs, b ~ Binomial(n, 0.5) under the null.
    The exact form is used rather than the chi-square approximation because our
    smallest cells have single-digit discordant counts, where the approximation
    is not trustworthy. Concordant pairs carry no information about the
    difference and correctly drop out.
    """
    n = b + c
    if n == 0:
        return 1.0  # nothing ever flipped: no evidence of any difference
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_bootstrap_ci(pairs: list[tuple[int, int]], iters: int = 10000,
                        alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap CI for (baseline rate - perturbed rate), in points.

    Resamples PAIRS, not episodes, so the initial-state matching is preserved in
    every resample. `pairs` is [(baseline_success, perturbed_success), ...].
    """
    n = len(pairs)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)  # fixed: the paper must not move between builds
    deltas = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            s += a - b
        deltas.append(100.0 * s / n)
    deltas.sort()
    lo = deltas[int((alpha / 2) * iters)]
    hi = deltas[min(iters - 1, int((1 - alpha / 2) * iters))]
    return (lo, hi)


def summarise(pairs: list[tuple[int, int]]) -> dict:
    """Everything the paper needs for one paired comparison."""
    b = sum(1 for a, p in pairs if a == 1 and p == 0)
    c = sum(1 for a, p in pairs if a == 0 and p == 1)
    lo, hi = paired_bootstrap_ci(pairs)
    return {
        "n_pairs": len(pairs),
        "b_baseline_only": b,
        "c_perturbed_only": c,
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "mcnemar_p": mcnemar_exact(b, c),
    }


if __name__ == "__main__":
    # Sanity checks against values that can be verified by hand.
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(10, 0) - 2 / 1024) < 1e-9   # all ten flips one way
    assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-9          # perfectly balanced
    lo, hi = paired_bootstrap_ci([(1, 1)] * 90 + [(1, 0)] * 10)
    assert lo < 10.0 < hi, (lo, hi)                       # 10 pp drop inside its CI
    print("paired_stats: self-checks pass")
