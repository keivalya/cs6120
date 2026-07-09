"""run/run_one.py — single (model, suite, condition, seed) eval -> logs (§8).

Fixes all seeds; asserts env reset-state hash == the `original` condition's for
the same task+seed (proves scene fixed, §5/§12); streams per-episode records
(success bool, achieved-goal predicates, trajectory summary, instruction,
wall-clock) to disk; writes summary.json atomically; appends to MANIFEST.json
(atomic + flock) with checkpoint hash, repo SHA, env lock hash, counts.

STATUS: stub (implemented in GATE 2).
"""
raise NotImplementedError("Implemented in GATE 2.")
