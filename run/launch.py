"""run/launch.py — autonomous sweep driver (§8).

Reads configs/grid.yaml, expands cells, computes deterministic run_id per cell,
SKIPS cells whose summary.json already has n_episodes >= requested (idempotent),
activates the correct conda env per model, dispatches run_one.py, catches
per-cell errors -> error.json and CONTINUES (§8.6). Then regenerates report/.

STATUS: stub (implemented in GATE 4).
"""
raise NotImplementedError("Implemented in GATE 4.")
