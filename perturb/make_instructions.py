"""perturb/make_instructions.py — causal-probe generator (CLAUDE.md §5).

For each LIBERO-Goal task, derive object O and action A from bddl/task metadata
(cross-checked vs LIBERO-Para/metrics/libero_para_metadata.csv) and emit the
fixed-scene conditions: original, blank, nonsense (length-matched), wrong_object,
wrong_action, wrong_task, repeated. Every string + provenance -> generated/<suite>.jsonl.

Only the instruction string changes; the simulator scene is never touched (§5, §12).

STATUS: stub (implemented in GATE 3).
"""
raise NotImplementedError("Implemented in GATE 3.")
