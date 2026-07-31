#!/usr/bin/env python
"""analyze/instruction_information.py — what does the instruction actually carry?

THE ARGUMENT THIS SCRIPT MAKES, AND THE NUMBERS BEHIND IT

The draft reports an "asymmetric grounding gap": VLAs bind object nouns strictly
but treat action verbs as near-optional (OpenVLA-OFT keeps ~97.5% TSR under
wrong_action). Read as a model property, that is a deficiency claim. It is not
supported, because it is exactly what an OPTIMAL agent does on LIBERO-Goal:

  * The unordered NOUN SET is unique across all 10 tasks -> H(task | nouns) = 0.
    Nouns carry the entire 3.32 bits of task identity.
  * The VERB is 6-way ambiguous ("put" covers tasks 1,2,4,6,8,9) ->
    H(task | verb) = 1.55 bits. Conditional on the nouns, the verb carries ZERO.

So verb-insensitivity is the information-theoretically correct policy here, and
wrong_action cannot detect verb-blindness: 7 of its 10 substitutions are
put->push, which leaves the noun set — and therefore the task — intact. Only
tasks 0, 3 and 7 swap a true antonym (open<->close, turn on<->turn off), where
the nouns cannot disambiguate. Those three are the only valid verb probes in the
suite, and this script reports them separately.

The mirror-image problem is wrong_object. It is not one condition but three:
  (a) the perturbed string IS another task's verbatim instruction (task4 ->
      task2's "put the wine bottle on top of the cabinet"),
  (b) a novel but coherent goal ("put the wine bottle on the stove"),
  (c) a DEGENERATE self-reference ("put the bowl in the bowl", "put the plate on
      the plate") — semantically impossible, so failure there is not evidence of
      noun binding at all.
Case (a) is the decisive one, and all three models COMPLY: they perform the task
the perturbed nouns name and never the original. Scoring that against the
original task calls it a 0% failure, when it is precise instruction-following.

CONCLUSION: both halves of the "asymmetric grounding gap" are explained by the
information structure of the benchmark plus correct instruction-following, with
no appeal to a grounding deficit. The residual real finding is narrower and
sharper — on the three genuine antonym pairs, OFT still ignores the verb.

The falsifiable follow-up is the nouns_masked / verb_dropped pair added to
perturb/make_instructions.py: removing the verb should cost ~nothing, masking the
nouns should be catastrophic. If verb_dropped collapses, this account is wrong.

Usage:  python analyze/instruction_information.py [--results_root results]
Writes: report/instruction_information.csv
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

MODELS = ("smolvla", "openvla", "openvla_oft")
# open->close, open->close, turn on->turn off: the verb is contrastive AND the
# noun set cannot disambiguate the goal. The only real verb probes in the suite.
ANTONYM_TASKS = {0, 3, 7}
NOUNS = ["middle drawer", "top drawer", "wine bottle", "cream cheese",
         "cabinet", "drawer", "bowl", "stove", "plate", "rack"]
VERBS = ["turn on", "turn off", "open", "close", "put", "push", "lift"]


def noun_set(s):
    found, rest = [], s.lower()
    for n in sorted(NOUNS, key=len, reverse=True):
        if n in rest:
            found.append(n)
            rest = rest.replace(n, " ")
    return frozenset(found)


def verb_set(s):
    return frozenset(v for v in VERBS if re.search(r"\b" + v + r"\b", s.lower()))


def load(results_root, model, cond):
    base = results_root / model / "libero_goal" / cond
    if not base.exists():
        return []
    out = []
    for p in sorted(base.rglob("task*.jsonl")):
        for line in p.open():
            if line.strip():
                out.append(json.loads(line))
    return out


def tsr(rs):
    return (sum(int(r["success"]) for r in rs) / len(rs)) if rs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default=str(REPO_ROOT / "results"))
    args = ap.parse_args()
    rr = pathlib.Path(args.results_root)
    rows = []

    # ---- 1. information content of the instruction vocabulary -----------------
    truth = {}
    for r in load(rr, "smolvla", "original"):
        truth[r["task_id"]] = r["true_instruction"]
    n_tasks = len(truth)
    ns = collections.Counter(noun_set(v) for v in truth.values())
    vs = collections.Counter(verb_set(v) for v in truth.values())
    H_prior = math.log2(n_tasks)
    # H(task | X) for a deterministic map task->X, uniform prior over tasks
    H_given_nouns = sum(c / n_tasks * math.log2(c) for c in ns.values())
    H_given_verbs = sum(c / n_tasks * math.log2(c) for c in vs.values())
    print(f"LIBERO-Goal instruction vocabulary ({n_tasks} tasks)")
    print(f"  H(task)             = {H_prior:.2f} bits")
    print(f"  H(task | noun set)  = {H_given_nouns:.2f} bits  "
          f"({len(ns)}/{n_tasks} distinct noun sets)")
    print(f"  H(task | verb set)  = {H_given_verbs:.2f} bits  "
          f"(largest collision: {max(vs.values())} tasks share a verb)")
    print(f"  => conditional on the nouns the verb carries "
          f"{H_given_verbs - H_given_nouns:.2f} bits beyond them"
          if H_given_nouns else
          f"  => nouns determine the task; the verb adds 0.00 bits")
    rows.append({"section": "information", "model": "", "subset": "",
                 "n": n_tasks, "value": round(H_given_nouns, 4),
                 "note": "H(task|nouns) bits"})
    rows.append({"section": "information", "model": "", "subset": "",
                 "n": n_tasks, "value": round(H_given_verbs, 4),
                 "note": "H(task|verbs) bits"})

    # ---- 2. wrong_action, split by whether the verb is contrastive ------------
    print("\nwrong_action TSR by verb contrastiveness "
          "(high TSR = the model ignored the swapped verb)")
    print(f"  {'model':<13}{'subset':<11}{'n':>5}{'TSR':>8}{'orig TSR':>10}{'delta pp':>10}")
    for m in MODELS:
        wa, orig = load(rr, m, "wrong_action"), load(rr, m, "original")
        if not wa:
            continue
        for name, in_ant in (("antonym", True), ("near-syn", False)):
            sub = [r for r in wa if (r["task_id"] in ANTONYM_TASKS) == in_ant]
            ob = [r for r in orig if (r["task_id"] in ANTONYM_TASKS) == in_ant]
            if not sub or not ob:
                continue
            t, o = tsr(sub), tsr(ob)
            print(f"  {m:<13}{name:<11}{len(sub):>5}{t:>8.3f}{o:>10.3f}{(o-t)*100:>10.1f}")
            rows.append({"section": "wrong_action_split", "model": m, "subset": name,
                         "n": len(sub), "value": round(t, 4),
                         "note": f"orig={o:.4f} delta_pp={(o-t)*100:.1f}"})

    # ---- 3. wrong_object: does the model do the task the new nouns name? ------
    by_text = {v.strip().lower(): k for k, v in truth.items()}
    print("\nwrong_object compliance: when the perturbed string IS another task's "
          "instruction,\ndoes the model perform THAT task?")
    print(f"  {'model':<13}{'task':<7}{'named':<7}{'n':>4}"
          f"{'did named':>11}{'did original':>14}")
    for m in MODELS:
        for r_task in sorted({r["task_id"] for r in load(rr, m, "wrong_object")}):
            rs = [r for r in load(rr, m, "wrong_object") if r["task_id"] == r_task]
            if not rs:
                continue
            named = by_text.get(rs[0]["instruction"].strip().lower())
            if named is None:
                continue          # only case (a) is decisive; skip (b) and (c)
            did_named = sum(1 for r in rs
                            if named in (r.get("achieved_task_ids") or []))
            did_orig = sum(1 for r in rs if r["success"])
            print(f"  {m:<13}{r_task:<7}{named:<7}{len(rs):>4}"
                  f"{did_named:>11}{did_orig:>14}")
            rows.append({"section": "wrong_object_compliance", "model": m,
                         "subset": f"task{r_task}->task{named}", "n": len(rs),
                         "value": round(did_named / len(rs), 4),
                         "note": f"did_original={did_orig}"})

    # ---- 4. the falsifiable follow-up, if it has been run ---------------------
    print("\nword-class ablation (prediction: verb_dropped ~ original, "
          "nouns_masked ~ 0)")
    any_run = False
    for m in MODELS:
        for cond in ("verb_dropped", "nouns_masked"):
            rs = load(rr, m, cond)
            if not rs:
                continue
            any_run = True
            o = tsr(load(rr, m, "original"))
            print(f"  {m:<13}{cond:<14}n={len(rs):<5}TSR={tsr(rs):.3f}  (original {o:.3f})")
            rows.append({"section": "word_class_ablation", "model": m, "subset": cond,
                         "n": len(rs), "value": round(tsr(rs), 4), "note": f"orig={o:.4f}"})
    if not any_run:
        print("  not yet run — see perturb/generated/libero_goal.jsonl "
              "(conditions generated, episodes pending)")

    out = REPO_ROOT / "report" / "instruction_information.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "model", "subset", "n", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
