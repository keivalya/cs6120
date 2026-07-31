#!/usr/bin/env python
"""perturb/make_instructions.py — causal-probe generator (CLAUDE.md §5).

For each LIBERO-Goal task (original instruction I = the env's task.language, i.e.
exactly what the policy is conditioned on), generate the fixed-scene conditions:
  original, blank, nonsense, wrong_object, wrong_action, wrong_task, repeated.

Object O / action A are derived from the task's bddl `:obj_of_interest` + `:objects`
(NOT by loosely parsing English, per §5). We cross-check the object noun against
LIBERO-Para/metrics/libero_para_metadata.csv when a match exists.

Only the instruction STRING changes; the simulator scene, object poses and init
state are never touched (§5, §12). run_one.py additionally asserts the reset-state
hash is identical across conditions for the same (task, episode).

Every generated string + its provenance is written to
perturb/generated/<suite>.jsonl for audit/reproducibility (§5). When a probe
cannot be made scene-valid (e.g. wrong_object for a fixture-only task like
"turn on the stove"), we emit instruction=null with a skip_reason rather than
invent an unexecutable string (§12: missing > invented).

Usage:
  python perturb/make_instructions.py --suite libero_goal [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCENE_MOVABLE_NOUNS = ["bowl", "cream cheese", "wine bottle", "plate"]

# verb -> a different, still-plausible action verb for the SAME object (§5).
WRONG_VERB = {
    "put": "push",
    "push": "lift",
    "open": "close",
    "turn on": "turn off",
}

# Every noun that can carry task identity in LIBERO-Goal, longest-first so that
# "wine bottle" and "top drawer" are matched before "bottle" / "drawer".
ALL_TASK_NOUNS = ["middle drawer", "top drawer", "wine bottle", "cream cheese",
                  "cabinet", "drawer", "bowl", "stove", "plate", "rack"]
NOUN_MASK = "thing"

# Verbs that can appear at the head of a LIBERO-Goal instruction.
ALL_TASK_VERBS = ["turn on", "turn off", "open", "close", "put", "push", "lift"]

# Out-of-domain word pools for length-matched nonsense (grammatical salad).
NONSENSE_VERBS = ["ponder", "orbit", "whisper", "calibrate", "summon", "dissolve", "archive", "juggle"]
NONSENSE_NOUNS = ["nebula", "syntax", "meadow", "quartz", "lantern", "algorithm", "monsoon", "trombone"]
NONSENSE_PREPS = ["beneath", "around", "beside", "within", "toward", "atop"]
NONSENSE_ADJ = ["velvet", "hollow", "crimson", "distant", "brittle", "luminous"]


def load_para_object_nouns(csv_path: Path) -> set[str]:
    nouns: set[str] = set()
    if not csv_path.exists():
        return nouns
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r.get("high") == "obj":
                nouns.add(r["original_instruction"].strip().lower())
    return nouns


def make_nonsense(instr: str, rng: random.Random) -> str:
    """Grammatical-but-meaningless imperative, length-matched to instr."""
    target = max(1, len(instr.split()))
    words = [rng.choice(NONSENSE_VERBS)]
    while len(words) < target:
        r = rng.random()
        if r < 0.55:
            words.append(rng.choice(NONSENSE_NOUNS))
        elif r < 0.75:
            words.append(rng.choice(NONSENSE_ADJ))
        else:
            words.append(rng.choice(NONSENSE_PREPS))
    return " ".join(words[:target])


def replace_object_noun(instr: str, obj_noun: str, new_noun: str) -> str | None:
    pat = re.compile(re.escape(obj_noun), re.IGNORECASE)
    if not pat.search(instr):
        return None
    return pat.sub(new_noun, instr, count=1)


def replace_verb(instr: str, verb: str, new_verb: str) -> str | None:
    pat = re.compile(r"^\s*" + re.escape(verb), re.IGNORECASE)
    if not pat.search(instr):
        return None
    repl = new_verb.capitalize() if instr[:1].isupper() else new_verb
    return pat.sub(repl, instr, count=1)


def mask_nouns(instr: str) -> str:
    """'put the bowl on the stove' -> 'put the thing on the thing'."""
    out = instr
    for n in ALL_TASK_NOUNS:                      # longest-first, see ALL_TASK_NOUNS
        out = re.compile(re.escape(n), re.IGNORECASE).sub(NOUN_MASK, out)
    return re.sub(r"\s+", " ", out).strip()


def drop_verb(instr: str) -> str | None:
    """'put the bowl on the stove' -> 'the bowl on the stove'. None if no verb found."""
    for v in sorted(ALL_TASK_VERBS, key=len, reverse=True):
        pat = re.compile(r"^\s*" + re.escape(v) + r"\b", re.IGNORECASE)
        if pat.search(instr):
            out = re.sub(r"\s+", " ", pat.sub("", instr)).strip()
            return out or None
    return None


def build_for_task(meta: dict, all_tasks: list[dict], rng: random.Random) -> list[dict]:
    tid = meta["task_id"]
    I = meta["language"].strip()
    obj_noun = meta.get("obj_noun")
    verb = meta.get("verb")
    recs = []

    def rec(condition, instruction, provenance, **extra):
        return {
            "suite": meta["suite"],
            "task_id": tid,
            "task_name": meta["name"],
            "condition": condition,
            "instruction": instruction,
            "original_instruction": I,
            "provenance": provenance,
            **extra,
        }

    recs.append(rec("original", I, "env task.language (unchanged)"))
    recs.append(rec("blank", "", "empty string"))
    recs.append(rec("nonsense", make_nonsense(I, rng),
                    f"length-matched token salad (n_tokens={len(I.split())}), seed-deterministic"))
    recs.append(rec("repeated", f"{I} {I}", "I concatenated with itself"))

    if obj_noun is not None:
        alts = [n for n in SCENE_MOVABLE_NOUNS if n != obj_noun]
        new_noun = alts[tid % len(alts)]
        new_instr = replace_object_noun(I, obj_noun, new_noun)
        if new_instr and new_instr != I:
            recs.append(rec("wrong_object", new_instr,
                            f"replaced object noun {obj_noun!r}->{new_noun!r} (both movable, in-scene)",
                            object_from=obj_noun, object_to=new_noun))
        else:
            recs.append(rec("wrong_object", None,
                            f"could not locate object noun {obj_noun!r} in instruction", skip=True))
    else:
        recs.append(rec("wrong_object", None,
                        "fixture-only task (no alternative movable object makes it scene-valid)", skip=True))

    if verb is not None and verb in WRONG_VERB:
        new_instr = replace_verb(I, verb, WRONG_VERB[verb])
        if new_instr and new_instr != I:
            recs.append(rec("wrong_action", new_instr,
                            f"replaced verb {verb!r}->{WRONG_VERB[verb]!r}",
                            verb_from=verb, verb_to=WRONG_VERB[verb]))
        else:
            recs.append(rec("wrong_action", None, f"verb {verb!r} not at instruction start", skip=True))
    else:
        recs.append(rec("wrong_action", None, f"no wrong-verb mapping for verb={verb!r}", skip=True))

    # --- ablate one word class at a time (the RQ2 information probe) -----------
    # In LIBERO-Goal the unordered NOUN SET uniquely identifies all 10 tasks, while
    # the verb leaves 1.55 bits of residual uncertainty ("put" covers 6 of 10). So
    # conditional on the nouns the verb carries ZERO task information. wrong_action
    # cannot detect verb-blindness for that reason: 7 of its 10 substitutions are
    # put->push, which preserves the noun set and therefore the task identity.
    # These two conditions ablate each word class directly and make the prediction
    # falsifiable:
    #   verb_dropped -> TSR should stay near original (0 bits removed)
    #   nouns_masked -> TSR should collapse to ~0 (all 3.32 bits removed)
    # If verb_dropped collapses, the information account is wrong and the models
    # genuinely need the verb token.
    nm = mask_nouns(I)
    if nm != I:
        recs.append(rec("nouns_masked", nm,
                        f"all task nouns -> {NOUN_MASK!r}; verb and syntax preserved "
                        f"(removes the 3.32 bits that identify the task)"))
    else:
        recs.append(rec("nouns_masked", None, "no task noun found in instruction", skip=True))

    vd = drop_verb(I)
    if vd:
        recs.append(rec("verb_dropped", vd,
                        "leading verb deleted; noun set and syntax otherwise preserved "
                        "(removes 0 bits conditional on the nouns)"))
    else:
        recs.append(rec("verb_dropped", None, "no leading verb found in instruction", skip=True))

    other = all_tasks[(tid + 5) % len(all_tasks)]
    if other["task_id"] == tid:
        other = all_tasks[(tid + 1) % len(all_tasks)]
    recs.append(rec("wrong_task", other["language"].strip(),
                    f"full instruction of goal task {other['task_id']} ({other['name']}); same scene",
                    wrong_task_id=other["task_id"]))
    return recs


# obj_noun (movable manipulated object) + leading verb per task, from bddl
# :obj_of_interest + env task.language. None obj_noun => fixture-only task.
TASK_GROUNDING = {
    0: {"obj_noun": None, "verb": "open"},
    1: {"obj_noun": "bowl", "verb": "put"},
    2: {"obj_noun": "wine bottle", "verb": "put"},
    3: {"obj_noun": None, "verb": "open"},
    4: {"obj_noun": "bowl", "verb": "put"},
    5: {"obj_noun": "plate", "verb": "push"},
    6: {"obj_noun": "cream cheese", "verb": "put"},
    7: {"obj_noun": None, "verb": "turn on"},
    8: {"obj_noun": "bowl", "verb": "put"},
    9: {"obj_noun": "wine bottle", "verb": "put"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path  # noqa

    bd = os.path.join(get_libero_path("bddl_files"), args.suite)
    ts = benchmark.get_benchmark_dict()[args.suite]()

    tasks = []
    for i in range(ts.get_num_tasks()):
        t = ts.get_task(i)
        txt = open(os.path.join(bd, t.name + ".bddl")).read()
        # Use the BENCHMARK task.language (exactly what the policy is conditioned
        # on at eval), NOT the bddl `:language` — they differ (case, and for some
        # tasks semantically, e.g. task 0 bddl "Open the middle layer of the
        # drawer" vs benchmark "open the middle drawer of the cabinet"). Feeding
        # the bddl string as `original` tanks TSR. bddl is still used for O/A.
        lang = t.language.strip()
        ooi = re.search(r"\(:obj_of_interest(.*?)\)", txt, re.S).group(1).split()
        objs = re.search(r"\(:objects(.*?)\)", txt, re.S).group(1)
        objlist = [l.split("-")[0].strip() for l in objs.strip().splitlines() if l.strip()]
        g = TASK_GROUNDING[i]
        tasks.append({
            "suite": args.suite, "task_id": i, "name": t.name, "language": lang,
            "obj_of_interest": ooi, "objects": objlist,
            "obj_noun": g["obj_noun"], "verb": g["verb"],
        })

    para_nouns = load_para_object_nouns(REPO_ROOT / "LIBERO-Para" / "metrics" / "libero_para_metadata.csv")

    out_dir = REPO_ROOT / "perturb" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.suite}.jsonl"

    all_recs = []
    for t in tasks:
        rng = random.Random(f"{args.suite}:{t['task_id']}:{args.seed}")
        recs = build_for_task(t, tasks, rng)
        for r in recs:
            r["para_object_crosscheck"] = (t["language"].strip().lower() in para_nouns)
        all_recs.extend(recs)

    with open(out_path, "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    c = Counter(r["condition"] for r in all_recs)
    skipped = Counter(r["condition"] for r in all_recs if r.get("skip"))
    print(f"Wrote {len(all_recs)} records to {out_path}")
    for cond in ["original", "blank", "nonsense", "wrong_object", "wrong_action", "wrong_task", "repeated", "nouns_masked", "verb_dropped"]:
        print(f"  {cond:14s} n={c[cond]:2d}  skipped={skipped.get(cond,0)}")


if __name__ == "__main__":
    main()
