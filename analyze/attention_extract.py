#!/usr/bin/env python3
"""analyze/attention_extract.py — REAL layer-wise noun-vs-verb attention (RQ4).

This replaces the fabricated analyze/attention_mechanics.py (removed; see
report/KNOWN_ISSUES.txt). That script emitted a closed-form curve
(noun = 0.40 + 0.20*(l/16) + 0.03*sin(l), verb = 0.38*exp(...)) with NO model
in the loop. This script instead measures attention from a real model.

DESIGN (what RQ4 must actually compute)
---------------------------------------
For each LIBERO-Goal instruction, run the model forward on the fixed initial
observation with attention capture ON, then, at every transformer layer l,
average the attention mass that language *noun* tokens vs language *verb* tokens
place on the visual tokens:

    A_noun(l) = mean over noun tokens of (attention to visual tokens)
    A_verb(l) = mean over verb tokens of (attention to visual tokens)
    AAR(l)    = A_noun(l) / A_verb(l)

Output: report/rq4_attention.csv with columns
    layer, noun_attention, verb_attention, AAR_noun_over_verb, n_instructions

INTEGRITY CONTRACT
------------------
This script MUST NOT emit a CSV unless a real model was loaded and produced real
attention tensors. There is deliberately NO synthetic fallback. If a model or its
attention cannot be obtained, it raises SystemExit. Missing > invented (CLAUDE.md
§12).

STATUS: on-device scaffold. The model load path and the noun/verb + visual token
index resolution are the two points that MUST be validated on a real GPU with the
real checkpoints before the numbers are trusted; both are marked below with
`VALIDATE-ON-DEVICE`. Do not paper over them.

Usage:
    python analyze/attention_extract.py --model openvla --suite libero_goal \
        --max_tasks 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _die(msg: str) -> "None":
    # Fail loudly. We never fabricate an attention curve.
    print(f"[attention_extract] FATAL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_instructions(suite: str, max_tasks: int | None):
    """Read the original (uncorrupted) instruction per task from perturb/generated."""
    gen = REPO_ROOT / "perturb" / "generated" / f"{suite}.jsonl"
    if not gen.exists():
        _die(f"missing {gen}; run perturb/make_instructions.py first")
    out = []
    for line in gen.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("condition") == "original" and r.get("instruction"):
            out.append((int(r["task_id"]), r["instruction"]))
    out.sort()
    if max_tasks:
        out = out[:max_tasks]
    if not out:
        _die("no original instructions found")
    return out


def tag_noun_verb_spans(instruction: str):
    """Return (noun_word_set, verb_word_set) for an instruction.

    VALIDATE-ON-DEVICE: reuse the exact tagging already used to build the RQ2
    paraphrase axes so RQ4 and RQ2 talk about the same tokens. perturb/
    make_paraphrases.py distinguishes the object-noun axis from the action-verb
    axis; import and reuse that logic rather than re-tagging here. A spaCy POS
    fallback is acceptable only if it is verified to agree with the RQ2 tagging
    on the LIBERO-Goal instructions.
    """
    try:
        sys.path.insert(0, str(REPO_ROOT / "perturb"))
        import make_paraphrases as MP  # noqa
    except Exception as e:  # pragma: no cover
        _die(f"cannot import perturb/make_paraphrases for noun/verb tagging: {e}")
    # The concrete function names live in make_paraphrases; wire them here once
    # confirmed on-device. Left explicit so a wrong guess cannot silently ship.
    raise NotImplementedError(
        "Wire tag_noun_verb_spans to perturb/make_paraphrases tagging "
        "(VALIDATE-ON-DEVICE) before running.")


def build_model(model_alias: str):
    """Load the real model with attention capture enabled.

    VALIDATE-ON-DEVICE per framework:
      * openvla / openvla_oft: transformers AutoModelForVision2Seq loaded with
        output_attentions=True (or attn_implementation='eager'; 'sdpa' does not
        return attention weights). Attentions come back as a tuple of
        (n_layers) tensors shaped [batch, heads, q_len, k_len].
      * smolvla: register forward hooks on the SmolVLM self-attention modules;
        lerobot's SmolVLAPolicy does not expose output_attentions directly.
    Must return an object exposing .attentions_for(instruction, obs) -> per-layer
    arrays plus the token-type mask (language/visual, noun/verb). This is the
    real work; it is intentionally not stubbed with fake tensors.
    """
    import yaml
    cfg_path = REPO_ROOT / "configs" / "models.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()).get(model_alias)
    if not cfg or not cfg.get("hf_repo"):
        _die(f"model '{model_alias}' has no hf_repo in configs/models.yaml")
    raise NotImplementedError(
        f"Implement real attention capture for framework "
        f"'{cfg.get('framework')}' (VALIDATE-ON-DEVICE). No synthetic fallback.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openvla")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "report" / "rq4_attention.csv"))
    args = ap.parse_args()

    instructions = load_instructions(args.suite, args.max_tasks)
    model = build_model(args.model)  # raises until implemented on-device

    per_layer_noun: dict[int, list] = {}
    per_layer_verb: dict[int, list] = {}
    for tid, text in instructions:
        nouns, verbs = tag_noun_verb_spans(text)
        layer_attn = model.attentions_for(text)  # real tensors, per layer
        for l, attn in layer_attn.items():
            per_layer_noun.setdefault(l, []).append(attn.noun_to_visual(nouns))
            per_layer_verb.setdefault(l, []).append(attn.verb_to_visual(verbs))

    layers = sorted(per_layer_noun)
    if not layers:
        _die("no attention captured; refusing to write an empty/synthetic CSV")
    rows = ["layer,noun_attention,verb_attention,AAR_noun_over_verb,n_instructions"]
    for l in layers:
        n = float(np.mean(per_layer_noun[l]))
        v = float(np.mean(per_layer_verb[l]))
        aar = n / v if v > 0 else float("inf")
        rows.append(f"{l},{n:.4f},{v:.4f},{aar:.4f},{len(per_layer_noun[l])}")
    Path(args.out).write_text("\n".join(rows) + "\n")
    print(f"[attention_extract] wrote {args.out} from {len(instructions)} real forward passes")


if __name__ == "__main__":
    main()
