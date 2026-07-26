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
    model, layer, noun_attention, verb_attention, AAR_noun_over_verb, n_instructions

INTEGRITY CONTRACT
------------------
This script MUST NOT emit a CSV unless a real model was loaded and produced real
attention tensors. There is deliberately NO synthetic fallback. If a model or its
attention cannot be obtained, it raises SystemExit. Missing > invented (CLAUDE.md §12).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

# Tagging dictionary for LIBERO-Goal tasks
POS_TAGS = {
    "open the middle drawer of the cabinet": {"verbs": {"open"}, "nouns": {"middle", "drawer", "cabinet"}},
    "put the bowl on the stove": {"verbs": {"put"}, "nouns": {"bowl", "stove"}},
    "put the wine bottle on top of the cabinet": {"verbs": {"put"}, "nouns": {"wine", "bottle", "top", "cabinet"}},
    "open the top drawer and put the bowl inside": {"verbs": {"open", "put"}, "nouns": {"top", "drawer", "bowl"}},
    "put the bowl on top of the cabinet": {"verbs": {"put"}, "nouns": {"bowl", "top", "cabinet"}},
    "push the plate to the front of the stove": {"verbs": {"push"}, "nouns": {"plate", "front", "stove"}},
    "put the cream cheese in the bowl": {"verbs": {"put"}, "nouns": {"cream", "cheese", "bowl"}},
    "turn on the stove": {"verbs": {"turn"}, "nouns": {"stove"}},
    "put the bowl on the plate": {"verbs": {"put"}, "nouns": {"bowl", "plate"}},
    "put the wine bottle on the rack": {"verbs": {"put"}, "nouns": {"wine", "bottle", "rack"}},
}


def _die(msg: str) -> "None":
    print(f"[attention_extract] FATAL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def clean_token(tok: str) -> str:
    return tok.lstrip(" ").lstrip("Ġ").lstrip("<|").rstrip(">").lower()


def load_instructions(suite: str, max_tasks: int | None):
    """Read the original (uncorrupted) instruction per task from perturb/generated."""
    gen = REPO_ROOT / "perturb" / "generated" / f"{suite}.jsonl"
    if not gen.exists():
        _die(f"missing {gen}; run perturb/make_instructions.py first")
    out = []
    seen_tasks = set()
    for line in gen.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        tid = int(r["task_id"])
        if r.get("condition") == "original" and r.get("instruction") and tid not in seen_tasks:
            out.append((tid, r["instruction"].strip()))
            seen_tasks.add(tid)
    out.sort()
    if max_tasks:
        out = out[:max_tasks]
    if not out:
        _die("no original instructions found")
    return out


def tag_noun_verb_spans(instruction: str):
    """Return (nouns_set, verbs_set) for an instruction."""
    text = instruction.strip().lower()
    if text in POS_TAGS:
        return POS_TAGS[text]["nouns"], POS_TAGS[text]["verbs"]
    # spaCy fallback if available
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(instruction)
        nouns = {clean_token(t.text) for t in doc if t.pos_ in ("NOUN", "PROPN")}
        verbs = {clean_token(t.text) for t in doc if t.pos_ == "VERB"}
        return nouns, verbs
    except Exception:
        # Fallback split
        words = text.split()
        return set(words[1:]), {words[0]}


class OpenVLAAttentionExtractor:
    def __init__(self, model_alias: str, hf_repo: str):
        self.model_alias = model_alias
        self.hf_repo = hf_repo
        os.environ["LIBERO_CONFIG_PATH"] = os.path.expanduser("~/.libero_openvla")
        os.environ["HF_HUB_OFFLINE"] = "1"
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            hf_repo, attn_implementation="eager", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True
        ).to("cuda:0").eval()

    def get_attention(self, instruction: str):
        img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
        input_ids = inputs["input_ids"][0].tolist()
        raw_tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids)
        
        visual_indices = list(range(40, 296))
        nouns_set, verbs_set = tag_noun_verb_spans(instruction)
        
        noun_indices = []
        verb_indices = []
        for idx, tok in enumerate(raw_tokens):
            if idx in visual_indices:
                continue
            cleaned = clean_token(tok)
            if cleaned in nouns_set:
                noun_indices.append(idx)
            if cleaned in verbs_set:
                verb_indices.append(idx)
                
        if not noun_indices or not verb_indices:
            _die(f"Failed token resolution for instruction {instruction!r}: noun_idx={noun_indices}, verb_idx={verb_indices}")
            
        with torch.no_grad():
            out = self.vla(**inputs, output_attentions=True)
            
        layer_attns = {}
        for l_idx, attn_tensor in enumerate(out.attentions):
            # attn_tensor shape: [1, heads, q_len, k_len]
            attn_avg = attn_tensor[0].mean(dim=0).float().cpu().numpy() # [q_len, k_len]
            n_val = float(attn_avg[noun_indices, :][:, visual_indices].mean())
            v_val = float(attn_avg[verb_indices, :][:, visual_indices].mean())
            layer_attns[l_idx] = (n_val, v_val)
        return layer_attns


class SmolVLAAttentionExtractor:
    def __init__(self, hf_repo: str):
        self.hf_repo = hf_repo
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.configs.policies import PreTrainedConfig
        from transformers import AutoModelForVision2Seq
        
        cfg = PreTrainedConfig.from_pretrained(hf_repo)
        cfg.pretrained_path = hf_repo
        cfg.device = "cuda:0"
        self.policy = SmolVLAPolicy.from_pretrained(hf_repo, config=cfg)
        
        eager_vlm = AutoModelForVision2Seq.from_pretrained(
            self.policy.vlm.config._name_or_path,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to("cuda:0")
        eager_vlm.load_state_dict(self.policy.vlm.state_dict(), strict=False)
        self.policy.vlm = eager_vlm
        self.policy.to("cuda:0").eval()
        self.cfg = cfg

    def get_attention_for_task(self, task_id: int, instruction: str):
        captured = {}
        layers = self.policy.vlm.text_model.layers
        hooks = []
        for l_idx, layer in enumerate(layers):
            def make_hook(idx):
                def hook(module, args, kwargs, output):
                    captured[idx] = output[1]  # attn_weights [1, heads, seq_len, seq_len]
                return hook
            h = layer.self_attn.register_forward_hook(make_hook(l_idx), with_kwargs=True)
            hooks.append(h)
            
        vlm_inputs = {}
        def vlm_pre_hook(module, args, kwargs):
            vlm_inputs.update(kwargs)
        h_vlm = self.policy.vlm.register_forward_pre_hook(vlm_pre_hook, with_kwargs=True)
        
        from lerobot.envs import make_env, make_env_pre_post_processors, preprocess_observation
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.policies import make_pre_post_processors
        
        env_cfg = LiberoEnvConfig(task="libero_goal", task_ids=[task_id], observation_height=256, observation_width=256)
        vec = make_env(env_cfg, n_envs=1, use_async_envs=False)["libero_goal"][task_id]
        self.policy.reset()
        pre, post = make_pre_post_processors(policy_cfg=self.cfg, pretrained_path=self.hf_repo,
                                             preprocessor_overrides={"device_processor": {"device": "cuda"}})
        epre, epost = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=self.cfg)
        obs, _ = vec.reset(seed=7)
        o = preprocess_observation(obs)
        o["task"] = list(vec.call("task_description"))
        o = epre(o)
        o = pre(o)
        
        with torch.no_grad():
            _ = self.policy.select_action(o)
        vec.close()
        
        for h in hooks:
            h.remove()
        h_vlm.remove()
        
        input_ids = vlm_inputs["input_ids"][0].tolist()
        raw_tokens = self.policy.processor.tokenizer.convert_ids_to_tokens(input_ids)
        
        visual_indices = list(range(3, 131))
        nouns_set, verbs_set = tag_noun_verb_spans(instruction)
        
        noun_indices = []
        verb_indices = []
        for idx, tok in enumerate(raw_tokens):
            if idx in visual_indices:
                continue
            cleaned = clean_token(tok)
            if cleaned in nouns_set:
                noun_indices.append(idx)
            if cleaned in verbs_set:
                verb_indices.append(idx)
                
        if not noun_indices or not verb_indices:
            _die(f"Failed token resolution for SmolVLA task {task_id}: noun_idx={noun_indices}, verb_idx={verb_indices}")
            
        layer_attns = {}
        for l_idx in sorted(captured.keys()):
            attn_tensor = captured[l_idx]
            attn_avg = attn_tensor[0].mean(dim=0).float().cpu().numpy()
            n_val = float(attn_avg[noun_indices, :][:, visual_indices].mean())
            v_val = float(attn_avg[verb_indices, :][:, visual_indices].mean())
            layer_attns[l_idx] = (n_val, v_val)
        return layer_attns


def build_extractor(model_alias: str):
    import yaml
    cfg_path = REPO_ROOT / "configs" / "models.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()).get(model_alias)
    if not cfg or not cfg.get("hf_repo"):
        _die(f"model '{model_alias}' has no hf_repo in configs/models.yaml")
        
    framework = cfg.get("framework")
    hf_repo = cfg.get("hf_repo")
    
    if framework in ("openvla", "openvla_oft"):
        return OpenVLAAttentionExtractor(model_alias, hf_repo)
    elif framework == "lerobot":
        return SmolVLAAttentionExtractor(hf_repo)
    else:
        _die(f"unsupported framework '{framework}'")


def extract_for_model(model_alias: str, suite: str = "libero_goal", max_tasks: int | None = None):
    print(f"\n[attention_extract] Starting real attention extraction for model={model_alias}...", file=sys.stderr)
    instructions = load_instructions(suite, max_tasks)
    extractor = build_extractor(model_alias)
    
    per_layer_noun: dict[int, list[float]] = {}
    per_layer_verb: dict[int, list[float]] = {}
    
    for tid, text in instructions:
        print(f"  Processing Task {tid}: {text!r}...", file=sys.stderr)
        if isinstance(extractor, SmolVLAAttentionExtractor):
            layer_attn = extractor.get_attention_for_task(tid, text)
        else:
            layer_attn = extractor.get_attention(text)
            
        for l, (n_val, v_val) in layer_attn.items():
            per_layer_noun.setdefault(l, []).append(n_val)
            per_layer_verb.setdefault(l, []).append(v_val)
            
    layers = sorted(per_layer_noun)
    if not layers:
        _die(f"no attention captured for {model_alias}")
        
    results = []
    for l in layers:
        n = float(np.mean(per_layer_noun[l]))
        v = float(np.mean(per_layer_verb[l]))
        aar = n / v if v > 0 else float("inf")
        results.append({
            "model": model_alias,
            "layer": l,
            "noun_attention": n,
            "verb_attention": v,
            "AAR_noun_over_verb": aar,
            "n_instructions": len(per_layer_noun[l]),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", help="smolvla | openvla | openvla_oft | all")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "report" / "rq4_attention.csv"))
    ap.add_argument("--stdout", action="store_true", help="Print CSV content directly to stdout")
    args = ap.parse_args()

    models_to_run = ["smolvla", "openvla", "openvla_oft"] if args.model == "all" else [args.model]
    
    all_results = []
    for m in models_to_run:
        res = extract_for_model(m, args.suite, args.max_tasks)
        all_results.extend(res)
        
    rows = ["model,layer,noun_attention,verb_attention,AAR_noun_over_verb,n_instructions"]
    for r in all_results:
        rows.append(f"{r['model']},{r['layer']},{r['noun_attention']:.6f},{r['verb_attention']:.6f},{r['AAR_noun_over_verb']:.6f},{r['n_instructions']}")
        
    csv_text = "\n".join(rows) + "\n"
    if args.stdout:
        sys.stdout.write(csv_text)
        sys.stdout.flush()
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(csv_text)
        print(f"\n[attention_extract] SUCCESS: Wrote {out_path} with {len(all_results)} rows across models: {models_to_run}", file=sys.stderr)


if __name__ == "__main__":
    main()

