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

Output: ONE SHARD PER MODEL, report/rq4_attention_<model>.csv, with columns
    model, layer, noun_attention, verb_attention, AAR_noun_over_verb, n_instructions
then `--merge` concatenates the shards into report/rq4_attention.csv.

INTEGRITY CONTRACT
------------------
This script MUST NOT emit a CSV unless a real model was loaded and produced real
attention tensors. There is deliberately NO synthetic fallback. If a model or its
attention cannot be obtained, it raises SystemExit. Missing > invented (CLAUDE.md §12).

WHY SHARDS (a real failure, not a hypothetical)
-----------------------------------------------
The first RQ4 run shipped a combined report/rq4_attention.csv whose openvla_oft
rows were BIT-IDENTICAL to its openvla rows across all 32 layers — impossible for
two different checkpoints (their safetensors differ by md5), so the OFT block was
a relabelled copy and RQ4 actually covered 2 models, not 3. Two things made that
possible and both are fixed here:
  1. this script used to write ONE combined CSV and overwrite it on every
     invocation. The three models need three different conda envs, so they cannot
     be measured in one process — every per-model run clobbered the last, and
     recombining was a manual step. Now each run writes its own shard and
     --merge refuses to emit a combined CSV if two models' rows are identical.
  2. the run was never logged. Each run now appends to MANIFEST.json like every
     rollout does, so an unprovenanced RQ4 number cannot reach the paper again.

Also fixed: the OpenVLA/OFT path used to forward an ALL-BLACK synthetic image
(np.zeros), while SmolVLA used the real env observation — so the two families
were never comparable and the near-uniform OpenVLA values (~1/256) were an
artifact of a degenerate input. Both families now use the real fixed initial
observation, as the DESIGN above and report/RQ4_RQ5_SPEC.txt require.
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
    """OpenVLA / OpenVLA-OFT. Both are LLaMA-backbone action-token models loaded
    through HF trust_remote_code, so one extractor covers both — but they are
    DIFFERENT checkpoints and must be run in their own envs, one process each."""

    NUM_WAIT = 10  # settle steps, same as run/eval_task_openvla.py

    def __init__(self, model_alias: str, hf_repo: str, obs_hw: int = 256):
        self.model_alias = model_alias
        self.hf_repo = hf_repo
        self.obs_hw = obs_hw
        os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_openvla"))
        os.environ["HF_HUB_OFFLINE"] = "1"
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(hf_repo, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            hf_repo, attn_implementation="eager", torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True
        ).to("cuda:0").eval()

    def _real_observation(self, task_id: int, suite: str = "libero_goal"):
        """The fixed initial observation for this task, preprocessed exactly as the
        rollout runner does. Replaces the np.zeros((224,224,3)) placeholder that
        made every OpenVLA-family attention number an artifact of a blank image.

        Mirrors run/eval_task_openvla.py end to end: get_libero_image (:156-160,
        180-degree rotate) -> resize_image (:148-153, Octo/RLDS jpeg roundtrip +
        lanczos3 to 224) -> the center_crop=True branch of predict_action
        (:179-190, crop_and_resize at 0.9). Attention is therefore measured on the
        exact tensor the policy acts on during a rollout. Duplicated rather than
        imported because that runner parses argv at import time; keep in sync."""
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")  # never contend with the policy
        from libero.libero import get_libero_path
        from libero.libero.benchmark import get_benchmark
        from libero.libero.envs import OffScreenRenderEnv

        bench = get_benchmark(suite)()
        task = bench.get_task(task_id)
        init_states = bench.get_task_init_states(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl,
                                 camera_heights=self.obs_hw, camera_widths=self.obs_hw)
        try:
            env.seed(0)
            env.reset()
            # Episode 0's scene, i.e. the same fixed initial state the rollouts start from.
            obs = env.set_init_state(init_states[0])
            for _ in range(self.NUM_WAIT):
                obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])  # dummy action
            img = obs["agentview_image"][::-1, ::-1]              # 180-degree rotate
            # resize_image(): jpeg roundtrip then lanczos3 to 224
            t = tf.image.encode_jpeg(tf.cast(img, tf.uint8))
            t = tf.io.decode_image(t, expand_animations=False, dtype=tf.uint8)
            t = tf.image.resize(t, (224, 224), method="lanczos3", antialias=True)
            t = tf.cast(tf.clip_by_value(tf.round(t), 0, 255), tf.uint8)
            # predict_action(center_crop=True): crop_and_resize at 0.9
            od = t.dtype
            t = tf.image.convert_image_dtype(t, tf.float32)
            t = tf.expand_dims(t, axis=0)
            new_hw = tf.reshape(tf.clip_by_value(tf.sqrt(0.9), 0, 1), shape=(1,))
            off = (1 - new_hw) / 2
            boxes = tf.stack([off, off, off + new_hw, off + new_hw], axis=1)
            t = tf.image.crop_and_resize(t, boxes, tf.range(1), (224, 224))[0]
            t = tf.clip_by_value(t, 0, 1)
            t = tf.image.convert_image_dtype(t, od, saturate=True)
            return Image.fromarray(t.numpy()).convert("RGB")
        finally:
            env.close()

    def get_attention_for_task(self, task_id: int, instruction: str, suite: str = "libero_goal"):
        img = self._real_observation(task_id, suite)
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
        layer_attn = extractor.get_attention_for_task(tid, text)

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


HEADER = "model,layer,noun_attention,verb_attention,AAR_noun_over_verb,n_instructions"
MODELS = ("smolvla", "openvla", "openvla_oft")


def shard_path(model_alias: str) -> Path:
    return REPO_ROOT / "report" / f"rq4_attention_{model_alias}.csv"


def _rows_to_csv(results: list[dict]) -> str:
    rows = [HEADER]
    for r in results:
        rows.append(f"{r['model']},{r['layer']},{r['noun_attention']:.6f},"
                    f"{r['verb_attention']:.6f},{r['AAR_noun_over_verb']:.6f},"
                    f"{r['n_instructions']}")
    return "\n".join(rows) + "\n"


def log_to_manifest(model_alias: str, suite: str, results: list[dict], out_path: Path) -> None:
    """Provenance for an RQ4 run, in the same MANIFEST.json every rollout uses.
    The first RQ4 run was absent from the manifest and had no job log, which is
    exactly how an unverifiable number reaches a paper."""
    sys.path.insert(0, str(REPO_ROOT / "run"))
    from run_one import append_manifest, git_sha  # noqa: E402

    append_manifest(REPO_ROOT / "MANIFEST.json", {
        "kind": "rq4_attention",
        "model": model_alias,
        "suite": suite,
        "n_layers": len(results),
        "n_instructions": results[0]["n_instructions"] if results else 0,
        "out": str(out_path.relative_to(REPO_ROOT)),
        "runner": "analyze/attention_extract.py",
        "repo_sha": git_sha(REPO_ROOT),
        "node": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp": __import__("datetime").datetime.now().astimezone().isoformat(),
    })


def merge_shards(suite: str, out_path: Path) -> None:
    """Concatenate the per-model shards, refusing to emit a duplicated model block."""
    blocks: dict[str, list[str]] = {}
    for m in MODELS:
        p = shard_path(m)
        if not p.exists():
            print(f"[attention_extract] skip {m}: no shard at {p}", file=sys.stderr)
            continue
        lines = [l for l in p.read_text().splitlines() if l.strip() and not l.startswith("model,")]
        if not lines:
            _die(f"shard {p} has a header but no data rows")
        blocks[m] = lines

    if not blocks:
        _die("no shards to merge; run --model <name> per env first")

    # The exact failure that shipped: openvla_oft's rows were bit-identical to
    # openvla's across all 32 layers. Two checkpoints cannot agree to 1e-6.
    measured = {m: [l.split(",", 1)[1] for l in ls] for m, ls in blocks.items()}
    names = sorted(measured)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if measured[a] == measured[b]:
                _die(f"shards for {a} and {b} are IDENTICAL row-for-row — that is a copy, "
                     f"not two measurements. Re-run whichever one is stale; do not ship this.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join([HEADER] + [l for m in names for l in blocks[m]]) + "\n")
    print(f"[attention_extract] merged {len(blocks)} model shard(s) ({', '.join(names)}) "
          f"-> {out_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                    help="smolvla | openvla | openvla_oft (ONE per process — each needs its own env)")
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--out", default=None, help="override the per-model shard path")
    ap.add_argument("--merge", action="store_true",
                    help="combine the per-model shards into report/rq4_attention.csv")
    ap.add_argument("--stdout", action="store_true", help="Print CSV content directly to stdout")
    args = ap.parse_args()

    combined = REPO_ROOT / "report" / "rq4_attention.csv"

    if args.merge:
        merge_shards(args.suite, combined)
        return

    if not args.model:
        _die("pass --model <smolvla|openvla|openvla_oft>, then --merge once all shards exist")
    if args.model == "all":
        # Refusing this is the point: the three models live in three different
        # conda envs, so "all" in one process either crashes or silently measures
        # one model and mislabels the rest.
        _die("--model all is not supported: run one model per env (each writes its own "
             "shard), then re-run with --merge")
    if args.model not in MODELS:
        _die(f"unknown model {args.model!r}; expected one of {MODELS}")

    results = extract_for_model(args.model, args.suite, args.max_tasks)
    csv_text = _rows_to_csv(results)

    if args.stdout:
        sys.stdout.write(csv_text)
        sys.stdout.flush()
        return

    out_path = Path(args.out) if args.out else shard_path(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(csv_text)
    log_to_manifest(args.model, args.suite, results, out_path)
    print(f"\n[attention_extract] SUCCESS: wrote {out_path} ({len(results)} layers, "
          f"{results[0]['n_instructions']} instructions) and logged it to MANIFEST.json.\n"
          f"  Run with --merge once every model's shard exists.", file=sys.stderr)


if __name__ == "__main__":
    main()

