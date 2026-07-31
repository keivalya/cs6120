#!/usr/bin/env python3
"""scratch/inspect_tokens.py — Evidence Gate diagnostic script for OpenVLA, OpenVLA-OFT, and SmolVLA."""

import sys
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def inspect_openvla():
    print("=== INSPECTING OPENVLA ===")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    ckpt = "openvla/openvla-7b-finetuned-libero-goal"
    print(f"Loading {ckpt} with eager attention...")
    
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to("cuda:0")
    model.eval()
    
    # Test instruction
    instruction = "open the middle drawer of the cabinet"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    
    # Create dummy image (256x256x3)
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    from PIL import Image
    image = Image.fromarray(dummy_img)
    
    inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
    input_ids = inputs["input_ids"][0]
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    
    print(f"Total sequence length: {len(tokens)}")
    print("Tokens with indices:")
    for idx, tok in enumerate(tokens):
        print(f"  idx {idx:3d}: {tok!r}")
        
    # Run forward pass with output_attentions=True
    with torch.inference_mode():
        outputs = model(**inputs, output_attentions=True)
        
    attentions = outputs.attentions
    print(f"Number of layers in attentions: {len(attentions)}")
    print(f"Layer 0 attention shape: {attentions[0].shape}")
    
    # Clean up GPU memory
    del model, processor, outputs, attentions
    torch.cuda.empty_cache()

if __name__ == "__main__":
    inspect_openvla()
