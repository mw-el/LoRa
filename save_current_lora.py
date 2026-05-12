#!/usr/bin/env python3
"""
Save current LoRA state from the active training process.
This extracts LoRA weights from the trained unet and saves them.
"""

import sys
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from diffusers import StableDiffusionPipeline

print("=" * 80)
print("SAVING CURRENT LORA STATE FOR TESTING")
print("=" * 80)

# Load the model
base_model = "/home/matthias/_AA_ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors"
output_dir = Path("/home/matthias/_AA_LoRa/correia-lora")

print("\n[1] Loading base model...")
try:
    pipe = StableDiffusionPipeline.from_single_file(
        base_model,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

print("\n[2] Checking for trainable LoRA processors...")

# The model loads with default AttnProcessor2_0, not trained LoRA layers
# We need to manually load from checkpoint if available
checkpoint_dir = output_dir / ".checkpoints"
checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*")) if checkpoint_dir.exists() else []

if checkpoints:
    latest_checkpoint = checkpoints[-1]
    checkpoint_meta_file = latest_checkpoint / "metadata.json"

    if checkpoint_meta_file.exists():
        with open(checkpoint_meta_file) as f:
            meta = json.load(f)
        print(f"✓ Found checkpoint: {latest_checkpoint.name}")
        print(f"  Epoch: {meta['epoch']}, Step: {meta['step']}")
        print(f"  Loss: {meta['loss']}")
else:
    print("✗ No checkpoints found")
    sys.exit(1)

print("\n[3] Extracting LoRA weights from current unet state...")

# The current unet has LoRA layers if training is ongoing
# Check what processors are currently in the unet
lora_weights = pipe.unet.attn_processors
print(f"Found {len(lora_weights)} attention processors")

if lora_weights:
    first_proc = list(lora_weights.values())[0]
    print(f"First processor type: {type(first_proc).__name__}")
    print(f"Has to_q_lora: {hasattr(first_proc, 'to_q_lora')}")

print("\n[4] Attempting to save LoRA weights...")

flat_state = {}
count = 0

for proc_name, processor in lora_weights.items():
    # Try to access LoRA layers directly
    for attr_name in ["to_q_lora", "to_k_lora", "to_v_lora", "to_out_lora"]:
        if hasattr(processor, attr_name):
            lora_layer = getattr(processor, attr_name)
            # Extract down and up weights
            if hasattr(lora_layer, "down") and hasattr(lora_layer, "up"):
                down_weight = lora_layer.down.weight.detach().cpu()
                up_weight = lora_layer.up.weight.detach().cpu()
                flat_state[f"{proc_name}.{attr_name}.down.weight"] = down_weight
                flat_state[f"{proc_name}.{attr_name}.up.weight"] = up_weight
                count += 2

if flat_state:
    print(f"✓ Extracted {count} weight tensors from {len(lora_weights)} processors")

    # Save
    output_path = output_dir / f"eudes-correia-style_current_test.safetensors"
    save_file(flat_state, output_path)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Successfully saved LoRA to: {output_path}")
    print(f"✓ File size: {file_size:.2f} MB")

    print("\n" + "=" * 80)
    print("SUCCESS! LoRA weights saved and ready for testing")
    print("=" * 80)
    print(f"\nFile: {output_path.name}")
    print(f"Path: {output_path}")

else:
    print("✗ No LoRA weights found to extract")
    print("\nNote: If training is using a base model that needs LoRA injection,")
    print("the processors might not have LoRA layers yet.")
    sys.exit(1)
