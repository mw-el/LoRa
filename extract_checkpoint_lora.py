#!/usr/bin/env python3
"""
Extract LoRA weights directly from the latest checkpoint.
This doesn't need the training process to be running.
"""

import sys
import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

print("=" * 80)
print("EXTRACTING LORA FROM LATEST CHECKPOINT")
print("=" * 80)

# Find latest checkpoint
checkpoint_dir = Path("/home/matthias/_AA_LoRa/correia-lora/.checkpoints")
checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*"))

if not checkpoints:
    print("✗ No checkpoints found")
    sys.exit(1)

latest_checkpoint = checkpoints[-1]
print(f"\nLatest checkpoint: {latest_checkpoint.name}")

# Load metadata
metadata_file = latest_checkpoint / "metadata.json"
if metadata_file.exists():
    with open(metadata_file) as f:
        meta = json.load(f)
    print(f"  Epoch: {meta['epoch']}, Step: {meta['step']}")
    print(f"  Loss: {meta['loss']:.6f}")

# Load LoRA weights from checkpoint
lora_weights_file = latest_checkpoint / "lora_weights.safetensors"
optimizer_file = latest_checkpoint / "optimizer.pt"

if not lora_weights_file.exists():
    print(f"\n✗ LoRA weights file not found: {lora_weights_file}")
    print("Available files in checkpoint:")
    for f in latest_checkpoint.iterdir():
        print(f"  - {f.name}")
    sys.exit(1)

print(f"\n[1] Loading checkpoint LoRA weights...")
print(f"    File: {lora_weights_file.name}")

try:
    # Load the safetensors file
    lora_state = load_file(lora_weights_file)
    print(f"✓ Loaded {len(lora_state)} tensors")

    # Show tensor info
    print("\nTensor breakdown:")
    total_params = 0
    for key, tensor in lora_state.items():
        total_params += tensor.numel()
        if total_params <= 3:  # Show first few
            print(f"  {key}: shape {tensor.shape}")

    print(f"  ... and {len(lora_state) - 3} more")
    print(f"\nTotal parameters: {total_params:,}")

    # Save as test file
    output_dir = Path("/home/matthias/_AA_LoRa/correia-lora")
    output_path = output_dir / f"eudes-correia-style_test_from_checkpoint.safetensors"

    print(f"\n[2] Saving as test LoRA file...")
    save_file(lora_state, output_path)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Successfully saved: {output_path.name}")
    print(f"✓ File size: {file_size:.2f} MB")

    # Also check if optimizer file exists
    if optimizer_file.exists():
        opt_size = optimizer_file.stat().st_size / (1024 * 1024)
        print(f"\n✓ Optimizer state also available: {opt_size:.2f} MB")
        print("  (Can be used to resume training)")

    print("\n" + "=" * 80)
    print("SUCCESS! Test LoRA extracted from checkpoint")
    print("=" * 80)
    print(f"\nTest file ready for validation:")
    print(f"  {output_path}")
    print(f"\nYou can now test this LoRA in your image generation tool.")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
