#!/usr/bin/env python3
"""
Test script to save LoRA weights from a checkpoint.
"""

import sys
import json
from pathlib import Path
import torch
from safetensors.torch import save_file

# Find latest checkpoint
checkpoint_dir = Path("/home/matthias/_AA_LoRa/correia-lora/.checkpoints")
checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*"))

if not checkpoints:
    print("No checkpoints found!")
    sys.exit(1)

latest = checkpoints[-1]
print(f"Testing save from: {latest.name}")

# Try to load the model and save
try:
    from lora_trainer_gui.training import train_lora
    from diffusers import StableDiffusionPipeline

    print("\nLoading model...")
    base_model = "/home/matthias/_AA_ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors"
    pipe = StableDiffusionPipeline.from_single_file(
        base_model,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")

    print("Model loaded successfully!")
    print(f"UNet type: {type(pipe.unet)}")
    print(f"Attn processors: {type(pipe.unet.attn_processors)}")

    # Check attn processor structure
    first_proc = list(pipe.unet.attn_processors.values())[0] if pipe.unet.attn_processors else None
    if first_proc:
        print(f"\nFirst processor type: {type(first_proc)}")
        print(f"Has to_q_lora: {hasattr(first_proc, 'to_q_lora')}")
        print(f"Dir: {[x for x in dir(first_proc) if 'lora' in x.lower() or 'weight' in x.lower()]}")

    # Try to save
    print("\n" + "=" * 80)
    print("ATTEMPTING TO SAVE LORA WEIGHTS")
    print("=" * 80)

    output_path = Path("/home/matthias/_AA_LoRa/correia-lora/test_lora_save.safetensors")

    # Method 1: Extract from attn_processors
    print("\nMethod 1: Direct attn_processor extraction...")
    flat_state = {}
    count = 0
    for name, module in pipe.unet.attn_processors.items():
        print(f"  Processing: {name} ({type(module).__name__})")
        if hasattr(module, "state_dict") and callable(module.state_dict):
            try:
                state = module.state_dict()
                for k, v in state.items():
                    flat_state[f"{name}.{k}"] = v
                    count += 1
                print(f"    ✓ Extracted {len(state)} parameters")
            except Exception as e:
                print(f"    ✗ Failed: {e}")

    if flat_state:
        print(f"\n✓ Extracted {count} total parameters")
        save_file(flat_state, output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Successfully saved to: {output_path}")
        print(f"✓ File size: {file_size:.2f} MB")
    else:
        print(f"\n✗ No parameters extracted from attn_processors")

        # Method 2: Try pipeline save
        print("\nMethod 2: Using pipeline save_lora_weights...")
        tmp_dir = Path("/tmp/lora_test")
        tmp_dir.mkdir(exist_ok=True)

        try:
            pipe.save_lora_weights(tmp_dir, safe_serialization=True)
            saved_file = next(tmp_dir.glob("*.safetensors"), None)
            if saved_file:
                saved_file.rename(output_path)
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ Successfully saved via pipeline to: {output_path}")
                print(f"✓ File size: {file_size:.2f} MB")
            else:
                print(f"✗ Pipeline save created no safetensors file")
        except Exception as e:
            print(f"✗ Pipeline save failed: {e}")
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n" + "=" * 80)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
