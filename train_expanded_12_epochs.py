#!/usr/bin/env python3
"""
LoRA training with expanded dataset (92 images) and 12 epochs.
Uses mirrored images to double the dataset.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

print("\n" + "=" * 80)
print("LORA TRAINING - EXPANDED DATASET (92 IMAGES) + 12 EPOCHS")
print("=" * 80)

PROJECT_ROOT = Path("/home/matthias/_AA_LoRa")
SOURCES_DIR = PROJECT_ROOT / "sources" / "correia"
OUTPUT_DIR = PROJECT_ROOT / "correia-lora"
BASE_MODEL = Path("/home/matthias/_AA_ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors")

logger.info(f"Starting training at {datetime.now()}")

try:
    from lora_trainer_gui.training import train_lora

    image_files = list(SOURCES_DIR.glob("*.png")) + list(SOURCES_DIR.glob("*.jpg")) + \
                  list(SOURCES_DIR.glob("*.jpeg")) + list(SOURCES_DIR.glob("*.webp"))

    logger.info(f"✓ Found {len(image_files)} training images")

    print("\n[Training Configuration - Expanded Dataset]")
    print(f"  Total Images: {len(image_files)} (46 original + 46 mirrored)")
    print(f"  Epochs: 12")
    print(f"  Batch Size: 1")
    print(f"  LoRA Rank: 64")
    print(f"  Total Steps: {len(image_files) * 12} (~{len(image_files) * 12 // 60} minutes)")
    print()

    def log_progress(msg):
        print(f"[PROGRESS] {msg}")
        logger.info(f"Progress: {msg}")

    def log_status(msg):
        print(f"[STATUS] {msg}")
        logger.info(f"Status: {msg}")

    print("=" * 80)
    print("STARTING TRAINING WITH EXPANDED DATASET (12 EPOCHS)...")
    print("=" * 80 + "\n")

    result = train_lora(
        base_model=str(BASE_MODEL),
        model_type="sd15",
        image_dir=SOURCES_DIR,
        output_dir=OUTPUT_DIR,
        style_token="eudes-correia-expanded-12",
        use_captions=True,
        resolution=768,
        lora_rank=64,
        learning_rate=1e-4,
        num_epochs=12,
        batch_size=1,
        use_gradient_checkpointing=True,
        use_fp16=True,
        max_train_steps=None,
        progress_callback=log_progress,
        status_callback=log_status,
        stop_flag=lambda: False,
    )

    lora_files = list(OUTPUT_DIR.glob("eudes-correia-expanded-12_*.safetensors"))
    if lora_files:
        latest = max(lora_files, key=lambda p: p.stat().st_mtime)
        file_size = latest.stat().st_size / (1024 * 1024)
        print(f"\n✓ TRAINING COMPLETE!")
        print(f"  File: {latest.name}")
        print(f"  Size: {file_size:.2f} MB")
    else:
        print(f"\n✗ ERROR: No LoRA file found!")
        sys.exit(1)

except Exception as e:
    logger.error(f"✗ Training failed: {e}", exc_info=True)
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
