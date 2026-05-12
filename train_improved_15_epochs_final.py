#!/usr/bin/env python3
"""
Improved LoRA training with captions, higher rank, and 15 epochs.
Uses the .txt caption files and increased LoRA capacity.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

print("\n" + "=" * 80)
print("IMPROVED LORA TRAINING - CAPTIONS + RANK 64 + 15 EPOCHS")
print("=" * 80)

# Setup
PROJECT_ROOT = Path("/home/matthias/_AA_LoRa")
SOURCES_DIR = PROJECT_ROOT / "sources" / "correia"
OUTPUT_DIR = PROJECT_ROOT / "correia-lora"
BASE_MODEL = Path("/home/matthias/_AA_ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors")

logger.info(f"Starting improved training at {datetime.now()}")
logger.info(f"Source images: {SOURCES_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")

try:
    from lora_trainer_gui.training import train_lora

    logger.info("✓ Imported training module")

    # Count images
    image_files = list(SOURCES_DIR.glob("*.png")) + \
                  list(SOURCES_DIR.glob("*.jpg")) + \
                  list(SOURCES_DIR.glob("*.jpeg")) + \
                  list(SOURCES_DIR.glob("*.webp"))

    logger.info(f"✓ Found {len(image_files)} training images")

    # Count captions
    caption_files = list(SOURCES_DIR.glob("*.txt"))
    logger.info(f"✓ Found {len(caption_files)} caption files")

    # Training configuration
    print("\n[Training Configuration - 15 Epochs]")
    print(f"  Epochs: 15 (aggressive learning)")
    print(f"  Batch Size: 1")
    print(f"  Resolution: 768")
    print(f"  LoRA Rank: 64")
    print(f"  Learning Rate: 1e-4")
    print(f"  Use Captions: True (46 descriptive captions)")
    print(f"  Model: SD 1.5")
    print(f"  Total Steps: ~690 (15 epochs × 46 images)")
    print()

    def log_progress(msg):
        print(f"[PROGRESS] {msg}")
        logger.info(f"Progress: {msg}")

    def log_status(msg):
        print(f"[STATUS] {msg}")
        logger.info(f"Status: {msg}")

    print("=" * 80)
    print("STARTING TRAINING WITH 15 EPOCHS...")
    print("=" * 80 + "\n")

    # Run training with 15 epochs
    result = train_lora(
        base_model=str(BASE_MODEL),
        model_type="sd15",
        image_dir=SOURCES_DIR,
        output_dir=OUTPUT_DIR,
        style_token="eudes-correia-v3",
        use_captions=True,
        resolution=768,
        lora_rank=64,
        learning_rate=1e-4,
        num_epochs=15,  # ← 15 EPOCHS
        batch_size=1,
        use_gradient_checkpointing=True,
        use_fp16=True,
        max_train_steps=None,
        progress_callback=log_progress,
        status_callback=log_status,
        stop_flag=lambda: False,
    )

    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)

    logger.info(f"✓ Training completed successfully")
    logger.info(f"✓ Result: {result}")

    # Check if LoRA file exists
    lora_files = list(OUTPUT_DIR.glob("eudes-correia-v3_*.safetensors"))
    if lora_files:
        latest = max(lora_files, key=lambda p: p.stat().st_mtime)
        file_size = latest.stat().st_size / (1024 * 1024)
        print(f"\n✓ LORA MODEL SAVED!")
        print(f"  File: {latest.name}")
        print(f"  Path: {latest}")
        print(f"  Size: {file_size:.2f} MB")
        logger.info(f"✓ LoRA file: {latest.name} ({file_size:.2f} MB)")
    else:
        print(f"\n✗ ERROR: No LoRA file found!")
        logger.error("✗ No LoRA file generated")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("COMPARISON - V1 vs V2 vs V3:")
    print("=" * 80)
    print("V1 (Baseline):        10 epochs, rank 32, no captions")
    print("V2 (Optimized):       12 epochs, rank 64, with captions")
    print("V3 (Aggressive):      15 epochs, rank 64, with captions")
    print("\nUse V3 for maximum style learning (may be slightly more specific)")
    print("Use V2 for best balance (optimal generalization)")
    print("=" * 80)

except Exception as e:
    logger.error(f"✗ Training failed: {e}", exc_info=True)
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
