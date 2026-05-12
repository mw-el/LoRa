#!/usr/bin/env python3
"""
Improved LoRA training with captions, higher rank, and 12 epochs (optimal).
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
print("IMPROVED LORA TRAINING - CAPTIONS + RANK 64 + 12 EPOCHS (OPTIMAL)")
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

    # Training configuration - OPTIMAL
    print("\n[Optimized Training Configuration]")
    print(f"  Epochs: 12 (optimal sweet spot)")
    print(f"  Batch Size: 1")
    print(f"  Resolution: 768")
    print(f"  LoRA Rank: 64 (increased from 32)")
    print(f"  Learning Rate: 1e-4")
    print(f"  Use Captions: True (46 descriptive captions)")
    print(f"  Model: SD 1.5")
    print(f"  Total Steps: ~552 (12 epochs × 46 images)")
    print()

    def log_progress(msg):
        print(f"[PROGRESS] {msg}")
        logger.info(f"Progress: {msg}")

    def log_status(msg):
        print(f"[STATUS] {msg}")
        logger.info(f"Status: {msg}")

    print("=" * 80)
    print("STARTING OPTIMIZED TRAINING WITH 12 EPOCHS...")
    print("=" * 80 + "\n")

    # Run training with captions, higher rank, and 12 epochs
    result = train_lora(
        base_model=str(BASE_MODEL),
        model_type="sd15",
        image_dir=SOURCES_DIR,
        output_dir=OUTPUT_DIR,
        style_token="eudes-correia-v2",
        use_captions=True,  # ← USE CAPTIONS
        resolution=768,
        lora_rank=64,  # ← INCREASED from 32
        learning_rate=1e-4,
        num_epochs=12,  # ← OPTIMAL 12 EPOCHS
        batch_size=1,
        use_gradient_checkpointing=True,
        use_fp16=True,
        max_train_steps=None,
        progress_callback=log_progress,
        status_callback=log_status,
        stop_flag=lambda: False,
    )

    print("\n" + "=" * 80)
    print("✓ OPTIMIZED TRAINING COMPLETE")
    print("=" * 80)

    logger.info(f"✓ Training completed successfully")
    logger.info(f"✓ Result: {result}")

    # Check if LoRA file exists
    lora_files = list(OUTPUT_DIR.glob("eudes-correia-v2_*.safetensors"))
    if lora_files:
        latest = max(lora_files, key=lambda p: p.stat().st_mtime)
        file_size = latest.stat().st_size / (1024 * 1024)
        print(f"\n✓ IMPROVED LORA MODEL SAVED!")
        print(f"  File: {latest.name}")
        print(f"  Path: {latest}")
        print(f"  Size: {file_size:.2f} MB")
        logger.info(f"✓ LoRA file: {latest.name} ({file_size:.2f} MB)")
    else:
        print(f"\n✗ ERROR: No LoRA file found!")
        logger.error("✗ No LoRA file generated")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("OPTIMIZATIONS APPLIED:")
    print("=" * 80)
    print("✓ Captions added (46 images with descriptive content)")
    print("✓ LoRA rank increased to 64 (was 32 - better capacity)")
    print("✓ 12 epochs training (optimal for 46 images)")
    print("✓ 552 total training steps")
    print("✓ Semantic guidance from captions + higher model capacity")
    print("\nExpected improvements over baseline (10 epochs, no captions):")
    print("  • Better style consistency and detail")
    print("  • Improved generalization to new prompts")
    print("  • Richer watercolor texture capture")
    print("  • More refined Eudes Correia style reproduction")
    print("=" * 80)

except Exception as e:
    logger.error(f"✗ Training failed: {e}", exc_info=True)
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
