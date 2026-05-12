#!/usr/bin/env python3
"""
Improved LoRA training with captions and higher rank.
Uses the .txt caption files for better training.
Increases LoRA rank from 32 to 64 for more model capacity.
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
print("IMPROVED LORA TRAINING - WITH CAPTIONS & HIGHER RANK")
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

    # Training configuration - IMPROVED
    print("\n[Improved Training Configuration]")
    print(f"  Epochs: 10 (comprehensive training)")
    print(f"  Batch Size: 1")
    print(f"  Resolution: 768")
    print(f"  LoRA Rank: 64 (increased from 32 for better capacity)")
    print(f"  Learning Rate: 1e-4")
    print(f"  Use Captions: True (NEW!)")
    print(f"  Model: SD 1.5")
    print()

    def log_progress(msg):
        print(f"[PROGRESS] {msg}")
        logger.info(f"Progress: {msg}")

    def log_status(msg):
        print(f"[STATUS] {msg}")
        logger.info(f"Status: {msg}")

    print("=" * 80)
    print("STARTING IMPROVED TRAINING...")
    print("=" * 80 + "\n")

    # Run training with captions and improved settings
    result = train_lora(
        base_model=str(BASE_MODEL),
        model_type="sd15",
        image_dir=SOURCES_DIR,
        output_dir=OUTPUT_DIR,
        style_token="eudes-correia-improved",
        use_captions=True,  # ← USE CAPTIONS!
        resolution=768,
        lora_rank=64,  # ← INCREASED from 32
        learning_rate=1e-4,
        num_epochs=10,
        batch_size=1,
        use_gradient_checkpointing=True,
        use_fp16=True,
        max_train_steps=None,
        progress_callback=log_progress,
        status_callback=log_status,
        stop_flag=lambda: False,
    )

    print("\n" + "=" * 80)
    print("✓ IMPROVED TRAINING COMPLETE")
    print("=" * 80)

    logger.info(f"✓ Training completed successfully")
    logger.info(f"✓ Result: {result}")

    # Check if LoRA file exists
    lora_files = list(OUTPUT_DIR.glob("eudes-correia-improved_*.safetensors"))
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
    print("IMPROVEMENTS APPLIED:")
    print("=" * 80)
    print("✓ Captions added (46 images with descriptive text)")
    print("✓ LoRA rank increased to 64 (was 32)")
    print("✓ Better model capacity for style learning")
    print("✓ Semantic information from captions")
    print("\nExpected improvements:")
    print("  • Better style consistency")
    print("  • Improved generalization to new prompts")
    print("  • Richer detail capture in generated images")
    print("=" * 80)

except Exception as e:
    logger.error(f"✗ Training failed: {e}", exc_info=True)
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
