#!/usr/bin/env python3
"""
Full training simulation with mock data.
Tests the actual training function and catches errors.
"""
import sys
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image
import numpy as np

# Setup logging to see errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("=" * 80)
print("STEP 1: Testing imports")
print("=" * 80)

try:
    from lora_trainer_gui.training import train_lora
    print("✓ Successfully imported train_lora")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("STEP 2: Creating mock training data")
print("=" * 80)

with TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)

    # Create image directory with test images
    image_dir = tmpdir / "images"
    image_dir.mkdir()

    # Create a few small test images
    for i in range(2):
        img = Image.new('RGB', (256, 256), color=(73, 109, 137))
        img.save(image_dir / f"test_{i}.png")
        print(f"✓ Created test image {i}")

    # Create output directory
    output_dir = tmpdir / "output"
    output_dir.mkdir()

    print(f"\nTest data created:")
    print(f"  - Image dir: {image_dir}")
    print(f"  - Images: {list(image_dir.glob('*.png'))}")
    print(f"  - Output dir: {output_dir}")

    print("\n" + "=" * 80)
    print("STEP 3: Simulating training button click")
    print("=" * 80)
    print("Running train_lora with test configuration...")

    try:
        # Create callbacks to log progress
        def log_progress(msg):
            print(f"[PROGRESS] {msg}")

        def log_status(msg):
            print(f"[STATUS] {msg}")

        # Try to run training with minimal config
        result = train_lora(
            base_model="runwayml/stable-diffusion-v1-5",  # Small model for testing
            model_type="sd15",
            image_dir=image_dir,
            output_dir=output_dir,
            style_token="test_style",
            use_captions=False,
            resolution=512,
            lora_rank=4,  # Minimal rank for testing
            learning_rate=1e-4,
            num_epochs=1,
            batch_size=1,
            use_gradient_checkpointing=True,
            use_fp16=True,
            max_train_steps=2,  # Just 2 steps for testing
            progress_callback=log_progress,
            status_callback=log_status,
            stop_flag=lambda: False,
        )

        print(f"\n✓ Training completed successfully!")
        print(f"Result: {result}")

    except Exception as e:
        print(f"\n✗ Training failed with error:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Message: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

        # Check error log
        error_log = Path.home() / ".lora_trainer_errors.log"
        if error_log.exists():
            print("\n" + "=" * 80)
            print("ERROR LOG CONTENTS:")
            print("=" * 80)
            print(error_log.read_text())

print("\n" + "=" * 80)
print("SIMULATION COMPLETE")
print("=" * 80)
