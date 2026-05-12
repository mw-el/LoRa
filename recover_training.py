#!/usr/bin/env python3
"""
Training Recovery Script

This script can recover from a interrupted training session by:
1. Finding the latest checkpoint
2. Loading the saved optimizer state and LoRA weights
3. Resuming training from where it stopped

Usage:
    python3 recover_training.py

The script will automatically detect:
- The output directory with checkpoints
- The latest checkpoint
- Configuration from saved metadata
- Starting epoch and step count
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def find_output_dir():
    """Find the output directory with checkpoints."""
    PROJECT_ROOT = Path("/home/matthias/_AA_LoRa")
    output_dir = PROJECT_ROOT / "correia-lora"

    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return None

    return output_dir

def find_latest_checkpoint(output_dir: Path) -> dict:
    """Find the latest checkpoint in the output directory."""
    checkpoint_dir = output_dir / ".checkpoints"

    if not checkpoint_dir.exists():
        logger.warning(f"No checkpoint directory found: {checkpoint_dir}")
        return None

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*"))

    if not checkpoints:
        logger.warning("No checkpoints found")
        return None

    latest_checkpoint = checkpoints[-1]
    logger.info(f"Found latest checkpoint: {latest_checkpoint.name}")

    # Load metadata
    metadata_file = latest_checkpoint / "metadata.json"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found in checkpoint: {metadata_file}")
        return None

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return {
        "path": latest_checkpoint,
        "metadata": metadata,
        "lora_weights": latest_checkpoint / "lora_weights.safetensors",
        "optimizer": latest_checkpoint / "optimizer.pt",
    }

def print_checkpoint_summary(checkpoint: dict):
    """Print a summary of the checkpoint."""
    meta = checkpoint["metadata"]

    print("\n" + "=" * 80)
    print("CHECKPOINT FOUND - TRAINING CAN BE RESUMED")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint['path'].name}")
    print(f"Timestamp: {meta['timestamp']}")
    print(f"Progress: Epoch {meta['epoch']}/{10}, Step {meta['step']}/{meta['total_steps']}")

    if meta['loss'] is not None:
        print(f"Last Loss: {meta['loss']:.6f}")

    print(f"\nFiles available:")
    print(f"  LoRA Weights: {checkpoint['lora_weights'].name}")
    print(f"  Optimizer State: {checkpoint['optimizer'].name}")
    print(f"  Metadata: metadata.json")

    progress_pct = (meta['step'] / meta['total_steps'] * 100) if meta['total_steps'] > 0 else 0
    print(f"\nProgress: {progress_pct:.1f}% complete")
    print(f"Remaining: ~{meta['total_steps'] - meta['step']} steps")
    print("=" * 80 + "\n")

def create_resume_instructions(checkpoint: dict, output_dir: Path):
    """Create instructions for resuming training."""
    meta = checkpoint["metadata"]

    instructions = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    TRAINING RECOVERY INSTRUCTIONS                        ║
╚══════════════════════════════════════════════════════════════════════════╝

Your training was interrupted but a checkpoint was found!

CHECKPOINT DETAILS:
  Location: {checkpoint['path']}
  Epoch: {meta['epoch']} / 10
  Step: {meta['step']} / {meta['total_steps']}
  Progress: {(meta['step'] / meta['total_steps'] * 100):.1f}% complete

NEXT STEPS:

1. The training will automatically resume from this checkpoint when you run:

   conda run -n lora-lab python3 /home/matthias/_AA_LoRa/run_training_simulation.py

2. The training process will:
   - Detect the latest checkpoint
   - Load the LoRA weights and optimizer state
   - Resume from Epoch {meta['epoch'] + 1}, Step {meta['step'] + 1}
   - Continue until all 10 epochs are complete

3. When training completes, you'll get a final report with:
   - LoRA model location
   - Final loss values
   - Training statistics

╔══════════════════════════════════════════════════════════════════════════╗
║  Your LoRA model will be saved to: {output_dir}                          ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

    return instructions.strip()

def check_for_crashed_training():
    """Check if a training session crashed or was interrupted."""
    output_dir = find_output_dir()

    if not output_dir:
        logger.error("Cannot find output directory")
        return False

    logger.info(f"Checking for checkpoints in: {output_dir}")

    checkpoint = find_latest_checkpoint(output_dir)

    if not checkpoint:
        logger.info("No previous training checkpoint found - would start fresh")
        return False

    # Check if training is actually complete
    meta = checkpoint["metadata"]
    if meta['step'] >= meta['total_steps']:
        logger.info("Previous training is already complete!")
        return False

    # Found an incomplete checkpoint
    print_checkpoint_summary(checkpoint)

    instructions = create_resume_instructions(checkpoint, output_dir)
    print(instructions)

    return True

def cleanup_checkpoints(output_dir: Path, keep_latest: int = 3):
    """Clean up old checkpoints to save disk space, keeping only the latest N."""
    checkpoint_dir = output_dir / ".checkpoints"

    if not checkpoint_dir.exists():
        return

    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*"))

    if len(checkpoints) <= keep_latest:
        return

    logger.info(f"Cleaning up old checkpoints (keeping latest {keep_latest})...")

    for old_checkpoint in checkpoints[:-keep_latest]:
        try:
            # Remove directory recursively
            import shutil
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed: {old_checkpoint.name}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")

def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("TRAINING RECOVERY SYSTEM - CHECKING FOR INTERRUPTED SESSIONS")
    print("=" * 80 + "\n")

    found = check_for_crashed_training()

    if not found:
        print("\nNo interrupted training sessions found.")
        print("You can start a new training session with:")
        print("  conda run -n lora-lab python3 /home/matthias/_AA_LoRa/run_training_simulation.py")
        return 0

    # Offer to clean up old checkpoints
    output_dir = find_output_dir()
    if output_dir:
        checkpoint_dir = output_dir / ".checkpoints"
        if checkpoint_dir.exists():
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*"))
            if len(checkpoints) > 3:
                print(f"\nYou have {len(checkpoints)} checkpoints. These take up disk space.")
                print("Old checkpoints will be automatically cleaned up once training completes.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
