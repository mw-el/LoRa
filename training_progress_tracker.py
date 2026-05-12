#!/usr/bin/env python3
"""
Real-time progress tracker for LoRA training simulation.
Monitors the log file and displays progress metrics.
"""

import sys
from pathlib import Path
import re
from datetime import datetime

def get_latest_log():
    """Find the most recent simulation log."""
    log_dir = Path.home() / ".lora_training_simulation"
    logs = sorted(log_dir.glob("simulation_*.log"), reverse=True)
    return logs[0] if logs else None

def parse_progress(log_path: Path):
    """Extract progress metrics from the log file."""
    content = log_path.read_text(errors='ignore')

    # Extract last step info
    steps = re.findall(r'Step (\d+)/(\d+)', content)
    if steps:
        current_step, total_steps = steps[-1]
        current_step = int(current_step)
        total_steps = int(total_steps)
        progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
    else:
        current_step = total_steps = progress_pct = 0

    # Extract current epoch
    epochs = re.findall(r'Epoch (\d+)/(\d+)', content)
    if epochs:
        curr_epoch, max_epochs = epochs[-1]
        curr_epoch = int(curr_epoch)
        max_epochs = int(max_epochs)
    else:
        curr_epoch = max_epochs = 0

    # Extract latest loss
    losses = re.findall(r'loss: ([\d.]+)', content)
    latest_loss = float(losses[-1]) if losses else None

    # Check status
    completed = 'Training completed successfully' in content
    has_errors = '[ERROR]' in content
    error_count = content.count('[ERROR]')

    return {
        'step': current_step,
        'total_steps': total_steps,
        'progress_pct': progress_pct,
        'epoch': curr_epoch,
        'max_epochs': max_epochs,
        'loss': latest_loss,
        'completed': completed,
        'errors': error_count,
        'has_errors': has_errors,
    }

def format_progress_bar(pct: float, width: int = 40) -> str:
    """Create a text progress bar."""
    filled = int(width * pct / 100)
    bar = '█' * filled + '░' * (width - filled)
    return f"[{bar}] {pct:.1f}%"

def display_progress():
    """Display training progress."""
    log_path = get_latest_log()

    if not log_path:
        print("No training log found. Training may not have started yet.")
        return

    progress = parse_progress(log_path)

    print("\n" + "=" * 70)
    print("           LORA TRAINING PROGRESS MONITOR".center(70))
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Progress bar
    if progress['total_steps'] > 0:
        bar = format_progress_bar(progress['progress_pct'])
        print(f"Training Progress: {bar}")
        print(f"Steps: {progress['step']:4d} / {progress['total_steps']:4d}")

        # ETA calculation
        if progress['step'] > 0 and progress['progress_pct'] < 100:
            # Rough estimate based on step rate
            steps_remaining = progress['total_steps'] - progress['step']
            # This is just a placeholder - actual ETA would need timing data
            print(f"Steps remaining: {steps_remaining}")

    # Epoch progress
    if progress['max_epochs'] > 0:
        epoch_pct = (progress['epoch'] / progress['max_epochs'] * 100)
        epoch_bar = format_progress_bar(epoch_pct)
        print()
        print(f"Epoch Progress:   {epoch_bar}")
        print(f"Epoch: {progress['epoch']} / {progress['max_epochs']}")

    # Loss
    if progress['loss'] is not None:
        print()
        print(f"Latest Loss: {progress['loss']:.6f}")

    # Status
    print()
    if progress['completed']:
        print("✓ TRAINING COMPLETED SUCCESSFULLY!")
    elif progress['has_errors']:
        print(f"⚠ Status: Running with {progress['errors']} error(s)")
    else:
        print("✓ Training in progress...")

    print("=" * 70)
    print()

if __name__ == "__main__":
    display_progress()
