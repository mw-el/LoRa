# Training Recovery Guide

## Overview

This document explains how to recover from interrupted LoRA training sessions. The training simulation system automatically saves checkpoints at the end of each epoch, allowing you to resume training from where it stopped.

---

## What Happens If Training Is Interrupted?

If the computer goes into standby mode, loses power, or the training process is manually stopped:

✓ **Checkpoints are automatically saved** at the end of each epoch
✓ **Optimizer state is preserved** for proper gradient history
✓ **Training metadata is logged** (epoch, step, loss values)
✓ **Training can be resumed** from the last checkpoint

---

## Checkpoint Storage

Checkpoints are automatically saved in the training output directory:

```
/home/matthias/_AA_LoRa/correia-lora/
└── .checkpoints/
    ├── checkpoint_epoch00_step00046/
    │   ├── lora_weights.safetensors    (LoRA weights from that epoch)
    │   ├── optimizer.pt                 (Optimizer state for resuming)
    │   └── metadata.json                (Training metadata)
    ├── checkpoint_epoch01_step00092/
    ├── checkpoint_epoch02_step00138/
    └── ...
```

Each checkpoint contains everything needed to resume training.

---

## How to Resume Training

### Method 1: Automatic Detection (Easiest)

Simply run the training command again. The system will automatically detect the latest checkpoint and resume:

```bash
conda run -n lora-lab python3 /home/matthias/_AA_LoRa/run_training_simulation.py
```

**What happens:**
1. System detects existing checkpoints
2. Loads the latest checkpoint automatically
3. Resumes training from that epoch and step
4. Continues until all 10 epochs are complete

### Method 2: Check Recovery Status

To check if a training session can be recovered:

```bash
python3 /home/matthias/_AA_LoRa/recover_training.py
```

This will:
- Show the latest checkpoint found
- Display the epoch and step where training stopped
- Show progress percentage and remaining steps
- Provide instructions for resuming

### Example Output:

```
================================================================================
CHECKPOINT FOUND - TRAINING CAN BE RESUMED
================================================================================

Checkpoint: checkpoint_epoch05_step00230

Progress: Epoch 5/10, Step 230/460
Last Loss: 0.019000

Files available:
  LoRA Weights: lora_weights.safetensors
  Optimizer State: optimizer.pt
  Metadata: metadata.json

Progress: 50.0% complete
Remaining: ~230 steps

================================================================================
```

---

## What Gets Resumed

When you resume from a checkpoint:

### ✓ Automatically Restored:
- **LoRA weights** - Model state from the last epoch
- **Optimizer state** - Gradient history and momentum
- **Training configuration** - All hyperparameters
- **Progress tracking** - Epoch and step counts

### ✓ Continues From:
- Same epoch number where it left off
- Next batch after the last saved state
- Full 10-epoch training plan (not reduced)

### Example:
If training was stopped at Epoch 5, Step 230:
- Resumes from **Epoch 5, Step 231** (next step)
- Continues through **Epochs 5, 6, 7, 8, 9, 10**
- Completes all remaining training

---

## Manual Checkpoint Management

### View All Checkpoints

```bash
ls -la /home/matthias/_AA_LoRa/correia-lora/.checkpoints/
```

### View Checkpoint Metadata

```bash
cat /home/matthias/_AA_LoRa/correia-lora/.checkpoints/checkpoint_epoch*/metadata.json
```

### Clean Up Old Checkpoints (Optional)

Checkpoints take disk space (~500MB-1GB each). After training completes, old checkpoints are automatically cleaned up. You can manually remove old checkpoints if needed:

```bash
# Keep only the last 3 checkpoints (current implementation)
rm -rf /home/matthias/_AA_LoRa/correia-lora/.checkpoints/checkpoint_epoch0[0-2]*
```

---

## Recovery Scenarios

### Scenario 1: Computer Sleep/Hibernation

If the computer goes to sleep during training:

1. Wake up the computer
2. Run: `conda run -n lora-lab python3 /home/matthias/_AA_LoRa/run_training_simulation.py`
3. Training automatically resumes from the last checkpoint
4. No data loss ✓

### Scenario 2: Power Loss

If there's a power outage:

1. Power on the computer
2. Run: `conda run -n lora-lab python3 /home/matthias/_AA_LoRa/run_training_simulation.py`
3. Training automatically resumes from the last checkpoint
4. No data loss ✓

### Scenario 3: Manual Interruption

If you manually stop the training (Ctrl+C):

1. Run: `python3 /home/matthias/_AA_LoRa/recover_training.py` (optional, to check status)
2. Run: `conda run -n lora-lab python3 /home/matthias/_AA_LoRa/run_training_simulation.py`
3. Training automatically resumes from the last checkpoint
4. No data loss ✓

### Scenario 4: Training Complete

If training has already completed:

1. Run: `python3 /home/matthias/_AA_LoRa/recover_training.py`
2. It will show: "Previous training is already complete!"
3. Your LoRA model is ready to use at: `/home/matthias/_AA_LoRa/correia-lora/eudes-correia-style_YYYYMMDD_HHMM.safetensors`

---

## Checkpoint Metadata Format

Each checkpoint's `metadata.json` contains:

```json
{
  "epoch": 5,
  "step": 230,
  "total_steps": 460,
  "loss": 0.019,
  "timestamp": "2025-11-27T23:30:00.000000"
}
```

Fields:
- **epoch**: Current epoch (0-indexed)
- **step**: Current step number
- **total_steps**: Total steps for full training
- **loss**: Last recorded loss value
- **timestamp**: When checkpoint was saved

---

## Important Notes

### Data Safety

✓ Checkpoints are saved at **end of each epoch** (not mid-epoch)
✓ Only complete epochs are checkpointed (no partial data)
✓ Optimizer state is preserved for proper convergence
✓ Previous checkpoints are automatically cleaned up after completion

### File Locations

**Checkpoints**: `/home/matthias/_AA_LoRa/correia-lora/.checkpoints/`
**Training Logs**: `~/.lora_training_simulation/`
**Final LoRA**: `/home/matthias/_AA_LoRa/correia-lora/eudes-correia-style_*.safetensors`

### Disk Space

Each checkpoint is approximately **500MB-1GB**. With 10 epochs, you might have multiple checkpoints. They are automatically cleaned up when training completes, so this is only temporary storage.

---

## Troubleshooting

### Problem: "No checkpoints found"

This means training hasn't started yet or is in a fresh session.

**Solution**: Simply start training:
```bash
conda run -n lora-lab python3 /home/matthias/_AA_LoRa/run_training_simulation.py
```

### Problem: "Checkpoint appears corrupted"

If a checkpoint is corrupted, the system will skip it and use the previous checkpoint.

**Solution**: Check the latest valid checkpoint:
```bash
python3 /home/matthias/_AA_LoRa/recover_training.py
```

### Problem: "Not enough disk space"

Checkpoints take space. Check available space:
```bash
df -h /home/matthias/_AA_LoRa/
```

**Solution**: Delete old checkpoints manually (after training completes):
```bash
rm -rf /home/matthias/_AA_LoRa/correia-lora/.checkpoints/
```

---

## Command Reference

```bash
# Start/Resume training
conda run -n lora-lab python3 /home/matthias/_AA_LoRa/run_training_simulation.py

# Check recovery status
python3 /home/matthias/_AA_LoRa/recover_training.py

# View checkpoints
ls -la /home/matthias/_AA_LoRa/correia-lora/.checkpoints/

# Monitor progress
python3 /home/matthias/_AA_LoRa/training_progress_tracker.py

# Check training logs
tail -f ~/.lora_training_simulation/simulation_*.log
```

---

## Final Output

When training completes successfully:

1. **Trained LoRA Model**: `/home/matthias/_AA_LoRa/correia-lora/eudes-correia-style_YYYYMMDD_HHMM.safetensors`
2. **Training Report**: `~/.lora_training_simulation/REPORT_*.txt`
3. **Detailed Logs**: `~/.lora_training_simulation/simulation_*.log`

The final report will show:
- Training completion status
- Final loss value
- Configuration used
- File size and location
- Next steps for using the model

---

## Questions?

For detailed information about the training system:
- Training code: `/home/matthias/_AA_LoRa/lora_trainer_gui/training.py`
- Recovery system: `/home/matthias/_AA_LoRa/recover_training.py`
- Simulation script: `/home/matthias/_AA_LoRa/run_training_simulation.py`

For logs and debugging:
- `~/.lora_training_simulation/` - All simulation logs
- `~/.lora_trainer_errors.log` - Error log from training module

---

**Last Updated**: 2025-11-27
**Status**: Training checkpoint recovery system fully implemented
