#!/usr/bin/env bash
# Monitor the training simulation progress

LOG_DIR="$HOME/.lora_training_simulation"
LATEST_LOG=$(ls -t "$LOG_DIR"/simulation_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No training log found yet"
    exit 1
fi

echo "=== TRAINING PROGRESS MONITOR ==="
echo "Log: $LATEST_LOG"
echo ""

# Count total steps
TOTAL_STEPS=$(grep -o "Step [0-9]*/[0-9]*" "$LATEST_LOG" | tail -1 | grep -o "/[0-9]*" | tr -d '/')
CURRENT_STEP=$(grep -o "Step [0-9]*/[0-9]*" "$LATEST_LOG" | tail -1 | grep -o "Step [0-9]*" | tr -d 'Step ')

if [ -n "$TOTAL_STEPS" ] && [ -n "$CURRENT_STEP" ]; then
    PERCENT=$((100 * CURRENT_STEP / TOTAL_STEPS))
    echo "Progress: $CURRENT_STEP / $TOTAL_STEPS steps ($PERCENT%)"
else
    echo "Training starting..."
fi

# Show latest loss
LATEST_LOSS=$(grep -o "loss: [0-9.]*" "$LATEST_LOG" | tail -1)
if [ -n "$LATEST_LOSS" ]; then
    echo "Latest $LATEST_LOSS"
fi

# Show current epoch
CURRENT_EPOCH=$(grep "Epoch [0-9]*/10" "$LATEST_LOG" | tail -1)
if [ -n "$CURRENT_EPOCH" ]; then
    echo "$CURRENT_EPOCH"
fi

# Check for errors
ERRORS=$(grep -c "\[ERROR\]" "$LATEST_LOG" 2>/dev/null || echo 0)
if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "⚠ Errors encountered: $ERRORS"
    echo "Last error:"
    grep "\[ERROR\]" "$LATEST_LOG" | tail -1
fi

# Check if completed
if grep -q "Training completed successfully" "$LATEST_LOG"; then
    echo ""
    echo "✓ TRAINING COMPLETED SUCCESSFULLY!"
    grep "Output saved to:" "$LATEST_LOG"
fi

echo ""
echo "Last update: $(date)"
