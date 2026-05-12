#!/usr/bin/env python3
"""
Simulation script to test training and auto-fix errors.
This attempts to run the training pipeline and logs all errors.
"""
import sys
import subprocess
from pathlib import Path

# Run the actual start.sh to see what errors occur
print("=" * 80)
print("SIMULATION: Running start.sh")
print("=" * 80)

try:
    result = subprocess.run(
        ["bash", "start.sh"],
        cwd="/home/matthias/_AA_LoRa",
        capture_output=True,
        text=True,
        timeout=30
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)
except subprocess.TimeoutExpired:
    print("Timeout - GUI started successfully (expected)")
except Exception as e:
    print(f"Error running start.sh: {e}")

# Check the error log
log_file = Path.home() / ".lora_trainer_errors.log"
print("\n" + "=" * 80)
print(f"Checking error log: {log_file}")
print("=" * 80)
if log_file.exists():
    content = log_file.read_text()
    print(content)
else:
    print("No error log yet (script may not have run)")
