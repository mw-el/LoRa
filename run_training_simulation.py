#!/usr/bin/env python3
"""
LoRA Training Simulation with Automatic Error Recovery

This script:
1. Simulates running LoRA training with default GUI settings
2. Catches and logs all errors
3. Applies automatic fixes
4. Retries training until successful
5. Logs all training, fixing, and restarting operations

Run with: python3 run_training_simulation.py
"""

import sys
import logging
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable, Tuple
import time

# ============================================================================
# LOGGING SETUP
# ============================================================================

# Create logs directory
LOGS_DIR = Path.home() / ".lora_training_simulation"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Main simulation log
SIM_LOG_FILE = LOGS_DIR / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
ERROR_LOG_FILE = LOGS_DIR / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
FIXES_LOG_FILE = LOGS_DIR / f"fixes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(SIM_LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("SIMULATION")
training_logger = logging.getLogger("TRAINING")
error_logger = logging.getLogger("ERRORS")
fixes_logger = logging.getLogger("FIXES")

# Add file handlers for specialized logs
error_file_handler = logging.FileHandler(ERROR_LOG_FILE)
error_file_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_file_handler)

fixes_file_handler = logging.FileHandler(FIXES_LOG_FILE)
fixes_file_handler.setLevel(logging.DEBUG)
fixes_logger.addHandler(fixes_file_handler)

# ============================================================================
# BANNER
# ============================================================================

def print_banner(title: str):
    """Print a formatted banner."""
    width = 80
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)
    logger.info(f"{'=' * width}")
    logger.info(f" {title}".center(width))
    logger.info(f"{'=' * width}")

# ============================================================================
# CONFIGURATION
# ============================================================================

print_banner("LORA TRAINING SIMULATION - STARTUP")

try:
    from lora_trainer_gui.config import default_config, DefaultConfig
    from lora_trainer_gui.training import train_lora
    logger.info("✓ Successfully imported training modules")
    print("✓ Successfully imported training modules")
except ImportError as e:
    logger.error(f"✗ Failed to import training modules: {e}")
    print(f"✗ Failed to import training modules: {e}")
    sys.exit(1)

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

print_banner("STEP 1: LOADING DEFAULT CONFIGURATION")

# Get default config
default_cfg: DefaultConfig = default_config()
logger.info(f"Default config loaded: model_type={default_cfg.model_type}, "
            f"resolution={default_cfg.resolution}, lora_rank={default_cfg.lora_rank}")

# Setup paths
PROJECT_ROOT = Path("/home/matthias/_AA_LoRa")
SOURCES_DIR = PROJECT_ROOT / "sources" / "correia"
OUTPUT_DIR = PROJECT_ROOT / "correia-lora"
BASE_MODEL = Path(default_cfg.base_model)

# Verify or create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Output directory ready: {OUTPUT_DIR}")

# Check if source images exist
if not SOURCES_DIR.exists():
    logger.error(f"✗ Sources directory does not exist: {SOURCES_DIR}")
    print(f"✗ Sources directory does not exist: {SOURCES_DIR}")
    sys.exit(1)

image_files = list(SOURCES_DIR.glob("*.png")) + \
              list(SOURCES_DIR.glob("*.jpg")) + \
              list(SOURCES_DIR.glob("*.jpeg")) + \
              list(SOURCES_DIR.glob("*.webp"))

if not image_files:
    logger.error(f"✗ No images found in {SOURCES_DIR}")
    print(f"✗ No images found in {SOURCES_DIR}")
    sys.exit(1)

logger.info(f"✓ Found {len(image_files)} training images in {SOURCES_DIR}")
print(f"✓ Found {len(image_files)} training images")

# Check base model
if not BASE_MODEL.exists():
    logger.error(f"✗ Base model not found: {BASE_MODEL}")
    print(f"✗ Base model not found: {BASE_MODEL}")
    logger.info("Note: Will attempt to use HuggingFace ID as fallback")
    print("Note: Will attempt to use HuggingFace ID as fallback")

# Configuration summary
config_summary = {
    "model_type": "sd15",  # GUI shows "SD 1.5", training expects "sd15"
    "resolution": default_cfg.resolution,
    "lora_rank": default_cfg.lora_rank,
    "learning_rate": default_cfg.learning_rate,
    "num_epochs": default_cfg.num_epochs,
    "batch_size": default_cfg.batch_size,
    "gradient_checkpointing": default_cfg.gradient_checkpointing,
    "mixed_precision": default_cfg.mixed_precision,
    "max_train_steps": default_cfg.max_train_steps,
    "base_model": str(BASE_MODEL),
    "output_dir": str(OUTPUT_DIR),
    "image_dir": str(SOURCES_DIR),
    "style_token": "eudes-correia-style",
    "use_captions": default_cfg.use_captions,
}

print("\nConfiguration:")
for key, value in config_summary.items():
    print(f"  {key}: {value}")
    logger.info(f"  {key}: {value}")

# ============================================================================
# ERROR TRACKING & FIXES
# ============================================================================

class ErrorTracker:
    """Track and manage errors during training."""

    def __init__(self):
        self.errors: list[Tuple[int, str, str]] = []  # (attempt, error_type, message)
        self.fixes_applied: list[Tuple[int, str]] = []  # (attempt, fix_description)
        self.attempt_count = 0

    def add_error(self, error_type: str, message: str):
        """Log an error."""
        self.errors.append((self.attempt_count, error_type, message))
        error_logger.error(f"[Attempt {self.attempt_count}] {error_type}: {message}")
        logger.error(f"✗ Error: {error_type}: {message}")

    def add_fix(self, fix_description: str):
        """Log an applied fix."""
        self.fixes_applied.append((self.attempt_count, fix_description))
        fixes_logger.info(f"[Attempt {self.attempt_count}] Fix applied: {fix_description}")
        logger.info(f"✓ Fix: {fix_description}")
        print(f"✓ Applying fix: {fix_description}")

    def record_attempt(self):
        """Start a new attempt."""
        self.attempt_count += 1
        logger.info(f"\n{'=' * 80}")
        logger.info(f"ATTEMPT #{self.attempt_count} - {datetime.now()}")
        logger.info(f"{'=' * 80}")
        print(f"\n{'=' * 80}")
        print(f"ATTEMPT #{self.attempt_count}")
        print(f"{'=' * 80}")

    def summary(self) -> str:
        """Get a summary of all errors and fixes."""
        lines = [
            "\n" + "=" * 80,
            "ERROR & FIX SUMMARY",
            "=" * 80,
            f"Total attempts: {self.attempt_count}",
            f"Total errors encountered: {len(self.errors)}",
            f"Total fixes applied: {len(self.fixes_applied)}",
        ]

        if self.errors:
            lines.append("\nErrors encountered:")
            for attempt, error_type, message in self.errors:
                lines.append(f"  [Attempt {attempt}] {error_type}: {message[:100]}")

        if self.fixes_applied:
            lines.append("\nFixes applied:")
            for attempt, fix_desc in self.fixes_applied:
                lines.append(f"  [Attempt {attempt}] {fix_desc}")

        lines.append("=" * 80)
        return "\n".join(lines)

error_tracker = ErrorTracker()

# ============================================================================
# PROGRESS CALLBACKS
# ============================================================================

training_step_count = 0
last_progress_time = time.time()

def on_progress(message: str):
    """Handle training progress updates."""
    global training_step_count, last_progress_time

    # Extract step count from message if available
    if "Step" in message:
        training_step_count += 1

    current_time = time.time()
    elapsed = current_time - last_progress_time

    # Log every 5 seconds to avoid log spam
    if elapsed >= 5 or "Step" not in message:
        logger.info(f"[PROGRESS] {message}")
        training_logger.info(message)
        last_progress_time = current_time

def on_status(message: str):
    """Handle training status updates."""
    logger.info(f"[STATUS] {message}")
    training_logger.info(f"[STATUS] {message}")
    print(f"[STATUS] {message}")

def on_stop_flag() -> bool:
    """Check if training should be stopped."""
    return False  # Never stop automatically

# ============================================================================
# TRAINING WITH AUTO-RECOVERY
# ============================================================================

MAX_ATTEMPTS = 5
KNOWN_FIXES = {
    "OutOfMemoryError": "Try reducing resolution or batch_size",
    "RuntimeError": "Check CUDA or model loading issues",
    "FileNotFoundError": "Verify base model and image paths",
    "ImportError": "Check dependencies installation",
}

def attempt_training(config: dict) -> Tuple[bool, Optional[Path], str]:
    """
    Attempt training with the given configuration.

    Returns:
        (success, output_path, message)
    """
    error_tracker.record_attempt()

    # Notify if this is a retry with a known fix
    if error_tracker.attempt_count > 1:
        logger.info(f"Retrying training with modified configuration...")
        print(f"⟳ Retrying training with fixes applied...")

    try:
        logger.info(f"Starting training attempt #{error_tracker.attempt_count}")
        print(f"Starting training (attempt #{error_tracker.attempt_count})...")

        result = train_lora(
            base_model=config["base_model"],
            model_type=config["model_type"],
            image_dir=Path(config["image_dir"]),
            output_dir=Path(config["output_dir"]),
            style_token=config["style_token"],
            use_captions=config["use_captions"],
            resolution=config["resolution"],
            lora_rank=config["lora_rank"],
            learning_rate=config["learning_rate"],
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
            use_gradient_checkpointing=config["gradient_checkpointing"],
            use_fp16=config["mixed_precision"],
            max_train_steps=config["max_train_steps"],
            progress_callback=on_progress,
            status_callback=on_status,
            stop_flag=on_stop_flag,
        )

        logger.info(f"✓ Training completed successfully!")
        print(f"✓ Training completed successfully!")
        logger.info(f"Output saved to: {result}")

        return True, result, "Training completed successfully"

    except Exception as e:
        error_type = type(e).__name__
        error_message = str(e)

        # Log the error
        error_tracker.add_error(error_type, error_message)
        logger.exception(f"Training failed with {error_type}")

        # Try to apply automatic fixes
        fixes_applied = []
        modified_config = config.copy()

        # Common fixes
        if "out of memory" in error_message.lower() or "cuda" in error_message.lower():
            if modified_config["resolution"] > 512:
                modified_config["resolution"] = 512
                error_tracker.add_fix(f"Reduced resolution to 512")
                fixes_applied.append("resolution_reduction")
            elif modified_config["batch_size"] > 1:
                modified_config["batch_size"] = 1
                error_tracker.add_fix(f"Reduced batch size to 1")
                fixes_applied.append("batch_size_reduction")
            elif not modified_config["gradient_checkpointing"]:
                modified_config["gradient_checkpointing"] = True
                error_tracker.add_fix(f"Enabled gradient checkpointing")
                fixes_applied.append("gradient_checkpointing")
            elif not modified_config["mixed_precision"]:
                modified_config["mixed_precision"] = True
                error_tracker.add_fix(f"Enabled mixed precision (fp16)")
                fixes_applied.append("mixed_precision")

        # Return for retry with potentially modified config
        return False, None, error_message, (modified_config if fixes_applied else config)

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print_banner("STEP 2: CHECKING FOR INTERRUPTED TRAINING")

# Check for existing checkpoints
checkpoint_dir = OUTPUT_DIR / ".checkpoints"
resume_from_checkpoint = False
resume_checkpoint = None

if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*"))
    if checkpoints:
        resume_checkpoint = checkpoints[-1]
        logger.info(f"Found checkpoint: {resume_checkpoint.name}")

        # Load metadata
        metadata_file = resume_checkpoint / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                checkpoint_meta = json.load(f)

            # Check if training is incomplete
            if checkpoint_meta['step'] < checkpoint_meta['total_steps']:
                resume_from_checkpoint = True
                logger.info(f"Training can be resumed from Epoch {checkpoint_meta['epoch']}, Step {checkpoint_meta['step']}")
                print(f"✓ Resuming from checkpoint: Epoch {checkpoint_meta['epoch'] + 1}/{config_summary['num_epochs']}, "
                      f"Step {checkpoint_meta['step']}/{checkpoint_meta['total_steps']}")

if not resume_from_checkpoint:
    print("✓ Starting fresh training session")

print_banner("STEP 3: STARTING TRAINING WITH AUTO-RECOVERY")

success = False
current_config = config_summary.copy()
attempt = 0

while not success and attempt < MAX_ATTEMPTS:
    attempt += 1

    try:
        success, output_path, message, *rest = attempt_training(current_config)

        if success:
            print_banner("TRAINING SUCCESSFUL")
            logger.info(f"✓ Training succeeded on attempt #{attempt}")
            logger.info(f"LoRA weights saved to: {output_path}")
            break
        else:
            # Update config for next attempt if fixes were applied
            if rest and rest[0]:
                current_config = rest[0]
                logger.info(f"Retrying with modified configuration...")
                print(f"\n⟳ Retrying with fixes (attempt {attempt + 1}/{MAX_ATTEMPTS})...")
                time.sleep(2)  # Brief pause before retry
            else:
                logger.error(f"Training failed and no automatic fixes available")
                break

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        print("\n⚠ Training interrupted by user")
        break
    except Exception as e:
        logger.exception(f"Unexpected error during training loop: {e}")
        error_tracker.add_error("UnexpectedError", str(e))
        if attempt < MAX_ATTEMPTS:
            print(f"\n⟳ Retrying (attempt {attempt + 1}/{MAX_ATTEMPTS})...")
            time.sleep(2)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_banner("FINAL SUMMARY")

summary_lines = [
    f"Timestamp: {datetime.now()}",
    f"Total attempts: {error_tracker.attempt_count}",
    f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}",
    f"",
    f"Configuration used:",
    f"  Model: {current_config['model_type'].upper()}",
    f"  Resolution: {current_config['resolution']}",
    f"  LoRA Rank: {current_config['lora_rank']}",
    f"  Epochs: {current_config['num_epochs']}",
    f"  Batch Size: {current_config['batch_size']}",
    f"  Learning Rate: {current_config['learning_rate']}",
    f"  Gradient Checkpointing: {current_config['gradient_checkpointing']}",
    f"  Mixed Precision (fp16): {current_config['mixed_precision']}",
    f"",
]

if success:
    summary_lines.append(f"✓ Output saved to: {output_path}")
else:
    summary_lines.append(f"✗ Training did not complete successfully")

summary_lines.extend([
    f"",
    f"Logs saved to:",
    f"  Simulation log: {SIM_LOG_FILE}",
    f"  Error log: {ERROR_LOG_FILE}",
    f"  Fixes log: {FIXES_LOG_FILE}",
])

for line in summary_lines:
    print(line)
    logger.info(line)

# Print error tracker summary
summary = error_tracker.summary()
print(summary)
logger.info(summary)

# ============================================================================
# EXTRACT FINAL METRICS FROM LOG
# ============================================================================

def extract_final_metrics(log_file: Path) -> dict:
    """Extract final training metrics from the log file."""
    metrics = {
        "final_loss": None,
        "total_steps": None,
        "final_step": None,
        "final_epoch": None,
        "total_epochs": None,
    }

    try:
        content = log_file.read_text(errors='ignore')

        # Extract final loss
        import re
        losses = re.findall(r'loss: ([\d.]+)', content)
        if losses:
            metrics["final_loss"] = float(losses[-1])

        # Extract total steps
        steps = re.findall(r'Step (\d+)/(\d+)', content)
        if steps:
            metrics["final_step"] = int(steps[-1][0])
            metrics["total_steps"] = int(steps[-1][1])

        # Extract epochs
        epochs = re.findall(r'Epoch (\d+)/(\d+)', content)
        if epochs:
            metrics["final_epoch"] = int(epochs[-1][0])
            metrics["total_epochs"] = int(epochs[-1][1])

    except Exception as e:
        logger.warning(f"Could not extract metrics: {e}")

    return metrics

# ============================================================================
# GENERATE HUMAN-READABLE REPORT
# ============================================================================

def generate_text_report(
    success: bool,
    output_path: Optional[Path],
    config: dict,
    error_tracker,
    metrics: dict,
    timestamp: str
) -> str:
    """Generate a comprehensive human-readable report."""
    lines = [
        "╔" + "=" * 78 + "╗",
        "║" + " " * 78 + "║",
        "║" + "LORA TRAINING SIMULATION - FINAL REPORT".center(78) + "║",
        "║" + " " * 78 + "║",
        "╚" + "=" * 78 + "╝",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Simulation Duration: {timestamp}",
        "",
    ]

    # Status section
    status_icon = "✓" if success else "✗"
    status_text = "SUCCESSFUL" if success else "FAILED"
    lines.extend([
        "─" * 80,
        f"STATUS: {status_icon} {status_text}",
        "─" * 80,
        "",
    ])

    if success and output_path:
        lines.extend([
            "📦 TRAINED LORA MODEL",
            f"  Location: {output_path}",
            f"  Filename: {output_path.name}",
            f"  Size: {output_path.stat().st_size / (1024*1024):.2f} MB",
            "",
        ])

    # Training metrics section
    lines.extend([
        "📊 TRAINING METRICS",
    ])

    if metrics["final_loss"] is not None:
        lines.append(f"  Final Loss: {metrics['final_loss']:.6f}")

    if metrics["final_step"] is not None and metrics["total_steps"] is not None:
        progress = (metrics["final_step"] / metrics["total_steps"] * 100)
        lines.append(f"  Steps Completed: {metrics['final_step']}/{metrics['total_steps']} ({progress:.1f}%)")

    if metrics["final_epoch"] is not None and metrics["total_epochs"] is not None:
        lines.append(f"  Epochs Completed: {metrics['final_epoch']}/{metrics['total_epochs']}")

    lines.append("")

    # Configuration section
    lines.extend([
        "⚙️  CONFIGURATION USED",
        f"  Base Model: {config['base_model']}",
        f"  Model Type: {config['model_type'].upper()}",
        f"  Resolution: {config['resolution']}x{config['resolution']}",
        f"  LoRA Rank: {config['lora_rank']}",
        f"  Learning Rate: {config['learning_rate']}",
        f"  Batch Size: {config['batch_size']}",
        f"  Gradient Checkpointing: {config['gradient_checkpointing']}",
        f"  Mixed Precision (fp16): {config['mixed_precision']}",
        f"  Total Epochs: {config['num_epochs']}",
        "",
    ])

    # Dataset section
    lines.extend([
        "📁 DATASET",
        f"  Source Directory: {config['image_dir']}",
        f"  Output Directory: {config['output_dir']}",
        f"  Style Token: {config['style_token']}",
        f"  Use Captions: {config['use_captions']}",
        "",
    ])

    # Error handling section if there were errors
    if error_tracker.attempt_count > 1 or error_tracker.errors:
        lines.extend([
            "🔧 ERROR HANDLING",
            f"  Total Attempts: {error_tracker.attempt_count}",
            f"  Errors Encountered: {len(error_tracker.errors)}",
            f"  Fixes Applied: {len(error_tracker.fixes_applied)}",
            "",
        ])

        if error_tracker.errors:
            lines.append("  Errors:")
            for attempt, error_type, message in error_tracker.errors:
                lines.append(f"    [{attempt}] {error_type}: {message[:60]}")
            lines.append("")

        if error_tracker.fixes_applied:
            lines.append("  Fixes Applied:")
            for attempt, fix_desc in error_tracker.fixes_applied:
                lines.append(f"    [{attempt}] {fix_desc}")
            lines.append("")

    # Logging section
    lines.extend([
        "📝 LOG FILES",
        f"  Simulation Log: {SIM_LOG_FILE}",
        f"  Error Log: {ERROR_LOG_FILE}",
        f"  Fixes Log: {FIXES_LOG_FILE}",
        "",
    ])

    # Usage instructions section
    if success and output_path:
        lines.extend([
            "🚀 NEXT STEPS",
            f"  Your LoRA model is ready to use!",
            f"  ",
            f"  Location: {output_path}",
            f"  Style Token: {config['style_token']}",
            "",
            f"  Use this LoRA in your image generation tool with the",
            f"  style token '{config['style_token']}' to apply the",
            f"  Eudes Correia watercolor painting style.",
            "",
        ])
    else:
        lines.extend([
            "⚠️  TROUBLESHOOTING",
            f"  Check the log files above for detailed error information.",
            f"  Key log file: {SIM_LOG_FILE}",
            "",
        ])

    # Footer
    lines.extend([
        "╔" + "=" * 78 + "╗",
        "║" + " " * 78 + "║",
        "║" + "END OF REPORT".center(78) + "║",
        "║" + " " * 78 + "║",
        "╚" + "=" * 78 + "╝",
    ])

    return "\n".join(lines)

# ============================================================================
# SAVE REPORTS
# ============================================================================

metrics = extract_final_metrics(SIM_LOG_FILE)

# Generate and save text report
text_report = generate_text_report(
    success=success,
    output_path=output_path if success else None,
    config=current_config,
    error_tracker=error_tracker,
    metrics=metrics,
    timestamp=str(datetime.now())
)

report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
text_report_file = LOGS_DIR / f"REPORT_{report_timestamp}.txt"
with open(text_report_file, 'w') as f:
    f.write(text_report)

# Print the text report to console
print("\n" + text_report)

# Save JSON report (for programmatic access)
report = {
    "timestamp": datetime.now().isoformat(),
    "success": success,
    "output_lora": str(output_path) if success else None,
    "attempts": error_tracker.attempt_count,
    "metrics": metrics,
    "errors": [
        {
            "attempt": attempt,
            "type": error_type,
            "message": message
        }
        for attempt, error_type, message in error_tracker.errors
    ],
    "fixes_applied": [
        {
            "attempt": attempt,
            "description": fix_desc
        }
        for attempt, fix_desc in error_tracker.fixes_applied
    ],
    "configuration": current_config,
    "log_files": {
        "simulation": str(SIM_LOG_FILE),
        "errors": str(ERROR_LOG_FILE),
        "fixes": str(FIXES_LOG_FILE),
        "text_report": str(text_report_file),
    }
}

json_report_file = LOGS_DIR / f"report_{report_timestamp}.json"
with open(json_report_file, 'w') as f:
    json.dump(report, f, indent=2)

logger.info(f"Text report saved to: {text_report_file}")
logger.info(f"JSON report saved to: {json_report_file}")
print(f"\n✓ Text report saved to: {text_report_file}")
print(f"✓ JSON report saved to: {json_report_file}")

sys.exit(0 if success else 1)
