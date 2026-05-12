from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import json


CONFIG_PATH = Path.home() / ".lora_trainer_gui_config.json"


@dataclass
class DefaultConfig:
    model_type: str = "SD 1.5"
    resolution: int = 768  # SD1.5 high-but-safer default for 8GB
    lora_rank: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    max_train_steps: int | None = None
    base_model: str = "/home/matthias/_AA_ComfyUI/models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors"
    output_dir: str = ""
    image_dir: str = ""
    style_token: str = ""
    use_captions: bool = False


def load_last_settings() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        # Do not crash the app if the config file is malformed.
        return {}
    return {}


def save_settings(data: Dict[str, Any]) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        # If persisting fails we simply ignore; this should not block the GUI.
        return


def default_config() -> DefaultConfig:
    return DefaultConfig()
