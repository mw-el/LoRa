# Style LoRA Trainer GUI

Local PyQt6 desktop app to train a style LoRA for Stable Diffusion models (SD 1.5 or SDXL). Point it at an image folder, pick a base model, enter a style token, and start training. The LoRA weights are saved as a `.safetensors` file.

## Setup

```bash
# Create/activate a conda env (adjust name/python as needed)
conda create -n lora-lab python=3.10 -y
conda activate lora-lab

pip install --upgrade pip
pip install -r lora_trainer_gui/requirements.txt
```

## Run

```bash
python -m lora_trainer_gui.main
# or
python lora_trainer_gui/main.py
```

## Using the app

1. Choose base model type (SDXL or SD 1.5) and enter a base model path or Hugging Face model ID.
2. Select an output folder where the `.safetensors` LoRA will be stored.
3. Pick your image folder. Optionally enable caption `.txt` files that sit next to images.
4. Enter a generic style token (e.g., `laura-style`).
5. Adjust training parameters if needed (resolution, rank, learning rate, epochs, etc.).
6. Click **Start Training**. Progress and errors appear in the log panel; the status bar shows the current phase.

The resulting LoRA file will be named `<style-token>_YYYYMMDD_HHMM.safetensors` in your chosen output folder.
