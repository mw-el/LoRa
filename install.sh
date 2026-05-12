#!/usr/bin/env bash
set -euo pipefail

# Quick installer for the Style LoRA Trainer.
# Usage: ./install.sh [env-name]
# Defaults to: lora-lab

ENV_NAME="${1:-lora-lab}"
REQ_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lora_trainer_gui/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

echo "Using conda environment: ${ENV_NAME}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} already exists. Skipping creation."
else
  echo "Creating environment..."
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

echo "Installing Python dependencies..."
conda run -n "${ENV_NAME}" pip install --upgrade pip
conda run -n "${ENV_NAME}" pip install -r "${REQ_FILE}"

echo
echo "Done."
echo "Activate the environment with: conda activate ${ENV_NAME}"
echo "Launching the app..."
conda run -n "${ENV_NAME}" python -m lora_trainer_gui.main
