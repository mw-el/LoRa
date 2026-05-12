#!/usr/bin/env bash
set -euo pipefail

# Usage: ./start.sh [env-name]
# Default environment name: lora-lab

ENV_NAME="${1:-lora-lab}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

echo "Activating conda environment: ${ENV_NAME}"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment ${ENV_NAME} not found. Run ./install.sh ${ENV_NAME} first." >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python -m lora_trainer_gui.main
