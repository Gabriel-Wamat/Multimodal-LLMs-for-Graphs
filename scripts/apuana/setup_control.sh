#!/usr/bin/env bash
#SBATCH --job-name=sih-control-setup
#SBATCH --partition=install
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export PIP_CACHE_DIR="$PWD/.cache/pip"
python3 -c 'import sys; assert sys.version_info[:2] == (3,10), "Python 3.10 required"'
python3 -m venv .venv-control
.venv-control/bin/python -m pip install -r configs/requirements-control.txt
.venv-control/bin/python -m pip check
mkdir -p .cache/control
.venv-control/bin/python -m pip freeze > .cache/control/environment.txt
.venv-control/bin/python -c 'from transformers import LlavaForConditionalGeneration,Qwen2VLForConditionalGeneration,DINOv3ViTModel; print("All architectures imported")'
