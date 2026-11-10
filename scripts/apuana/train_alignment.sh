#!/usr/bin/env bash
#SBATCH --job-name=sih-alignment-train
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
model=${1:?Use llava or qwen}
arm=${2:?Use clip_double or clip_dino}
run_id=${3:?Provide a unique training run ID}
shift 3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
# The Python runner verifies all four GPU diagnostics and the CPU resume gate.
# Optional trailing --resume restores last.pt at an optimizer boundary.
.venv-control/bin/python -m src.training.run --model "$model" --arm "$arm" --run-id "$run_id" "$@"
