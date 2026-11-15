#!/usr/bin/env bash
#SBATCH --job-name=graph-proj-control
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the training snapshot}"
model=${1:?Provide the model}
training=${2:?Provide the duplicate-CLIP training run}
control=${3:?Provide a distinct evaluation run ID}
fusion=${4:?Provide the completed fusion evaluation ID}
analysis=${5:?Provide the directory with the comparison code}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
.venv-control/bin/python -m src.evaluation.trained --training-run "$training" --run-id "$control" --resume
cd "$analysis"
.venv-control/bin/python -m src.evaluation.compare_projector_pair --model "$model" \
    --control "results/$control" --fusion "results/$fusion"
