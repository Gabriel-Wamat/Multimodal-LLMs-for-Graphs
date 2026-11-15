#!/usr/bin/env bash
#SBATCH --job-name=graph-full-sft-check
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from a frozen project snapshot}"
model=${1:?Use qwen or llava}
prefix=${2:?Provide a new GPU-check prefix}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv
for arm in clip_single clip_double clip_dino; do
 .venv-control/bin/python -m src.training.check_full_sft --model "$model" --arm "$arm" --run-id "${prefix}-${model}-${arm}"
done
