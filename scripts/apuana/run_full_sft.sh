#!/usr/bin/env bash
#SBATCH --job-name=graph-full-sft
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from a frozen project snapshot}"
model=${1:?Use qwen or llava}
prefix=${2:?Provide a new campaign prefix}
checks=${3:?Provide the validated GPU-check prefix}
: "${SFT_BACKUP_ROOT:?Set a private backup directory outside the project}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
.venv-control/bin/python -m scripts.apuana.full_sft_campaign --model "$model" --prefix "$prefix" --checks-prefix "$checks" --backup-root "$SFT_BACKUP_ROOT"
