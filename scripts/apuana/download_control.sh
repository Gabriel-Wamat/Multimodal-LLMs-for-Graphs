#!/usr/bin/env bash
#SBATCH --job-name=sih-control-weights
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export HF_HUB_DISABLE_XET=1
.venv-control/bin/python -m src.models.control.download "$@"
