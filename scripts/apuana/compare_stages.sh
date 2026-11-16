#!/usr/bin/env bash
#SBATCH --job-name=graph-stage-report
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the comparison snapshot}"
.venv-control/bin/python -m src.evaluation.compare_training_stages \
    --prefix "${1:?Provide SFT prefix}" --model "${2:?Provide model}" --before "${3:?Provide untrained run ID}"
