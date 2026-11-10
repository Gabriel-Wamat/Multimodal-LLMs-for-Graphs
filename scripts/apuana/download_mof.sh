#!/usr/bin/env bash
#SBATCH --job-name=sih-mof-weights
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Download separado para não consumir o limite de 30 minutos da partição install.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
bash scripts/baseline.sh download
