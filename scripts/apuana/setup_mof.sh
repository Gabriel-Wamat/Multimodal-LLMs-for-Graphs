#!/usr/bin/env bash
#SBATCH --job-name=sih-mof-setup
#SBATCH --partition=install
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
bash scripts/baseline.sh install --env-only
