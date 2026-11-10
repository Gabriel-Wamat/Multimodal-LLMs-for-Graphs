#!/usr/bin/env bash
#SBATCH --job-name=sih-control-audit
#SBATCH --partition=short-simple
#SBATCH --qos=simple
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
.venv-control/bin/python scripts/local/validate_control_smoke.py "$@"
