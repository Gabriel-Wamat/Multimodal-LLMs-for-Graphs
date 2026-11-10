#!/usr/bin/env bash
#SBATCH --job-name=sih-control-tests
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.venv-control/bin/python -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(4,device="cuda",dtype=torch.bfloat16); print(torch.__version__, torch.version.cuda, x.sum().item())'
.venv-control/bin/python -m unittest discover -s tests -v
