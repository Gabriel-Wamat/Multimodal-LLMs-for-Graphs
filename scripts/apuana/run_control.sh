#!/usr/bin/env bash
#SBATCH --job-name=sih-control-mof
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TORCH_HOME="$PWD/.cache/control/torch"
export PYTHONUNBUFFERED=1
argv=("$@")
run_id=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id) run_id="${2:?Missing run ID}"; shift 2 ;;
        *) shift ;;
    esac
done
.venv-control/bin/python -m src.evaluation.control.run "${argv[@]}"
.venv-control/bin/python -m src.evaluation.control.score "$run_id"
