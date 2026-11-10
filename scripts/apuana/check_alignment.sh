#!/usr/bin/env bash
#SBATCH --job-name=sih-alignment-check
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
.venv-control/bin/python -m unittest discover -s tests -p test_alignment_masks.py -v
for model in llava qwen; do
  for arm in clip_double clip_dino; do
    .venv-control/bin/python -m src.training.check_alignment --model "$model" --arm "$arm" --run-id "alignment-check-v1-${model}-${arm}"
  done
done
