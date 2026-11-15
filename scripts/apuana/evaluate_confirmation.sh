#!/usr/bin/env bash
#SBATCH --job-name=graph-confirm
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
prefix=${1:?Provide training campaign prefix}
model=${2:?Provide model name}
case "$model" in qwen|llava) ;; *) exit 2 ;; esac
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
.venv-control/bin/python -m src.evaluation.compare_full_sft --prefix "$prefix" --model "$model"
for arm in clip_single clip_double clip_dino; do
    train_id="$prefix-$model-$arm"
    result_id="$train_id-confirmation"
    resume=()
    if [[ -d "results/$result_id" ]]; then resume+=(--resume); fi
    .venv-control/bin/python -m src.evaluation.full_sft \
        --dataset confirmation_v1 --training-run "$train_id" --run-id "$result_id" "${resume[@]}"
done
.venv-control/bin/python -m src.evaluation.compare_full_sft \
    --prefix "$prefix" --model "$model" --dataset confirmation_v1
for dataset in pilot50_v1 confirmation_v1; do
    .venv-control/bin/python -m scripts.local.export_full_sft_results \
        --prefix "$prefix" --model "$model" --dataset "$dataset"
done
