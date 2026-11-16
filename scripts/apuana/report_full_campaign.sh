#!/usr/bin/env bash
#SBATCH --job-name=graph-sft-report
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the analysis snapshot}"
primary=${1:?Provide the primary campaign prefix}
replica1=${2:?Provide the second campaign prefix}
replica2=${3:?Provide the third campaign prefix}
project=${4:?Provide project root with notebook environment}
for dataset in pilot50_v1 confirmation_v1; do
    case "$dataset" in pilot50_v1) suffix=pilot ;; confirmation_v1) suffix=confirmation ;; esac
    runs=()
    for model in qwen llava; do
        .venv-control/bin/python -m src.evaluation.seed_stability --model "$model" \
            --dataset "$dataset" --prefixes "$primary" "$replica1" "$replica2"
        for arm in clip_single clip_double clip_dino; do
            runs+=("$primary-$model-$arm-$suffix")
        done
    done
    .venv-control/bin/python -m src.evaluation.semantic_review --dataset "$dataset" \
        --runs "${runs[@]}" --output "results/$primary-semantic-review/$dataset"
done
cd "$project"
export PATH="$PWD/.venv-analysis/bin:$PATH"
mkdir -p results/notebook-previews
.venv-analysis/bin/jupyter-nbconvert --to html --execute --ExecutePreprocessor.timeout=300 \
    --output-dir results/notebook-previews notebooks/resultados_experimentos.ipynb
