#!/usr/bin/env bash
#SBATCH --job-name=sih-mof-pilot
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
run_id=''
action=run
resume=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id) run_id="${2:?Missing run ID}"; shift 2 ;;
        --smoke) action=smoke; shift ;;
        --resume) resume=(--resume); shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
test -n "$run_id"
bash scripts/baseline.sh "$action" --run-id "$run_id" "${resume[@]}"
