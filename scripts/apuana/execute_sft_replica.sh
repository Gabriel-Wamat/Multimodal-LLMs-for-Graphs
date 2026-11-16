#!/usr/bin/env bash
#SBATCH --job-name=graph-replica
#SBATCH --partition=long-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
model=${1:?Provide model name}
seed=${2:?Provide replica seed}
analysis_snapshot=${3:?Provide frozen confirmation evaluator directory}
backup=${4:?Provide private backup directory}
case "$model" in qwen|llava) ;; *) exit 2 ;; esac
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen training replica}"
python3 - "$seed" <<'PY'
import json
from pathlib import Path
import sys
seed = int(sys.argv[1])
record = json.loads(Path("replica.json").read_text())
if seed != record["seed"]:
    raise ValueError("Replica seed differs from the submission")
PY
prefix="graph-replica-$seed"
checks="graph-check-$seed"
bash scripts/apuana/check_full_sft.sh "$model" "$checks"
export SFT_BACKUP_ROOT="$backup/$prefix"
bash scripts/apuana/run_full_sft.sh "$model" "$prefix" "$checks"
cd "$analysis_snapshot"
SLURM_SUBMIT_DIR="$PWD" bash scripts/apuana/evaluate_confirmation.sh "$prefix" "$model"
