#!/usr/bin/env bash
#SBATCH --job-name=sih-paired-full
#SBATCH --partition=short-complex
#SBATCH --qos=complex
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
model=${1:?Use llava or qwen}
run_id=${2:?Provide a unique paired run ID}
shift 2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
.venv-control/bin/python - "$model" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
model = sys.argv[1]
root = Path.cwd()
path = root / 'results' / ('paired-mitigations-smoke-v1-' + model)
record = json.loads((path / 'run.json').read_text())
complete = json.loads((path / 'complete.json').read_text())
if record['model'] != model or not record['smoke'] or complete['records'] != 72:
    raise ValueError('The paired GPU smoke test is incomplete')
if hashlib.sha256((path / 'predictions.jsonl').read_bytes()).hexdigest() != complete['predictions_sha256']:
    raise ValueError('Smoke predictions changed')
for name in ['src/evaluation/paired/conditions.py', 'src/models/control/vlm.py']:
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != record['source_sha256'][name]:
        raise ValueError('Inference implementation differs from smoke test: ' + name)
PY
.venv-control/bin/python -m src.evaluation.paired.run --model "$model" --run-id "$run_id" "$@"
.venv-control/bin/python -m src.evaluation.paired.score "$run_id"
