#!/usr/bin/env bash
# Run on an allocated CPU; the analysis environment is separate from model training.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv-analysis/bin/python -m src.data.confirmation
.venv-analysis/bin/python - <<'PY'
import json
from collections import Counter
from src.data.pilot import ROOT, sha
from src.evaluation.score import grade, read_jsonl
folder = ROOT / "data/manifests/confirmation_v1"
complete = json.loads((folder/"complete.json").read_text())
for name in ("sources", "inputs", "labels"):
    assert sha(folder/(name+".jsonl")) == complete[name+"_sha256"]
inputs = read_jsonl(folder/"inputs.jsonl")
labels = read_jsonl(folder/"labels.jsonl")
assert len(inputs) == len(labels) == 1080
assert len({row["grafico"] for row in inputs}) == 360
assert {r["id"] for r in inputs} == {r["id"] for r in labels} and len({r["id"] for r in inputs}) == len(inputs)
assert all(grade(row, str(row["resposta"]))[0] for row in labels)
assert set(Counter((r["tipo"],r["estilo"]) for r in inputs).values()) == {30}
for image, expected in {(r["image"],r["image_sha256"]) for r in inputs}:
    assert sha(ROOT/image) == expected
print(json.dumps(complete))
PY
