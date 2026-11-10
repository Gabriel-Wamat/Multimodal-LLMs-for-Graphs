"""Audit candidate split against source integrity and cross-split near duplicates."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    split_path = ROOT / "data/manifests/train_dev_v1/split.json"
    split = json.loads(split_path.read_text())
    raw = ROOT / "dados_projeto_multi_modais/dados"
    role = {
        g: k
        for k in ("train", "development", "evaluation_exploratory")
        for g in split[k]
    }
    assert len(role) == sum(
        len(split[k]) for k in ("train", "development", "evaluation_exploratory")
    )
    assert (
        digest(ROOT / "data/manifests/pilot50_v1/split.json")
        == split["pilot_split_sha256"]
    )
    buckets = defaultdict(list)
    for graph, expected in split["source_sha256"].items():
        path = raw / (graph + ".csv")
        assert digest(path) == expected, graph
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        numbers = []
        layout = []
        for row in rows[1:]:
            for value in row:
                try:
                    v = float(value)
                    assert np.isfinite(v)
                    numbers.append(v)
                    layout.append("<numeric>")
                except ValueError:
                    layout.append(value)
        vector = np.asarray(numbers, dtype=float).reshape(len(rows) - 1, -1)
        key = (graph.split("-")[0], tuple(rows[0]), tuple(layout))
        if graph in role:
            buckets[key].append((graph, vector))
    candidates = []
    comparisons = 0
    for group in buckets.values():
        for i, (a, x) in enumerate(group):
            for b, y in group[i + 1 :]:
                comparisons += 1
                active = np.any(np.abs(x - y) > 1e-12, axis=0)
                xx, yy = x[:, active], y[:, active]
                raw_distance, normalized = 0.0, 0.0
                if active.any():
                    sx, sy = np.std(xx, axis=0), np.std(yy, axis=0)
                    scale = np.maximum(np.maximum(sx, sy), 1e-8)
                    raw_distance = float(np.sqrt(np.mean(((xx - yy) / scale) ** 2)))
                    valid = (sx > 1e-8) & (sy > 1e-8)
                    normalized = None
                    if xx.shape[0] >= 6 and valid.any():
                        nx = (xx[:, valid] - xx[:, valid].mean(axis=0)) / sx[valid]
                        ny = (yy[:, valid] - yy[:, valid].mean(axis=0)) / sy[valid]
                        normalized = float(np.sqrt(np.mean((nx - ny) ** 2)))
                if raw_distance <= 0.01 or (
                    normalized is not None and normalized <= 0.02
                ):
                    candidates.append(
                        {
                            "a": a,
                            "b": b,
                            "split_a": role[a],
                            "split_b": role[b],
                            "raw_standardized_rmse": raw_distance,
                            "affine_normalized_rmse": normalized,
                        }
                    )
    manifest = json.loads((ROOT / "data/manifests/imagens.json").read_text())
    source_checks = []
    for relative, expected in manifest["input_sha256"].items():
        p = ROOT / "dados_projeto_multi_modais" / relative
        assert digest(p) == expected, relative
        source_checks.append(relative)
    result = {
        "split_sha256": digest(split_path),
        "auditor_sha256": digest(Path(__file__)),
        "source_integrity_pass": True,
        "checked_sources": len(source_checks),
        "all_split_comparisons": comparisons,
        "near_duplicate_candidates": candidates,
        "thresholds": {
            "raw_standardized_rmse_max": 0.01,
            "affine_normalized_rmse_max": 0.02,
            "min_affine_values": 6,
        },
        "limitations": [
            "Heuristic screening, not proof of independent data generation.",
            "Common synthetic templates are shared; evaluation is exploratory.",
            "Dataset supplied by user; upstream generation history is unavailable.",
        ],
        "training_gate": "review_candidates"
        if candidates
        else "screening_pass_exploratory_only",
    }
    out = ROOT / "experiments/five-stage-campaign/data_audit_v3_all_families.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if out.exists():
        assert out.read_text() == text, (
            "Audit differs; version the audit before replacing"
        )
    else:
        out.write_text(text)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["checked_sources", "all_split_comparisons", "training_gate"]
            }
        )
    )
    print("near_duplicate_candidates", len(candidates))


if __name__ == "__main__":
    main()
