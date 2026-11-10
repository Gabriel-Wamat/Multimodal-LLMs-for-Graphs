"""Development-only relative thresholds; retain every evaluation pair."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from src.data.pilot import ROOT, json_text, write_once
from src.models.control.artifacts import digest


ENCODERS = ("clip", "dinov3")
QUANTILES = (0.5, 0.75, 0.9)


def validate_scores(rows):
    if not rows or len({r["pair_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate similarity rows")
    for row in rows:
        for encoder in ENCODERS:
            score = row[encoder]
            if not np.isfinite(score) or not -1 <= score <= 1:
                raise ValueError("Invalid cosine similarity")


def calibrate(rows):
    validate_scores(rows)
    development = [r for r in rows if r["split"] == "development"]
    if not development:
        raise ValueError("Development pairs are required; no evaluation fallback")
    return {
        "definition": "high means at or above the selected development quantile; relative rank, not perceptual equivalence",
        "primary_quantile": 0.75,
        "quantiles": {
            str(q): {
                key: float(
                    np.quantile([r[key] for r in development], q, method="linear")
                )
                for key in ENCODERS
            }
            for q in QUANTILES
        },
        "development_pair_ids": sorted(r["pair_id"] for r in development),
        "calibration_uses_model_predictions": False,
        "limitations": "No semantic high/low labels are available; quantiles describe the development distribution. Correlated families remain grouped for outcome uncertainty.",
    }


def quadrant(row, thresholds):
    a, b = (row[k] >= thresholds[k] for k in ENCODERS)
    return {
        (True, True): "high_both",
        (True, False): "high_clip_only",
        (False, True): "high_dinov3_only",
        (False, False): "low_both",
    }[(a, b)]


def summarize(rows, calibration):
    validate_scores(rows)
    evaluation = [r for r in rows if r["split"] == "evaluation_exploratory"]
    if not evaluation:
        raise ValueError("No evaluation pairs")
    if set(calibration["development_pair_ids"]) & {r["pair_id"] for r in evaluation}:
        raise ValueError("Calibration overlaps evaluation")
    sensitivity = {}
    for q, thresholds in calibration["quantiles"].items():
        counts = Counter(quadrant(r, thresholds) for r in evaluation)
        strata = {}
        for field in ["attribute", "intensity", "estilo"]:
            strata[field] = {
                value: dict(
                    Counter(
                        quadrant(r, thresholds) for r in evaluation if r[field] == value
                    )
                )
                for value in sorted({r[field] for r in evaluation})
            }
        sensitivity[q] = {
            "thresholds": thresholds,
            "counts": dict(counts),
            "by_stratum": strata,
        }
    return {
        "pairs": len(evaluation),
        "all_pairs_retained": True,
        "sensitivity": sensitivity,
        "historical_threshold_adaptation": {
            "clip_gt_095_and_dinov3_lt_06": sum(
                r["clip"] > 0.95 and r["dinov3"] < 0.6 for r in evaluation
            ),
            "not_original_filter_replication": "The paper used DINOv2 ViT-L/14; this experiment uses DINOv3 ViT-B/16 and fixed chart pairs.",
            "source": "https://arxiv.org/html/2401.06209v2#S2.SS1",
        },
        "interpretation": "High cosine in both encoders is a candidate global-representation limitation, not proof that all visual information is absent.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    folder = args.run if args.run.is_absolute() else ROOT / args.run
    complete = json.loads((folder / "complete.json").read_text())
    scores_path = folder / "scores.json"
    if digest(scores_path) != complete["scores_sha256"]:
        raise ValueError("Representation scores changed")
    rows = json.loads(scores_path.read_text())
    if len(rows) != complete["pairs"]:
        raise ValueError("Incomplete representations")
    run_path = folder / "run.json"
    if digest(run_path) != complete["run_sha256"]:
        raise ValueError("Representation protocol changed")
    inputs_path = ROOT / "data/manifests/pairs_v1/inputs.jsonl"
    run = json.loads(run_path.read_text())
    if digest(inputs_path) != run["inputs_sha256"]:
        raise ValueError("Expected full fixed pairs manifest; smoke cannot calibrate")
    inputs = [json.loads(line) for line in inputs_path.read_text().splitlines()]
    expected = {r["pair_id"]: r for r in inputs}
    if set(expected) != {r["pair_id"] for r in rows}:
        raise ValueError("Pair coverage differs from fixed manifest")
    for row in rows:
        for field in ["origin", "split", "tipo", "estilo", "attribute", "intensity"]:
            if row[field] != expected[row["pair_id"]][field]:
                raise ValueError("Pair metadata changed")
    calibration = calibrate(rows)
    report = summarize(rows, calibration)
    lineage = {
        "scores_sha256": digest(scores_path),
        "analysis_source_sha256": digest(Path(__file__)),
    }
    write_once(folder / "calibration.json", json_text(calibration | lineage))
    write_once(folder / "similarity_report.json", json_text(report | lineage))


if __name__ == "__main__":
    main()
