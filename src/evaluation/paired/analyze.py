"""Join completed visual inference and representations into an H1-H3 report."""

import argparse
import json
from pathlib import Path

from src.data.pilot import ROOT, json_text, write_once
from src.models.control.artifacts import digest
from .hypotheses import analyze
from .metrics import compare, score_pairs
from .similarity import calibrate, summarize


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Invalid run ID")
    folder = ROOT / "results" / args.run_id
    run = json.loads((folder / "run.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    if run["smoke"]:
        raise ValueError("A technical smoke run cannot test the hypotheses")
    base = ROOT / "data/manifests/pairs_v1"
    for name in ["inputs", "labels"]:
        if digest(base / (name + ".jsonl")) != run[name + "_sha256"]:
            raise ValueError("Pair manifest changed")
    prediction_path = folder / "predictions.jsonl"
    if digest(prediction_path) != complete["predictions_sha256"]:
        raise ValueError("Predictions changed")
    inputs = [
        r for r in read(base / "inputs.jsonl") if r["split"] == "evaluation_exploratory"
    ]
    if set(run["item_ids"]) != {r["id"] for r in inputs}:
        raise ValueError("Full evaluation coverage is required")
    labels = read(base / "labels.jsonl")
    predictions = read(prediction_path)
    conditions = {"original", "crops_2x2", "transcription"}
    if (
        set(run["conditions"]) != conditions
        or {r["condition"] for r in predictions} != conditions
    ):
        raise ValueError("Three planned conditions are required")
    if len(predictions) != complete["records"] or len(predictions) != len(inputs) * 3:
        raise ValueError("Prediction coverage differs")
    split_path = ROOT / "data/manifests/train_dev_v2/split.json"
    split = json.loads(split_path.read_text())
    families = {g: f for f, gs in split["family_members"].items() for g in gs}
    details = {
        c: score_pairs(
            inputs, labels, [r for r in predictions if r["condition"] == c], families
        )
        for c in sorted(conditions)
    }
    representation = ROOT / "results/representations-pairs_v1-v1"
    rc = json.loads((representation / "complete.json").read_text())
    rr = json.loads((representation / "run.json").read_text())
    if (
        digest(representation / "scores.json") != rc["scores_sha256"]
        or digest(representation / "run.json") != rc["run_sha256"]
        or rr["inputs_sha256"] != run["inputs_sha256"]
    ):
        raise ValueError("Representations have incompatible provenance")
    scores = json.loads((representation / "scores.json").read_text())
    expected_pairs = {r["pair_id"] for r in read(base / "inputs.jsonl")}
    if len(scores) != rc["pairs"] or {r["pair_id"] for r in scores} != expected_pairs:
        raise ValueError("Representations incomplete")
    calibration = calibrate(scores)
    report = {
        "model": run["model"],
        "H1": summarize(scores, calibration),
        "calibration": calibration,
        "H2_and_similarity_error": {
            c: analyze(rows, scores) for c, rows in details.items()
        },
        "H3": {
            c: compare(details["original"], details[c])
            for c in sorted(conditions - {"original"})
        },
        "provenance": {
            "predictions": digest(prediction_path),
            "representations": rc["scores_sha256"],
            "families": digest(split_path),
            "analysis_sources": {
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "analyze.py",
                    "hypotheses.py",
                    "metrics.py",
                    "similarity.py",
                ]
            },
        },
        "limitations": [
            "Exploratory evaluation; upstream independence unknown",
            "No human answerability certification; rho +/-0.1 dispersion can be visually ambiguous",
            "All strata reported; McNemar p-values are unadjusted exploratory comparisons",
            "High global cosine alone does not establish absence of information in patch tokens",
        ],
    }
    write_once(folder / "scored/hypotheses.json", json_text(report))


if __name__ == "__main__":
    main()
