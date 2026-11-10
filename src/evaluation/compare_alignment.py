"""Re-score matched native, unaligned and aligned predictions with shared rules."""

import json
from pathlib import Path

from src.data.pilot import ROOT, json_text, write_once
from src.models.control.artifacts import digest
from src.evaluation.score import evaluate, read_jsonl
from src.evaluation.control.score import normalize_binary
from src.evaluation.paired.metrics import family_bootstrap


def contrast(details, coefficients, families):
    if not coefficients or sum(coefficients.values()) != 0:
        raise ValueError("A contrast must have coefficients summing to zero")
    indexed = {}
    for name in coefficients:
        rows = details[name]
        indexed[name] = {r["id"]: r for r in rows}
        if len(indexed[name]) != len(rows):
            raise ValueError("Duplicate question IDs")
    reference = next(iter(indexed.values()))
    if any(set(rows) != set(reference) for rows in indexed.values()):
        raise ValueError("Question sets differ")
    values = []
    for qid, row in reference.items():
        if any(r[qid]["grafico"] != row["grafico"] for r in indexed.values()):
            raise ValueError("Question source differs")
        values.append(
            {
                "family": families[row["grafico"]],
                "delta": sum(
                    coefficients[k] * int(indexed[k][qid]["correct"])
                    for k in coefficients
                ),
            }
        )
    result = family_bootstrap(values, "delta")
    result["questions"] = result.pop("pairs")
    result["coefficients"] = coefficients
    result["percentage_points"] = 100 * result["estimate"]
    if result["families"] < 2:
        result["ci95"] = None
        result["uncertainty_note"] = "At least two independent families are required"
    return result


def load_run(run_id, labels, inputs_sha, expected_model):
    folder = ROOT / "results" / run_id
    run = json.loads((folder / "run.json").read_text())
    if run["model"] != expected_model:
        raise ValueError("Model identity differs")
    complete = json.loads((folder / "complete.json").read_text())
    path = folder / "predictions.jsonl"
    if (
        digest(path) != complete["predictions_sha256"]
        or run["inputs_sha256"] != inputs_sha
    ):
        raise ValueError("Prediction/input provenance mismatch")
    rows = read_jsonl(path)
    training_summary = None
    if "training_run" in run:
        training = ROOT / "experiments/training" / run["training_run"]
        if digest(training / "complete.json") != run["training_complete_sha256"]:
            raise ValueError("Training provenance changed")
        history = json.loads((training / "history.json").read_text())
        epochs = [r for r in history if r["phase"] == "epoch_complete"]
        if not epochs:
            raise ValueError("Training history is incomplete")
        training_summary = {
            "selected_epoch": run["selected_epoch"],
            "trainable_parameters": epochs[-1]["trainable_parameters"],
            "epoch_count": len(epochs),
            "training_seconds": sum(r["train_seconds"] for r in epochs),
            "history_sha256": digest(training / "history.json"),
        }
    arms = run.get("arms", [run.get("arm")])
    if any(r["arm"] not in arms for r in rows):
        raise ValueError("Unexpected arm")
    if set(run["item_ids"]) != {r["id"] for r in labels}:
        raise ValueError("Evaluation membership differs")
    output = {}
    metrics = {}
    for arm in arms:
        subset = [r for r in rows if r["arm"] == arm]
        primary, details = evaluate(labels, normalize_binary(labels, subset))
        strict, _ = evaluate(labels, subset)
        if not primary["all_records_successful"]:
            raise ValueError("Incomplete model results")
        output[arm] = details
        metrics[arm] = {
            "primary": primary,
            "strict": strict,
            "seconds": sum(r["seconds"] for r in subset),
            "generated_tokens": sum(r["generated_tokens"] for r in subset),
            "peak_gpu_bytes": max(r["peak_gpu_bytes"] for r in subset),
        }
    return (
        output,
        metrics,
        {
            "run_id": run_id,
            "run_sha256": digest(folder / "run.json"),
            "predictions_sha256": digest(path),
            "selected_epoch": run.get("selected_epoch"),
            "training": training_summary,
        },
    )


def main():
    base = ROOT / "data/manifests/pilot50_v1"
    split = json.loads((base / "split.json").read_text())
    if (
        digest(base / "labels.jsonl") != split["labels_sha256"]
        or digest(base / "inputs.jsonl") != split["inputs_sha256"]
    ):
        raise ValueError("Fixed pilot changed")
    labels = read_jsonl(base / "labels.jsonl")
    family_path = ROOT / "data/manifests/train_dev_v2/split.json"
    family_split = json.loads(family_path.read_text())
    families = {g: f for f, gs in family_split["family_members"].items() for g in gs}
    report = {}
    for model in ["llava", "qwen"]:
        details, metrics, lineage = load_run(
            "mof-dinov3-stage0-pilot50-v1-" + model, labels, split["inputs_sha256"], model
        )
        sources = [lineage]
        for arm in ["clip_double", "clip_dino"]:
            trained, tm, source = load_run(
                "aligned-pilot50-v1-" + model + "-" + arm,
                labels,
                split["inputs_sha256"],
                model,
            )
            if set(trained) != {arm}:
                raise ValueError("Trained arm differs")
            details["trained_" + arm] = trained[arm]
            metrics["trained_" + arm] = tm[arm]
            sources.append(source)
        specs = {
            "fusion_vs_native_before": {"native": -1, "clip_dino": 1},
            "fusion_vs_native_after": {"native": -1, "trained_clip_dino": 1},
            "clip_alignment_gain": {"clip_double": -1, "trained_clip_double": 1},
            "fusion_alignment_gain": {"clip_dino": -1, "trained_clip_dino": 1},
            "fusion_vs_clip_after": {"trained_clip_double": -1, "trained_clip_dino": 1},
            "difference_in_alignment_gains": {
                "trained_clip_dino": 1,
                "clip_dino": -1,
                "trained_clip_double": -1,
                "clip_double": 1,
            },
        }
        report[model] = {
            "metrics": metrics,
            "contrasts": {
                name: contrast(details, c, families) for name, c in specs.items()
            },
            "sources": sources,
        }
    destination = ROOT / "results/alignment-comparison-v1/report.json"
    write_once(
        destination,
        json_text(
            {
                "models": report,
                "families_sha256": digest(family_path),
                "sources": {
                    str(p.relative_to(ROOT)): digest(p)
                    for p in [
                        Path(__file__),
                        ROOT / "src/evaluation/score.py",
                        ROOT / "src/evaluation/control/score.py",
                        ROOT / "src/evaluation/paired/metrics.py",
                    ]
                },
                "interpretation": "Exploratory matched question accuracy, whole-family bootstrap. Different trainable parameter counts; single seed. No causal claim or automatic hypothesis acceptance.",
            }
        ),
    )


if __name__ == "__main__":
    main()
