"""Export an allowlisted public summary without raw predictions or machine paths."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def completed(run_id):
    folder = ROOT / "results" / run_id
    complete = read(folder / "complete.json")
    run = read(folder / "run.json")
    metrics = read(folder / "scored/metrics.json")
    prediction = folder / "predictions.jsonl"
    sha = digest(prediction)
    if sha != complete["predictions_sha256"] or sha != metrics["predictions_sha256"]:
        raise ValueError("Prediction digest mismatch: " + run_id)
    rows = [json.loads(line) for line in prediction.read_text().splitlines()]
    keys = [(row["id"], row.get("arm", row.get("condition"))) for row in rows]
    if len(keys) != len(set(keys)) or any(row["status"] != "ok" for row in rows):
        raise ValueError("Duplicate or failed responses: " + run_id)
    source = {"run_id": run_id, "prediction_sha256": sha,
              "metrics_sha256": digest(folder / "scored/metrics.json"),
              "run_sha256": digest(folder / "run.json"), "records": len(rows)}
    return metrics, run, source


def measure(row):
    return {key: row[key] for key in ("n", "correct", "accuracy")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", required=True, type=date.fromisoformat)
    parser.add_argument("--output", type=Path, default=ROOT / "results/published/summary.json")
    args = parser.parse_args()
    sources, pilot, validation = [], [], []
    metrics, run, source = completed("mof13b-pilot50-v1")
    sources.append(source)
    diagnostic = read(ROOT / "results/mof13b-pilot50-v1/diagnostics/binary_aliases_posthoc.json")
    if diagnostic["predictions_sha256"] != source["prediction_sha256"]:
        raise ValueError("Binary diagnostic source differs")
    baseline = {"model": "I-MoF 13B", "model_id": run["config"]["model_id"],
                "revision": run["config"]["model_revision"], "strict_v1": measure(metrics),
                "binary_aliases_posthoc": {"n": diagnostic["n"],
                    "correct": diagnostic["diagnostic_correct"],
                    "accuracy": diagnostic["diagnostic_accuracy"]}}
    metrics, _, source = completed("mof-dinov3-stage0-pilot50-v1-llava")
    sources.append(source)
    selected = ["native", "native_double", "clip_double", "clip_dino"]
    names = {"native": "Nativo", "native_double": "Nativo duplicado",
             "clip_double": "CLIP duplicado", "clip_dino": "CLIP + DINOv3"}
    for arm in selected:
        pilot.append({"model": "LLaVA-1.5-7B", "arm": arm, "configuration": names[arm],
                      "before": measure(metrics["metrics"][arm]["primary_binary_aliases_v2"]),
                      "after": None})
    for model in ("llava", "qwen"):
        for arm in ("clip_double", "clip_dino"):
            run_id = f"aligned-pilot50-v1-{model}-{arm}"
            if not (ROOT / "results" / run_id / "complete.json").exists():
                if model == "qwen":
                    pilot.append({"model": "Qwen2-VL-2B", "arm": arm,
                                  "configuration": names[arm], "before": None, "after": None,
                                  "status": "final_evaluation_not_confirmed"})
                continue
            metrics, run, source = completed(run_id)
            sources.append(source)
            result = measure(metrics["primary_binary_aliases_v2"])
            if model == "llava":
                next(row for row in pilot if row["model"] == "LLaVA-1.5-7B" and row["arm"] == arm)["after"] = result
            else:
                pilot.append({"model": "Qwen2-VL-2B", "arm": arm,
                              "configuration": names[arm], "before": None, "after": result})
            training = ROOT / "experiments/training" / run["training_run"]
            history = read(training / "history.json")
            chosen = next(row for row in history if row["epoch"] == run["selected_epoch"])
            validation.append({"model": model, "arm": arm, "epoch": run["selected_epoch"],
                               "n": chosen["development"]["questions"],
                               "accuracy": chosen["development"]["accuracy"],
                               "loss": chosen["development"]["loss"],
                               "checkpoint_sha256": run["checkpoint_sha256"]})
    pairs = []
    for model in ("llava", "qwen"):
        folder = ROOT / "results" / ("paired-mitigations-full-v1-" + model)
        metrics = read(folder / "scored/metrics.json")
        complete = read(folder / "complete.json")
        if digest(folder / "predictions.jsonl") != complete["predictions_sha256"]:
            raise ValueError("Paired prediction digest mismatch")
        for condition, row in metrics["metrics"].items():
            pairs.append({"model": model, "condition": condition,
                          "question_accuracy": row["question_accuracy"],
                          "pair_accuracy": row["paired_accuracy"]["estimate"],
                          "pair_ci95": row["paired_accuracy"]["ci95"],
                          "invalid_responses": row["invalid_responses"], "pairs": 720})
        sources.append({"run_id": folder.name, "prediction_sha256": complete["predictions_sha256"],
                        "metrics_sha256": digest(folder / "scored/metrics.json")})
    result = {"schema_version": 1, "evidence_snapshot_date": args.snapshot_date.isoformat(),
              "scope": "exploratory; historical evidence, not live scheduler state",
              "pilot_questions": 3173, "pilot_graphs": 360,
              "scorer": "primary_binary_aliases_v2; original I-MoF strict score kept separately",
              "official_baseline": baseline, "pilot_results": pilot,
              "validation_results": validation, "paired_results": pairs,
              "sources": sources,
              "limitations": ["One initialization per arm; no robustness claim across seeds",
                "Pilot previously explored; unknown upstream independence",
                "Duplicated CLIP controls tokens, not trainable parameters or initialization",
                "Frozen deterministic grader; full semantic audit remains pending; no BERTScore",
                "Raw data, model weights and full private run archives are not distributed here",
                "Unavailable results are not zeros and not evidence of running jobs"]}
    counts = Counter((row["model"], row["arm"]) for row in pilot)
    if any(count != 1 for count in counts.values()):
        raise ValueError("Repeated result configuration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
