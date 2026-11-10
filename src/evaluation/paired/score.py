"""Score completed paired runs with family-aware uncertainty and fixed contrasts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from src.data.pilot import ROOT, write_once, json_text
from src.models.control.artifacts import digest
from .metrics import score_pairs, family_bootstrap, compare


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    args = p.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Invalid ID")
    folder = ROOT / "results" / args.run_id
    run = json.loads((folder / "run.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    assert digest(folder / "predictions.jsonl") == complete["predictions_sha256"]
    base = ROOT / "data/manifests" / ("pairs_v1_smoke" if run["smoke"] else "pairs_v1")
    assert (
        digest(base / "inputs.jsonl") == run["inputs_sha256"]
        and digest(base / "labels.jsonl") == run["labels_sha256"]
    )
    ids = set(run["item_ids"])
    inputs = [r for r in read(base / "inputs.jsonl") if r["id"] in ids]
    labels = read(base / "labels.jsonl")
    split_path = ROOT / "data/manifests/train_dev_v2/split.json"
    split = json.loads(split_path.read_text())
    families = {g: f for f, gs in split["family_members"].items() for g in gs}
    predictions = read(folder / "predictions.jsonl")
    assert (
        len(predictions) == len(inputs) * len(run["conditions"]) == complete["records"]
    )
    assert {r["condition"] for r in predictions} == set(run["conditions"])
    details = {}
    metrics = {}
    cost = {}
    for condition in run["conditions"]:
        subset = [r for r in predictions if r["condition"] == condition]
        pairs = score_pairs(inputs, labels, subset, families)
        details[condition] = pairs
        groups = defaultdict(list)
        for r in pairs:
            groups[r["attribute"] + "/" + r["intensity"]].append(r)
        metrics[condition] = {
            "paired_accuracy": family_bootstrap(pairs),
            "question_accuracy": sum(r["question_hits"] for r in pairs)
            / (2 * len(pairs)),
            "invalid_responses": sum(r["invalid_responses"] for r in pairs),
            "by_attribute_intensity": {
                k: family_bootstrap(v) for k, v in groups.items()
            },
        }
        cost[condition] = {
            "seconds": sum(r["seconds"] for r in subset),
            "model_calls": sum(len(r["calls"]) for r in subset),
            "input_tokens": sum(r["input_tokens"] for r in subset),
            "generated_tokens": sum(r["generated_tokens"] for r in subset),
            "peak_gpu_bytes": max(r["peak_gpu_bytes"] for r in subset),
        }
    report = {
        "model": run["model"],
        "smoke": run["smoke"],
        "metrics": metrics,
        "costs": cost,
        "contrasts": {
            c: compare(details["original"], details[c])
            for c in run["conditions"]
            if c != "original"
        },
        "predictions_sha256": digest(folder / "predictions.jsonl"),
        "families_sha256": digest(split_path),
        "scorer_sha256": digest(Path(__file__)),
        "metric_source_sha256": digest(Path(__file__).with_name("metrics.py")),
        "interpretation": "Technical diagnosis only"
        if run["smoke"]
        else "Exploratory paired comparison; family independence not guaranteed",
        "mcnemar_reference": "https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html",
    }
    write_once(folder / "scored/metrics.json", json_text(report))
    write_once(
        folder / "scored/pairs.jsonl",
        "".join(
            json.dumps(dict(r, condition=c), ensure_ascii=False) + "\n"
            for c, rs in details.items()
            for r in rs
        ),
    )
    print(
        json_text(
            {
                "model": run["model"],
                "pairs": len(details["original"]),
                "smoke": run["smoke"],
                "status": "scored",
            }
        )
    )


if __name__ == "__main__":
    main()
