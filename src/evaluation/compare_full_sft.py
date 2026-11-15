"""Compare three full-SFT arms on the fixed pilot, with paired uncertainty."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.compare_alignment import contrast
from src.evaluation.control.score import normalize_binary, paired_graph_delta
from src.evaluation.score import canonical, evaluate, norm, read_jsonl

ARMS = ("clip_single", "clip_double", "clip_dino")


def read(path):
    return json.loads(path.read_text())


def error_type(row):
    if row["correct"]:
        return "correct"
    if row["reason"] == "empty":
        return "empty"
    if row["reason"] == "invalid_numeric_format":
        return "invalid_numeric_format"
    if row["reason"] == "numeric_interval":
        return "outside_numeric_interval"
    response, fmt = row["response"], row["formato"]
    if canonical(response, fmt) is None or (
        fmt == "sim/não" and norm(response) not in {"sim", "nao"}
    ):
        return "invalid_format"
    return "answer_mismatch_semantics_unreviewed"


def collect(root, prefix, model, arm, labels, split, dataset):
    training_id = f"{prefix}-{model}-{arm}"
    training = root / "experiments/full_sft" / training_id
    suffix = "-pilot" if dataset == "pilot50_v1" else "-confirmation"
    out = root / "results" / (training_id + suffix)
    run = read(out / "run.json")
    record = read(training / "run.json")
    complete = read(out / "complete.json")
    expected = {
        "training_kind": "full_sft",
        "training_run": training_id,
        "arm": arm,
        "model": model,
        "inputs_sha256": split["inputs_sha256"],
        "labels_sha256": split["labels_sha256"],
        "training_record_sha256": sha(training / "run.json"),
        "training_complete_sha256": sha(training / "complete.json"),
        "checkpoint_sha256": read(training / "best.json")["sha256"],
    }
    if run.get("dataset", "pilot50_v1") != dataset:
        raise ValueError("Unexpected evaluation dataset")
    if (
        dataset == "confirmation_v1"
        and run["confirmation_protocol_sha256"] != split["protocol_sha256"]
    ):
        raise ValueError("Unexpected confirmation protocol")
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Incompatible evaluation record for {training_id}")
    if (record["arm"], record["model"]) != (arm, model):
        raise ValueError("Unexpected training arm")
    for filename, key in [
        ("predictions.jsonl", "predictions_sha256"),
        ("metrics.json", "metrics_sha256"),
    ]:
        if sha(out / filename) != complete[key]:
            raise ValueError(f"Modified {filename} for {training_id}")
    rows = read_jsonl(out / "predictions.jsonl")
    metrics, details = evaluate(labels, normalize_binary(labels, rows))
    if (
        not metrics["all_records_successful"]
        or metrics != read(out / "metrics.json")["primary_binary_aliases_v2"]
    ):
        raise ValueError("Incomplete or inconsistent scores")
    times = np.array([r["seconds"] for r in rows])
    summary = {
        "primary": metrics,
        "error_counts": dict(Counter(map(error_type, details))),
        "inference_seconds": float(times.sum()),
        "median_seconds": float(np.median(times)),
        "p95_seconds": float(np.quantile(times, 0.95)),
        "peak_gpu_bytes": max(r["peak_gpu_bytes"] for r in rows),
        "generated_tokens": sum(r["generated_tokens"] for r in rows),
        "selected_epoch": run["selected_epoch"],
        "checkpoint_sha256": run["checkpoint_sha256"],
        "predictions_sha256": complete["predictions_sha256"],
    }
    shared = [
        record[key] for key in ("config", "data_sha256", "weights", "base_config", "source_sha256")
    ]
    shared.extend([run["scorers"], run["source_sha256"]])
    return details, summary, shared


def compare_model(root, prefix, model, dataset="pilot50_v1"):
    if dataset not in ("pilot50_v1", "confirmation_v1"):
        raise ValueError("Unknown evaluation dataset")
    pilot = root / "data/manifests" / dataset
    split = read(pilot / ("split.json" if dataset == "pilot50_v1" else "complete.json"))
    for key in ("inputs", "labels"):
        if sha(pilot / f"{key}.jsonl") != split[f"{key}_sha256"]:
            raise ValueError("Pilot manifest changed")
    labels = read_jsonl(pilot / "labels.jsonl")
    if dataset == "pilot50_v1":
        family_path = root / "data/manifests/train_dev_v2/split.json"
        families = {
            g: f for f, graphs in read(family_path)["family_members"].items() for g in graphs
        }
    else:
        family_path = pilot / "sources.jsonl"
        if (
            sha(family_path) != split["sources_sha256"]
            or sha(pilot / "protocol.json") != split["protocol_sha256"]
        ):
            raise ValueError("Confirmation sources changed")
        families = {row["grafico"]: row["grafico"] for row in read_jsonl(family_path)}
        if set(families) != {row["grafico"] for row in labels} or len(families) != split["graphs"]:
            raise ValueError("Confirmation chart membership differs")
    details, summary, protocols = {}, {}, []
    for arm in ARMS:
        details[arm], summary[arm], protocol = collect(
            root, prefix, model, arm, labels, split, dataset
        )
        protocols.append(protocol)
    if not all(p == protocols[0] for p in protocols):
        raise ValueError("Arms differ in training or evaluation protocol")
    comparisons = {}
    for control in ARMS[:2]:
        comparisons[control] = {
            "fusion_minus_control_by_graph": paired_graph_delta(
                labels, details[control], details["clip_dino"]
            ),
            "fusion_minus_control_by_family": contrast(
                details, {control: -1, "clip_dino": 1}, families
            ),
        }
    train_cfg, data_hashes, weights, visual_cfg, sources, scorers, inference = protocols[0]
    seed = train_cfg["seed"]
    adapter_seed = visual_cfg.get("adapter_seed")
    fixed = {
        "training": {k: v for k, v in train_cfg.items() if k != "seed"},
        "vision": {k: v for k, v in visual_cfg.items() if k != "adapter_seed"},
        "weights": weights,
        "training_data": data_hashes,
        "training_code": sources,
        "scorers": scorers,
        "inference_code": inference,
        "dataset": dataset,
        "evaluation_data": [split["inputs_sha256"], split["labels_sha256"], sha(family_path)],
    }
    return {
        "model": model,
        "dataset": dataset,
        "seed": seed,
        "adapter_seed": adapter_seed,
        "matched_protocol_sha256": hashlib.sha256(
            json.dumps(fixed, sort_keys=True).encode()
        ).hexdigest(),
        "prefix": prefix,
        "arms": summary,
        "comparisons": comparisons,
        "family_split_sha256": sha(family_path),
        "analysis_sha256": sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--model", choices=["qwen", "llava"], required=True)
    parser.add_argument(
        "--dataset", choices=["pilot50_v1", "confirmation_v1"], default="pilot50_v1"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,48}", args.prefix):
        parser.error("Invalid campaign prefix")
    report = compare_model(ROOT, args.prefix, args.model, args.dataset)
    path = ROOT / "results" / f"{args.prefix}-comparison" / f"{args.model}.json"
    if args.dataset == "confirmation_v1":
        path = path.with_name(f"{args.model}-confirmation.json")
    write_once(path, json_text(report))
    print(path)


if __name__ == "__main__":
    main()
