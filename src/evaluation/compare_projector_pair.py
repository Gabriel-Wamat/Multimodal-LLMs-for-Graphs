"""Compare projector-only fusion with its explicitly matched duplicate-CLIP control."""

import argparse
from pathlib import Path

from src.data.pilot import ROOT, json_text, write_once
from src.evaluation.compare_alignment import contrast, load_run
from src.evaluation.compare_full_sft import read
from src.evaluation.score import read_jsonl
from src.models.control.artifacts import digest


def training_protocol(training):
    keys = (
        "model",
        "config",
        "base_config",
        "manifest_sha256",
        "split_sha256",
        "weights",
        "source_sha256",
    )
    return {key: training[key] for key in keys}


def compare(model, control, fusion):
    base = ROOT / "data/manifests/pilot50_v1"
    split = read(base / "split.json")
    for name in ("inputs", "labels"):
        if digest(base / (name + ".jsonl")) != split[name + "_sha256"]:
            raise ValueError("Pilot manifest changed")
    labels = read_jsonl(base / "labels.jsonl")
    details, sources, scores, protocols = {}, {}, {}, []
    family_path = ROOT / "data/manifests/train_dev_v2/split.json"
    for arm, folder in (("clip_double", control), ("clip_dino", fusion)):
        if folder.resolve().parent != (ROOT / "results").resolve():
            raise ValueError("Expected a run directory under results")
        record = read(folder / "run.json")
        training_dir = ROOT / "experiments/training" / record["training_run"]
        training = read(training_dir / "run.json")
        complete = read(training_dir / "complete.json")
        if (training["arm"], training["model"], record["arm"]) != (arm, model, arm):
            raise ValueError("Wrong training arm or model")
        weights = training_dir / "best.safetensors"
        if weights.exists() and digest(weights) != record["checkpoint_sha256"]:
            raise ValueError("Selected projector weights changed")
        if (
            record["training_record_sha256"] != digest(training_dir / "run.json")
            or record["labels_sha256"] != split["labels_sha256"]
            or training["split_sha256"] != digest(family_path)
            or record["selected_epoch"] != complete["best"]["epoch"]
            or record["checkpoint_sha256"] != complete["best_sha256"]
            or complete["epochs"] != training["config"]["epochs"]
            or record["selection_used_evaluation"] is not False
        ):
            raise ValueError("Unverified training or checkpoint lineage")
        protocols.append((training_protocol(training), record["scorers"], record["source_sha256"]))
        rows, metrics, source = load_run(folder.name, labels, split["inputs_sha256"], model)
        details[arm] = rows[arm]
        scores[arm] = {
            rule: {k: metrics[arm][rule][k] for k in ("correct", "n", "accuracy")}
            for rule in ("strict", "primary")
        }
        sources[arm] = {
            **source,
            "checkpoint_file_verified": weights.exists(),
            "training_run": record["training_run"],
            "checkpoint_sha256": record["checkpoint_sha256"],
        }
    if protocols[0] != protocols[1]:
        raise ValueError("Projector control and fusion protocols differ")
    families = {g: f for f, graphs in read(family_path)["family_members"].items() for g in graphs}
    return {
        "model": model,
        "scores": scores,
        "sources": sources,
        "split_sha256": digest(base / "split.json"),
        "family_split_sha256": digest(family_path),
        "fusion_minus_control": contrast(details, {"clip_double": -1, "clip_dino": 1}, families),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["qwen", "llava"], required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.model, args.control, args.fusion)
    destination = args.control / "projector_pair" / args.fusion.name / "comparison.json"
    write_once(destination, json_text(result))
    print(destination)


if __name__ == "__main__":
    main()
