"""Evaluate a selected full-decoder SFT checkpoint on the unchanged chart pilot."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import torch
from src.data.pilot import ROOT, json_text, write_once
from src.models.control.artifacts import digest, installed, write_json
from src.training.graph_sft import GraphSFTModel
from src.evaluation.trained import measured_answer
from src.evaluation.score import evaluate
from src.evaluation.control.score import normalize_binary


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--training-run", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dataset", choices=["pilot50_v1", "confirmation_v1"], default="pilot50_v1")
    a = p.parse_args()
    import re

    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", x) for x in [a.training_run, a.run_id]
    ):
        raise ValueError("Invalid ID")
    training = ROOT / "experiments/full_sft" / a.training_run
    record = json.loads((training / "run.json").read_text())
    complete = json.loads((training / "complete.json").read_text())
    best = json.loads((training / "best.json").read_text())
    cfg = record["config"]
    base = record["base_config"]
    if complete["stage2_epochs"] != cfg["stage2"]["epochs"]:
        raise ValueError("Incomplete training budget")
    checkpoint = training / "best-model.pt"
    if digest(checkpoint) != best["sha256"]:
        raise ValueError("Selected full checkpoint changed")
    for name in [
        "src/training/graph_sft.py",
        "src/training/alignment.py",
        "src/models/control/vlm.py",
        "src/models/control/encoders.py",
        "src/models/control/fusion.py",
        "src/evaluation/run.py",
    ]:
        if digest(ROOT / name) != record["source_sha256"][name]:
            raise ValueError("Inference source changed")
    weights = {k: installed(k, verify=True) for k in [record["model"], "clip", "dinov3"]}
    if weights != record["weights"]:
        raise ValueError("Pretrained visual/foundation source changed")
    if a.dataset == "confirmation_v1":
        dataset_dir = ROOT / "data/manifests/confirmation_v1"
        path = dataset_dir / "inputs.jsonl"
        labels_path = dataset_dir / "labels.jsonl"
        split = json.loads((dataset_dir / "complete.json").read_text())
        if (
            digest(dataset_dir / "protocol.json") != split["protocol_sha256"]
            or split["graphs"] != 360
        ):
            raise ValueError("Confirmation protocol changed")
        expected_count = 1080
    else:
        path = ROOT / cfg["evaluation"]
        split = json.loads((ROOT / base["split"]).read_text())
        labels_path = ROOT / base["labels"]
        expected_count = 3173
    items = read(path)
    if (
        digest(path) != split["inputs_sha256"]
        or digest(labels_path) != split["labels_sha256"]
        or len(items) != expected_count
    ):
        raise ValueError("Evaluation data changed")
    for image, sha in {r["image"]: r["image_sha256"] for r in items}.items():
        if digest(ROOT / image) != sha:
            raise ValueError("Pilot image changed")
    out = ROOT / "results" / a.run_id
    if out.exists() and not a.resume:
        raise ValueError("Use a new ID or explicit resume")
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    metadata = {
        "model": record["model"],
        "arm": record["arm"],
        "training_run": a.training_run,
        "training_kind": "full_sft",
        "selected_epoch": best["epoch"],
        "checkpoint_sha256": best["sha256"],
        "training_record_sha256": digest(training / "run.json"),
        "training_complete_sha256": digest(training / "complete.json"),
        "inputs_sha256": digest(path),
        "labels_sha256": digest(labels_path),
        "source_sha256": digest(Path(__file__)),
        "scorers": {
            n: digest(ROOT / n)
            for n in ["src/evaluation/score.py", "src/evaluation/control/score.py"]
        },
        "selection_used_evaluation": False,
    }
    if a.dataset == "confirmation_v1":
        metadata.update(dataset=a.dataset, confirmation_protocol_sha256=split["protocol_sha256"])
    write_once(out / "run.json", json_text(metadata))
    predictions = out / "predictions.jsonl"
    previous = read(predictions) if predictions.exists() else []
    done = {r["id"] for r in previous}
    if len(done) != len(previous) or done - {r["id"] for r in items}:
        raise ValueError("Invalid partial predictions")
    if len(done) < len(items):
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = GraphSFTModel(record["model"], base, record["arm"], cfg["max_length"])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["stage"] != 2:
            raise ValueError("Expected full decoder checkpoint")
        model.load_weights(state)
        del state
        model.components.eval()
        with predictions.open("a") as stream:
            for row in items:
                if row["id"] in done:
                    continue
                result = {"id": row["id"], "status": "ok", **measured_answer(model, row)}
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream.flush()
                if (len(done) + 1) % 25 == 0:
                    print(json.dumps({"questions": len(done) + 1, "total": len(items)}), flush=True)
                done.add(row["id"])
            os.fsync(stream.fileno())
    actual = read(predictions)
    labels = read(labels_path)
    strict, _ = evaluate(labels, actual)
    primary, details = evaluate(labels, normalize_binary(labels, actual))
    write_json(
        out / "metrics.json",
        {
            "strict_v1": strict,
            "primary_binary_aliases_v2": primary,
            "scope": (
                "independent synthetic confirmation"
                if a.dataset == "confirmation_v1"
                else "full decoder/projector two-stage graph adaptation; exploratory pilot"
            ),
        },
    )
    write_json(out / "details.json", details)
    write_json(
        out / "complete.json",
        {
            "predictions_sha256": digest(predictions),
            "questions": len(actual),
            "metrics_sha256": digest(out / "metrics.json"),
            "completed_at_unix": time.time(),
        },
    )
    print(json.dumps(primary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
