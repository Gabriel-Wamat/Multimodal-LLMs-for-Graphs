"""Evaluate a development-selected projector checkpoint on the frozen 50% pilot."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import torch

from src.data.pilot import ROOT, json_text, write_once
from src.models.control.artifacts import digest, installed
from src.training.alignment import AlignmentModel
from src.evaluation.score import evaluate
from src.evaluation.control.score import normalize_binary


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def measured_answer(model, row):
    """Capture generation token counts without changing the aligned inference path."""
    original = model.vlm.model.generate
    counts = []

    def generate(*args, **kwargs):
        prefix = kwargs["input_ids"].shape[1]
        output = original(*args, **kwargs)
        counts.append((prefix, output.shape[1] - prefix))
        return output

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    model.vlm.model.generate = generate
    try:
        answer = model.answer(row)
    finally:
        model.vlm.model.generate = original
    torch.cuda.synchronize()
    if len(counts) != 1:
        raise RuntimeError("Expected exactly one independent generation")
    return dict(
        response=answer,
        seconds=time.perf_counter() - started,
        input_tokens=counts[0][0],
        generated_tokens=counts[0][1],
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import re

    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", s)
        for s in [args.run_id, args.training_run]
    ):
        raise ValueError("Invalid run ID")
    training = ROOT / "experiments/training" / args.training_run
    train_run = json.loads((training / "run.json").read_text())
    complete = json.loads((training / "complete.json").read_text())
    if complete["epochs"] != train_run["config"]["epochs"]:
        raise ValueError("Training budget incomplete")
    checkpoint = training / "best.safetensors"
    if digest(checkpoint) != complete["best_sha256"]:
        raise ValueError("Selected checkpoint changed")
    for name in [
        "src/training/alignment.py",
        "src/models/control/vlm.py",
        "src/models/control/encoders.py",
        "src/models/control/fusion.py",
        "src/evaluation/run.py",
    ]:
        if digest(ROOT / name) != train_run["source_sha256"][name]:
            raise ValueError("Inference code differs from training: " + name)
    cfg = train_run["base_config"]
    path = ROOT / train_run["config"]["evaluation_data"]
    rows = read(path)
    split_path = ROOT / cfg["split"]
    split = json.loads(split_path.read_text())
    if (
        digest(path) != split["inputs_sha256"]
        or len(rows) != split["questions_evaluated"]
    ):
        raise ValueError("Evaluation pilot changed")
    labels_path = ROOT / cfg["labels"]
    if digest(labels_path) != split["labels_sha256"]:
        raise ValueError("Evaluation labels changed")
    for row in rows:
        if digest(ROOT / row["image"]) != row["image_sha256"]:
            raise ValueError("Evaluation image changed")
    weights = {
        key: installed(key, verify=True)
        for key in [train_run["model"], "clip", "dinov3"]
    }
    if weights != train_run["weights"]:
        raise ValueError("Frozen backbone weights differ from training")
    record = {
        "training_run": args.training_run,
        "model": train_run["model"],
        "arm": train_run["arm"],
        "training_record_sha256": digest(training / "run.json"),
        "training_complete_sha256": digest(training / "complete.json"),
        "checkpoint_sha256": digest(checkpoint),
        "selected_epoch": complete["best"]["epoch"],
        "selection_used_evaluation": False,
        "inputs_sha256": digest(path),
        "labels_sha256": digest(labels_path),
        "item_ids": [r["id"] for r in rows],
        "source_sha256": digest(Path(__file__)),
        "scorers": {
            name: digest(ROOT / name)
            for name in ["src/evaluation/score.py", "src/evaluation/control/score.py"]
        },
        "weights": weights,
        "config": cfg,
    }
    folder = ROOT / "results" / args.run_id
    if folder.exists() and not args.resume:
        raise ValueError("Run exists; explicit --resume required")
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_once(folder / "run.json", json_text(record))
    write_once(folder / "source.py", Path(__file__).read_text())
    prediction_path = folder / "predictions.jsonl"
    previous = read(prediction_path) if prediction_path.exists() else []
    done = {r["id"] for r in previous}
    if (
        len(done) != len(previous)
        or done - set(record["item_ids"])
        or any(r["status"] != "ok" for r in previous)
    ):
        raise ValueError("Invalid previous predictions")
    if len(done) != len(rows):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = AlignmentModel(train_run["model"], cfg, train_run["arm"])
        model.load_projectors(checkpoint)
        model.projectors.eval().requires_grad_(False)
        versions = {n: p._version for n, p in model.projectors.named_parameters()}
        with prediction_path.open("a") as stream:
            for row in rows:
                if row["id"] in done:
                    continue
                result = measured_answer(model, row)
                if any(
                    p._version != versions[n]
                    for n, p in model.projectors.named_parameters()
                ):
                    raise RuntimeError("Projector mutated during evaluation")
                prediction = dict(
                    id=row["id"], status="ok", arm=train_run["arm"], **result
                )
                stream.write(
                    json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
                done.add(row["id"])
                print(len(done), len(rows), row["id"], flush=True)
    predictions = read(prediction_path)
    truth = {r["id"]: r for r in read(labels_path)}
    labels = [truth[r["id"]] for r in rows]
    strict, _ = evaluate(labels, predictions)
    primary, graded = evaluate(labels, normalize_binary(labels, predictions))
    if not primary["all_records_successful"]:
        raise ValueError("Incomplete evaluation")
    report = {
        "strict_v1": strict,
        "primary_binary_aliases_v2": primary,
        "selected_epoch": complete["best"]["epoch"],
        "selection_note": "Epoch zero may win; no improvement is presumed",
        "predictions_sha256": digest(prediction_path),
        "cost": {
            "seconds": sum(r["seconds"] for r in predictions),
            "input_tokens": sum(r["input_tokens"] for r in predictions),
            "generated_tokens": sum(r["generated_tokens"] for r in predictions),
            "peak_gpu_bytes": max(r["peak_gpu_bytes"] for r in predictions),
        },
    }
    write_once(folder / "scored/metrics.json", json_text(report))
    write_once(
        folder / "scored/answers_scored.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in graded),
    )
    write_once(
        folder / "complete.json",
        json_text(
            {"questions": len(rows), "predictions_sha256": digest(prediction_path)}
        ),
    )


if __name__ == "__main__":
    main()
