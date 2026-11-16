"""Compare two projector-only and full-SFT validations at the same QA epoch."""

import argparse
from collections import Counter
from pathlib import Path

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.compare_full_sft import read
from src.evaluation.control.score import normalize_binary, paired_graph_delta
from src.evaluation.score import evaluate, read_jsonl


def compare(projector, full, epoch):
    old, new = (read(p / "run.json") for p in (projector, full))
    if (old["model"], old["arm"]) != (new["model"], new["arm"]):
        raise ValueError("Expected the same model and arm")
    dev = Path(old["config"]["development_data"])
    manifest = ROOT / dev.parent / "manifest.json"
    if (
        sha(manifest) != old["manifest_sha256"]
        or str(dev) != new["config"]["development"]
        or sha(ROOT / dev) != read(manifest)["counts"]["development"]["sha256"]
        or sha(ROOT / dev) != new["data_sha256"][str(dev)]
    ):
        raise ValueError("Validation manifest differs between recipes")
    labels = read_jsonl(ROOT / dev)
    results, details = {}, []
    for name, folder in (("projectors", projector), ("full", full)):
        validation = folder / "validation" / f"epoch-{epoch:02d}"
        path = validation / "predictions.json"
        metrics, rows = evaluate(labels, normalize_binary(labels, read(path)))
        if (
            not metrics["all_records_successful"]
            or metrics != read(validation / "metrics.json")["metrics"]
        ):
            raise ValueError("Incomplete or inconsistent validation")
        results[name] = {
            "run": folder.name,
            "correct": metrics["correct"],
            "predictions_sha256": sha(path),
        }
        details.append(rows)
    counts = Counter((a["correct"], b["correct"]) for a, b in zip(*details))
    return {
        "dataset": "development",
        "model": old["model"],
        "arm": old["arm"],
        "epoch": epoch,
        "n": len(labels),
        "labels_sha256": sha(ROOT / dev),
        "recipes": results,
        "projectors_only_correct": counts[True, False],
        "full_only_correct": counts[False, True],
        "both_correct": counts[True, True],
        "both_wrong": counts[False, False],
        "paired": paired_graph_delta(labels, *details),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projector", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("epoch", type=int)
    args = parser.parse_args()
    if args.epoch < 1:
        parser.error("Choose a completed QA epoch")
    report = compare(args.projector, args.full, args.epoch)
    out = (
        ROOT
        / "results/published/development_recipes"
        / f"{args.projector.name}--{args.full.name}-epoch-{args.epoch:02d}.json"
    )
    write_once(out, json_text(report))
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
