"""Re-score two completed development epochs and count answer-level transitions."""

import argparse
from collections import Counter
from pathlib import Path

from src.data.pilot import ROOT, sha, json_text, write_once
from src.evaluation.compare_full_sft import error_type, read
from src.evaluation.control.score import normalize_binary, paired_graph_delta
from src.evaluation.score import evaluate, read_jsonl


def compare(folder, earlier, later):
    if not 0 <= earlier < later:
        raise ValueError("Expected two epochs in chronological order")
    record = read(folder / "run.json")
    dev = record["config"]["development"]
    if sha(ROOT / dev) != record["data_sha256"][dev]:
        raise ValueError("Development questions changed")
    labels = read_jsonl(ROOT / dev)
    history = {
        row["epoch"]: row["development"]
        for row in read(folder / "history.json")
        if row["stage"] == 2 and row.get("development")
    }
    summaries, details = [], []
    for epoch in (earlier, later):
        path = folder / "validation" / f"epoch-{epoch:02d}" / "predictions.json"
        predictions = read(path)
        metrics, rows = evaluate(labels, normalize_binary(labels, predictions))
        if metrics != history[epoch]["metrics"] or not metrics["all_records_successful"]:
            raise ValueError("Predictions differ from completed validation")
        details.append(rows)
        summaries.append(
            {
                "epoch": epoch,
                "correct": metrics["correct"],
                "auto_error_groups": dict(Counter(map(error_type, rows))),
                "predictions_sha256": sha(path),
                "accuracy": metrics["accuracy"],
            }
        )
    changes = Counter((a["correct"], b["correct"]) for a, b in zip(*details))
    return {
        "scope": "development only; epoch 0 follows caption alignment",
        "run": folder.name,
        "epochs": summaries,
        "n": len(labels),
        "gained": changes[False, True],
        "lost": changes[True, False],
        "unchanged_correct": changes[True, True],
        "unchanged_error": changes[False, False],
        "paired": paired_graph_delta(labels, *details),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_folder", type=Path)
    parser.add_argument("earlier", type=int)
    parser.add_argument("later", type=int)
    parser.add_argument(
        "--public", action="store_true", help="Save the aggregate in the public notebook data"
    )
    args = parser.parse_args()
    report = compare(args.training_folder, args.earlier, args.later)
    base = ROOT / "results"
    if args.public:
        base = base / "published" / "development"
    destination = (
        base
        / (report["run"] + "-development")
        / f"epoch-{args.earlier:02d}-vs-{args.later:02d}.json"
    )
    if destination.exists() and read(destination) == report:
        print(destination)
        return
    write_once(destination, json_text(report))
    print(destination)


if __name__ == "__main__":
    main()
