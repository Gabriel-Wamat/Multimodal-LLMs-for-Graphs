"""Image-free control that predicts the most frequent training answer to a question."""

import argparse
from collections import Counter, defaultdict
import re

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.compare_alignment import contrast
from src.evaluation.compare_full_sft import ARMS, collect, read
from src.evaluation.control.score import normalize_binary
from src.evaluation.score import evaluate, norm, read_jsonl


def predict(train, questions):
    by_question, by_format = defaultdict(Counter), defaultdict(Counter)
    overall = Counter()
    for row in train:
        answer = row["resposta_txt"]
        fmt = row["formato"]
        by_question[fmt, norm(row["pergunta"])][answer] += 1
        by_format[fmt][answer] += 1
        overall[answer] += 1

    predictions = []
    sources = Counter()
    for row in questions:
        fmt = row["formato"]
        counts = by_question.get((fmt, norm(row["pergunta"])))
        source = "matched_question"
        if not counts:
            counts = by_format.get(fmt)
            source = "format_fallback"
        if not counts:
            counts = overall
            source = "global_fallback"
        # Lexical order resolves ties independently of the evaluation set and input ordering.
        answer = min(counts, key=lambda value: (-counts[value], value))
        predictions.append({"id": row["id"], "status": "ok", "response": answer})
        sources[source] += 1
    return predictions, dict(sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", help="Completed campaign for a paired pilot comparison")
    parser.add_argument("--model", choices=("qwen", "llava"))
    args = parser.parse_args()
    if bool(args.model) != bool(args.prefix) or (
        args.prefix and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,48}", args.prefix)
    ):
        parser.error("Provide both a valid campaign prefix and model, or neither")
    manifests = ROOT / "data/manifests"
    dev = manifests / "train_dev_v2"
    pilot = manifests / "pilot50_v1"
    manifest = read(dev / "manifest.json")
    split = read(dev / "split.json")
    pilot_info = read(pilot / "split.json")
    files = {
        dev / "train.jsonl": manifest["counts"]["train"]["sha256"],
        dev / "development.jsonl": manifest["counts"]["development"]["sha256"],
        dev / "split.json": manifest["split_sha256"],
        pilot / "split.json": split["pilot_split_sha256"],
        pilot / "inputs.jsonl": pilot_info["inputs_sha256"],
        pilot / "labels.jsonl": pilot_info["labels_sha256"],
    }
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Changed manifest {path.name}")
    train = read_jsonl(dev / "train.jsonl")
    train_graphs = {r["grafico"] for r in train}
    if train_graphs != set(split["train"]):
        raise ValueError("Training membership changed")
    datasets = {}
    for name, field, labels_path in [
        ("development", "development", dev / "development.jsonl"),
        ("pilot50_v1", "evaluation_exploratory", pilot / "labels.jsonl"),
    ]:
        labels = read_jsonl(labels_path)
        rows = read_jsonl(pilot / "inputs.jsonl") if name == "pilot50_v1" else labels
        if {q["grafico"] for q in labels} != set(split[field]) or train_graphs.intersection(
            split[field]
        ):
            raise ValueError("Unexpected evaluation membership")
        questions = [{key: row[key] for key in ("id", "pergunta", "formato")} for row in rows]
        preds, sources = predict(train, questions)
        metrics, details = evaluate(labels, normalize_binary(labels, preds))
        if not metrics["all_records_successful"]:
            raise ValueError("Incomplete question-only evaluation")
        datasets[name] = {
            "correct": metrics["correct"],
            "n": metrics["n"],
            "accuracy": metrics["accuracy"],
            "ci95_graph_bootstrap": metrics["accuracy_ci95_graph_bootstrap"],
            "by_chart_type": metrics["by_tipo"],
            "by_format": metrics["by_formato"],
            "prediction_sources": sources,
        }
        if args.model and name == "pilot50_v1":
            families = {g: f for f, graphs in split["family_members"].items() for g in graphs}
            comparisons = {}
            for arm in ARMS:
                adapted, info, protocol = collect(
                    ROOT, args.prefix, args.model, arm, labels, pilot_info, name
                )
                if (
                    protocol[1]["data/manifests/train_dev_v2/train.jsonl"]
                    != files[dev / "train.jsonl"]
                ):
                    raise ValueError("Model and control training questions differ")
                comparisons[arm] = {
                    "model_correct": info["primary"]["correct"],
                    "predictions_sha256": info["predictions_sha256"],
                    "difference": contrast(
                        {"model": adapted, "control": details},
                        {"model": 1, "control": -1},
                        families,
                    ),
                }
            paired_file = (
                ROOT / "results/published/question_prior" / f"{args.prefix}-{args.model}.json"
            )
            write_once(
                paired_file,
                json_text(
                    {
                        "prefix": args.prefix,
                        "model": args.model,
                        "dataset": name,
                        "control_correct": metrics["correct"],
                        "questions": metrics["n"],
                        "pilot_sha256": files[pilot / "split.json"],
                        "arms": comparisons,
                    }
                ),
            )
    output = ROOT / "results/published/question-prior-v1.json"
    write_once(
        output,
        json_text(
            {
                "training_questions": len(train),
                "training_graphs": len(train_graphs),
                "train_sha256": files[dev / "train.jsonl"],
                "development_sha256": files[dev / "development.jsonl"],
                "pilot_sha256": files[pilot / "split.json"],
                "scorer": "primary_binary_aliases_v2",
                "scope": "exploratory question-only control; train answer frequencies, no image features",
                "datasets": datasets,
            }
        ),
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
