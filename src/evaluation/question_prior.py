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
    parser.add_argument("--prefix", help="Completed campaign for a paired comparison")
    parser.add_argument("--model", choices=("qwen", "llava"))
    parser.add_argument(
        "--confirmation", action="store_true", help="Evaluate the fixed control on confirmation_v1"
    )
    args = parser.parse_args()
    if bool(args.model) != bool(args.prefix) or (
        args.prefix and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,48}", args.prefix)
    ):
        parser.error("Provide both a valid campaign prefix and model, or neither")
    manifests = ROOT / "data/manifests"
    dev = manifests / "train_dev_v2"
    manifest = read(dev / "manifest.json")
    split = read(dev / "split.json")
    files = {
        dev / "train.jsonl": manifest["counts"]["train"]["sha256"],
        dev / "development.jsonl": manifest["counts"]["development"]["sha256"],
        dev / "split.json": manifest["split_sha256"],
    }
    paired_dataset = "confirmation_v1" if args.confirmation else "pilot50_v1"
    dataset = manifests / paired_dataset
    manifest_key = "confirmation_sha256" if args.confirmation else "pilot_sha256"
    manifest_path = dataset / ("complete.json" if args.confirmation else "split.json")
    dataset_info = read(manifest_path)
    for key in ("inputs", "labels"):
        files[dataset / (key + ".jsonl")] = dataset_info[key + "_sha256"]
    if args.confirmation:
        files[dataset / "sources.jsonl"] = dataset_info["sources_sha256"]
        files[dataset / "protocol.json"] = dataset_info["protocol_sha256"]
        families = {r["grafico"]: r["grafico"] for r in read_jsonl(dataset / "sources.jsonl")}
        if len(families) != dataset_info["graphs"]:
            raise ValueError("Unexpected confirmation graph count")
        tasks = [(paired_dataset, set(families), dataset / "labels.jsonl")]
    else:
        files[manifest_path] = split["pilot_split_sha256"]
        families = {g: f for f, graphs in split["family_members"].items() for g in graphs}
        tasks = [
            ("development", set(split["development"]), dev / "development.jsonl"),
            (paired_dataset, set(split["evaluation_exploratory"]), dataset / "labels.jsonl"),
        ]
    for path, expected in files.items():
        if sha(path) != expected:
            raise ValueError(f"Changed manifest {path.name}")
    train = read_jsonl(dev / "train.jsonl")
    train_graphs = {r["grafico"] for r in train}
    if train_graphs != set(split["train"]):
        raise ValueError("Training membership changed")
    datasets = {}
    for name, expected_graphs, labels_path in tasks:
        labels = read_jsonl(labels_path)
        rows = labels if name == "development" else read_jsonl(dataset / "inputs.jsonl")
        if {q["grafico"] for q in labels} != expected_graphs or train_graphs.intersection(
            expected_graphs
        ):
            raise ValueError("Unexpected evaluation membership")
        if args.confirmation and len(labels) != dataset_info["questions"]:
            raise ValueError("Unexpected confirmation question count")
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
        if args.model and name == paired_dataset:
            comparisons = {}
            for arm in ARMS:
                adapted, info, protocol = collect(
                    ROOT, args.prefix, args.model, arm, labels, dataset_info, name
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
            suffix = "-confirmation" if args.confirmation else ""
            paired_file = (
                ROOT
                / "results/published/question_prior"
                / f"{args.prefix}-{args.model}{suffix}.json"
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
                        manifest_key: sha(manifest_path),
                        "arms": comparisons,
                    }
                ),
            )
    filename = (
        "question-prior-confirmation-v1.json" if args.confirmation else "question-prior-v1.json"
    )
    output = ROOT / "results/published" / filename
    write_once(
        output,
        json_text(
            {
                "training_questions": len(train),
                "training_graphs": len(train_graphs),
                "train_sha256": files[dev / "train.jsonl"],
                "development_sha256": files[dev / "development.jsonl"],
                manifest_key: sha(manifest_path),
                "scorer": "primary_binary_aliases_v2",
                "scope": "exploratory question-only control; train answer frequencies, no image features",
                "datasets": datasets,
            }
        ),
    )
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
