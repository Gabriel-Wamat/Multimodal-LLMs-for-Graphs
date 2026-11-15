"""Compare graph-adapted checkpoints with the official I-MoF on identical pilot items."""

import argparse
from collections import Counter
from pathlib import Path

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.compare_alignment import contrast
from src.evaluation.compare_full_sft import ARMS, collect, read
from src.evaluation.control.score import normalize_binary
from src.evaluation.score import evaluate, read_jsonl


def compare(prefix, model):
    pilot = ROOT / "data/manifests/pilot50_v1"
    split_path = pilot / "split.json"
    split = read(split_path)
    for key in ("inputs", "labels"):
        if sha(pilot / f"{key}.jsonl") != split[f"{key}_sha256"]:
            raise ValueError(f"Pilot {key} changed")
    labels = read_jsonl(pilot / "labels.jsonl")
    official = ROOT / "results/mof13b-pilot50-v1"
    run = read(official / "run.json")
    complete = read(official / "complete.json")
    predictions = official / "predictions.jsonl"
    if (
        run["smoke"]
        or run["config"] != read(ROOT / "configs/baseline_mof.json")
        or run["split_sha256"] != sha(split_path)
        or run["inputs_sha256"] != split["inputs_sha256"]
        or sha(predictions) != complete["predictions_sha256"]
        or complete["questions"] != len(labels)
    ):
        raise ValueError("Official baseline differs from pilot protocol")
    metrics, baseline = evaluate(labels, normalize_binary(labels, read_jsonl(predictions)))
    if not metrics["all_records_successful"]:
        raise ValueError("Incomplete official pilot")
    family_path = ROOT / "data/manifests/train_dev_v2/split.json"
    families = {g: f for f, graphs in read(family_path)["family_members"].items() for g in graphs}
    arms = {}
    for arm in ARMS:
        adapted, info, protocol = collect(ROOT, prefix, model, arm, labels, split, "pilot50_v1")
        if protocol[1]["data/manifests/train_dev_v2/split.json"] != sha(family_path):
            raise ValueError("Family split differs from training")
        paired = contrast(
            {"official": baseline, "adapted": adapted}, {"official": -1, "adapted": 1}, families
        )
        by_id = {r["id"]: r["correct"] for r in adapted}
        changes = Counter((r["correct"], by_id[r["id"]]) for r in baseline)
        arms[arm] = {
            "correct": info["primary"]["correct"],
            "by_chart_type": info["primary"]["by_tipo"],
            "predictions_sha256": info["predictions_sha256"],
            "adapted_only": changes[False, True],
            "official_only": changes[True, False],
            "both_correct": changes[True, True],
            "both_wrong": changes[False, False],
            "paired_family_difference": paired,
        }
    return {
        "dataset": "pilot50_v1",
        "prefix": prefix,
        "model": model,
        "questions": len(labels),
        "scorer": "primary_binary_aliases_v2",
        "baseline_correct": metrics["correct"],
        "baseline_by_chart_type": metrics["by_tipo"],
        "baseline_predictions_sha256": complete["predictions_sha256"],
        "pilot_sha256": sha(split_path),
        "family_split_sha256": sha(family_path),
        "analysis_sha256": sha(Path(__file__)),
        "arms": arms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--model", required=True, choices=("llava", "qwen"))
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in (".", ".."):
        parser.error("Expected a campaign name")
    result = compare(args.prefix, args.model)
    out = ROOT / "results/published/official_comparison" / f"{args.prefix}-{args.model}.json"
    write_once(out, json_text(result))
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
