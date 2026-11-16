"""Compare the unaligned, projector-only and full-SFT evaluations on matched pilot questions."""

import argparse
from pathlib import Path
import re

from src.data.pilot import ROOT, sha, json_text, write_once
from src.evaluation.compare_alignment import load_run, contrast
from src.evaluation.compare_full_sft import collect, read
from src.evaluation.control.score import paired_graph_delta
from src.evaluation.score import read_jsonl


def build(model, primary_prefix, before_id):
    dataset = ROOT / "data/manifests/pilot50_v1"
    split = read(dataset / "split.json")
    for name in ["inputs", "labels"]:
        if sha(dataset / (name + ".jsonl")) != split[name + "_sha256"]:
            raise ValueError(f"Pilot {name} changed")
    labels = read_jsonl(dataset / "labels.jsonl")
    families_path = ROOT / "data/manifests/train_dev_v2/split.json"
    families = {g: f for f, graphs in read(families_path)["family_members"].items() for g in graphs}
    before_record = read(ROOT / "results" / before_id / "run.json")
    if (
        before_record.get("config", {}).get("training_performed") is not False
        or before_record["smoke"]
    ):
        raise ValueError("Expected complete untrained pilot")
    before, before_metrics, before_source = load_run(
        before_id, labels, split["inputs_sha256"], model
    )
    rows = {}
    for arm in ["clip_double", "clip_dino"]:
        after, summary, protocol = collect(
            ROOT, primary_prefix, model, arm, labels, split, "pilot50_v1"
        )
        config, _, weights, base_config, *_ = protocol
        hardware_fields = {"inference_min_vram_gib", "training_min_vram_gib"}
        old_protocol = {
            k: v for k, v in before_record["config"].items() if k not in hardware_fields
        }
        current_protocol = {k: v for k, v in base_config.items() if k not in hardware_fields}
        if old_protocol != current_protocol or config["seed"] != base_config["adapter_seed"]:
            raise ValueError("Baseline and trained configuration differ")
        for key, trained in weights.items():
            old = before_record["weights"][key]
            # Stored installation directories may differ, so compare pinned registry and content hashes.
            if trained["config"] != old["config"] or sorted(trained["files"].values()) != sorted(
                old["files"].values()
            ):
                raise ValueError("Foundation/encoder identity differs")
        groups = {"before": before[arm], "full": after}
        scores = {"before": before_metrics[arm]["primary"], "full": summary["primary"]}
        sources = {
            "before": before_source["predictions_sha256"],
            "full": summary["predictions_sha256"],
        }
        projector_id = f"aligned-pilot50-v1-{model}-{arm}"
        if (ROOT / "results" / projector_id / "complete.json").is_file():
            old, metrics, source = load_run(projector_id, labels, split["inputs_sha256"], model)
            groups["projectors"] = old[arm]
            scores["projectors"] = metrics[arm]["primary"]
            sources["projectors"] = source["predictions_sha256"]
        comparisons = {}
        for reference in sorted(groups.keys() - {"full"}):
            comparisons[reference] = {
                "full_minus_reference_by_graph": paired_graph_delta(
                    labels, groups[reference], after
                ),
                "full_minus_reference_by_family": contrast(
                    groups, {reference: -1, "full": 1}, families
                ),
            }
        rows[arm] = {
            "by_stage": {
                stage: {
                    k: m[k] for k in ["correct", "n", "accuracy", "accuracy_ci95_graph_bootstrap"]
                }
                for stage, m in scores.items()
            },
            "contrasts": comparisons,
            "predictions_sha256": sources,
        }
    return {
        "model": model,
        "dataset": "pilot50_v1",
        "prefix": primary_prefix,
        "arms": rows,
        "inputs_sha256": split["inputs_sha256"],
        "labels_sha256": split["labels_sha256"],
        "families_sha256": sha(families_path),
        "analysis_sha256": sha(Path(__file__)),
        "scope": "Matched descriptive comparisons. Training supervision and trainable components differ across stages. LLaVA historical CLIP uses its native branch; full SFT uses external CLIP.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["llava", "qwen"], required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--before", required=True)
    args = parser.parse_args()
    if not all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", v) for v in [args.prefix, args.before]
    ):
        parser.error("Invalid run identifier")
    report = build(args.model, args.prefix, args.before)
    path = ROOT / "results" / (args.prefix + "-stages") / (args.model + ".json")
    write_once(path, json_text(report))
    print(path)


if __name__ == "__main__":
    main()
