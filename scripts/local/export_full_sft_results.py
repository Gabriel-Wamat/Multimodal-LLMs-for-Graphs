"""Publish aggregate results from three complete, matched full-SFT evaluations."""

import argparse
import re

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.compare_full_sft import ARMS, compare_model


def public_summary(report):
    arms = {}
    for arm in ARMS:
        row = report["arms"][arm]
        metrics = row["primary"]
        arms[arm] = {
            "correct": metrics["correct"],
            "questions": metrics["n"],
            "accuracy": metrics["accuracy"],
            "accuracy_ci95_graph_bootstrap": metrics["accuracy_ci95_graph_bootstrap"],
            "graphs": metrics["graph_count"],
            "by_chart_type": metrics["by_tipo"],
            "by_response_format": metrics["by_formato"],
            "by_question_level": metrics["by_nivel"],
            "automatic_error_counts": row["error_counts"],
            "selected_epoch": row["selected_epoch"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "predictions_sha256": row["predictions_sha256"],
            "total_generation_seconds": row["inference_seconds"],
            "median_generation_seconds": row["median_seconds"],
            "p95_generation_seconds": row["p95_seconds"],
            "peak_gpu_bytes": row["peak_gpu_bytes"],
            "generated_tokens": row["generated_tokens"],
        }
    differences = {}
    for control, comparison in report["comparisons"].items():
        result = comparison["fusion_minus_control_by_family"]
        differences[control] = {
            "difference_pp": result["percentage_points"],
            "ci95_pp": None if result["ci95"] is None else [100 * v for v in result["ci95"]],
            "families": result["families"],
            "questions": result["questions"],
        }
    return {
        "schema": "full_sft_aggregates_v1",
        "dataset": report["dataset"],
        "model": report["model"],
        "seed": report["seed"],
        "adapter_seed": report["adapter_seed"],
        "correction": "primary_binary_aliases_v2",
        "training": "caption alignment followed by full decoder and projector instruction tuning",
        "protocol_sha256": report["matched_protocol_sha256"],
        "analysis_sha256": report["analysis_sha256"],
        "arms": arms,
        "fusion_minus_control": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--model", choices=["llava", "qwen"], required=True)
    parser.add_argument("--dataset", choices=["pilot50_v1", "confirmation_v1"], required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,48}", args.prefix):
        parser.error("Invalid campaign ID")
    report = compare_model(ROOT, args.prefix, args.model, args.dataset)
    public = public_summary(report)
    out = ROOT / "results/published/full_sft" / f"{args.prefix}-{args.model}-{args.dataset}.json"
    write_once(out, json_text(public))
    print(out.relative_to(ROOT), sha(out))


if __name__ == "__main__":
    main()
