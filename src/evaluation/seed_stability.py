"""Summarize controlled changes in accuracy across three matched training seeds."""

import argparse
from pathlib import Path
import re
from statistics import mean, stdev

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.compare_full_sft import ARMS, compare_model

SEEDS = (20261102, 20261103, 20261104)


def distribution(values):
    return {
        "by_seed": dict(zip(SEEDS, values)),
        "mean": mean(values),
        "sample_sd": stdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(reports):
    by_seed = {r["seed"]: r for r in reports}
    if len(reports) != len(SEEDS) or set(by_seed) != set(SEEDS):
        raise ValueError("Expected one result for each registered seed")
    ordered = [by_seed[s] for s in SEEDS]
    if any(r["adapter_seed"] != r["seed"] for r in ordered):
        raise ValueError("Training order and adapter seeds must match")
    for field in ["matched_protocol_sha256", "analysis_sha256", "model", "dataset"]:
        if len({r[field] for r in ordered}) != 1:
            raise ValueError(f"Replicas differ in {field}")
    arms = {
        arm: distribution([100 * r["arms"][arm]["primary"]["accuracy"] for r in ordered])
        for arm in ARMS
    }
    contrasts = {
        arm: distribution(
            [
                r["comparisons"][arm]["fusion_minus_control_by_family"]["percentage_points"]
                for r in ordered
            ]
        )
        for arm in ARMS[:2]
    }
    return {
        "model": ordered[0]["model"],
        "dataset": ordered[0]["dataset"],
        "accuracy_percent": arms,
        "fusion_minus_control_pp": contrasts,
        "runs": {r["seed"]: r["prefix"] for r in ordered},
        "matched_protocol_sha256": ordered[0]["matched_protocol_sha256"],
        "analysis_sha256": sha(Path(__file__)),
        "scope": "Sample SD across training seeds at fixed inputs; graph-bootstrap intervals are in the individual reports.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefixes", nargs=3, required=True)
    parser.add_argument("--model", choices=["llava", "qwen"], required=True)
    parser.add_argument("--dataset", choices=["pilot50_v1", "confirmation_v1"], required=True)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,48}", p) for p in args.prefixes):
        parser.error("Invalid campaign ID")
    report = summarize(
        [compare_model(ROOT, prefix, args.model, args.dataset) for prefix in args.prefixes]
    )
    target = (
        ROOT
        / "results"
        / (report["runs"][SEEDS[0]] + "-stability")
        / f"{args.model}-{args.dataset}.json"
    )
    write_once(target, json_text(report))
    print(target)


if __name__ == "__main__":
    main()
