"""Prepare matched, model-blind semantic review forms for a frozen evaluation."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.score import grade, read_jsonl
from src.evaluation.control.score import normalize_binary

VERSION = "semantic-review-v1"


def rank(*parts):
    return hashlib.sha256(json.dumps([VERSION, *parts]).encode()).hexdigest()


def select_questions(items):
    strata = defaultdict(list)
    for row in items:
        strata[(row["tipo"], row["estilo"])].append(row)
    if len(strata) != 36 or len({r["id"] for r in items}) != len(items):
        raise ValueError("Expected six types, six styles and unique questions")
    return [min(rows, key=lambda r: rank(r["id"])) for _, rows in sorted(strata.items())]


def prepare(dataset, run_ids, destination):
    base = ROOT / "data/manifests" / dataset
    split = json.loads(
        (base / ("split.json" if dataset == "pilot50_v1" else "complete.json")).read_text()
    )
    for name in ("inputs", "labels"):
        if sha(base / (name + ".jsonl")) != split[name + "_sha256"]:
            raise ValueError("Data changed")
    items = read_jsonl(base / "inputs.jsonl")
    labels = {r["id"]: r for r in read_jsonl(base / "labels.jsonl")}
    selected = select_questions(items)
    if {r["id"] for r in items} != set(labels):
        raise ValueError("Inputs and labels differ")
    forms, key = {}, {}
    for name in sorted(run_ids):
        folder = ROOT / "results" / name
        record = json.loads((folder / "run.json").read_text())
        complete = json.loads((folder / "complete.json").read_text())
        if (
            record.get("training_kind") != "full_sft"
            or record.get("dataset", "pilot50_v1") != dataset
            or record["inputs_sha256"] != split["inputs_sha256"]
            or record["labels_sha256"] != split["labels_sha256"]
            or sha(folder / "predictions.jsonl") != complete["predictions_sha256"]
        ):
            raise ValueError(f"Incompatible run: {name}")
        preds = read_jsonl(folder / "predictions.jsonl")
        by_id = {r["id"]: r for r in preds}
        if (
            set(by_id) != set(labels)
            or len(preds) != len(labels)
            or any(r["status"] != "ok" for r in preds)
        ):
            raise ValueError("Incomplete predictions")
        for item in selected:
            qid = item["id"]
            raw = by_id[qid]["response"]
            token = rank(qid, raw)  # Identical answers to the same question receive one decision.
            q = labels[qid]
            forms[token] = dict(
                case_id=token,
                question=q["pergunta"],
                image=item["image"],
                reference=q["resposta"],
                accepted_min=q["min"],
                accepted_max=q["max"],
                response=raw,
                semantic_verdict="",
                evidence="",
                reviewer="",
            )
            normalized = normalize_binary([q], [by_id[qid]])[0]["response"]
            accepted, reason = grade(q, normalized)
            key.setdefault(token, []).append(
                {
                    "run": name,
                    "automatic_correct": accepted,
                    "automatic_rule": reason,
                    "id": qid,
                    "predictions_sha256": complete["predictions_sha256"],
                }
            )
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(next(iter(forms.values()))), lineterminator="\n")
    writer.writeheader()
    writer.writerows(forms[k] for k in sorted(forms))
    write_once(destination / "blank_review.csv", buf.getvalue())
    write_once(destination / "private_key.json", json_text(key))
    write_once(
        destination / "protocol.json",
        json_text(
            dict(
                version=VERSION,
                dataset=dataset,
                questions=[r["id"] for r in selected],
                scope="descriptive sample: one fixed question per chart type/style",
                form_sha256=sha(destination / "blank_review.csv"),
                key_sha256=sha(destination / "private_key.json"),
                allowed_verdicts=["correta", "incorreta", "parcial", "indeterminada"],
            )
        ),
    )
    print(destination)


def finalize(destination, reviewed):
    protocol = json.loads((destination / "protocol.json").read_text())
    blank = destination / "blank_review.csv"
    key_path = destination / "private_key.json"
    if sha(blank) != protocol["form_sha256"] or sha(key_path) != protocol["key_sha256"]:
        raise ValueError("Review form or key changed")
    key = json.loads(key_path.read_text())
    with blank.open(newline="") as stream:
        original = {r["case_id"]: r for r in csv.DictReader(stream)}
    fields = {"reviewer", "evidence", "semantic_verdict"}
    decisions = {}
    with reviewed.open(newline="") as stream:
        for row in csv.DictReader(stream):
            token = row["case_id"]
            if token not in original or token in decisions:
                raise ValueError("Unexpected or repeated case")
            if any(row[k] != value for k, value in original[token].items() if k not in fields):
                raise ValueError("Question, reference or response changed")
            if row["semantic_verdict"] not in protocol["allowed_verdicts"] or not all(
                row[k].strip() for k in fields
            ):
                raise ValueError("Each case requires a verdict, evidence and reviewer")
            decisions[token] = row["semantic_verdict"]
    if set(decisions) != set(key):
        raise ValueError("Missing decisions")
    counts, cross = defaultdict(Counter), defaultdict(Counter)
    for token, aliases in key.items():
        for entry in aliases:
            run = entry["run"]
            verdict = decisions[token]
            counts[run][verdict] += 1
            cross[run][("auto_1" if entry["automatic_correct"] else "auto_0") + "/" + verdict] += 1
    summary = {
        run: {
            "reviewed": c.total(),
            "verdicts": {k: c[k] for k in protocol["allowed_verdicts"]},
            "semantic_correct_fraction_in_sample": c["correta"] / c.total(),
            "automatic_by_semantic": dict(cross[run]),
        }
        for run, c in counts.items()
    }
    write_once(
        destination / "reviewed_summary.json",
        json_text(
            {
                "dataset": protocol["dataset"],
                "scope": protocol["scope"],
                "reviewed_sheet_sha256": sha(reviewed),
                "protocol_sha256": sha(destination / "protocol.json"),
                "runs": summary,
            }
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pilot50_v1", "confirmation_v1"], required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.runs)) != len(args.runs) or not all(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", r) for r in args.runs
    ):
        parser.error("Expected distinct run IDs")
    prepare(args.dataset, args.runs, args.output)


if __name__ == "__main__":
    main()
