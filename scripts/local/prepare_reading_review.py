"""Fix a descriptive review sample without reading any model predictions."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / "data/manifests/pairs_v1/inputs.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    evaluation = [r for r in rows if r["split"] == "evaluation_exploratory"]
    strata = defaultdict(dict)
    for row in evaluation:
        key = (row["tipo"], row["estilo"], row["intensity"])
        strata[key].setdefault(row["pair_id"], []).append(row)
    if len(strata) != 72:
        raise ValueError("Expected 6 types x 6 styles x 2 intensities")
    selected = []
    for key, pairs in sorted(strata.items()):
        pair_id = min(pairs, key=lambda p: hashlib.sha256(
            ("reading-review-v1:" + p).encode()).hexdigest())
        members = sorted(pairs[pair_id], key=lambda r: r["member"])
        if [r["member"] for r in members] != ["A", "B"]:
            raise ValueError("Pair does not contain exactly A and B")
        selected.append({"pair_id": pair_id, "type": key[0], "style": key[1],
                         "intensity": key[2], "items": [r["id"] for r in members]})
    output = {
        "protocol": "reading-review-v1", "status": "sample_fixed_review_pending",
        "inputs_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection": "minimum SHA256(reading-review-v1:pair_id) in each evaluation type/style/intensity stratum",
        "uses_predictions": False, "pairs": len(selected), "images": len(selected)*2,
        "models": ["llava", "qwen"], "selected": selected,
        "interpretation": "Descriptive stratified sample, not a population prevalence estimate; equal stratum weights differ from dataset weights.",
        "additional_cases": "Outcome-selected helped/harmed/unchanged examples may be added in a separate illustrative section; never mix with fixed-sample denominators.",
        "timing": "Fixed while full inference was running; earlier smoke results and progress logs had already been inspected. Not a prospective confirmatory preregistration.",
    }
    destination = ROOT / "experiments/five-stage-campaign/reading_review_sample_v1.json"
    payload = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if destination.exists():
        if destination.read_text() != payload:
            raise ValueError("Existing fixed sample differs; do not overwrite")
    else:
        destination.write_text(payload)
    print(destination)


if __name__ == "__main__":
    main()
