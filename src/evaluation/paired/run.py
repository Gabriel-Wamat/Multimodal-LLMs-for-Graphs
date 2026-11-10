"""Run the slide's three visual conditions on independent A/B queries."""

import argparse
import json
import os
from src.data.pilot import ROOT, write_once, json_text
from src.models.control.artifacts import digest, installed
from src.evaluation.paired.conditions import evaluate_condition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llava", "qwen"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        raise ValueError("Invalid run id")
    base = ROOT / "data/manifests" / ("pairs_v1_smoke" if args.smoke else "pairs_v1")
    complete = json.loads((base / "complete.json").read_text())
    assert digest(base / "inputs.jsonl") == complete["inputs_sha256"]
    rows = [json.loads(s) for s in (base / "inputs.jsonl").read_text().splitlines()]
    if not args.smoke:
        rows = [r for r in rows if r["split"] == "evaluation_exploratory"]
    assert rows and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert digest(ROOT / r["image"]) == r["image_sha256"]
    cfg = json.loads((ROOT / "configs/control_mof_stage0.json").read_text())
    weights = installed(args.model, verify=True)
    conditions = ["original", "crops_2x2", "transcription"]
    sources = list((ROOT / "src/evaluation/paired").glob("*.py")) + list(
        (ROOT / "src/models/control").glob("*.py")
    )
    stamp = {
        "model": args.model,
        "smoke": args.smoke,
        "config": cfg,
        "weights": weights,
        "inputs_sha256": digest(base / "inputs.jsonl"),
        "labels_sha256": complete["labels_sha256"],
        "conditions": conditions,
        "item_ids": [r["id"] for r in rows],
        "source_sha256": {str(f.relative_to(ROOT)): digest(f) for f in sources},
    }
    folder = ROOT / "results" / args.run_id
    if folder.exists() and not args.resume:
        raise ValueError("Run exists")
    write_once(folder / "run.json", json_text(stamp))
    for f in sources:
        write_once(folder / "source" / f.relative_to(ROOT), f.read_text())
    pred = folder / "predictions.jsonl"
    previous = (
        [json.loads(s) for s in pred.read_text().splitlines()] if pred.exists() else []
    )
    done = {(r["id"], r["condition"]) for r in previous}
    expected = {(r["id"], c) for r in rows for c in conditions}
    assert len(done) == len(previous) and not done - expected
    if done != expected:
        import torch
        from src.models.control.vlm import ControlledVLM

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = ControlledVLM(args.model, cfg)
        write_once(
            folder / "runtime.json",
            json_text(
                {
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "training_steps": 0,
                }
            ),
        )
        with pred.open("a") as stream:
            for index, row in enumerate(rows):
                order = conditions[index % 3 :] + conditions[: index % 3]
                for condition in order:
                    if (row["id"], condition) in done:
                        continue
                    result = evaluate_condition(
                        model,
                        row,
                        ROOT / row["image"],
                        condition,
                        ROOT / ".cache/paired/crops",
                    )
                    record = {
                        "id": row["id"],
                        "pair_id": row["pair_id"],
                        "origin": row["origin"],
                        "condition": condition,
                        "status": "ok",
                        **result,
                    }
                    stream.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    done.add((row["id"], condition))
                    print(len(done), len(expected), row["id"], condition, flush=True)
    write_once(
        folder / "complete.json",
        json_text(
            {
                "items": len(rows),
                "records": len(done),
                "predictions_sha256": digest(pred),
            }
        ),
    )


if __name__ == "__main__":
    main()
