"""Epoch training with development-only selection and resumable optimizer boundaries."""

import argparse
import fcntl
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from src.data.pilot import ROOT, write_once, json_text
from src.models.control.artifacts import digest, write_json, installed
from src.evaluation.score import evaluate
from src.evaluation.control.score import normalize_binary
from .alignment import AlignmentModel
from .checkpoint import save, restore


def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def validate(model, rows, folder, epoch):
    start = time.perf_counter()
    losses = []
    predictions = []
    with torch.no_grad():
        for completed, row in enumerate(rows, 1):
            loss, _ = model.loss(row)
            losses.append(float(loss))
            predictions.append(
                {"id": row["id"], "response": model.answer(row), "status": "ok"}
            )
            if completed % 25 == 0 or completed == len(rows):
                progress = {
                    "phase": "validation",
                    "epoch": epoch,
                    "completed": completed,
                    "total": len(rows),
                    "mean_loss": sum(losses) / len(losses),
                    "seconds": time.perf_counter() - start,
                    "updated_at_unix": time.time(),
                }
                write_json(folder / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    strict, _ = evaluate(rows, predictions)
    metric, _ = evaluate(rows, normalize_binary(rows, predictions))
    out = folder / "validation" / f"epoch-{epoch:02d}"
    out.mkdir(parents=True, exist_ok=True)
    # Recompute interrupted validation; completed epochs are committed in last.pt.
    write_json(out / "predictions.json", predictions)
    result = {
        "loss": sum(losses) / len(losses),
        "accuracy": metric["accuracy"],
        "strict_accuracy": strict["accuracy"],
        "metrics": metric,
        "seconds": time.perf_counter() - start,
        "questions": len(rows),
    }
    write_json(out / "metrics.json", result)
    return result


def require_gates(prefix):
    evidence = {}
    for key in ["llava", "qwen"]:
        for arm in ["clip_double", "clip_dino"]:
            d = (
                ROOT
                / "experiments/five-stage-campaign/training_checks"
                / f"{prefix}-{key}-{arm}"
            )
            report = json.loads((d / "complete.json").read_text())
            run = json.loads((d / "run.json").read_text())
            if report["status"] != "loss_gradient_update_reload_pass" or run[
                "source_sha256"
            ] != digest(Path(__file__).with_name("alignment.py")):
                raise ValueError("Alignment gates missing or stale")
            evidence[str(d.relative_to(ROOT))] = digest(d / "complete.json")
    p = ROOT / "experiments/five-stage-campaign/resume_check.json"
    r = json.loads(p.read_text())
    if not r["passed"] or r["checkpoint_source_sha256"] != digest(
        Path(__file__).with_name("checkpoint.py")
    ):
        raise ValueError("Resume gate missing or stale")
    evidence[str(p.relative_to(ROOT))] = digest(p)
    return evidence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llava", "qwen"], required=True)
    p.add_argument("--arm", choices=["clip_double", "clip_dino"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checks-prefix", default="alignment-check-v1")
    args = p.parse_args()
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        raise ValueError("Invalid run ID")
    gates = require_gates(args.checks_prefix)
    cfg = json.loads((ROOT / "configs/alignment_training_v1.json").read_text())
    base_cfg = json.loads((ROOT / "configs/control_mof_stage0.json").read_text())
    manifest = ROOT / "data/manifests/train_dev_v2/manifest.json"
    metadata = json.loads(manifest.read_text())
    train_path = ROOT / cfg["training_data"]
    dev_path = ROOT / cfg["development_data"]
    assert (
        digest(train_path) == metadata["counts"]["train"]["sha256"]
        and digest(dev_path) == metadata["counts"]["development"]["sha256"]
    )
    train = read(train_path)
    dev = read(dev_path)
    assert not ({r["family_id"] for r in train} & {r["family_id"] for r in dev})
    split_path = manifest.with_name("split.json")
    if digest(split_path) != metadata["split_sha256"]:
        raise ValueError("Split manifest changed")
    split = json.loads(split_path.read_text())
    if split.get("eligible_for_training") is not True:
        raise ValueError("Data audit has not authorized training")
    protected = set(split["evaluation_exploratory"]) | set(
        split["quarantined_near_evaluation"]
    )
    verified_images = {}
    for role, rows in [("train", train), ("development", dev)]:
        if {r["grafico"] for r in rows} != set(split[role]):
            raise ValueError("Question membership differs from audited split")
        for row in rows:
            if protected.intersection(split["family_members"][row["family_id"]]):
                raise ValueError("Training/development family overlaps evaluation")
            expected = row["image_sha256"]
            if row["image"] not in verified_images:
                verified_images[row["image"]] = digest(ROOT / row["image"])
            if verified_images[row["image"]] != expected:
                raise ValueError("Training/development image changed: " + row["image"])
    sources = list((ROOT / "src/training").glob("*.py")) + list(
        (ROOT / "src/models/control").glob("*.py")
    )
    sources += [
        ROOT / "src/evaluation/run.py",
        ROOT / "src/evaluation/score.py",
        ROOT / "src/evaluation/control/score.py",
    ]
    record = {
        "model": args.model,
        "arm": args.arm,
        "config": cfg,
        "base_config": base_cfg,
        "manifest_sha256": digest(manifest),
        "split_sha256": digest(split_path),
        "verified_image_count": len(verified_images),
        "gates": gates,
        "weights": {
            k: installed(k, verify=True) for k in [args.model, "clip", "dinov3"]
        },
        "source_sha256": {str(f.relative_to(ROOT)): digest(f) for f in sources},
    }
    signature = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    folder = ROOT / "experiments/training" / args.run_id
    if folder.exists() and not args.resume:
        raise ValueError("Run exists; use explicit resume")
    if args.resume and not (folder / "last.pt").exists():
        raise ValueError("No recoverable checkpoint")
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_once(folder / "run.json", json_text(record))
    for f in sources:
        write_once(folder / "source" / f.relative_to(ROOT), f.read_text())
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AlignmentModel(args.model, base_cfg, args.arm)
    optimizer = torch.optim.AdamW(
        model.projectors.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    total = cfg["epochs"] * math.ceil(len(train) / cfg["gradient_accumulation"])
    warm = max(1, round(total * cfg["warmup_fraction"]))

    def multiplier(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (
            1 + math.cos(math.pi * min(1, (step - warm) / max(1, total - warm)))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    state = {
        "epoch": 0,
        "cursor": 0,
        "updates": 0,
        "loss_sum": 0.0,
        "loss_count": 0,
        "epoch_train_seconds": 0.0,
        "history": [],
        "best": None,
    }

    def persist():
        save(
            folder / "last.pt", model.projectors, optimizer, scheduler, state, signature
        )

    def select(result, epoch):
        score = [result["accuracy"], -result["loss"]]
        if state["best"] is None or tuple(score) > tuple(state["best"]["score"]):
            state["best"] = {
                "epoch": epoch,
                "score": score,
                "updates": state["updates"],
            }
            save(
                folder / "best.pt",
                model.projectors,
                optimizer,
                scheduler,
                state,
                signature,
            )
            temp = folder / "best.tmp.safetensors"
            model.save_projectors(temp)
            temp.replace(folder / "best.safetensors")

    if args.resume:
        state = restore(
            folder / "last.pt", model.projectors, optimizer, scheduler, signature
        )
    else:
        initial = validate(model, dev, folder, 0)
        select(initial, 0)
        state["history"].append(
            {"epoch": 0, "phase": "initial", "development": initial}
        )
        persist()
    while state["epoch"] < cfg["epochs"]:
        epoch = state["epoch"]
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(cfg["seed"] + epoch)
        ).tolist()
        torch.cuda.reset_peak_memory_stats()
        while state["cursor"] < len(order):
            started = time.perf_counter()
            batch = order[
                state["cursor"] : state["cursor"] + cfg["gradient_accumulation"]
            ]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss, _ = model.loss(train[index])
                state["loss_sum"] += float(loss.detach())
                state["loss_count"] += 1
                (loss / len(batch)).backward()
                del loss
            for name, module in model.projectors.items():
                if not all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in module.parameters()
                ):
                    raise RuntimeError("Invalid gradients: " + name)
            norm = torch.nn.utils.clip_grad_norm_(
                model.projectors.parameters(),
                cfg["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            scheduler.step()
            model.assert_frozen()
            state["cursor"] += len(batch)
            state["updates"] += 1
            state["epoch_train_seconds"] += time.perf_counter() - started
            if state["updates"] % 10 == 0:
                progress = {
                    "phase": "train",
                    "epoch": epoch + 1,
                    "updates": state["updates"],
                    "cursor": state["cursor"],
                    "total": len(order),
                    "mean_loss": state["loss_sum"] / state["loss_count"],
                    "updated_at_unix": time.time(),
                }
                write_json(folder / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if state["updates"] % cfg["checkpoint_every_updates"] == 0 or state[
                "cursor"
            ] == len(order):
                persist()
                print(
                    json.dumps(
                        {
                            "phase": "train",
                            "epoch": epoch + 1,
                            "cursor": state["cursor"],
                            "updates": state["updates"],
                            "loss": state["loss_sum"] / state["loss_count"],
                            "lr": scheduler.get_last_lr(),
                            "gradient_norm": float(norm),
                        }
                    ),
                    flush=True,
                )
        result = validate(model, dev, folder, epoch + 1)
        select(result, epoch + 1)
        entry = {
            "epoch": epoch + 1,
            "phase": "epoch_complete",
            "train_loss": state["loss_sum"] / state["loss_count"],
            "development": result,
            "lr": scheduler.get_last_lr(),
            "train_seconds": state["epoch_train_seconds"],
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "updates": state["updates"],
            "trainable_parameters": sum(
                p.numel() for p in model.projectors.parameters()
            ),
        }
        state["history"].append(entry)
        state.update(
            epoch=epoch + 1,
            cursor=0,
            loss_sum=0.0,
            loss_count=0,
            epoch_train_seconds=0.0,
        )
        persist()
        write_json(folder / "history.json", state["history"])
        print(json_text(entry), flush=True)
    write_once(
        folder / "complete.json",
        json_text(
            {
                "epochs": state["epoch"],
                "updates": state["updates"],
                "best": state["best"],
                "best_sha256": digest(folder / "best.safetensors"),
                "signature": signature,
                "held_out_evaluation": "pending",
            }
        ),
    )


if __name__ == "__main__":
    main()
