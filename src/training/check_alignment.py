"""GPU gates: real loss/backward/update, frozen backbones, exact checkpoint reload."""

import argparse
import gc
import json
from pathlib import Path
import torch
from src.data.pilot import ROOT, write_once, json_text
from src.models.control.artifacts import digest
from .alignment import AlignmentModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llava", "qwen"], required=True)
    p.add_argument("--arm", choices=["clip_double", "clip_dino"], required=True)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Invalid ID")
    folder = ROOT / "experiments/five-stage-campaign/training_checks" / args.run_id
    if folder.exists():
        raise ValueError("Use a new check ID")
    manifest = ROOT / "data/manifests/train_dev_v2/manifest.json"
    metadata = json.loads(manifest.read_text())
    path = manifest.parent / "train.jsonl"
    assert digest(path) == metadata["counts"]["train"]["sha256"]
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    chosen = {}
    for row in rows:
        chosen.setdefault(row["tipo"], row)
    cfg = json.loads((ROOT / "configs/control_mof_stage0.json").read_text())
    torch.manual_seed(cfg["adapter_seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AlignmentModel(args.model, cfg, args.arm)
    write_once(
        folder / "run.json",
        json_text(
            {
                "model": args.model,
                "arm": args.arm,
                "train_manifest_sha256": digest(manifest),
                "items": [r["id"] for r in chosen.values()],
                "source_sha256": digest(Path(__file__).with_name("alignment.py")),
                "initial_adapter_sha256": model.initial_adapter_sha256,
            }
        ),
    )
    optimizer = torch.optim.AdamW(
        model.projectors.parameters(), lr=1e-4, weight_decay=0.0
    )
    diagnostics = []
    for row in chosen.values():
        before = {n: p.detach().clone() for n, p in model.projectors.named_parameters()}
        optimizer.zero_grad(set_to_none=True)
        loss, tokens = model.loss(row)
        loss.backward()
        gradients = {}
        for name, module in model.projectors.items():
            parameters = list(module.parameters())
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in parameters
            ), name
            norm = sum(float(p.grad.float().square().sum()) for p in parameters) ** 0.5
            assert norm > 0, name
            gradients[name] = norm
        torch.nn.utils.clip_grad_norm_(model.projectors.parameters(), 1.0)
        optimizer.step()
        model.assert_frozen()
        changed = {
            name: any(
                not torch.equal(before[n], p)
                for n, p in model.projectors.named_parameters()
                if n.startswith(name + ".")
            )
            for name in model.projectors
        }
        assert all(changed.values()), changed
        diagnostics.append(
            {
                "id": row["id"],
                "loss": float(loss.detach()),
                "supervised_tokens": tokens,
                "gradient_norm": gradients,
                "updated": changed,
            }
        )
        del loss, before
        gc.collect()
    model.save_projectors(folder / "projectors.safetensors")
    with torch.no_grad():
        reference = float(model.loss(rows[0])[0])
    for p in model.projectors.parameters():
        with torch.no_grad():
            p.add_(0.01)
    model.load_projectors(folder / "projectors.safetensors")
    with torch.no_grad():
        restored = float(model.loss(rows[0])[0])
    assert abs(reference - restored) < 1e-6, (reference, restored)
    report = {
        "status": "loss_gradient_update_reload_pass",
        "updates": len(diagnostics),
        "diagnostics": diagnostics,
        "checkpoint_sha256": digest(folder / "projectors.safetensors"),
        "reload_loss_before": reference,
        "reload_loss_after": restored,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(0),
        "limitation": "Diagnostic updates only; not the production training initialization or a finished training campaign.",
    }
    write_once(folder / "complete.json", json_text(report))
    print(json_text(report))


if __name__ == "__main__":
    main()
