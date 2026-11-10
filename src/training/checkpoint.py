"""Atomic local checkpoints for projectors, optimizer, scheduler, RNG and data cursor."""

from pathlib import Path
import os
import torch


def save(path, projectors, optimizer, scheduler, state, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "projectors": projectors.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "state": state,
        "signature": signature,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def restore(path, projectors, optimizer, scheduler, signature):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["signature"] != signature:
        raise ValueError("Checkpoint protocol/data/source differs")
    projectors.load_state_dict(payload["projectors"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["state"]
