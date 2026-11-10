"""Intercalação espacial no estilo I-MoF; adaptadores novos NÃO estão alinhados."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .artifacts import digest, write_json

ARMS = (
    "native",
    "native_double",
    "clip_double",
    "dino_double",
    "clip_dino",
    "clip_dino_shuffled",
)


def interleave(left, right):
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("Features incompatíveis para intercalar")
    return torch.stack((left, right), dim=1).reshape(2 * left.shape[0], left.shape[1])


def shuffle_patches(features, image_id, seed):
    hashed = int(hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()[:16], 16)
    g = torch.Generator(device="cpu")
    g.manual_seed(hashed)
    idx = torch.randperm(len(features), generator=g).to(features.device)
    return features[idx]


def expand_image_tokens(inputs, token_id):
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("Somente uma imagem/item por chamada")
    ids = inputs["input_ids"][0]
    mask = ids == token_id
    if mask.sum().item() != 576:
        raise ValueError("Esperados 576 placeholders nativos")
    repeats = torch.where(mask, 2, 1)
    result = dict(inputs)
    result["input_ids"] = torch.repeat_interleave(ids, repeats).unsqueeze(0)
    result["attention_mask"] = torch.repeat_interleave(
        inputs["attention_mask"][0], repeats
    ).unsqueeze(0)
    result.pop("token_type_ids", None)
    return result


class StageZeroAdapters(nn.Module):
    def __init__(self, model_key, hidden_size, seed, std=0.02):
        super().__init__()
        self.model_key = model_key
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if model_key == "qwen":
                self.clip = nn.Sequential(
                    nn.Linear(1024, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
            self.dino = nn.Sequential(
                nn.Linear(768, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=std)
                    nn.init.zeros_(module.bias)
        self.eval().requires_grad_(False)

    def persist(self, folder, metadata):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "initial_adapters.safetensors"
        stamp = folder / "initialization.json"
        if path.exists():
            saved = json.loads(stamp.read_text())
            if saved["metadata"] != metadata or digest(path) != saved["sha256"]:
                raise ValueError("Inicialização existente divergente")
            self.load_state_dict(load_file(str(path)), strict=True)
        else:
            if stamp.exists():
                raise ValueError("Inicialização incompleta")
            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in self.state_dict().items()
                },
                str(path),
            )
            write_json(
                stamp,
                {
                    "metadata": metadata,
                    "sha256": digest(path),
                    "training_steps": 0,
                    "aligned": False,
                },
            )
        return digest(path)
