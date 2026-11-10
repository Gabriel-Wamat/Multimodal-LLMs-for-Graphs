"""Instala os quatro modelos em cache próprio, sem alterar o ambiente I-MoF."""

import argparse
import json
import shutil
from pathlib import Path

from src.data.pilot import ROOT

from .artifacts import INSTALL, REGISTRY, digest, write_json


def prepare_hf(key, cfg):
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().model_info(
        cfg["repo"], revision=cfg["revision"], files_metadata=True
    )
    if info.sha != cfg["revision"]:
        raise ValueError("Revisão divergente")
    files = {}
    folder = None
    for f in info.siblings:
        name = f.rfilename
        keep = name.endswith((".json", ".txt", ".model")) or (
            name.endswith(".safetensors")
            if cfg["weights"] == "safetensors"
            else name == "pytorch_model.bin"
        )
        if (
            not keep
            or "/" in name
            or name in ("training_args.bin", "trainer_state.json")
        ):
            continue
        print(f"{key}: {name}", flush=True)
        try:
            path = Path(
                hf_hub_download(
                    cfg["repo"],
                    name,
                    revision=cfg["revision"],
                    cache_dir=ROOT / ".cache/huggingface/hub",
                )
            )
        except OSError as exc:
            if "Consistency check failed" not in str(exc):
                raise
            path = Path(
                hf_hub_download(
                    cfg["repo"],
                    name,
                    revision=cfg["revision"],
                    cache_dir=ROOT / ".cache/huggingface/hub",
                    force_download=True,
                )
            )
        sha = digest(path)
        lfs = f.lfs
        expected = (
            lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        )
        if expected and sha != expected:
            raise ValueError(f"SHA-256 LFS inválido: {name}")
        if f.size is not None and path.stat().st_size != f.size:
            raise ValueError(f"Tamanho inválido: {name}")
        files[str(path)] = sha
        folder = path.parent
    if not any(n.endswith((".safetensors", ".bin")) for n in files):
        raise ValueError("Pesos ausentes")
    return {
        "config": cfg,
        "status": "ready",
        "path": str(folder),
        "files": files,
        "origin": "official_huggingface_fixed_revision",
    }


def prepare_dino(cfg, source):
    from huggingface_hub import HfApi

    from .encoders import load_dinov3

    if not source:
        raise ValueError(
            "Informe --dinov3-source /caminho/model.safetensors (o arquivo local .pth também é aceito)."
        )
    source = Path(source).resolve()
    sha = digest(source)
    if sha != cfg["expected_sha256"] or source.stat().st_size != cfg["expected_bytes"]:
        raise ValueError(
            f"Checkpoint não corresponde ao safetensors oficial ViT-B/16: SHA-256={sha}"
        )
    info = HfApi().model_info(
        cfg["repo"], revision=cfg["revision"], files_metadata=True
    )
    official = next(f for f in info.siblings if f.rfilename == "model.safetensors")
    lfs = official.lfs
    expected = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
    if (
        info.sha != cfg["revision"]
        or expected != sha
        or official.size != cfg["expected_bytes"]
    ):
        raise ValueError("Metadados oficiais divergentes")
    folder = ROOT / ".cache/control/dinov3"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (sha + ".safetensors")
    if not target.exists():
        tmp = target.with_suffix(".tmp")
        shutil.copyfile(source, tmp)
        tmp.replace(target)
    if digest(target) != sha:
        raise ValueError("Cópia local corrompida")
    model = load_dinov3(target, cfg)
    del model
    return {
        "config": cfg,
        "status": "ready",
        "path": str(target),
        "files": {str(target): sha},
        "strict_architecture_load": True,
        "origin": "existing_local_safetensors_full_official_lfs_sha256_verified",
        "source": str(source),
        "source_sha256": sha,
        "official_metadata": {
            "repo": cfg["repo"],
            "revision": info.sha,
            "file": official.rfilename,
            "lfs_sha256": expected,
            "size": official.size,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models",
        nargs="+",
        choices=["llava", "qwen", "clip", "dinov3"],
        default=["llava", "qwen", "clip", "dinov3"],
    )
    p.add_argument("--dinov3-source", type=Path)
    args = p.parse_args()
    registry = json.loads(REGISTRY.read_text())
    record = json.loads(INSTALL.read_text()) if INSTALL.exists() else {"models": {}}
    failures = []
    for key in args.models:
        cfg = registry["models"][key]
        try:
            entry = (
                prepare_dino(cfg, args.dinov3_source)
                if key == "dinov3"
                else prepare_hf(key, cfg)
            )
            record["models"][key] = entry
            write_json(INSTALL, record)
            print(f"{key}: READY, arquivos e hashes registrados.", flush=True)
        except Exception as exc:
            # Não registrar tokens ou URLs assinadas de exceções HTTP.
            record["models"][key] = {
                "config": cfg,
                "status": "error",
                "error_type": type(exc).__name__,
            }
            write_json(INSTALL, record)
            failures.append(key)
            print(f"{key}: FAILED ({type(exc).__name__})", flush=True)
            if isinstance(exc, ValueError):
                print(str(exc), flush=True)
    if failures:
        raise SystemExit("Instalação incompleta: " + ",".join(failures))


if __name__ == "__main__":
    main()
