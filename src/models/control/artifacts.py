import hashlib
import json
from pathlib import Path

from src.data.pilot import ROOT

REGISTRY = ROOT / "configs/models_control.json"
INSTALL = ROOT / ".cache/control/installed.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def installed(key, verify=False):
    config = json.loads(REGISTRY.read_text())["models"][key]
    record = json.loads(INSTALL.read_text())["models"][key]
    if record["config"] != config or record.get("status") != "ready":
        raise ValueError(f"Instalação divergente ou incompleta: {key}")
    if not record.get("files"):
        raise ValueError(f"Manifesto vazio: {key}")
    for name, sha in record["files"].items():
        if not Path(name).is_file():
            raise FileNotFoundError(name)
        if verify and digest(name) != sha:
            raise ValueError(f"Hash inválido: {name}")
    return record
