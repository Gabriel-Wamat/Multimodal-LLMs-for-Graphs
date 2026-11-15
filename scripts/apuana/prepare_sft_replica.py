"""Prepare a seed replicate from a primary run's frozen source snapshot."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


REPLICATION_SEEDS = (20261103, 20261104)
TRAIN_CONFIG = "configs/graph_full_sft_v1.json"
VISION_CONFIG = "configs/control_mof_stage0.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def prepare(source, destination, primary_record, seed):
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Choose a new snapshot path outside the source snapshot")
    if seed not in REPLICATION_SEEDS:
        raise ValueError("Seed is outside the declared replication series")
    primary = read(primary_record)
    training = read(source / TRAIN_CONFIG)
    vision = read(source / VISION_CONFIG)
    if training != primary["config"] or vision != primary["base_config"]:
        raise ValueError("Snapshot does not match the primary training configuration")
    for name, expected in {**primary["source_sha256"], **primary["data_sha256"]}.items():
        if sha(source / name) != expected:
            raise ValueError(f"Primary training source or data changed: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        snapshot = Path(temp) / "snapshot"
        for name in ["src", "configs", "scripts/apuana"]:
            shutil.copytree(
                source / name,
                snapshot / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        for name in ["data", ".venv-control", "experiments", "results", "logs"]:
            target = (source / name).resolve(strict=True)
            (snapshot / name).symlink_to(target, target_is_directory=True)
        # Keep writable feature caches separate so concurrent replicas cannot race on cache misses.
        cache = snapshot / ".cache/control"
        cache.mkdir(parents=True)
        (cache / "installed.json").symlink_to(
            (source / ".cache/control/installed.json").resolve(strict=True)
        )
        training["seed"] = seed
        vision["adapter_seed"] = seed
        for name, value in [(TRAIN_CONFIG, training), (VISION_CONFIG, vision)]:
            (snapshot / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        record = {
            "seed": seed,
            "primary_training_record_sha256": sha(primary_record),
            "primary_seed": primary["config"]["seed"],
            "primary_adapter_seed": primary["base_config"]["adapter_seed"],
            "changed_fields": ["graph_full_sft_v1.seed", "control_mof_stage0.adapter_seed"],
            "files": {
                str(p.relative_to(snapshot)): sha(p)
                for directory in ("configs", "src", "scripts")
                for p in sorted((snapshot / directory).rglob("*"))
                if p.is_file()
            },
        }
        (snapshot / "replica.json").write_text(json.dumps(record, indent=2) + "\n")
        snapshot.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--primary-record", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=REPLICATION_SEEDS, required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.primary_record, args.seed)
    print(args.destination)


if __name__ == "__main__":
    main()
