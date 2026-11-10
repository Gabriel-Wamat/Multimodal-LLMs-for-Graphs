"""Read saved training progress and shared feature-cache coverage; no job mutations."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def inspect(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("Invalid training run ID")
    folder = ROOT / "experiments/training" / run_id
    run_path = folder / "run.json"
    run = read(run_path)
    cfg = run["base_config"]
    clip, dino = run["weights"]["clip"], run["weights"]["dinov3"]
    stamp = {
        "clip": clip["config"], "dino": dino["config"],
        "clip_files": clip["files"], "dino_files": dino["files"],
        "clip_size": cfg["clip_resolution"], "dino_size": cfg["dino_resolution"],
        "layer": cfg["clip_layer"], "dino_features": cfg["dino_features"],
        "source": run["source_sha256"]["src/models/control/encoders.py"],
    }
    cache_id = hashlib.sha256(json.dumps(stamp, sort_keys=True).encode()).hexdigest()
    cache = ROOT / ".cache/control/features" / cache_id
    result = {
        "observation_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_run": run_id, "model": run["model"], "arm": run["arm"],
        "saved_validations": {}, "shared_feature_cache": {"cache_id": cache_id},
        "limitations": [
            "This snapshot does not query SLURM or GPU activity; check both separately.",
            "Live files may change during this non-atomic observation; start and end times are recorded.",
            "Cache coverage is not completed question count, validation accuracy or training completion.",
            "Cache counters check presence/mtime only; the encoder runner validates content hashes.",
            "Caches are shared: files newer than run.json may originate from other processes.",
            "CLIP-only LLaVA does not use this external feature cache.",
        ],
    }
    for role, key in [("train", "training_data"), ("development", "development_data")]:
        data = ROOT / run["config"][key]
        graphs = {r["grafico"]: r["image_sha256"] for r in
                  (json.loads(line) for line in data.read_text().splitlines())}
        available, new = [], []
        for graph, sha in graphs.items():
            path = cache / (sha + ".safetensors")
            if path.exists() and path.with_suffix(".json").exists():
                available.append(graph)
                if path.stat().st_mtime >= run_path.stat().st_mtime:
                    new.append(graph)
        result["shared_feature_cache"][role] = {
            "graphs": len(graphs), "with_cache": len(available),
            "created_since_run_record": len(new),
            "new_by_type": dict(Counter(g.split("-")[0] for g in new)),
            "last_new_in_manifest_order": new[-1] if new else None,
        }
    for epoch in range(run["config"]["epochs"] + 1):
        path = folder / "validation" / f"epoch-{epoch:02d}" / "metrics.json"
        if path.exists():
            metric = read(path)
            result["saved_validations"][str(epoch)] = {
                key: metric[key] for key in ("questions", "accuracy", "loss", "seconds")
            }
    history = read(folder / "history.json") if (folder / "history.json").exists() else []
    result["completed_epochs"] = [r["epoch"] for r in history if r["phase"] == "epoch_complete"]
    result["completion_marker_present"] = (folder / "complete.json").exists()
    result["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_run")
    args = parser.parse_args()
    print(json.dumps(inspect(args.training_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
