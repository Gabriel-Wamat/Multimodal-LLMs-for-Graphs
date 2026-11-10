"""Matriz pareada por item, com resultados imutáveis e inicialização rastreável."""

import argparse
import hashlib
import importlib.metadata
import json
import os

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.run import prompt_for
from src.evaluation.score import read_jsonl

from src.models.control.artifacts import REGISTRY, installed
from src.models.control.fusion import ARMS

CONFIG = ROOT / "configs/control_mof_stage0.json"


def select_rows(cfg, smoke):
    rows = read_jsonl(ROOT / cfg["inputs"])
    split = json.loads((ROOT / cfg["split"]).read_text())
    if sha(ROOT / cfg["inputs"]) != split["inputs_sha256"]:
        raise ValueError("Inputs divergentes")
    if smoke:
        selected = {}
        for row in rows:
            selected.setdefault(row["tipo"], row)
        rows = list(selected.values())
    for name, digest in {(r["image"], r["image_sha256"]) for r in rows}:
        if sha(ROOT / name) != digest:
            raise ValueError("Imagem alterada")
    allowed = {
        "id",
        "grafico",
        "tipo",
        "estilo",
        "nivel",
        "pergunta",
        "formato",
        "image",
        "image_sha256",
    }
    if any(set(row) != allowed for row in rows):
        raise ValueError("Campos indevidos no input")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["llava", "qwen"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--arms", nargs="+", choices=ARMS)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.run_id):
        raise ValueError("ID inválido")
    cfg = json.loads(CONFIG.read_text())
    if args.seed is not None:
        cfg["adapter_seed"] = args.seed
    arms = args.arms or cfg["arms"]
    if len(set(arms)) != len(arms):
        raise ValueError("Braços duplicados")
    folder = ROOT / "results" / args.run_id
    if folder.exists() and not args.resume:
        raise ValueError("Run já existe; usar novo ID ou --resume")
    if args.resume and not (folder / "run.json").exists():
        raise ValueError("Retomada sem registro")
    rows = select_rows(cfg, args.smoke)
    needed = {args.model}
    if any(a not in ("native", "native_double") for a in arms):
        needed |= {"clip", "dinov3"}
    weights = {key: installed(key, verify=True) for key in sorted(needed)}
    sources = list((ROOT / "src/models/control").glob("*.py")) + list(
        (ROOT / "src/evaluation/control").glob("*.py")
    )
    sources += [
        ROOT / "src/evaluation/run.py",
        ROOT / "src/evaluation/score.py",
        ROOT / "src/data/pilot.py",
        CONFIG,
        REGISTRY,
        ROOT / "configs/requirements-control.txt",
        ROOT / "scripts/apuana/run_control.sh",
    ]
    stamp = {
        "config": cfg,
        "model": args.model,
        "arms": arms,
        "smoke": args.smoke,
        "expected_items": len(rows),
        "item_ids": [r["id"] for r in rows],
        "registry_sha256": sha(REGISTRY),
        "inputs_sha256": sha(ROOT / cfg["inputs"]),
        "split_sha256": sha(ROOT / cfg["split"]),
        "source_sha256": {str(f.relative_to(ROOT)): sha(f) for f in sorted(sources)},
        "weights": weights,
        "prompt_sha256": hashlib.sha256(
            "".join(prompt_for(r) for r in rows).encode()
        ).hexdigest(),
    }
    write_once(folder / "run.json", json_text(stamp))
    for source in sources:
        write_once(folder / "source" / source.relative_to(ROOT), source.read_text())
    pred_path = folder / "predictions.jsonl"
    previous = read_jsonl(pred_path) if pred_path.exists() else []
    expected = {(r["id"], arm) for r in rows for arm in arms}
    done = {(r["id"], r["arm"]) for r in previous}
    if (
        len(done) != len(previous)
        or done - expected
        or any(r["status"] != "ok" for r in previous)
    ):
        raise ValueError("Retomada inválida")
    if done != expected:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA necessário")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from src.models.control.vlm import ControlledVLM

        model = ControlledVLM(args.model, cfg)
        metadata = {
            "packages": {
                k: importlib.metadata.version(k)
                for k in ["torch", "torchvision", "transformers", "numpy", "Pillow"]
            },
            "gpu": torch.cuda.get_device_name(0),
            "adapter_sha256": model.adapter_sha,
            "all_parameters_frozen": True,
            "training_steps": 0,
        }
        write_once(folder / "runtime.json", json_text(metadata))
        with pred_path.open("a") as stream:
            # Ordem rotativa fixa para reduzir confusão entre braço e aquecimento.
            for index, row in enumerate(rows):
                order = arms[index % len(arms) :] + arms[: index % len(arms)]
                for arm in order:
                    if (row["id"], arm) in done:
                        continue
                    response = model.answer(
                        ROOT / row["image"],
                        prompt_for(row),
                        arm,
                        row["grafico"],
                        verify_native=args.smoke and index == 0 and arm == "native",
                    )
                    record = {
                        "id": row["id"],
                        "grafico": row["grafico"],
                        "model": args.model,
                        "arm": arm,
                        "prompt": prompt_for(row),
                        "status": "ok",
                        **response,
                    }
                    stream.write(
                        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                    done.add((row["id"], arm))
                    print(
                        f"{len(done)}/{len(expected)} {arm} {row['id']}: {response['response']}",
                        flush=True,
                    )
    write_once(
        folder / "complete.json",
        json_text(
            {
                "items": len(rows),
                "records": len(done),
                "predictions_sha256": sha(pred_path),
            }
        ),
    )


if __name__ == "__main__":
    main()
