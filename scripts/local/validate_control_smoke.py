"""Audita duas execuções curtas completas; não estabelece eficácia científica."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.pilot import json_text, write_once

from src.models.control.artifacts import digest

ARMS = {
    "native",
    "native_double",
    "clip_double",
    "dino_double",
    "clip_dino",
    "clip_dino_shuffled",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(run_id):
    require(Path(run_id).name == run_id, "ID inválido")
    folder = ROOT / "results" / run_id
    run = json.loads((folder / "run.json").read_text())
    runtime = json.loads((folder / "runtime.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    metrics = json.loads((folder / "scored/metrics.json").read_text())
    rows = [
        json.loads(line)
        for line in (folder / "predictions.jsonl").read_text().splitlines()
    ]
    require(run["smoke"] and metrics["smoke"], "Somente testes curtos")
    require(
        set(run["arms"]) == ARMS and run["expected_items"] == 6, "Matriz incompleta"
    )
    expected = {(item, arm) for item in run["item_ids"] for arm in ARMS}
    require(
        len(rows) == 36 and {(r["id"], r["arm"]) for r in rows} == expected,
        "Cobertura/duplicação",
    )
    require(
        complete["records"] == 36 and complete["items"] == 6,
        "Marcador de conclusão inválido",
    )
    predictions_sha = digest(folder / "predictions.jsonl")
    require(
        predictions_sha
        == complete["predictions_sha256"]
        == metrics["predictions_sha256"],
        "Hash das previsões",
    )
    require(digest(folder / "run.json") == metrics["run_sha256"], "Hash do registro")
    for name, sha in run["source_sha256"].items():
        require(digest(folder / "source" / name) == sha, f"Snapshot divergente: {name}")
    require(
        runtime["all_parameters_frozen"] and runtime["training_steps"] == 0,
        "Treinamento indevido",
    )
    require(
        all(r["status"] == "ok" and r["training_steps"] == 0 for r in rows),
        "Falha de inferência",
    )
    require(
        all(r["adapter_sha256"] == runtime["adapter_sha256"] for r in rows),
        "Inicializações divergentes",
    )
    initializer = (
        ROOT
        / "experiments"
        / run["config"]["experiment"]
        / "initializers"
        / run["model"]
        / str(run["config"]["adapter_seed"])
    )
    require(
        digest(initializer / "initial_adapters.safetensors")
        == runtime["adapter_sha256"],
        "Pesos iniciais divergentes",
    )
    for row in rows:
        trace = row["visual_trace"]
        require(trace["native_tokens"] == 576, "Grade nativa inválida")
        require(
            trace["output_tokens"] == (576 if row["arm"] == "native" else 1152),
            "Intercalação inválida",
        )
        require(
            trace["hidden_size"] == (4096 if run["model"] == "llava" else 1536),
            "Dimensão inválida",
        )
    parity = [r for r in rows if r.get("native_reference_parity") is not None]
    require(
        len(parity) == 1 and parity[0]["native_reference_parity"],
        "Equivalência nativa não demonstrada",
    )
    lookup = {(r["id"], r["arm"]): r for r in rows}
    agreement = None
    if run["model"] == "llava":
        agreement = all(
            lookup[(i, "native_double")]["response"]
            == lookup[(i, "clip_double")]["response"]
            for i in run["item_ids"]
        )
        require(agreement, "Controles duplicados LLaVA divergem")
    return {
        "run_id": run_id,
        "model": run["model"],
        "items": 6,
        "responses": len(rows),
        "run_sha256": digest(folder / "run.json"),
        "predictions_sha256": predictions_sha,
        "adapter_sha256": runtime["adapter_sha256"],
        "gpu": runtime["gpu"],
        "packages": runtime["packages"],
        "native_reference_parity": True,
        "llava_duplicate_control_agreement": agreement,
        "peak_gpu_bytes": max(r["peak_gpu_bytes"] for r in rows),
        "observed_seconds": sum(r["seconds"] for r in rows),
        "source_sha256": run["source_sha256"],
        "item_ids": run["item_ids"],
        "inputs_sha256": run["inputs_sha256"],
        "seed": run["config"]["adapter_seed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id, "ID inválido")
    models = [validate(args.run_id + "-" + model) for model in ("llava", "qwen")]
    for field in ("item_ids", "inputs_sha256", "seed", "source_sha256"):
        require(models[0][field] == models[1][field], f"Comparação divergente: {field}")
    report = {
        "status": "technical_validation_pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_steps": 0,
        "efficacy_evaluated": False,
        "full_pilot_evaluated": False,
        "D5": "pass",
        "D6": "pass",
        "scope": "installation and six-arm smoke only",
        "models": models,
        "limitations": [
            "New adapters are unaligned.",
            "Six questions do not establish accuracy.",
            "Qwen uses a documented virtual mRoPE grid.",
            "50% campaign not included in this validation.",
        ],
    }
    folder = ROOT / "experiments/control-mof-dinov3-stage0-v1/validations" / args.run_id
    write_once(folder / "validation.json", json_text(report))
    lines = [
        "# Validação técnica CLIP + DINOv3",
        "",
        "Dois modelos, seis perguntas por modelo e seis condições: **72 respostas verificadas**.",
        "Nenhum treinamento. Projetores novos sem alinhamento. Este teste não mede eficácia.",
        "",
        "| Modelo | Respostas | Tokens original/fusão | Equivalência nativa |",
        "|---|---:|---:|---|",
    ]
    for model in models:
        lines.append(f"| {model['model']} | 36 | 576 / 1152 | Passou |")
    lines += [
        "",
        "Validados: cobertura, snapshots do código, hashes, inicialização persistida,",
        "congelamento, dimensões e equivalência do controle original. Os dois controles",
        "duplicados do LLaVA concordam nas seis respostas.",
        "",
        "A campanha de 3173 perguntas, réplicas e treinamento permanecem etapas separadas.",
        "Custos do teste incluem carregamento, cache e a inferência extra da checagem de equivalência.",
        "",
        "Comando para a campanha completa, a partir da raiz no Apuana:",
        "",
        "```sh",
        "python3 scripts/apuana/submit_control.py --run-id controle-piloto50",
        "```",
        "",
    ]
    write_once(folder / "REPORT.md", "\n".join(lines))
    print(
        json_text({"status": report["status"], "responses": 72, "folder": str(folder)})
    )


if __name__ == "__main__":
    main()
