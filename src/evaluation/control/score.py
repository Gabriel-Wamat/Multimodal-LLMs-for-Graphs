"""Correção prospectiva uniforme; idioma binário explícito, dados brutos intactos."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.score import evaluate, norm, read_jsonl


def normalize_binary(labels, predictions):
    truth = {q["id"]: q for q in labels}
    output = []
    for row in predictions:
        copy = dict(row)
        if truth[row["id"]]["formato"] == "sim/não":
            copy["response"] = {"yes": "sim", "no": "não"}.get(
                norm(row["response"]), row["response"]
            )
        output.append(copy)
    return output


def paired_graph_delta(labels, first, second, seed=20261102, replicates=2000):
    if len(first) != len(second) or len(first) != len(labels):
        raise ValueError("Cobertura incompatível")
    by_id = {r["id"]: r for r in second}
    groups = defaultdict(list)
    for q, a in zip(labels, first):
        if a["id"] != q["id"] or a["id"] not in by_id:
            raise ValueError("IDs incompatíveis")
        b = by_id[a["id"]]
        groups[q["grafico"]].append(int(b["correct"]) - int(a["correct"]))
    strata = defaultdict(list)
    metadata = {q["grafico"]: (q["tipo"], q["estilo"]) for q in labels}
    for graph, values in groups.items():
        strata[metadata[graph]].append((sum(values), len(values)))
    rng = np.random.default_rng(seed)
    hits = np.zeros(replicates)
    totals = np.zeros(replicates)
    for group in strata.values():
        data = np.array(group)
        sample = data[rng.integers(0, len(data), (replicates, len(data)))]
        hits += sample[:, :, 0].sum(1)
        totals += sample[:, :, 1].sum(1)
    delta = sum(sum(g) for g in groups.values()) / len(labels)
    return {
        "delta_accuracy": delta,
        "delta_percentage_points": 100 * delta,
        "ci95_graph_bootstrap": np.quantile(hits / totals, [0.025, 0.975]).tolist(),
        "unit": "same questions; graph-clustered, stratified type/style; exploratory",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id")
    args = p.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("ID inválido")
    folder = ROOT / "results" / args.run_id
    run = json.loads((folder / "run.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    if sha(folder / "predictions.jsonl") != complete["predictions_sha256"]:
        raise ValueError("Previsões alteradas")
    cfg = run["config"]
    if sha(ROOT / cfg["split"]) != run["split_sha256"]:
        raise ValueError("Split alterado desde a inferência")
    if sha(ROOT / cfg["inputs"]) != run["inputs_sha256"]:
        raise ValueError("Inputs alterados desde a inferência")
    for name in ("src/evaluation/control/score.py", "src/evaluation/score.py"):
        if sha(ROOT / name) != run["source_sha256"][name]:
            raise ValueError(
                "Corretor diferente do registrado; use uma nova versão de avaliação"
            )
    split = json.loads((ROOT / cfg["split"]).read_text())
    if sha(ROOT / cfg["labels"]) != split["labels_sha256"]:
        raise ValueError("Gabarito alterado")
    all_labels = {q["id"]: q for q in read_jsonl(ROOT / cfg["labels"])}
    labels = [all_labels[i] for i in run["item_ids"]]
    rows = read_jsonl(folder / "predictions.jsonl")
    if len(rows) != len(labels) * len(run["arms"]):
        raise ValueError("Execução incompleta")
    if any(r["arm"] not in run["arms"] or r["status"] != "ok" for r in rows):
        raise ValueError("Braço ou status inválido")
    metrics = {}
    details = {}
    costs = {}
    for arm in run["arms"]:
        predictions = [r for r in rows if r["arm"] == arm]
        strict, _ = evaluate(labels, predictions)
        primary, graded = evaluate(labels, normalize_binary(labels, predictions))
        if not primary["all_records_successful"]:
            raise ValueError("Cobertura incompleta")
        metrics[arm] = {"primary_binary_aliases_v2": primary, "strict_v1": strict}
        details[arm] = graded
        costs[arm] = {
            "total_seconds_observed": sum(r["seconds"] for r in predictions),
            "median_seconds_observed": float(
                np.median([r["seconds"] for r in predictions])
            ),
            "input_tokens": sum(r["input_tokens"] for r in predictions),
            "generated_tokens": sum(r["generated_tokens"] for r in predictions),
            "peak_gpu_bytes": max(r["peak_gpu_bytes"] for r in predictions),
            "note": "Observed includes lazy loading/cache and optional native parity check; not a warmed throughput benchmark.",
        }
    contrasts = {}
    for name in [
        "primary_contrast",
        "token_budget_control",
        "alignment_control",
        "spatial_negative_control",
    ]:
        before, after = cfg[name]
        if before in details and after in details:
            contrasts[name] = {
                "before": before,
                "after": after,
                **paired_graph_delta(labels, details[before], details[after]),
            }
    report = {
        "stage": cfg["stage"],
        "model": run["model"],
        "smoke": run["smoke"],
        "interpretation": "Technical smoke only; not efficacy evidence."
        if run["smoke"]
        else "Exploratory unaligned stage zero; do not infer encoder quality from random adapter performance.",
        "scorer_version": "control-v2-exact-binary-aliases",
        "metrics": metrics,
        "contrasts": contrasts,
        "costs": costs,
        "predictions_sha256": sha(folder / "predictions.jsonl"),
        "run_sha256": sha(folder / "run.json"),
        "scorer_sha256": sha(Path(__file__)),
    }
    write_once(folder / "scored/metrics.json", json_text(report))
    raw = {(r["id"], r["arm"]): r["response"] for r in rows}
    scored = [
        dict(r, arm=arm, raw_response=raw[(r["id"], arm)])
        for arm, items in details.items()
        for r in items
    ]
    write_once(
        folder / "scored/answers_scored.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scored),
    )
    print(
        json_text(
            {
                "model": run["model"],
                "smoke": run["smoke"],
                "items": len(labels),
                "arms": len(metrics),
                "complete": True,
            }
        )
    )


if __name__ == "__main__":
    main()
