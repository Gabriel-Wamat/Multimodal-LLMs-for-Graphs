"""Post-inference reading-error inspection; never supplies CSV values to a model."""

import argparse
import csv
import html
import io
import json
import os
from pathlib import Path
from urllib.parse import quote

from src.data.pilot import ROOT, json_text, write_once
from src.models.control.artifacts import digest
from .conditions import binary


def escaped(value):
    return html.escape(str(value), quote=True)


def csv_preview(text, limit=40):
    rows = list(csv.reader(io.StringIO(text)))
    body = "".join(
        "<tr>" + "".join("<td>" + escaped(cell) + "</td>" for cell in row) + "</tr>"
        for row in rows[: limit + 1]
    )
    return (
        "<table>"
        + body
        + "</table>"
        + f"<p>{min(limit, max(0, len(rows) - 1))} de {max(0, len(rows) - 1)} linhas de dados exibidas.</p>"
    )


def reading_record(row, label, predictions):
    by_condition = {r["condition"]: r for r in predictions}
    if (
        len(by_condition) != 3
        or len(predictions) != 3
        or set(by_condition) != {"original", "crops_2x2", "transcription"}
    ):
        raise ValueError("Three unique conditions are required")
    calls = [
        c for c in by_condition["transcription"]["calls"] if c["stage"] == "transcribe"
    ]
    if len(calls) != 1:
        raise ValueError("One visual transcription is required")
    return {
        "id": row["id"],
        "pair_id": row["pair_id"],
        "tipo": row["tipo"],
        "attribute": row["attribute"],
        "intensity": row["intensity"],
        "question": row["pergunta"],
        "reference_answer": label["answer"],
        "reference_numeric_check": label["numeric_check"],
        "transcription": calls[0]["response"],
        "responses": {c: r["response"] for c, r in by_condition.items()},
        "correct": {
            c: binary(r["response"]) == label["answer"] for c, r in by_condition.items()
        },
        "reading_error_review": {
            "status": "unreviewed",
            "categories": [],
            "evidence": "",
            "note": "Wrong final answer alone does not establish a reading error.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Invalid ID")
    folder = ROOT / "results" / args.run_id
    run = json.loads((folder / "run.json").read_text())
    complete = json.loads((folder / "complete.json").read_text())
    base = ROOT / "data/manifests" / ("pairs_v1_smoke" if run["smoke"] else "pairs_v1")
    for name in ["inputs", "labels"]:
        if digest(base / (name + ".jsonl")) != run[name + "_sha256"]:
            raise ValueError("Dataset changed")
    path = folder / "predictions.jsonl"
    if digest(path) != complete["predictions_sha256"]:
        raise ValueError("Predictions changed")
    read = lambda p: [json.loads(line) for line in p.read_text().splitlines()]
    labels = {r["id"]: r for r in read(base / "labels.jsonl")}
    inputs = [r for r in read(base / "inputs.jsonl") if r["id"] in set(run["item_ids"])]
    predictions = read(path)
    if len(predictions) != len(inputs) * 3 or len(predictions) != complete["records"]:
        raise ValueError("Incomplete run")
    grouped = {r["id"]: [] for r in inputs}
    for r in predictions:
        if r["status"] != "ok":
            raise ValueError("Failed inference")
        grouped[r["id"]].append(r)
    output = folder / "review"
    cards = []
    records = []
    for row in sorted(inputs, key=lambda r: r["id"]):
        label = labels[row["id"]]
        csv_path = ROOT / label["derived_csv"]
        image_path = ROOT / row["image"]
        if (
            digest(csv_path) != label["csv_sha256"]
            or digest(image_path) != row["image_sha256"]
        ):
            raise ValueError("Reference artifact changed")
        record = reading_record(row, label, grouped[row["id"]])
        records.append(record)
        link = lambda p: quote(os.path.relpath(p, output), safe="/")
        responses = "".join(
            "<tr><td>"
            + escaped(c)
            + "</td><td>"
            + escaped(answer)
            + "</td><td>"
            + ("correta" if record["correct"][c] else "incorreta/inválida")
            + "</td></tr>"
            for c, answer in record["responses"].items()
        )
        cards.append(f'''<details><summary>{escaped(row["id"])} — {escaped(row["tipo"])} / {escaped(row["intensity"])}</summary>
<p>{escaped(row["pergunta"])}</p><img loading="lazy" src="{link(image_path)}" alt="{escaped(row["id"])}">
<h3>Respostas</h3><table>{responses}</table>
<h3>Transcrição visual do modelo</h3><pre>{escaped(record["transcription"])}</pre>
<h3>Referência posterior à inferência</h3><p>Resposta: {escaped(label["answer"])}</p>
<pre>{escaped(json.dumps(label["numeric_check"], ensure_ascii=False, indent=2))}</pre>
<details><summary>CSV de referência</summary>{csv_preview(csv_path.read_text())}<a href="{link(csv_path)}">CSV completo</a></details>
<p>Erro de leitura: ainda não revisado. Não inferir erro de leitura apenas da resposta final.</p></details>''')
    page = (
        """<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Inspeção de erros de leitura</title>
<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px;color:#222}details{border:1px solid #bbb;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:600}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f4f4;padding:12px}td{border:1px solid #ccc;padding:6px}table{border-collapse:collapse}h1{font-size:26px}</style>
<h1>Inspeção de transcrição e resposta</h1><p>Referências CSV exibidas apenas após a inferência. Todos os itens da execução estão disponíveis; nenhuma revisão humana foi presumida.</p>"""
        + "".join(cards)
        + "</html>"
    )
    write_once(output / "index.html", page)
    write_once(
        output / "reading_review.jsonl",
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
    )
    write_once(
        output / "manifest.json",
        json_text(
            {
                "items": len(records),
                "review_status": "unreviewed",
                "allowed_categories": [
                    "axis_or_unit",
                    "legend_or_category",
                    "value_reading",
                    "omission",
                    "unsupported_value",
                    "no_observed_reading_error",
                    "uncertain",
                ],
                "predictions_sha256": digest(path),
                "source_sha256": digest(Path(__file__)),
                "html_sha256": digest(output / "index.html"),
                "records_sha256": digest(output / "reading_review.jsonl"),
                "instructions": "Record concrete transcript spans and CSV/plot evidence in a separate versioned audit. Do not overwrite raw responses or infer error solely from answer correctness.",
            }
        ),
    )


if __name__ == "__main__":
    main()
