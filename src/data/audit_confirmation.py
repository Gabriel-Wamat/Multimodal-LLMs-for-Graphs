"""Recompute confirmation answers from saved CSVs and question text."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
from statistics import mean, median

from src.data.pilot import ROOT, json_text, sha, write_once
from src.evaluation.score import read_jsonl


def compare_names(values, names):
    left, right = names
    return "Sim" if values[left] > values[right] else "Não"


def answer(question, rows):
    text = question["pergunta"]
    kind = question["grafico"].split("-")[0]
    if kind in {"BAR", "PIZ", "BOX"}:
        if kind == "BOX":
            groups = defaultdict(list)
            for row in rows:
                groups[row["grupo"]].append(float(row["valor"]))
            values = {g: median(v) for g, v in groups.items()}
        else:
            field = "valor" if kind == "BAR" else "percentual"
            values = {r["categoria"]: float(r[field]) for r in rows}
        names = re.findall(r"Grupo [A-Z]\b", text)
        if len(names) == 2:
            return compare_names(values, names)
        if len(names) == 1:
            return values[names[0]]
        if "Quantas fatias" in text:
            return len(values)
        if "maior" in text:
            winners = [g for g, v in values.items() if v == max(values.values())]
            if len(winners) != 1:
                raise ValueError("Ambiguous maximum")
            return winners[0]
    elif kind == "HIS":
        counts = {(r["faixa_inicio"], r["faixa_fim"]): int(r["contagem"]) for r in rows}
        ranges = re.findall(r"faixa (\d+) a (\d+)", text)
        if ranges:
            return compare_names(counts, ranges)
        if "Quantas faixas" in text:
            return len(counts)
        if "faixa mais alta" in text:
            return max(counts.values())
    elif kind == "LIN":
        if "ano Alfa" in text:
            return int(max(rows, key=lambda r: float(r["Alfa"]))["ano"])
        if "Beta em" in text:
            (year,) = re.findall(r"\d{4}", text)
            (row,) = [r for r in rows if r["ano"] == year]
            return float(row["Beta"])
        if "termina acima" in text:
            last = max(rows, key=lambda r: int(r["ano"]))
            return "Sim" if float(last["Alfa"]) > float(last["Beta"]) else "Não"
    elif kind == "DIS":
        points = [(float(r["x"]), float(r["y"])) for r in rows]
        if "Quantos pontos" in text:
            return len(set(points))
        if "mais à direita" in text:
            return max(points)[1]
        if "tendência linear" in text:
            xbar = mean(x for x, y in points)
            ybar = mean(y for x, y in points)
            return "Sim" if sum((x - xbar) * (y - ybar) for x, y in points) > 0 else "Não"
    raise ValueError(f"Unknown confirmation question: {text}")


def main():
    folder = ROOT / "data/manifests/confirmation_v1"
    complete = json.loads((folder / "complete.json").read_text())
    for name in ["sources", "inputs", "labels"]:
        if sha(folder / (name + ".jsonl")) != complete[name + "_sha256"]:
            raise ValueError(f"{name} manifest changed")
    sources = {}
    for row in read_jsonl(folder / "sources.jsonl"):
        path = ROOT / row["csv"]
        if sha(path) != row["csv_sha256"]:
            raise ValueError(f"CSV changed: {row['grafico']}")
        with path.open(newline="") as f:
            sources[row["grafico"]] = list(csv.DictReader(f))
    labels = {r["id"]: r for r in read_jsonl(folder / "labels.jsonl")}
    counts = Counter()
    seen = set()
    for q in read_jsonl(folder / "inputs.jsonl"):
        ref = labels[q["id"]]
        if q["id"] in seen or any(
            q[k] != ref[k] for k in ("pergunta", "grafico", "formato", "tipo", "estilo")
        ):
            raise ValueError(f"Inconsistent question: {q['id']}")
        seen.add(q["id"])
        value = answer(q, sources[q["grafico"]])
        if value != ref["resposta"]:
            raise ValueError(f"Answer disagrees with CSV: {q['id']}")
        if q["formato"] == "número":
            limits = sorted([value * 0.95, value * 1.05])
            if ref["correcao"] != "±5%" or not all(
                math.isclose(a, b) for a, b in zip(limits, [ref["min"], ref["max"]])
            ):
                raise ValueError(f"Incorrect numeric interval: {q['id']}")
        elif ref["correcao"] != "exata" or ref["min"] is not None or ref["max"] is not None:
            raise ValueError(f"Unexpected exact-match rule: {q['id']}")
        counts[q["tipo"]] += 1
    if (
        seen != set(labels)
        or len(sources) != 360
        or set(counts.values()) != {180}
        or len(counts) != 6
    ):
        raise ValueError("Incomplete confirmation coverage")
    report = dict(
        questions=len(seen),
        graphs=len(sources),
        by_type=counts,
        manifest_sha256=sha(folder / "complete.json"),
        auditor_sha256=sha(Path(__file__)),
    )
    write_once(folder / "csv_answer_audit.json", json_text(report))
    print(json_text(report))


if __name__ == "__main__":
    main()
