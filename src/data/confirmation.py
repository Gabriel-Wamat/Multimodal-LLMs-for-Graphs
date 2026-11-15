"""Freeze a new synthetic evaluation sample before seeing checkpoint predictions."""

import csv
import fcntl
import io
import json
from pathlib import Path

import numpy as np

from src.data.pilot import ROOT, json_text, sha, write_once
from src.data.render import THEMES, TYPES, plt, render, render_spec_sha256

VERSION = "confirmation_v1"
SEED = 8037651


def generate(kind, rng):
    """Generate a fresh source and three chart-grounded questions with known answers."""
    n = int(rng.integers(4, 9))
    groups = [f"Grupo {chr(65 + i)}" for i in range(n)]
    values = rng.choice(np.arange(12, 95), n, replace=False)
    first, second = rng.choice(n, 2, replace=False)
    largest = groups[int(values.argmax())]
    higher = "Sim" if values[first] > values[second] else "Não"
    if kind in {"BAR", "PIZ", "HIS"}:
        if kind == "BAR":
            rows = [dict(categoria=g, valor=int(v)) for g, v in zip(groups, values)]
            questions = [
                ("Qual categoria tem a maior barra?", "categoria", largest),
                (
                    f"Qual o valor aproximado da barra {groups[first]}?",
                    "número",
                    int(values[first]),
                ),
                (
                    f"A barra {groups[first]} é maior que a barra {groups[second]}?",
                    "sim/não",
                    higher,
                ),
            ]
        elif kind == "PIZ":
            rows = [
                dict(categoria=g, percentual=100 * float(v) / sum(map(int, values)))
                for g, v in zip(groups, values)
            ]
            questions = [
                ("Qual categoria tem a maior fatia?", "categoria", largest),
                ("Quantas fatias há no gráfico?", "número inteiro", n),
                (
                    f"A fatia de {groups[first]} é maior que a de {groups[second]}?",
                    "sim/não",
                    higher,
                ),
            ]
        else:
            width = int(rng.integers(3, 12))
            rows = [
                dict(faixa_inicio=i * width, faixa_fim=(i + 1) * width, contagem=int(v))
                for i, v in enumerate(values)
            ]
            a, b = map(int, (first, second))
            questions = [
                ("Quantas faixas há no histograma?", "número inteiro", n),
                ("Qual a contagem aproximada na faixa mais alta?", "número", int(values.max())),
                (
                    f"A faixa {a * width} a {(a + 1) * width} tem mais observações que a faixa {b * width} a {(b + 1) * width}?",
                    "sim/não",
                    higher,
                ),
            ]
    elif kind == "LIN":
        year = int(rng.integers(1995, 2018))
        other = int(rng.choice([x for x in range(12, 95) if x != values[-1]]))
        ys = rng.integers(12, 95, n)
        ys[-1] = other
        rows = [
            dict(ano=year + i, Alfa=int(v), Beta=int(w)) for i, (v, w) in enumerate(zip(values, ys))
        ]
        questions = [
            ("Em que ano Alfa atinge o maior valor?", "ano", year + int(values.argmax())),
            (f"Qual o valor aproximado de Beta em {year + n - 1}?", "número", other),
            ("Alfa termina acima de Beta?", "sim/não", "Sim" if values[-1] > other else "Não"),
        ]
    elif kind == "BOX":
        rows = []
        for g, v in zip(groups, values):
            offsets = rng.uniform(1, 15, 12)
            for value in np.r_[v - offsets, v, v + offsets]:
                rows.append(dict(grupo=g, valor=float(value)))
        questions = [
            ("Qual grupo tem a maior mediana?", "categoria", largest),
            (f"Qual a mediana aproximada de {groups[first]}?", "número", int(values[first])),
            (f"A mediana de {groups[first]} supera a de {groups[second]}?", "sim/não", higher),
        ]
    else:
        count = int(rng.integers(10, 26))
        x = np.linspace(1, 100, count)
        slope = float(rng.uniform(0.4, 1.5)) * rng.choice([-1, 1])
        y = np.round(100 + slope * (x - 50) + rng.normal(0, 4, count), 2)
        rows = [dict(x=float(a), y=float(b)) for a, b in zip(x, y)]
        questions = [
            ("Quantos pontos há no gráfico?", "número inteiro", count),
            ("Qual o valor aproximado de Y do ponto mais à direita?", "número", float(y[-1])),
            (
                "A tendência linear entre X e Y é positiva?",
                "sim/não",
                "Sim" if np.corrcoef(x, y)[0, 1] > 0 else "Não",
            ),
        ]
    return rows, questions


def main():
    data = ROOT / "data/processed" / VERSION
    manifest = ROOT / "data/manifests" / VERSION
    data.mkdir(parents=True, exist_ok=True)
    manifest.mkdir(parents=True, exist_ok=True)
    lock = (manifest / "generator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = json.loads((ROOT / "data/manifests/imagens.json").read_text())["images"]
    seen_img = {r["png_sha256"] for r in prior}
    seen_csv = {r["csv_sha256"] for r in prior}
    paired_inputs = ROOT / "data/manifests/pairs_v1/inputs.jsonl"
    seen_img.update(
        json.loads(line)["image_sha256"] for line in paired_inputs.read_text().splitlines()
    )
    write_once(
        manifest / "protocol.json",
        json_text(
            dict(
                seed=SEED,
                generator_sha256=sha(Path(__file__)),
                render_spec_sha256=render_spec_sha256(),
                per_kind_style=10,
                purpose="evaluation only",
                independence_unit="new source chart",
            )
        ),
    )
    inputs, labels, source_rows = [], [], []
    for k, kind in enumerate(sorted(TYPES)):
        for s, style in enumerate(sorted(THEMES)):
            for i in range(10):
                rng = np.random.default_rng(np.random.SeedSequence([SEED, k, s, i]))
                rows, questions = generate(kind, rng)
                graph = f"{kind}-CONF-{s}-{i}"
                source = data / (graph + ".csv")
                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
                write_once(source, buffer.getvalue())
                image = source.with_suffix(".png")
                fig = render(source, style)
                tmp = image.with_suffix(".partial.png")
                fig.savefig(tmp)
                plt.close(fig)
                if image.exists() and sha(tmp) != sha(image):
                    raise ValueError(f"Confirmation image changed: {graph}")
                tmp.replace(image)
                ih, ch = sha(image), sha(source)
                if ih in seen_img or ch in seen_csv:
                    raise ValueError(f"Repeated chart: {graph}")
                seen_img.add(ih)
                seen_csv.add(ch)
                source_rows.append(
                    dict(
                        grafico=graph,
                        csv=str(source.relative_to(ROOT)),
                        csv_sha256=ch,
                        image_sha256=ih,
                    )
                )
                for j, (question, fmt, target) in enumerate(questions, 1):
                    common = dict(
                        id=f"{graph}-Q{j}",
                        grafico=graph,
                        tipo=TYPES[kind],
                        estilo=style,
                        nivel="confirmation",
                        formato=fmt,
                        pergunta=question,
                    )
                    inputs.append(
                        dict(**common, image=str(image.relative_to(ROOT)), image_sha256=ih)
                    )
                    numeric = fmt == "número"
                    labels.append(
                        dict(
                            **common,
                            resposta=target,
                            correcao="±5%" if numeric else "exata",
                            min=float(target) * 0.95 if numeric else None,
                            max=float(target) * 1.05 if numeric else None,
                        )
                    )
    for name, rows in [("inputs", inputs), ("labels", labels), ("sources", source_rows)]:
        write_once(
            manifest / (name + ".jsonl"),
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )
    record = {
        name + "_sha256": sha(manifest / (name + ".jsonl"))
        for name in ("inputs", "labels", "sources")
    }
    write_once(
        manifest / "complete.json",
        json_text(
            dict(
                **record,
                graphs=len(source_rows),
                questions=len(inputs),
                protocol_sha256=sha(manifest / "protocol.json"),
            )
        ),
    )


if __name__ == "__main__":
    main()
