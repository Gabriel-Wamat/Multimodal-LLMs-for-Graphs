"""Generate controlled A/B chart pairs without consulting encoders or VLM outputs."""

import argparse
import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.data.render import ROOT, render, plt, render_spec_sha256


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def closest(values, adjacent=False):
    return min(
        (
            (i, j)
            for i in range(len(values))
            for j in range(i + 1, len(values))
            if not adjacent or j == i + 1
        ),
        key=lambda ij: (abs(values[ij[0]] - values[ij[1]]), ij),
    )


def make_pair(kind, rows, intensity, positive_a):
    variants = []
    facts = []
    fraction = {"subtle": 0.02, "large": 0.2}[intensity]
    if kind == "DIS":
        x = np.array([float(r["x"]) for r in rows])
        y = np.array([float(r["y"]) for r in rows])
        z = (x - x.mean()) / x.std()
        res = y - y.mean()
        res -= np.dot(res, z) / np.dot(z, z) * z
        if res.std() < 1e-8:
            raise ValueError("Degenerate scatter residual")
        res /= res.std()
        rho = 0.1 if intensity == "subtle" else 0.8
        question = "A tendência linear entre X e Y é positiva?"
        for sign in [1, -1] if positive_a else [-1, 1]:
            out = copy.deepcopy(rows)
            new = y.mean() + y.std() * (sign * rho * z + math.sqrt(1 - rho * rho) * res)
            for r, v in zip(out, new):
                r["y"] = str(float(v))
            measured = float(np.corrcoef(x, new)[0, 1])
            assert abs(measured - sign * rho) < 1e-8
            variants.append(out)
            facts.append({"correlation": measured, "positive": measured > 0})
        return variants, facts, question, "trend_sign"
    if kind == "LIN":
        keys = [k for k in rows[0] if k != "ano"]
        values = [float(rows[-1][k]) for k in keys]
    elif kind == "BOX":
        keys = list(dict.fromkeys(r["grupo"] for r in rows))
        values = [
            float(np.median([float(r["valor"]) for r in rows if r["grupo"] == k]))
            for k in keys
        ]
    else:
        field = {"BAR": "valor", "PIZ": "percentual", "HIS": "contagem"}[kind]
        values = [float(r[field]) for r in rows]
        keys = list(range(len(rows)))
    i, j = closest(values, kind == "HIS")
    mid = (values[i] + values[j]) / 2
    scale = max(max(values) - min(values), max(abs(v) for v in values), 1.0)
    delta = fraction * scale
    if kind == "PIZ":
        delta = min(fraction * 100, 0.8 * (values[i] + values[j]))
    if kind in ("BAR", "HIS"):
        mid = max(mid, delta / 2 + 1)
    if kind == "HIS":
        delta = max(2, 2 * math.ceil(delta / 2))
        mid = round(max(mid, delta / 2 + 1))
    if kind == "BAR":
        question = f"A barra {rows[i]['categoria']} é mais alta que a barra {rows[j]['categoria']}?"
    elif kind == "PIZ":
        question = f"A proporção de {rows[i]['categoria']} é maior que a de {rows[j]['categoria']}?"
    elif kind == "HIS":
        question = f"A contagem na faixa {rows[i]['faixa_inicio']} a {rows[i]['faixa_fim']} é maior que na faixa {rows[j]['faixa_inicio']} a {rows[j]['faixa_fim']}?"
    elif kind == "LIN":
        question = f"No ano {rows[-1]['ano']}, {keys[i]} tem valor maior que {keys[j]}?"
    else:
        question = f"A mediana de {keys[i]} é maior que a mediana de {keys[j]}?"
    for sign in [1, -1] if positive_a else [-1, 1]:
        out = copy.deepcopy(rows)
        a, b = mid + sign * delta / 2, mid - sign * delta / 2
        if kind == "LIN":
            out[-1][keys[i]] = str(a)
            out[-1][keys[j]] = str(b)
        elif kind == "BOX":
            for r in out:
                if r["grupo"] == keys[i]:
                    r["valor"] = str(float(r["valor"]) + a - values[i])
                elif r["grupo"] == keys[j]:
                    r["valor"] = str(float(r["valor"]) + b - values[j])
        else:
            out[i][field] = str(int(a) if kind == "HIS" else a)
            out[j][field] = str(int(b) if kind == "HIS" else b)
        if kind == "LIN":
            actual = [float(out[-1][keys[k]]) for k in (i, j)]
        elif kind == "BOX":
            actual = [
                float(
                    np.median([float(r["valor"]) for r in out if r["grupo"] == keys[k]])
                )
                for k in (i, j)
            ]
        else:
            actual = [float(out[k][field]) for k in (i, j)]
        assert (actual[0] > actual[1]) == (sign > 0)
        if kind == "PIZ":
            assert abs(sum(float(r[field]) for r in out) - sum(values)) < 1e-8
        variants.append(out)
        facts.append(
            {
                "compared_values": actual,
                "positive": actual[0] > actual[1],
                "actual_difference": abs(actual[0] - actual[1]),
                "scale": scale,
            }
        )
    return (
        variants,
        facts,
        question,
        {
            "BAR": "bar_height",
            "LIN": "last_year_order",
            "HIS": "bin_count",
            "PIZ": "slice_proportion",
            "BOX": "median_order",
        }[kind],
    )


def write_once(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_bytes().decode() == text, str(p)
    else:
        p.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    version = "pairs_v1_smoke" if args.smoke else "pairs_v1"
    split_path = ROOT / "data/manifests/train_dev_v2/split.json"
    split = json.loads(split_path.read_text())
    assert split["eligible_for_training"]
    roles = {
        g: k
        for k in ["train", "development", "evaluation_exploratory"]
        for g in split[k]
    }
    groups = defaultdict(list)
    for g in sorted(roles):
        groups[(roles[g], g[:3])].append(g)
    signs = {g: bool(i % 2) for gs in groups.values() for i, g in enumerate(gs)}
    selected = sorted(roles)
    if args.smoke:
        selected = [
            next(g for g in selected if g.startswith(k))
            for k in ["BAR", "LIN", "HIS", "PIZ", "BOX", "DIS"]
        ]
    folder = ROOT / "data/processed" / version
    manifest = ROOT / "data/manifests" / version
    manifest.mkdir(parents=True, exist_ok=True)
    stamp = {
        "generator_sha256": digest(Path(__file__)),
        "renderer_spec_sha256": render_spec_sha256(),
        "split_sha256": digest(split_path),
        "smoke": args.smoke,
        "originals": selected,
        "review_status": "pending_visual_review",
    }
    write_once(manifest / "generation.json", json.dumps(stamp, indent=2) + "\n")
    params = json.loads(
        (ROOT / "dados_projeto_multi_modais/dados/parametros.json").read_text()
    )
    inputs = []
    labels = []
    for graph in selected:
        source = ROOT / "dados_projeto_multi_modais/dados" / (graph + ".csv")
        assert digest(source) == split["source_sha256"][graph]
        with source.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for intensity in ["subtle", "large"]:
            pair_id = f"{graph}-{intensity}"
            variants, facts, question, attribute = make_pair(
                graph[:3], rows, intensity, signs[graph]
            )
            figures = []
            paths = []
            for member, derived in zip("AB", variants):
                path = folder / f"{pair_id}-{member}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                import io

                buffer = io.StringIO()
                writer = csv.DictWriter(buffer, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(derived)
                write_once(path, buffer.getvalue())
                figures.append(render(path, params[graph]["estilo"]))
                paths.append(path)
            if graph[:3] != "PIZ":
                lo = min(f.axes[0].get_ylim()[0] for f in figures)
                hi = max(f.axes[0].get_ylim()[1] for f in figures)
                for f in figures:
                    f.axes[0].set_ylim(lo, hi)
            hashes = []
            for member, f, path, fact in zip("AB", figures, paths, facts):
                png = path.with_suffix(".png")
                f.savefig(png)
                plt.close(f)
                h = digest(png)
                hashes.append(h)
                item = {
                    "id": f"{pair_id}-{member}",
                    "pair_id": pair_id,
                    "origin": graph,
                    "split": roles[graph],
                    "tipo": graph[:3],
                    "estilo": params[graph]["estilo"],
                    "attribute": attribute,
                    "intensity": intensity,
                    "member": member,
                    "pergunta": question,
                    "options": ["Sim", "Não"],
                    "formato": "sim/não",
                    "image": str(png.relative_to(ROOT)),
                    "image_sha256": h,
                }
                inputs.append(item)
                labels.append(
                    {
                        "id": item["id"],
                        "pair_id": pair_id,
                        "answer": "Sim" if fact["positive"] else "Não",
                        "numeric_check": fact,
                        "derived_csv": str(path.relative_to(ROOT)),
                        "csv_sha256": digest(path),
                    }
                )
            assert hashes[0] != hashes[1], pair_id
        print(graph, len(inputs), flush=True)
    for name, rows_out in [("inputs", inputs), ("labels", labels)]:
        write_once(
            manifest / (name + ".jsonl"),
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_out),
        )
    write_once(
        manifest / "complete.json",
        json.dumps(
            {
                "images": len(inputs),
                "pairs": len(inputs) // 2,
                "inputs_sha256": digest(manifest / "inputs.jsonl"),
                "labels_sha256": digest(manifest / "labels.jsonl"),
                "visual_review": "pending",
            },
            indent=2,
        )
        + "\n",
    )


if __name__ == "__main__":
    main()
