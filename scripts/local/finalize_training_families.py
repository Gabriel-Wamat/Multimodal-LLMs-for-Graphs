"""Preserve evaluation; quarantine connected lookalike families and freeze training inputs."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def once(p, obj):
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        assert p.read_text() == text, "Existing artifact differs"
    else:
        p.write_text(text)


def main():
    old_path = ROOT / "data/manifests/train_dev_v1/split.json"
    audit_path = (
        ROOT / "experiments/five-stage-campaign/data_audit_v3_all_families.json"
    )
    old = json.loads(old_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert audit["split_sha256"] == sha(old_path) and audit["source_integrity_pass"]
    names = set(old["train"] + old["development"] + old["evaluation_exploratory"])
    parent = {x: x for x in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for edge in audit["near_duplicate_candidates"]:
        a, b = find(edge["a"]), find(edge["b"])
        parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for x in sorted(names):
        groups[find(x)].append(x)
    evaluation = set(old["evaluation_exploratory"])
    old_dev = set(old["development"])
    train = []
    dev = []
    quarantined = []
    for members in groups.values():
        if evaluation.intersection(members):
            quarantined.extend(set(members) - evaluation)
        elif old_dev.intersection(members):
            dev.extend(members)
        else:
            train.extend(members)
    roles = {
        x: k
        for k, items in [
            ("train", train),
            ("development", dev),
            ("evaluation", evaluation),
            ("quarantine", quarantined),
        ]
        for x in items
    }
    for edge in audit["near_duplicate_candidates"]:
        a, b = roles[edge["a"]], roles[edge["b"]]
        assert a == b or "quarantine" in (a, b), (edge, a, b)
    assert len(roles) == len(names)
    record = {
        **old,
        "version": "train-dev-v2-family-screened",
        "train": sorted(train),
        "development": sorted(dev),
        "quarantined_near_evaluation": sorted(quarantined),
        "family_members": dict(groups),
        "family_rule": "connected components of exact/affine similarity screen across all split roles",
        "audit_sha256": sha(audit_path),
        "parent_split_sha256": sha(old_path),
        "eligible_for_training": True,
        "next_gate": "training loader, loss, gradients, checkpoint and telemetry tests",
        "limitation": "Exploratory comparison only. Heuristic family screen does not establish unknown upstream independence.",
    }
    out = ROOT / "data/manifests/train_dev_v2"
    once(out / "split.json", record)
    questions = json.loads(
        (ROOT / "dados_projeto_multi_modais/dados/gabarito.json").read_text()
    )
    images = {
        r["id"]: r
        for r in json.loads((ROOT / "data/manifests/imagens.json").read_text())[
            "images"
        ]
    }
    counts = {}
    for role, ids in [("train", train), ("development", dev)]:
        ids = set(ids)
        rows = []
        for q in questions:
            if q["grafico"] not in ids:
                continue
            image = images[q["grafico"]]
            rows.append(
                {
                    **q,
                    "image": image["png"],
                    "image_sha256": image["png_sha256"],
                    "family_id": find(q["grafico"]),
                    "split": role,
                }
            )
        p = out / (role + ".jsonl")
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if p.exists():
            assert p.read_text() == text
        else:
            p.write_text(text)
        counts[role] = {"graphs": len(ids), "questions": len(rows), "sha256": sha(p)}
    once(
        out / "manifest.json",
        {
            "split_sha256": sha(out / "split.json"),
            "counts": counts,
            "labels_sha256": sha(
                ROOT / "dados_projeto_multi_modais/dados/gabarito.json"
            ),
            "generator_sha256": sha(Path(__file__)),
        },
    )
    print(
        json.dumps(
            {
                "counts": counts,
                "quarantined": len(quarantined),
                "evaluation_graphs": len(evaluation),
            }
        )
    )


if __name__ == "__main__":
    main()
