"""Both images must be correct; uncertainty resamples whole source families."""

import hashlib
import math
from collections import defaultdict
import numpy as np
from .conditions import binary


def score_pairs(inputs, labels, predictions, families):
    truth = {r["id"]: r for r in labels}
    pred = {r["id"]: r for r in predictions}
    if len(pred) != len(predictions) or set(pred) != {r["id"] for r in inputs}:
        raise ValueError("Missing, duplicate or unexpected predictions")
    grouped = defaultdict(list)
    for row in inputs:
        p = pred[row["id"]]
        if p["status"] != "ok":
            raise ValueError("Failed response")
        parsed = binary(p["response"])
        correct = parsed == truth[row["id"]]["answer"]
        grouped[row["pair_id"]].append({**row, "correct": correct, "parsed": parsed})
    pairs = []
    for pair_id, rows in sorted(grouped.items()):
        if len(rows) != 2 or {r["member"] for r in rows} != {"A", "B"}:
            raise ValueError("Incomplete pair")
        a, b = rows
        for field in [
            "origin",
            "attribute",
            "intensity",
            "tipo",
            "estilo",
            "split",
            "pergunta",
        ]:
            if a[field] != b[field]:
                raise ValueError("Pair metadata differs")
        if truth[a["id"]]["answer"] == truth[b["id"]]["answer"]:
            raise ValueError("Pair answers must differ")
        pairs.append(
            {
                "pair_id": pair_id,
                "origin": a["origin"],
                "family": families[a["origin"]],
                "attribute": a["attribute"],
                "intensity": a["intensity"],
                "tipo": a["tipo"],
                "estilo": a["estilo"],
                "correct": all(r["correct"] for r in rows),
                "question_hits": sum(r["correct"] for r in rows),
                "invalid_responses": sum(r["parsed"] is None for r in rows),
            }
        )
    return pairs


def family_bootstrap(rows, field="correct", seed=20261102, repetitions=2000):
    if not rows:
        raise ValueError("Empty sample")
    groups = defaultdict(list)
    for r in rows:
        groups[r["family"]].append(float(r[field]))
    values = np.array([(sum(v), len(v)) for v in groups.values()])
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(repetitions, len(values)))]
    means = samples[:, :, 0].sum(1) / samples[:, :, 1].sum(1)
    return {
        "estimate": sum(float(r[field]) for r in rows) / len(rows),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "families": len(groups),
        "pairs": len(rows),
        "unit": "source similarity family",
        "exploratory": True,
    }


def exact_mcnemar(before, after):
    if len(before) != len(after):
        raise ValueError("Unpaired arrays")
    lost = sum(bool(a) and not bool(b) for a, b in zip(before, after))
    gained = sum(not bool(a) and bool(b) for a, b in zip(before, after))
    n = lost + gained
    p = (
        1.0
        if not n
        else min(
            1.0, 2 * sum(math.comb(n, k) for k in range(min(lost, gained) + 1)) / 2**n
        )
    )
    return {
        "lost": lost,
        "gained": gained,
        "discordant": n,
        "p_exact_two_sided": p,
        "n": len(before),
    }


def compare(before, after):
    b = {r["pair_id"]: r for r in before}
    a = {r["pair_id"]: r for r in after}
    if len(b) != len(before) or len(a) != len(after) or set(a) != set(b):
        raise ValueError("Pair sets differ")
    delta = []
    strata = defaultdict(dict)
    for pid, r in b.items():
        s = a[pid]
        if any(r[k] != s[k] for k in ["family", "attribute", "intensity"]):
            raise ValueError("Metadata differs")
        delta.append({**r, "delta": int(s["correct"]) - int(r["correct"])})
        key = (r["attribute"], r["intensity"])
        family = r["family"]
        current = strata[key].get(family)
        if (
            current is None
            or hashlib.sha256(pid.encode()).digest()
            < hashlib.sha256(current.encode()).digest()
        ):
            strata[key][family] = pid
    tests = {}
    for key, representatives in strata.items():
        ids = sorted(representatives.values())
        tests["/".join(key)] = {
            **exact_mcnemar(
                [b[i]["correct"] for i in ids], [a[i]["correct"] for i in ids]
            ),
            "pair_ids": ids,
            "selection": "one pair per family, smallest SHA256(pair_id), independent of responses",
        }
    return {
        "paired_delta": family_bootstrap(delta, "delta"),
        "mcnemar_by_attribute_intensity": tests,
        "multiplicity": "p values unadjusted; exploratory, report all comparisons",
    }
