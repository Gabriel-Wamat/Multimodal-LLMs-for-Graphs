"""Exploratory H2 and similarity/error estimates with whole-family uncertainty."""

from collections import defaultdict
import numpy as np
from scipy.stats import rankdata

from .metrics import family_bootstrap


def rank_correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 1
        or x.shape != y.shape
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Invalid paired observations")
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(
        np.corrcoef(rankdata(x, method="average"), rankdata(y, method="average"))[0, 1]
    )


def similarity_error(rows, encoder, repetitions=2000, seed=20261102):
    if encoder not in {"clip", "dinov3"} or not rows:
        raise ValueError("Missing observations or unknown encoder")
    x = np.array([r[encoder] for r in rows])
    y = np.array([1 - int(r["correct"]) for r in rows])
    estimate = rank_correlation(x, y)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["family"]].append(i)
    groups = [np.array(indices) for _, indices in sorted(groups.items())]
    result = {
        "rho": estimate,
        "pairs": len(rows),
        "families": len(groups),
        "outcome": "pair error (one or both images incorrect)",
        "resampling_unit": "whole source family",
        "ci95": None,
        "bootstrap_requested": repetitions,
        "bootstrap_valid": 0,
        "p_value": None,
        "exploratory": True,
    }
    if estimate is None:
        result["reason"] = "constant variable or fewer than two observations"
        return result
    if len(groups) < 2:
        result["reason"] = "fewer than two independent families"
        return result
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        indices = np.concatenate(
            [groups[i] for i in rng.integers(len(groups), size=len(groups))]
        )
        value = rank_correlation(x[indices], y[indices])
        if value is not None:
            values.append(value)
    result["bootstrap_valid"] = len(values)
    # Do not hide degeneracy behind a conditional interval from few resamples.
    if len(values) >= 0.95 * repetitions:
        result["ci95"] = np.quantile(values, [0.025, 0.975]).tolist()
    else:
        result["reason"] = "more than 5% of family resamples have undefined correlation"
    return result


def intensity_effect(rows):
    """H2: large-minus-subtle paired accuracy, matched on the same chart origin."""
    origins = defaultdict(dict)
    for r in rows:
        intensity = r["intensity"]
        if intensity not in {"subtle", "large"} or intensity in origins[r["origin"]]:
            raise ValueError("Unknown or duplicate intensity for origin")
        origins[r["origin"]][intensity] = r
    delta = []
    for origin, pair in sorted(origins.items()):
        if set(pair) != {"subtle", "large"}:
            raise ValueError("Unmatched intensities")
        a, b = pair["subtle"], pair["large"]
        if any(a[k] != b[k] for k in ["family", "attribute", "tipo", "estilo"]):
            raise ValueError("Intensity metadata mismatch")
        delta.append(dict(a, delta=int(b["correct"]) - int(a["correct"])))
    result = family_bootstrap(delta, "delta")
    result["contrast"] = "large minus subtle paired accuracy"
    result["origins"] = len(delta)
    return result


def analyze(pairs, scores):
    """Join exact coverage; call separately for each model and inference condition."""
    representations = {
        r["pair_id"]: r for r in scores if r["split"] == "evaluation_exploratory"
    }
    if len({r["pair_id"] for r in pairs}) != len(pairs) or set(representations) != {
        r["pair_id"] for r in pairs
    }:
        raise ValueError("Similarity/outcome coverage differs")
    if len(representations) != sum(
        r["split"] == "evaluation_exploratory" for r in scores
    ):
        raise ValueError("Duplicate representations")
    joined = []
    for p in pairs:
        s = representations[p["pair_id"]]
        if any(
            p[k] != s[k] for k in ["origin", "attribute", "intensity", "tipo", "estilo"]
        ):
            raise ValueError("Similarity/outcome metadata differs")
        joined.append(dict(p, clip=s["clip"], dinov3=s["dinov3"]))
    strata = {"all": joined}
    for field in ["attribute", "intensity", "estilo"]:
        for value in sorted({r[field] for r in joined}):
            strata[field + "/" + value] = [r for r in joined if r[field] == value]
    return {
        "H2": {
            "all": intensity_effect(joined),
            "by_attribute": {
                value: intensity_effect([r for r in joined if r["attribute"] == value])
                for value in sorted({r["attribute"] for r in joined})
            },
        },
        "similarity_error": {
            name: {
                encoder: similarity_error(rs, encoder) for encoder in ["clip", "dinov3"]
            }
            for name, rs in strata.items()
        },
        "interpretation": "Exploratory effects and family bootstrap intervals; no automatic hypothesis acceptance, causal claim or asymptotic Spearman p-value.",
    }
