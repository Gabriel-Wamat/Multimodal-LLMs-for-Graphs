"""Render verified, complete representation results without changing calibration."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    folder = ROOT / "results/representations-pairs_v1-v1"
    complete = json.loads((folder / "complete.json").read_text())
    for name in ("scores", "run"):
        if sha(folder / f"{name}.json") != complete[f"{name}_sha256"]:
            raise ValueError(f"Changed {name}")
    scores = json.loads((folder / "scores.json").read_text())
    calibration = json.loads((folder / "calibration.json").read_text())
    report = json.loads((folder / "similarity_report.json").read_text())
    for artifact in (calibration, report):
        if artifact["scores_sha256"] != complete["scores_sha256"]:
            raise ValueError("Calibration/report lineage mismatch")
    if len(scores) != complete["pairs"] or len({r["pair_id"] for r in scores}) != len(scores):
        raise ValueError("Incomplete or duplicate pairs")
    rows = [r for r in scores if r["split"] == "evaluation_exploratory"]
    if len(rows) != report["pairs"] or len(rows) != 720:
        raise ValueError("Expected all 720 evaluation pairs")
    out = folder / "plots"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
    for intensity, color, label in [("subtle", "#0072B2", "Sutil"),
                                    ("large", "#D55E00", "Ampla")]:
        subset = [r for r in rows if r["intensity"] == intensity]
        axes[0].scatter([r["clip"] for r in subset], [r["dinov3"] for r in subset],
                        s=16, alpha=.55, color=color, label=f"{label} (n={len(subset)})")
    thresholds = calibration["quantiles"]["0.75"]
    axes[0].axvline(thresholds["clip"], color="#555555", ls="--", lw=1)
    axes[0].axhline(thresholds["dinov3"], color="#555555", ls="--", lw=1)
    axes[0].set(xlabel="Cosseno CLIP", ylabel="Cosseno DINOv3",
                title="Similaridade por par — eixos ampliados")
    axes[0].ticklabel_format(useOffset=False)
    axes[0].legend(loc="lower left")
    quadrants = [("high_both", "Alta nos dois", "#009E73"),
                 ("high_clip_only", "Alta só CLIP", "#0072B2"),
                 ("high_dinov3_only", "Alta só DINOv3", "#E69F00"),
                 ("low_both", "Baixa nos dois", "#999999")]
    quantiles = ["0.5", "0.75", "0.9"]
    bottom = np.zeros(3)
    for key, label, color in quadrants:
        counts = np.array([report["sensitivity"][q]["counts"].get(key, 0) for q in quantiles])
        axes[1].bar(quantiles, counts, bottom=bottom, color=color, label=label)
        for i, n in enumerate(counts):
            if n >= 40:
                axes[1].text(i, bottom[i] + n / 2, str(n), ha="center", va="center", fontsize=9)
        bottom += counts
    if not np.all(bottom == len(rows)):
        raise ValueError("Quadrants do not cover all evaluation pairs")
    axes[1].set(xlabel="Quantil de desenvolvimento usado como limiar",
                ylabel="Número de pares", title="Sensibilidade dos grupos aos limiares")
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=2)
    fig.suptitle("CLIP + DINOv3: 720 pares de avaliação; limiares em 142 pares de desenvolvimento\n"
                 "Alta/baixa são classes relativas; nenhum par foi removido", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"similarity.{suffix}", dpi=180)
    plt.close(fig)
    manifest = {"scores_sha256": complete["scores_sha256"],
                "calibration_sha256": sha(folder / "calibration.json"),
                "report_sha256": sha(folder / "similarity_report.json"),
                "source_sha256": sha(Path(__file__)), "evaluation_pairs": len(rows),
                "files": {p.name: sha(p) for p in sorted(out.glob("similarity.*"))},
                "limitation": "Descriptive representations only; no model-error association or H1 conclusion."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
