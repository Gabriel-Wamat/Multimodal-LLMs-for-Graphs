"""Exercise the public comparison path, including reordered questions and drift."""

import json
from pathlib import Path
import tempfile
import unittest

from src.data.pilot import sha
from src.evaluation.compare_full_sft import ARMS, compare_model
from src.evaluation.score import evaluate


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class PilotComparisonTest(unittest.TestCase):
    def test_matched_questions_and_incompatible_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "data/manifests/pilot50_v1"
            base.mkdir(parents=True)
            labels = [
                dict(
                    id=g,
                    grafico=g,
                    tipo="BAR",
                    estilo="A",
                    nivel="1",
                    formato="ano",
                    correcao="exata",
                    resposta="2020",
                    min=None,
                    max=None,
                )
                for g in ["x", "y"]
            ]
            for name, rows in [("inputs", [{"id": r["id"]} for r in labels]), ("labels", labels)]:
                (base / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            split = {f"{n}_sha256": sha(base / f"{n}.jsonl") for n in ("inputs", "labels")}
            save(base / "split.json", split)
            save(
                root / "data/manifests/train_dev_v2/split.json",
                {"family_members": {"X": ["x"], "Y": ["y"]}},
            )
            for arm in ARMS:
                run = f"test-qwen-{arm}"
                train = root / "experiments/full_sft" / run
                result = root / "results" / (run + "-pilot")
                record = dict(
                    model="qwen",
                    arm=arm,
                    config={"seed": 42},
                    data_sha256={},
                    weights={},
                    base_config={},
                    source_sha256={},
                )
                save(train / "run.json", record)
                save(train / "complete.json", {"stage2_epochs": 3})
                save(train / "best.json", {"sha256": "abc"})
                metadata = dict(
                    model="qwen",
                    arm=arm,
                    training_kind="full_sft",
                    training_run=run,
                    scorers={},
                    source_sha256="xyz",
                    checkpoint_sha256="abc",
                    selected_epoch=1,
                    training_record_sha256=sha(train / "run.json"),
                    training_complete_sha256=sha(train / "complete.json"),
                    **split,
                )
                save(result / "run.json", metadata)
                preds = [
                    dict(
                        id=r["id"],
                        response="2020" if arm == "clip_dino" else "invalid",
                        status="ok",
                        seconds=2,
                        peak_gpu_bytes=100,
                        generated_tokens=1,
                    )
                    for r in reversed(labels)
                ]
                (result / "predictions.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in preds)
                )
                metrics, _ = evaluate(labels, preds)
                save(result / "metrics.json", {"primary_binary_aliases_v2": metrics})
                save(
                    result / "complete.json",
                    dict(
                        predictions_sha256=sha(result / "predictions.jsonl"),
                        metrics_sha256=sha(result / "metrics.json"),
                    ),
                )
            report = compare_model(root, "test", "qwen")
            self.assertEqual(
                report["comparisons"]["clip_double"]["fusion_minus_control_by_family"][
                    "percentage_points"
                ],
                100,
            )
            self.assertEqual(report["arms"]["clip_single"]["error_counts"], {"invalid_format": 2})
            record["config"]["seed"] = 7
            save(train / "run.json", record)
            metadata["training_record_sha256"] = sha(train / "run.json")
            save(result / "run.json", metadata)
            with self.assertRaisesRegex(ValueError, "differ"):
                compare_model(root, "test", "qwen")
