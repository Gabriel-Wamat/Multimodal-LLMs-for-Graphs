"""The stability report must compare the same experiment across unique seeds."""

import unittest

from src.evaluation.compare_full_sft import ARMS
from src.evaluation.seed_stability import SEEDS, summarize


class StabilityTest(unittest.TestCase):
    def reports(self):
        return [
            dict(
                seed=s,
                adapter_seed=s,
                model="qwen",
                dataset="pilot50_v1",
                prefix=str(s),
                analysis_sha256="a",
                matched_protocol_sha256="p",
                arms={arm: {"primary": {"accuracy": 0.15}} for arm in ARMS},
                comparisons={
                    arm: {"fusion_minus_control_by_family": {"percentage_points": d}}
                    for arm in ARMS[:2]
                },
            )
            for s, d in zip(SEEDS, (1.0, 4.0, 7.0))
        ]

    def test_paired_seed_spread(self):
        result = summarize(self.reports()[::-1])
        self.assertEqual(result["fusion_minus_control_pp"]["clip_double"]["mean"], 4.0)
        self.assertEqual(result["fusion_minus_control_pp"]["clip_double"]["sample_sd"], 3.0)

    def test_drift_and_duplicate_seed(self):
        rows = self.reports()
        rows[1]["matched_protocol_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "differ"):
            summarize(rows)
        with self.assertRaisesRegex(ValueError, "each registered seed"):
            summarize([rows[0], rows[0], rows[2]])
