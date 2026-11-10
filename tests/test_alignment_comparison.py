import unittest
from src.evaluation.compare_alignment import contrast


class AlignmentComparisonTest(unittest.TestCase):
    def test_matches_by_id_not_order(self):
        a = [
            dict(id="a", grafico="g1", correct=False),
            dict(id="b", grafico="g2", correct=True),
        ]
        b = [
            dict(id="b", grafico="g2", correct=True),
            dict(id="a", grafico="g1", correct=True),
        ]
        result = contrast({"a": a, "b": b}, {"a": -1, "b": 1}, {"g1": "f1", "g2": "f2"})
        self.assertEqual(result["estimate"], 0.5)
        self.assertEqual(result["questions"], 2)

    def test_rejects_missing_and_duplicate(self):
        r = dict(id="a", grafico="g", correct=True)
        for bad in [[], [r, r]]:
            with self.assertRaises(ValueError):
                contrast({"a": [r], "b": bad}, {"a": -1, "b": 1}, {"g": "f"})

    def test_difference_in_gains(self):
        details = {
            k: [dict(id="a", grafico="g", correct=v)]
            for k, v in [("f0", False), ("f1", True), ("c0", True), ("c1", True)]
        }
        result = contrast(
            details, {"f1": 1, "f0": -1, "c1": -1, "c0": 1}, {"g": "family"}
        )
        self.assertEqual(result["estimate"], 1)
        self.assertIsNone(result["ci95"])


if __name__ == "__main__":
    unittest.main()
