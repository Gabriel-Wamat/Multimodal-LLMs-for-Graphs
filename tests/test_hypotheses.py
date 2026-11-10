import unittest
from src.evaluation.paired.hypotheses import (
    rank_correlation,
    similarity_error,
    intensity_effect,
)


class HypothesesTest(unittest.TestCase):
    def test_scipy_documented_ties_example_and_constant(self):
        self.assertAlmostEqual(
            rank_correlation([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]), 0.8207826816681233
        )
        self.assertIsNone(rank_correlation([1, 2, 3], [0, 0, 0]))

    def test_degenerate_family_is_not_false_precision(self):
        rows = [dict(family="shared", clip=i, correct=i > 1) for i in range(4)]
        result = similarity_error(rows, "clip", repetitions=100)
        self.assertIsNone(result["ci95"])
        self.assertLess(result["rho"], 0)

    def test_intensity_matching_and_sign(self):
        base = dict(
            origin="g", family="f", attribute="height", tipo="BAR", estilo="classic"
        )
        rows = [
            dict(base, intensity="subtle", correct=False),
            dict(base, intensity="large", correct=True),
        ]
        self.assertEqual(intensity_effect(rows)["estimate"], 1)
        with self.assertRaises(ValueError):
            intensity_effect(rows[:1])
        with self.assertRaises(ValueError):
            intensity_effect(rows + [rows[0]])


if __name__ == "__main__":
    unittest.main()
