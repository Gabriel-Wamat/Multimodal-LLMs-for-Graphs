import unittest

from src.evaluation.paired.similarity import calibrate, quadrant, summarize


def row(pid, split, clip, dino):
    return dict(
        pair_id=pid,
        split=split,
        clip=clip,
        dinov3=dino,
        attribute="height",
        intensity="subtle",
        estilo="classic",
    )


class SimilarityTest(unittest.TestCase):
    def test_evaluation_cannot_change_calibration(self):
        dev = [row("d1", "development", 0.2, 0.4), row("d2", "development", 0.6, 0.8)]
        low = dev + [row("e1", "evaluation_exploratory", -1, -1)]
        high = dev + [row("e1", "evaluation_exploratory", 1, 1)]
        self.assertEqual(calibrate(low), calibrate(high))
        self.assertAlmostEqual(calibrate(dev)["quantiles"]["0.75"]["clip"], 0.5)
        self.assertEqual(summarize(high, calibrate(dev))["pairs"], 1)
        self.assertTrue(summarize(low, calibrate(dev))["all_pairs_retained"])

    def test_no_development_no_fallback(self):
        with self.assertRaises(ValueError):
            calibrate([row("e", "evaluation_exploratory", 1, 1)])

    def test_boundary_and_invalid_inputs(self):
        self.assertEqual(
            quadrant(row("a", "development", 0.5, 0.6), {"clip": 0.5, "dinov3": 0.6}),
            "high_both",
        )
        with self.assertRaises(ValueError):
            calibrate([row("a", "development", float("nan"), 0.6)])
        r = row("a", "development", 0.5, 0.6)
        with self.assertRaises(ValueError):
            calibrate([r, r])


if __name__ == "__main__":
    unittest.main()
