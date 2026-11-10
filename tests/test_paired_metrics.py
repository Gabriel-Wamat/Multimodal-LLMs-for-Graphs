import unittest
from src.evaluation.paired.metrics import (
    exact_mcnemar,
    family_bootstrap,
    score_pairs,
    compare,
)


class PairedMetricsTests(unittest.TestCase):
    def test_pair_requires_two_hits_and_rejects_missing(self):
        rows = [
            {
                "id": m,
                "pair_id": "p",
                "member": m,
                "origin": "g",
                "attribute": "height",
                "intensity": "subtle",
                "tipo": "BAR",
                "estilo": "plain",
                "split": "test",
                "pergunta": "maior?",
            }
            for m in "AB"
        ]
        labels = [{"id": "A", "answer": "Sim"}, {"id": "B", "answer": "Não"}]
        predictions = [{"id": m, "response": "Sim", "status": "ok"} for m in "AB"]
        result = score_pairs(rows, labels, predictions, {"g": "family"})
        self.assertFalse(result[0]["correct"])
        self.assertEqual(result[0]["question_hits"], 1)
        predictions[1]["response"] = "Não"
        self.assertTrue(
            score_pairs(rows, labels, predictions, {"g": "family"})[0]["correct"]
        )
        with self.assertRaises(ValueError):
            score_pairs(rows, labels, predictions[:1], {"g": "family"})

    def test_resampling_keeps_correlated_pairs_together(self):
        r = family_bootstrap(
            [{"family": "f", "correct": True}, {"family": "f", "correct": False}]
        )
        self.assertEqual(r["ci95"], [0.5, 0.5])
        self.assertEqual(r["families"], 1)

    def test_exact_discordance(self):
        r = exact_mcnemar([False] * 6, [True] * 6)
        self.assertEqual(r["p_exact_two_sided"], 0.03125)
        self.assertEqual(exact_mcnemar([True], [True])["p_exact_two_sided"], 1)

    def test_mcnemar_uses_one_pair_per_family(self):
        a = [
            {
                "pair_id": str(i),
                "family": "shared",
                "attribute": "height",
                "intensity": "subtle",
                "correct": False,
            }
            for i in range(5)
        ]
        b = [dict(r, correct=True) for r in a]
        result = compare(a, b)
        self.assertEqual(
            result["mcnemar_by_attribute_intensity"]["height/subtle"]["n"], 1
        )
        self.assertEqual(result["paired_delta"]["estimate"], 1)


if __name__ == "__main__":
    unittest.main()
