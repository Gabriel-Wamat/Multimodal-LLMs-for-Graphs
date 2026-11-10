import unittest
from src.evaluation.paired.error_gallery import csv_preview, reading_record


class ReadingReviewTest(unittest.TestCase):
    def test_reference_html_is_escaped(self):
        rendered = csv_preview("name,value\n<script>alert(1)</script>,3\n")
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)

    def test_wrong_answer_does_not_fabricate_reading_error(self):
        row = dict(
            id="i",
            pair_id="p",
            tipo="BAR",
            attribute="height",
            intensity="subtle",
            pergunta="A > B?",
        )
        label = dict(answer="Sim", numeric_check={"compared_values": [2, 1]})
        predictions = [
            dict(
                condition=c,
                response="Não",
                calls=[dict(stage="transcribe", response="A=2, B=1")],
            )
            for c in ["original", "crops_2x2", "transcription"]
        ]
        result = reading_record(row, label, predictions)
        self.assertFalse(result["correct"]["transcription"])
        self.assertEqual(result["reading_error_review"]["status"], "unreviewed")
        self.assertEqual(result["reading_error_review"]["categories"], [])
        with self.assertRaises(ValueError):
            reading_record(row, label, predictions[:2])


if __name__ == "__main__":
    unittest.main()
