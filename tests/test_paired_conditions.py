import tempfile
import unittest
from pathlib import Path
from PIL import Image
from src.evaluation.paired.conditions import (
    aggregate,
    binary,
    crop_images,
    evaluate_condition,
)


class ConditionsTests(unittest.TestCase):
    def test_strict_majority_and_invalid(self):
        self.assertEqual(aggregate(["Yes", "Sim.", "sim", "Não"]), "Sim")
        self.assertIsNone(aggregate(["Sim", "Sim", "Não", "Não"]))
        self.assertIsNone(aggregate(["Sim", "Sim", "talvez", ""]))
        self.assertIsNone(binary("Sim, porque a barra é maior"))

    def test_quadrants_preserve_odd_border_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "image.png"
            im = Image.new("RGB", (5, 7))
            im.putpixel((4, 6), (255, 0, 0))
            im.save(p)
            paths = crop_images(p, Path(d) / "cache")
            self.assertEqual(len(paths), 4)
            for f in paths:
                with Image.open(f) as image:
                    self.assertEqual(image.size, (5, 7))
            with Image.open(paths[-1]) as image:
                self.assertEqual(image.getpixel((4, 6)), (255, 0, 0))

    def test_transcription_is_visual_and_budget_restored(self):
        class Fake:
            cfg = {"max_new_tokens": 96}

            def answer(self, path, prompt, arm, image_id):
                return {
                    "response": "Eixo Y ilegível"
                    if self.cfg["max_new_tokens"] == 384
                    else "Sim",
                    "seconds": 1,
                    "input_tokens": 10,
                    "generated_tokens": 2,
                    "peak_gpu_bytes": 100,
                }

        model = Fake()
        r = evaluate_condition(
            model,
            {"id": "x", "pergunta": "A é maior?"},
            "image.png",
            "transcription",
            "cache",
        )
        self.assertEqual(len(r["calls"]), 2)
        self.assertIn("Eixo Y ilegível", r["calls"][1]["prompt"])
        self.assertNotIn("A é maior?", r["calls"][0]["prompt"])
        self.assertEqual(model.cfg["max_new_tokens"], 96)
        self.assertEqual(r["input_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
