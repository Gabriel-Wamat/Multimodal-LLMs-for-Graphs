import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch profile required")
class AlignmentMaskTests(unittest.TestCase):
    def test_only_answer_is_supervised(self):
        import torch
        from src.training.alignment import supervised_inputs

        source = {
            "input_ids": torch.tensor([[1, 99, 99, 2]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }
        out = supervised_inputs(source, [7, 8, 3])
        self.assertEqual(out["labels"].tolist(), [[-100, -100, -100, -100, 7, 8, 3]])
        self.assertEqual(out["input_ids"].tolist(), [[1, 99, 99, 2, 7, 8, 3]])
        self.assertEqual(source["input_ids"].shape[1], 4)

    def test_empty_targets_rejected(self):
        import torch
        from src.training.alignment import supervised_inputs

        with self.assertRaises(ValueError):
            supervised_inputs({"input_ids": torch.ones(1, 2, dtype=torch.long)}, [])


if __name__ == "__main__":
    unittest.main()
