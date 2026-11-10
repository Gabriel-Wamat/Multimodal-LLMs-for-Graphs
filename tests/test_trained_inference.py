"""Generation instrumentation must preserve the inference method on success/failure."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch
try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.evaluation.trained import measured_answer


@unittest.skipUnless(torch is not None, "Install the control environment for PyTorch tests")
class TrainedInferenceTest(unittest.TestCase):
    def test_counts_and_method_restoration(self):
        def generate(**kwargs):
            return torch.zeros((1, kwargs["input_ids"].shape[1] + 3), dtype=torch.long)

        core = SimpleNamespace(generate=generate)

        def answer(row):
            core.generate(input_ids=torch.zeros((1, 7), dtype=torch.long))
            return "Sim"

        model = SimpleNamespace(vlm=SimpleNamespace(model=core), answer=answer)
        with (
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.max_memory_allocated", return_value=1024),
        ):
            result = measured_answer(model, {})
        self.assertIs(core.generate, generate)
        self.assertEqual((result["input_tokens"], result["generated_tokens"]), (7, 3))
        self.assertEqual(result["response"], "Sim")
        self.assertEqual(result["peak_gpu_bytes"], 1024)

    def test_failure_does_not_leave_patched_generation(self):
        def generate(**kwargs):
            raise RuntimeError("simulated generation failure")

        core = SimpleNamespace(generate=generate)

        def answer(row):
            core.generate(input_ids=torch.zeros((1, 7), dtype=torch.long))

        model = SimpleNamespace(vlm=SimpleNamespace(model=core), answer=answer)
        with (
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.synchronize"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                measured_answer(model, {})
        self.assertIs(core.generate, generate)


if __name__ == "__main__":
    unittest.main()
