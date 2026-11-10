import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch do ambiente .venv-control necessário")
class ControlMoFTests(unittest.TestCase):
    def test_dino_extracts_pre_norm_not_normalized_hidden_state(self):
        from types import SimpleNamespace

        import torch

        from src.models.control.encoders import dinov3_pre_norm

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                return SimpleNamespace(last_hidden_state=self.norm(x * 3 + 4))

        model = Encoder()
        pixels = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        actual = dinov3_pre_norm(model, pixels)
        torch.testing.assert_close(actual, pixels * 3 + 4)
        self.assertEqual(len(model.norm._forward_pre_hooks), 0)

    def test_dino_hook_removed_on_failure(self):
        import torch

        from src.models.control.encoders import dinov3_pre_norm

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                raise RuntimeError("example")

        model = Encoder()
        with self.assertRaises(RuntimeError):
            dinov3_pre_norm(model, torch.ones(1, 4))
        self.assertEqual(len(model.norm._forward_pre_hooks), 0)

    def test_interleave_spatial_order(self):
        import torch

        from src.models.control.fusion import interleave

        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        self.assertTrue(
            torch.equal(
                interleave(a, b),
                torch.tensor([[1.0, 2.0], [5.0, 6.0], [3.0, 4.0], [7.0, 8.0]]),
            )
        )
        with self.assertRaises(ValueError):
            interleave(a, b[:1])

    def test_placeholder_expansion_preserves_text(self):
        import torch

        from src.models.control.fusion import expand_image_tokens

        ids = torch.tensor([[7] + [99] * 576 + [8, 9]])
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        out = expand_image_tokens(inputs, 99)
        self.assertEqual(out["input_ids"].tolist(), [[7] + [99] * 1152 + [8, 9]])
        self.assertEqual(out["attention_mask"].shape, out["input_ids"].shape)
        self.assertEqual(inputs["input_ids"].shape[1], 579)

    def test_frozen_seeded_initialization(self):
        import torch

        from src.models.control.fusion import StageZeroAdapters

        a = StageZeroAdapters("qwen", 16, 42)
        b = StageZeroAdapters("qwen", 16, 42)
        c = StageZeroAdapters("qwen", 16, 43)
        self.assertTrue(
            all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
        )
        self.assertFalse(torch.equal(a.clip[0].weight, c.clip[0].weight))
        self.assertFalse(any(p.requires_grad for p in a.parameters()))

    def test_shuffle_is_fixed_per_image(self):
        import torch

        from src.models.control.fusion import shuffle_patches

        x = torch.arange(64).reshape(32, 2)
        a = shuffle_patches(x, "BAR-001", 42)
        b = shuffle_patches(x, "BAR-001", 42)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, x))
        self.assertEqual(sorted(a[:, 0].tolist()), x[:, 0].tolist())


class ControlScoreTests(unittest.TestCase):
    def test_boolean_aliases_only_for_boolean_questions(self):
        from src.evaluation.control.score import normalize_binary

        labels = [
            {"id": "a", "formato": "sim/não"},
            {"id": "b", "formato": "categoria"},
        ]
        raw = [{"id": "a", "response": "Yes."}, {"id": "b", "response": "Yes"}]
        result = normalize_binary(labels, raw)
        self.assertEqual(result[0]["response"], "sim")
        self.assertEqual(result[1]["response"], "Yes")
        self.assertEqual(raw[0]["response"], "Yes.")

    def test_paired_delta_keeps_whole_graph(self):
        from src.evaluation.control.score import paired_graph_delta

        labels = [
            {"id": i, "grafico": "g", "tipo": "bar", "estilo": "a"} for i in ["a", "b"]
        ]
        first = [{"id": i, "correct": False} for i in ["a", "b"]]
        second = [{"id": i, "correct": True} for i in ["a", "b"]]
        result = paired_graph_delta(labels, first, second)
        self.assertEqual(result["delta_accuracy"], 1)
        self.assertEqual(result["ci95_graph_bootstrap"], [1, 1])


if __name__ == "__main__":
    unittest.main()
