"""Compare uninterrupted AdamW training with a fresh-process-style restore."""

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.training.checkpoint import restore, save


@unittest.skipUnless(torch is not None, "Install the control environment for PyTorch tests")
class ResumeTest(unittest.TestCase):
    def test_exact_continuation_and_protocol_rejection(self):
        torch.manual_seed(48)

        def build():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 7), torch.nn.Dropout(0.25), torch.nn.Linear(7, 2)
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.7)
            return model, optimizer, scheduler

        def advance(parts, state, count):
            model, optimizer, scheduler = parts
            losses = []
            for _ in range(count):
                x, y = torch.randn(3, 4), torch.randn(3, 2)
                optimizer.zero_grad(set_to_none=True)
                loss = (model(x) - y).square().mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
                state['cursor'] += 3
                state['updates'] += 1
                losses.append(loss.item())
            return losses

        original = build()
        state = {'cursor': 0, 'updates': 0}
        advance(original, state, 3)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'last.pt'
            save(checkpoint, *original, state, 'protocol-A')
            expected_losses = advance(original, state, 5)
            expected_rng = torch.rand(9)
            resumed = build()  # Deliberately consumes RNG before restoration.
            resumed_state = restore(checkpoint, *resumed, 'protocol-A')
            self.assertEqual(resumed_state, {'cursor': 9, 'updates': 3})
            self.assertEqual(advance(resumed, resumed_state, 5), expected_losses)
            self.assertTrue(torch.equal(torch.rand(9), expected_rng))
            self.assertEqual(resumed_state, state)
            for left, right in zip(original[0].parameters(), resumed[0].parameters()):
                self.assertTrue(torch.equal(left, right))
            self.assertEqual(original[2].state_dict(), resumed[2].state_dict())
            for key, expected in original[1].state_dict()['state'].items():
                actual = resumed[1].state_dict()['state'][key]
                for field, value in expected.items():
                    self.assertTrue(torch.equal(value, actual[field]))
            with self.assertRaisesRegex(ValueError, 'protocol/data/source'):
                restore(checkpoint, *resumed, 'protocol-B')


if __name__ == '__main__':
    unittest.main()
