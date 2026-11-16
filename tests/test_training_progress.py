"""Progress output preserves scoring, order and RNG; mirrors exclude unpublished files."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.apuana.run_alignment_pair import mirror


class ArtifactMirrorTests(unittest.TestCase):
    def test_published_checkpoint_replacement_and_temporary_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';source.mkdir()
            backup=Path(tmp)/'backup'
            (source/'last.pt').write_bytes(b'checkpoint-one')
            (source/'last.tmp').write_bytes(b'unpublished')
            (source/'best.tmp.safetensors').write_bytes(b'unpublished')
            self.assertEqual(mirror(source,backup),1)
            self.assertEqual(mirror(source,backup),0)
            self.assertFalse((backup/'last.tmp').exists())
            self.assertFalse((backup/'best.tmp.safetensors').exists())
            replacement=source/'last.tmp';replacement.write_bytes(b'checkpoint-two-complete')
            replacement.replace(source/'last.pt')
            self.assertEqual(mirror(source,backup),1)
            self.assertEqual((backup/'last.pt').read_bytes(),b'checkpoint-two-complete')


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch profile required')
class ValidationProgressTests(unittest.TestCase):
    def test_progress_keeps_prediction_order_loss_and_rng(self):
        import torch
        from src.training.run import validate
        class Model:
            def loss(self,row):
                if torch.is_grad_enabled():raise AssertionError('Validation enabled gradients')
                return torch.tensor(row['value']),1
            def answer(self,row):return row['response']
        rows=[{'id':str(i),'value':i/10,'response':'Sim'} for i in range(27)]
        def evaluate(items,predictions):
            self.assertEqual([r['id'] for r in items],[p['id'] for p in predictions])
            self.assertEqual([p['response'] for p in predictions],['Sim']*27)
            return {'accuracy':1.0},[]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            rng=torch.get_rng_state().clone()
            with patch('src.training.run.evaluate',evaluate),patch('src.training.run.normalize_binary',lambda r,p:p):
                result=validate(Model(),rows,folder,0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertAlmostEqual(result['loss'],1.3,places=6)
            self.assertEqual(result['questions'],27)
            progress=json.loads((folder/'progress.json').read_text())
            self.assertEqual((progress['completed'],progress['total']),(27,27))
