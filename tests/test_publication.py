"""The publication boundary rejects private data and notebook state."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'publication', Path(__file__).resolve().parents[1] / 'scripts/local/check_publication.py'
)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def test_private_data_and_checkpoint_rejected(self):
        self.assertTrue(publication.check_file('data/manifests/labels.jsonl', b'{}'))
        self.assertTrue(publication.check_file('results/raw/run.json', b'{}'))
        self.assertTrue(publication.check_file('weights/model.safetensors', b'weights'))
        self.assertEqual(publication.check_file('results/published/summary.json', b'{}'), [])

    def test_generated_notebook_output_rejected(self):
        n={'cells':[{'cell_type':'code','execution_count':1,'outputs':[]}]}
        self.assertTrue(publication.check_file('notebooks/test.ipynb',json.dumps(n).encode()))
        n['cells'][0]['execution_count']=None
        self.assertEqual(publication.check_file('notebooks/test.ipynb',json.dumps(n).encode()),[])

    def test_missing_readme_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(publication.check_file('README.md',b'[missing](missing.md)',Path(tmp)))
            (Path(tmp)/'existing.md').write_text('ok')
            self.assertEqual(publication.check_file('README.md',b'[exists](existing.md)',Path(tmp)),[])
