import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

from src.models import artifacts

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline_launcher', ROOT/'scripts/baseline.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
submit_spec = importlib.util.spec_from_file_location('submit_launcher', ROOT/'scripts/apuana/submit_baseline.py')
submit = importlib.util.module_from_spec(submit_spec)
submit_spec.loader.exec_module(submit)


class LauncherTests(unittest.TestCase):
    def test_rejects_paths_and_protected_result(self):
        for name in ('..', '../other', '/tmp/run', 'foo/bar', 'mof13b-pilot50-v1', ''):
            with self.subTest(name=name), self.assertRaises(ValueError):
                launcher.validate_run_id(name)
        launcher.validate_run_id('replica-01')

    def test_existing_run_requires_explicit_resume_before_gpu(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(launcher, 'ROOT', Path(tmp)):
            (Path(tmp)/'results/existing').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'já existe'):
                launcher.run('existing', smoke=False, resume=False)
            with self.assertRaisesRegex(ValueError, 'run.json'):
                launcher.run('existing', smoke=False, resume=True)

    def test_weight_corruption_and_changed_reference_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(artifacts, 'ROOT', Path(tmp)):
            root = Path(tmp)
            (root/'configs').mkdir()
            folder = root/'weights'; folder.mkdir()
            weight = folder/'model.bin'; weight.write_bytes(b'original weights')
            digest = artifacts.file_digest(weight)
            (root/'configs/baseline_mof.json').write_text('{}')
            (root/'configs/checkpoints-mof.json').write_text(json.dumps({'files':[
                {'role':'model','name':weight.name,'sha256':digest}]}))
            record = {'paths':{'model':str(folder)},'config':{},'sha256':{str(weight):digest}}
            self.assertEqual(artifacts.verify_weights(record),1)
            weight.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'corrompido'):
                artifacts.verify_weights(record)
            record['sha256'][str(weight)] = artifacts.file_digest(weight)
            with self.assertRaisesRegex(ValueError,'experimento original'):
                artifacts.verify_weights(record)

    @unittest.skipUnless(
        (ROOT/'results/mof13b-pilot50-v1/run.json').is_file(),
        'Historical inference archives are distributed separately',
    )
    def test_saved_original_source_matches_measured_run(self):
        run = json.loads((ROOT/'results/mof13b-pilot50-v1/run.json').read_text())
        source = ROOT/'experiments/mof13b-pilot50-v1/source'
        for rel, digest in run['source_sha256'].items():
            self.assertEqual(artifacts.file_digest(source/rel),digest)

    def test_slurm_dependencies_and_record(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(submit, 'ROOT', Path(tmp)), \
                patch.object(submit, 'verify_data'), \
                patch.object(submit.sys, 'argv', ['submit','--run-id','replica']), \
                patch.object(submit.subprocess, 'check_output', side_effect=['101\n','102\n','103\n','104\n']) as sbatch, \
                contextlib.redirect_stdout(io.StringIO()):
            submit.main()
            calls = [call.args[0] for call in sbatch.call_args_list]
            self.assertNotIn('--dependency=afterok:101',calls[0])
            for args, dependency in zip(calls[1:], ['101','102','103']):
                self.assertIn('--dependency=afterok:'+dependency,args)
                self.assertIn('--kill-on-invalid-dep=yes',args)
            record = json.loads((Path(tmp)/'experiments/submissions/replica.json').read_text())
            self.assertEqual(record['status'],'submitted')
            self.assertEqual([j['stage'] for j in record['jobs']],['setup','weights','smoke','evaluation'])

    def test_failed_submission_preserves_already_submitted_job(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(submit, 'ROOT', Path(tmp)), \
                patch.object(submit, 'verify_data'), \
                patch.object(submit.sys, 'argv', ['submit','--run-id','replica','--smoke-only']), \
                patch.object(submit.subprocess, 'check_output', side_effect=['101\n',subprocess.CalledProcessError(1,['sbatch'])]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                submit.main()
            record = json.loads((Path(tmp)/'experiments/submissions/replica.json').read_text())
            self.assertEqual(record['status'],'submission_failed')
            self.assertEqual([j['job_id'] for j in record['jobs']],['101'])


if __name__=='__main__':
    unittest.main()
