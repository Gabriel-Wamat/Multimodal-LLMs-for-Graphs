"""Run the CPU continuation contract and persist evidence only on success."""

import hashlib
import json
import os
import platform
import unittest
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parents[2]
    suite = unittest.defaultTestLoader.discover(
        str(root / 'tests'), pattern='test_training_resume.py'
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.testsRun != 1:
        raise SystemExit(1)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        'passed': True,
        'tests': result.testsRun,
        'checkpoint_source_sha256': sha(Path(__file__).with_name('checkpoint.py')),
        'test_source_sha256': sha(root / 'tests/test_training_resume.py'),
        'torch_version': str(torch.__version__),
        'python_version': platform.python_version(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'scope': 'CPU bitwise continuation of weights, AdamW moments, scheduler, RNG and cursor; CUDA full-model checks are separate',
    }
    target = root / 'experiments/five-stage-campaign/resume_check.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
