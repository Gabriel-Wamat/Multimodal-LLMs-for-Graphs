"""Run matched alignment arms sequentially and mirror atomic artifacts privately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def mirror(source, destination):
    """Copy a published inode once; replacement by the trainer cannot mix versions."""
    if not source.exists():
        return 0
    copied = 0
    for path in source.rglob('*'):
        if not path.is_file() or '.tmp' in path.name or path.name.endswith('.lock'):
            continue
        target = destination / path.relative_to(source)
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if target.exists() and target.stat().st_size == stat.st_size and target.stat().st_mtime_ns == stat.st_mtime_ns:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.copying')
        digest = hashlib.sha256()
        try:
            reader = path.open('rb')
        except FileNotFoundError:
            continue
        with reader, temporary.open('wb') as writer:
            stable = os.fstat(reader.fileno())
            for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                writer.write(chunk)
                digest.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        os.utime(temporary, ns=(stable.st_atime_ns, stable.st_mtime_ns))
        temporary.replace(target)
        if target.suffix in ('.pt', '.safetensors'):
            stamp = target.with_name(target.name + '.sha256')
            stamp.write_text(digest.hexdigest() + '\n')
        copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['llava', 'qwen'], required=True)
    parser.add_argument('--run-prefix', required=True)
    parser.add_argument('--checks-prefix', required=True)
    parser.add_argument('--backup-root', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,48}', args.run_prefix):
        parser.error('Invalid run prefix')
    if args.backup_root.resolve().is_relative_to(ROOT.resolve()):
        parser.error('Backup must be outside the project')
    args.backup_root.mkdir(parents=True, exist_ok=True)
    for arm in ['clip_double', 'clip_dino']:
        run_id = f'{args.run_prefix}-{args.model}-{arm}'
        source = ROOT / 'experiments/training' / run_id
        if source.exists():
            raise SystemExit('Run exists: use the documented explicit resume workflow')
        command = [sys.executable, '-m', 'src.training.run', '--model', args.model,
                   '--arm', arm, '--run-id', run_id, '--checks-prefix', args.checks_prefix]
        child = subprocess.Popen(command, cwd=ROOT)
        def stop(signum, frame):
            if child.poll() is None:
                try:
                    child.send_signal(signum)
                except ProcessLookupError:
                    pass
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        print(json.dumps({'phase': 'arm_started', 'run_id': run_id, 'pid': child.pid}), flush=True)
        while child.poll() is None:
            try:
                mirror(source, args.backup_root / run_id)
            except OSError as exc:
                print(json.dumps({'phase': 'backup_error', 'run_id': run_id,
                                  'error_type': type(exc).__name__}), flush=True)
            time.sleep(15)
        mirror(source, args.backup_root / run_id)
        print(json.dumps({'phase': 'arm_finished', 'run_id': run_id, 'exit_code': child.returncode}), flush=True)
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
