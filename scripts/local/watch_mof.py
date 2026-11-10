"""Read-only remote monitoring with explicit connection settings and bounded waits."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def remote_status(host, remote_root, run_id, jobs=(), timeout=45):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9@._-]*", host):
        raise ValueError("Invalid SSH host or alias")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id) or run_id in (".", ".."):
        raise ValueError("Invalid run ID")
    if not Path(remote_root).is_absolute():
        raise ValueError("Remote root must be an absolute path supplied by the operator")
    if any(not str(job).isdigit() for job in jobs):
        raise ValueError("Job IDs must be numeric")
    payload = json.dumps({"root": remote_root, "run_id": run_id, "jobs": list(jobs)})
    script = """
import getpass, hashlib, json, subprocess
from pathlib import Path
cfg = json.loads(PAYLOAD)
root = Path(cfg['root'])
if not root.is_dir():
    raise FileNotFoundError('Remote checkout not found: ' + str(root))
folder = root / 'results' / cfg['run_id']
queue = subprocess.run(['squeue','-h','-u',getpass.getuser(),'-o','%i|%T|%r'],
                       capture_output=True,text=True,check=True,timeout=15)
wanted = set(cfg['jobs'])
rows = [r for r in queue.stdout.splitlines() if not wanted or r.split('|')[0] in wanted]
prediction = folder / 'predictions.jsonl'
count = 0
if prediction.exists():
    with prediction.open('rb') as stream:
        count = sum(1 for line in stream if line.endswith(b'\\n') and line.strip())
state = 'active' if rows else 'unknown'
marker = folder / 'complete.json'
if marker.exists():
    complete = json.loads(marker.read_text())
    if not prediction.exists() or hashlib.sha256(prediction.read_bytes()).hexdigest() != complete.get('predictions_sha256'):
        raise ValueError('Completion marker/prediction hash mismatch')
    state = 'complete'
elif any('DependencyNeverSatisfied' in row for row in rows):
    state = 'needs_attention'
print(json.dumps({'state':state,'queue':rows,'prediction_records':count}))
""".replace("PAYLOAD", repr(payload))
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
         host, "python3", "-"],
        input=script, capture_output=True, text=True, timeout=timeout, check=True,
    )
    return json.loads(result.stdout)


def collect(host, remote_root, run_id, timeout=180):
    target = ROOT / "results" / run_id
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-az", "--protect-args", "--", f"{host}:{remote_root}/results/{run_id}/",
         str(target) + "/"], check=True, timeout=timeout,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("SIH_SSH_HOST"))
    parser.add_argument("--remote-root", default=os.environ.get("SIH_REMOTE_ROOT"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--watch", action="store_true", help="Poll until complete or duration expires")
    parser.add_argument("--duration", type=float, default=300)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--collect", action="store_true", help="Copy only this completed run with rsync >=3")
    args = parser.parse_args()
    if not args.host or not args.remote_root:
        parser.error("Supply --host and --remote-root or SIH_SSH_HOST/SIH_REMOTE_ROOT")
    if args.duration <= 0 or args.interval <= 0:
        parser.error("Duration and interval must be positive")
    deadline = time.monotonic() + args.duration
    while True:
        try:
            status = remote_status(args.host, args.remote_root, args.run_id, args.job_id)
            if args.collect and status["state"] == "complete":
                collect(args.host, args.remote_root, args.run_id)
                status["collected"] = True
        except (subprocess.SubprocessError, ValueError, OSError) as exc:
            status = {"state": "connection_or_collection_error", "error": str(exc)}
        status["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if not args.watch or status["state"] in ("complete", "needs_attention"):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(args.interval, remaining))


if __name__ == "__main__":
    main()
