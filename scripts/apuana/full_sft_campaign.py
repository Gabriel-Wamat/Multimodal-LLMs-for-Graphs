"""Run gated full-SFT arms and final evaluation; keep full checkpoints on the cluster."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from scripts.apuana.run_alignment_pair import mirror

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','llava'],required=True);p.add_argument('--prefix',required=True);p.add_argument('--checks-prefix',required=True);p.add_argument('--backup-root',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.backup_root.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Backup must be outside project')
    for arm in ['clip_single','clip_double','clip_dino']:
        run_id=f'{a.prefix}-{a.model}-{arm}';source=ROOT/'experiments/full_sft'/run_id
        command=[sys.executable,'-m','src.training.run_full_sft','--model',a.model,'--arm',arm,'--run-id',run_id,'--checks-prefix',a.checks_prefix]
        if (source/'complete.json').exists():
            if not a.resume:raise ValueError('Run already exists')
            command=None
        elif source.exists():
            if not a.resume:raise ValueError('Partial run exists; explicit resume required')
            command.append('--resume')
        if command:
            child=subprocess.Popen(command,cwd=ROOT)
            # Checkpoint copies are atomic, at most once per published version.
            while child.poll() is None:
                try:mirror(source,a.backup_root/run_id)
                except OSError as exc:print(json.dumps({'phase':'backup_error','error_type':type(exc).__name__,'run':run_id}),flush=True)
                time.sleep(30)
            mirror(source,a.backup_root/run_id)
            if child.returncode:raise SystemExit(child.returncode)
        result_id=run_id+'-pilot'
        evaluate=[sys.executable,'-m','src.evaluation.full_sft','--training-run',run_id,'--run-id',result_id]
        if a.resume and (ROOT/'results'/result_id).exists():evaluate.append('--resume')
        subprocess.run(evaluate,cwd=ROOT,check=True)
        mirror(ROOT/'results'/result_id,a.backup_root/'results'/result_id)
        print(json.dumps({'phase':'arm_evaluated','model':a.model,'arm':arm}),flush=True)


if __name__=='__main__':main()
