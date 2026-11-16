"""Bounded SLURM observation and private off-cluster artifact copies; never resubmit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--run-prefix', required=True)
    p.add_argument('--model', choices=['llava', 'qwen'], required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--backup-root', type=Path, required=True)
    p.add_argument('--interval', type=int, default=60)
    p.add_argument('--max-hours', type=float, default=48)
    a = p.parse_args()
    for value, pattern in [(a.host,r'[A-Za-z0-9@_.-]+'),(a.remote_root,r'/[A-Za-z0-9_./-]+'),
                           (a.run_prefix,r'[A-Za-z0-9][A-Za-z0-9_.-]{0,48}'),(a.job_id,r'[0-9]+')]:
        if not re.fullmatch(pattern,value):p.error('Invalid connection or experiment identifier')
    if a.interval < 15 or not 0 < a.max_hours <= 72:p.error('Invalid monitoring bounds')
    a.backup_root.mkdir(parents=True,exist_ok=True)
    state_root=a.backup_root/('.monitor-'+a.run_prefix)
    state_root.mkdir(parents=True,exist_ok=True)
    run_ids=[f'{a.run_prefix}-{a.model}-{arm}' for arm in ['clip_double','clip_dino']]
    cfg={'root':a.remote_root,'job':a.job_id,'run_ids':run_ids}
    script='''import json,subprocess,time
from pathlib import Path
c=CONFIG
root=Path(c['root'])
q=subprocess.run(['squeue','-j',c['job'],'-h','-o','%i|%T|%M|%R'],capture_output=True,text=True,timeout=15)
if q.returncode and 'Invalid job id' not in q.stderr:raise RuntimeError('Scheduler query failed')
r={'observed_at_unix':time.time(),'queue':q.stdout.splitlines(),'runs':{}}
for name in c['run_ids']:
 f=root/'experiments/training'/name
 d={'exists':f.is_dir(),'complete':(f/'complete.json').is_file()}
 if (f/'progress.json').is_file():d['progress']=json.loads((f/'progress.json').read_text())
 r['runs'][name]=d
if any('|RUNNING|' in line for line in r['queue']):
 try:
  g=subprocess.run(['srun','--jobid='+c['job'],'--overlap','-N','1','-n','1','-c','1','nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True,timeout=15)
  r['gpu']=g.stdout.strip() if g.returncode==0 else 'unavailable'
 except subprocess.TimeoutExpired:r['gpu']='query_timeout'
print(json.dumps(r))
'''.replace('CONFIG',repr(cfg))
    started=time.monotonic();errors=0
    while time.monotonic()-started < a.max_hours*3600:
        try:
            result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',a.host,'python3','-'],input=script,text=True,capture_output=True,timeout=45,check=True)
            state=json.loads(result.stdout)
            for name,info in state['runs'].items():
                if not info['exists']:continue
                result=subprocess.run(['rsync','-rt','--delay-updates','--exclude=*.tmp*','--exclude=runner.lock',
                    '-e','ssh -o BatchMode=yes -o ConnectTimeout=10',
                    f'{a.host}:{a.remote_root}/experiments/training/{name}',str(a.backup_root)+'/'],timeout=120)
                if result.returncode not in (0,24):raise RuntimeError('Artifact transfer failed')
            checksums={}
            for name in run_ids:
                folder=a.backup_root/name
                for path in folder.rglob('*') if folder.exists() else []:
                    if path.is_file() and path.suffix in ('.pt','.safetensors'):
                        digest=hashlib.sha256()
                        with path.open('rb') as f:
                            for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
                        checksums[str(path.relative_to(a.backup_root))]=digest.hexdigest()
            state['local_checkpoint_sha256']=checksums
            state['state']='complete' if all(v['complete'] for v in state['runs'].values()) else ('active' if state['queue'] else 'stopped_without_completion')
            errors=0
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            errors+=1;state={'state':'connection_or_backup_error','error_type':type(exc).__name__,'consecutive_errors':errors,'observed_at_unix':time.time()}
        temp=state_root/'monitor.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_root/'monitor.json')
        with (state_root/'observations.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        print(json.dumps(state),flush=True)
        if state['state'] in ('complete','stopped_without_completion') or errors>=3:return
        time.sleep(a.interval)
    (state_root/'monitor-ended.json').write_text(json.dumps({'reason':'monitor_time_budget_exhausted','job_not_cancelled':True})+'\n')


if __name__=='__main__':main()
