"""Bounded, read-only model observation on Apuana; records status without resubmission."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]


def read_optional(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--prefix',required=True);p.add_argument('--checks-prefix',required=True);p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    if not a.job_id.isdigit() or not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,48}',s) for s in [a.prefix,a.checks_prefix]) or a.interval<15:raise ValueError('Invalid observer identity/bounds')
    target=ROOT/'experiments/submissions'/f'{a.prefix}.monitor.json';journal=target.with_suffix('.jsonl');last_key=None;last_change=time.monotonic();started=last_change;errors=0
    while time.monotonic()-started<49*3600:
        now=time.time();data={'job_id':a.job_id,'observed_at_unix':now,'scope':'full two-stage graph SFT; checkpoints remain on Apuana','runs':{},'gates':{}}
        try:
            q=subprocess.run(['squeue','-j',a.job_id,'-h','-o','%i|%T|%M|%R'],capture_output=True,text=True,timeout=15)
            if q.returncode and 'Invalid job id' not in q.stderr:raise RuntimeError('Scheduler unavailable')
            data['queue']=q.stdout.splitlines()
            for model in ['qwen','llava']:
                for arm in ['clip_single','clip_double','clip_dino']:
                    name=f'{a.prefix}-{model}-{arm}';folder=ROOT/'experiments/full_sft'/name
                    data['runs'][name]={'exists':folder.exists(),'progress':read_optional(folder/'progress.json'),'checkpoint':read_optional(folder/'checkpoint.json'),'best':read_optional(folder/'best.json'),'complete':(folder/'complete.json').is_file(),'pilot_complete':(ROOT/'results'/(name+'-pilot')/'complete.json').is_file()}
                    predictions=ROOT/'results'/(name+'-pilot')/'predictions.jsonl'
                    try:
                        stat=predictions.stat()
                        data['runs'][name]['pilot_predictions']={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
                    except FileNotFoundError:
                        data['runs'][name]['pilot_predictions']=None
                    gate=ROOT/'experiments/full_sft_checks'/f'{a.checks_prefix}-{model}-{arm}'
                    marker=read_optional(gate/'complete.json')
                    data['gates'][model+'-'+arm]={'passed':bool(marker),'progress':read_optional(gate/'progress.json')}
                    if marker:data['gates'][model+'-'+arm].update(decoder_trainable_parameters=marker['decoder_trainable_parameters'],status=marker['status'])
            gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv'],capture_output=True,text=True,timeout=15)
            data['gpu']=gpu.stdout.strip() if gpu.returncode==0 else 'unavailable'
            temps={}
            for path in Path('/tmp').glob('graph-sft-resume-*/full.pt'):
                try:
                    st=path.stat()
                    if st.st_uid==os.getuid():temps[str(path)]={'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
                except FileNotFoundError:pass
            for folder in (ROOT/'experiments/full_sft').glob(a.prefix+'*'):
                for path in folder.glob('*.tmp'):
                    try:temps[str(path)]={'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns}
                    except FileNotFoundError:pass
            data['active_checkpoint_io']=temps
            key=json.dumps([data['runs'],data['gates'],temps],sort_keys=True)
            if key!=last_key:last_key=key;last_change=time.monotonic()
            data['seconds_without_artifact_change']=round(time.monotonic()-last_change)
            if all(v['complete'] and v['pilot_complete'] for v in data['runs'].values()):data['state']='complete'
            elif not data['queue']:data['state']='stopped_without_all_results'
            elif data['seconds_without_artifact_change']>1200:data['state']='needs_attention';data['note']='No recent artifact or checkpoint I/O change; investigate before any restart'
            else:data['state']='active'
            errors=0
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            errors+=1;data.update(state='observation_error',error_type=type(exc).__name__,consecutive_errors=errors)
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(target)
        with journal.open('a') as f:f.write(json.dumps(data)+'\n')
        if data['state'] in ['complete','stopped_without_all_results'] or errors>=3:break
        time.sleep(a.interval)


if __name__=='__main__':main()
