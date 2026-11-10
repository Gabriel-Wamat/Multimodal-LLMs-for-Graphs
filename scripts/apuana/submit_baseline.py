"""Submete a cadeia SLURM; --dry-run mostra os comandos sem submeter jobs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from baseline import validate_run_id, verify_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--prepared', action='store_true', help='Ambiente e pesos já instalados: pula essas duas etapas.')
    parser.add_argument('--smoke-only', action='store_true', help='Encerra após as seis perguntas de diagnóstico.')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--node', help='Nó GPU opcional; por padrão o SLURM escolhe a alocação.')
    args = parser.parse_args()
    validate_run_id(args.run_id)
    validate_run_id(args.run_id+'-smoke')
    for run_id in (args.run_id, args.run_id+'-smoke'):
        if (ROOT/'results'/run_id).exists():
            raise ValueError(f'Resultados já existem para {run_id}; escolha outro nome.')
    record_path = ROOT/'experiments/submissions'/f'{args.run_id}.json'
    if record_path.exists() and not args.dry_run:
        raise ValueError('Submissão já registrada; consulte os jobs antes de submeter novamente.')
    verify_data()
    record = {'run_id':args.run_id, 'submitted_at_utc':datetime.now(timezone.utc).isoformat(),
              'project':str(ROOT), 'node':args.node, 'jobs':[], 'status':'submitting'}
    if not args.dry_run:
        (ROOT/'logs').mkdir(exist_ok=True)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with record_path.open('x') as stream:
            json.dump(record, stream, indent=2)

    def save():
        temporary = record_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(record_path)

    previous = None
    stages = ([] if args.prepared else [('setup', 'setup_mof.sh', []), ('weights', 'download_mof.sh', [])])
    stages += [('smoke', 'run_mof.sh', ['--run-id',args.run_id+'-smoke','--smoke'])]
    if not args.smoke_only:
        stages += [('evaluation', 'run_mof.sh', ['--run-id',args.run_id])]
    try:
        for stage, script, extra in stages:
            cmd = ['sbatch','--parsable',f'--chdir={ROOT}']
            if previous:
                cmd.append(f'--dependency=afterok:{previous}')
                cmd.append('--kill-on-invalid-dep=yes')
            if stage in ('smoke','evaluation'):
                if args.node:
                    cmd.append(f'--nodelist={args.node}')
                if stage=='smoke':
                    cmd.append('--time=00:30:00')
            cmd += [f'scripts/apuana/{script}', *extra]
            print(shlex.join(cmd), flush=True)
            if args.dry_run:
                previous = f'<job_{stage}>'
            else:
                output = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
                previous = output.split(';')[0]
                if not previous.isdigit():
                    raise ValueError(f'Resposta inesperada do sbatch: {output}')
                record['jobs'].append({'stage':stage,'job_id':previous,'command':cmd})
                save()
    except (OSError, ValueError, subprocess.CalledProcessError):
        if not args.dry_run:
            record['status']='submission_failed'
            save()
            print(f'Submissão interrompida. Jobs já enviados estão registrados em {record_path}.', file=sys.stderr)
        raise
    if not args.dry_run:
        record['status']='submitted'
        save()
        print(f'Jobs registrados em {record_path}. Acompanhe com squeue -u "$USER".')


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print(f'ERRO: {exc}',file=sys.stderr)
        sys.exit(1)
