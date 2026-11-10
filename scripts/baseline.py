"""Instalação e execução reproduzível do piloto I-MoF, sem caminhos de usuário."""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.pilot import sha
from src.models.artifacts import verify_repositories, verify_weights, weights_manifest_path

ENV_PYTHON = ROOT/'.venv-mof/bin/python'


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=ROOT, check=True, **kwargs)


def configure_cache():
    # Todos os downloads e metadados desta instalação pertencem ao projeto.
    for name, relative in [('HF_HOME', '.cache/huggingface'), ('TORCH_HOME', '.cache/torch'),
                           ('PIP_CACHE_DIR', '.cache/pip')]:
        os.environ[name] = str(ROOT/relative)
    os.environ['SIH_WEIGHTS_MANIFEST'] = str(ROOT/'.cache/mof/weights.json')
    os.environ['PYTHONUNBUFFERED'] = '1'


def require_linux():
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise ValueError('Este ambiente de inferência requer Linux x86_64 + NVIDIA CUDA. '
                         'No Mac, use check --data-only e submeta a execução ao Apuana.')


def require_environment():
    if not ENV_PYTHON.is_file():
        raise ValueError('Ambiente ausente: execute bash scripts/baseline.sh install.')
    command([ENV_PYTHON, '-c',
             'import sys; assert sys.version_info[:2]==(3,10), "Use Python 3.10"'])


def verify_data():
    base = ROOT/'data/manifests/pilot50_v1'
    split = json.loads((base/'split.json').read_text())
    for name in ('inputs', 'labels'):
        if sha(base/f'{name}.jsonl') != split[f'{name}_sha256']:
            raise ValueError(f'{name}.jsonl diverge da amostra congelada')
    inputs = [json.loads(line) for line in (base/'inputs.jsonl').read_text().splitlines()]
    labels = [json.loads(line) for line in (base/'labels.jsonl').read_text().splitlines()]
    ids = {r['id'] for r in inputs}
    if len(inputs) != 3173 or len(ids) != 3173 or ids != {r['id'] for r in labels}:
        raise ValueError('IDs ou cobertura divergentes do piloto')
    if {r['grafico'] for r in inputs} != set(split['selected_graphs']):
        raise ValueError('Seleção de gráficos divergente')
    allowed = {'id','grafico','tipo','estilo','nivel','pergunta','formato','image','image_sha256'}
    if any(set(r) != allowed for r in inputs):
        raise ValueError('Campos inesperados nas entradas do modelo')
    images = {(r['image'], r['image_sha256']) for r in inputs}
    if len(images) != 360:
        raise ValueError('Esperados 360 gráficos')
    for name, digest in images:
        if sha(ROOT/name) != digest:
            raise ValueError(f'Imagem divergente: {name}')
    print('Dados OK: 360 gráficos, 3.173 perguntas; hashes e campos conferidos.', flush=True)


def checkout(name, url, revision):
    destination = ROOT/'work/vendor'/name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        command(['git', 'clone', '--no-checkout', url, destination])
    elif not (destination/'.git').exists():
        raise ValueError(f'Diretório já existe e não é um clone Git: {destination}')
    else:
        dirty = command(['git', '-C', destination, 'status', '--porcelain'],
                        capture_output=True, text=True).stdout.strip()
        if dirty:
            raise ValueError(f'Preservando alterações locais em {destination}; resolva antes de instalar.')
    command(['git', '-C', destination, 'checkout', '--detach', revision])


def install(env_only):
    require_linux()
    if sys.version_info[:2] != (3, 10):
        raise ValueError('Instale Python 3.10 e use SIH_PYTHON=python3.10 bash scripts/baseline.sh install.')
    if not ENV_PYTHON.exists():
        command([sys.executable, '-m', 'venv', ROOT/'.venv-mof'])
    require_environment()
    command([ENV_PYTHON, '-m', 'pip', 'install', '-r', 'configs/requirements-mof.txt',
             '-c', 'configs/constraints-mof.txt'])
    command([ENV_PYTHON, '-m', 'pip', 'check'])
    cfg = json.loads((ROOT/'configs/baseline_mof.json').read_text())
    checkout('MMVP', cfg['code_repository'], cfg['code_revision'])
    checkout('dinov2', 'https://github.com/facebookresearch/dinov2.git', cfg['dino_revision'])
    output = ROOT/'.cache/mof'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'environment.txt').open('w') as stream:
        command([ENV_PYTHON, '-m', 'pip', 'freeze'], stdout=stream)
    if not env_only:
        download()
    print('Ambiente preparado.' if env_only else 'Ambiente e checkpoints preparados.', flush=True)


def download():
    require_linux()
    require_environment()
    command([ENV_PYTHON, 'scripts/apuana/prepare_weights.py'])


def check_weights():
    cfg = json.loads((ROOT/'configs/baseline_mof.json').read_text())
    verify_repositories(cfg)
    record = json.loads(weights_manifest_path().read_text())
    count = verify_weights(record)
    print(f'Pesos OK: {count} arquivos; SHA-256 idênticos ao experimento original.', flush=True)


def check_gpu():
    command([ENV_PYTHON, '-c', '''import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA indisponível: verifique driver NVIDIA e alocação da GPU.')
gpu = torch.cuda.get_device_properties(0)
if gpu.total_memory < 32*1024**3:
    raise SystemExit('FP16 requer pelo menos 32 GiB de VRAM; preferir A100 80 GB.')
print(f'GPU: {gpu.name}; {gpu.total_memory/1024**3:.1f} GiB')
'''])


def validate_run_id(run_id):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,79}', run_id):
        raise ValueError('run-id deve ter 1–80 letras, números, ponto, hífen ou sublinhado, sem caminhos.')
    if run_id == 'mof13b-pilot50-v1':
        raise ValueError('O experimento original está preservado. Escolha outro --run-id.')


def run(run_id, smoke, resume):
    validate_run_id(run_id)
    folder = ROOT/'results'/run_id
    if folder.exists() and not resume:
        raise ValueError('run-id já existe. Use outro nome ou --resume para retomar o mesmo protocolo.')
    if resume and not (folder/'run.json').is_file():
        raise ValueError('--resume exige uma execução anterior com run.json.')
    require_linux()
    require_environment()
    verify_data()
    check_weights()
    check_gpu()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    command([ENV_PYTHON, '-m', 'src.evaluation.run', '--run-id', run_id] + (['--smoke'] if smoke else []))
    if not smoke:
        metrics = folder/'scored/metrics.json'
        if metrics.exists():
            existing = json.loads(metrics.read_text())
            if (existing['predictions_sha256'] != sha(folder/'predictions.jsonl')
                    or existing['scorer_sha256'] != sha(ROOT/'src/evaluation/score.py')
                    or existing['run_manifest_sha256'] != sha(folder/'run.json')
                    or existing['split_sha256'] != sha(ROOT/'data/manifests/pilot50_v1/split.json')
                    or not existing['all_records_successful']):
                raise ValueError('Métricas existentes divergentes; nada foi sobrescrito.')
            print('Métricas completas já verificadas; correção não repetida.')
        else:
            command([ENV_PYTHON, '-m', 'src.evaluation.score', folder/'predictions.jsonl',
                     '--output', folder/'scored'])
    print(f'Resultados: {folder}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    setup = commands.add_parser('install', help='Cria .venv-mof, instala dependências e baixa checkpoints.')
    setup.add_argument('--env-only', action='store_true', help='Instala ambiente/código; download em etapa separada.')
    commands.add_parser('download', help='Baixa/retoma os pesos e verifica os hashes oficiais e do baseline.')
    check = commands.add_parser('check', help='Valida dados, código externo e checkpoints, sem inferência.')
    check.add_argument('--data-only', action='store_true', help='Confere apenas os dados; funciona no Mac.')
    check.add_argument('--gpu', action='store_true', help='Verifica também a GPU CUDA da alocação atual.')
    for action in ('smoke', 'run'):
        sub = commands.add_parser(action, help='6 perguntas de diagnóstico.' if action=='smoke' else 'Avalia os 360 gráficos / 3.173 perguntas.')
        sub.add_argument('--run-id', required=True, help='Nome novo para preservar resultados anteriores.')
        sub.add_argument('--resume', action='store_true', help='Retoma somente configuração/código idênticos.')
    args = parser.parse_args()
    configure_cache()
    if args.action=='install':
        install(args.env_only)
    elif args.action=='download':
        download()
    elif args.action=='check':
        verify_data()
        if not args.data_only:
            check_weights()
        if args.gpu:
            require_linux()
            require_environment()
            check_gpu()
    else:
        run(args.run_id, args.action=='smoke', args.resume)


if __name__=='__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'ERRO: {exc}', file=sys.stderr)
        sys.exit(1)
