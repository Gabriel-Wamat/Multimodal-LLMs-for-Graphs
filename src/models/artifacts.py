"""Caminhos da instalação e validação dos bytes usados no baseline."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

from src.data.pilot import ROOT


def weights_manifest_path():
    return Path(os.environ.get('SIH_WEIGHTS_MANIFEST', ROOT/'.cache/mof/weights.json')).resolve()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_repositories(config):
    for name, revision in [('MMVP', config['code_revision']), ('dinov2', config['dino_revision'])]:
        path = ROOT/'work/vendor'/name
        head = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip()
        if head != revision or dirty:
            raise ValueError(f'Código externo alterado ou revisão divergente: {path}')


def verify_weights(record, check_hashes=True):
    config = json.loads((ROOT/'configs/baseline_mof.json').read_text())
    if record['config'] != config:
        raise ValueError('Configuração dos pesos diverge do baseline')
    expected = json.loads((ROOT/'configs/checkpoints-mof.json').read_text())['files']
    actual = {}
    for path, digest in record['sha256'].items():
        p = Path(path)
        role = next((name for name, folder in record['paths'].items()
                     if p.parent == Path(folder)), 'dino')
        key = (role, p.name)
        if key in actual:
            raise ValueError(f'Artefato duplicado: {key}')
        actual[key] = digest
        if not p.is_file():
            raise FileNotFoundError(f'Checkpoint ausente: {p}; execute download novamente.')
        if check_hashes and file_digest(p) != digest:
            raise ValueError(f'Checkpoint corrompido: {p}')
    if actual != {(item['role'], item['name']): item['sha256'] for item in expected}:
        raise ValueError('Os arquivos não correspondem aos hashes do experimento original')
    return len(actual)
