"""Baixa revisões fixas e registra hashes dos pesos. Executar no nó de instalação."""
import hashlib
import json
import sys
import re
from pathlib import Path
import torch
from huggingface_hub import snapshot_download, hf_hub_download, get_hf_file_metadata, hf_hub_url
from huggingface_hub.constants import HF_HUB_CACHE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.artifacts import weights_manifest_path, verify_weights, verify_repositories
cfg = json.loads((ROOT/'configs/baseline_mof.json').read_text())
verify_repositories(cfg)
# Verifica dependências da classe oficial antes de continuar os downloads grandes.
sys.path.insert(0, str(ROOT/'work/vendor/MMVP/LLaVA'))
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
print(f'Importação da arquitetura oficial: {LlavaLlamaForCausalLM.__name__} OK.', flush=True)
paths = {}
for name, repo, revision in [('model',cfg['model_id'],cfg['model_revision']),
                              ('clip',cfg['clip_id'],cfg['clip_revision'])]:
    filenames = ([f'pytorch_model-{i:05d}-of-00003.bin' for i in range(1,4)]
                 if name == 'model' else ['pytorch_model.bin'])
    for filename in filenames:
        metadata = get_hf_file_metadata(hf_hub_url(repo, filename, revision=revision))
        partial = Path(HF_HUB_CACHE)/('models--'+repo.replace('/','--'))/'blobs'/(metadata.etag+'.incomplete')
        if partial.exists() and metadata.size is not None and partial.stat().st_size > metadata.size:
            print(f'Removendo apenas fragmento maior que o arquivo oficial: {filename}', flush=True)
            partial.unlink()
    for attempt in range(4):
        try:
            paths[name] = snapshot_download(repo_id=repo, revision=revision,
                allow_patterns=['*.json','*.model','pytorch_model*.bin'],
                ignore_patterns=['training_args.bin','trainer_state.json'], max_workers=3)
            break
        except OSError as exc:
            # Alguns gateways não retomam Range corretamente; reinicia só o shard inválido.
            match = re.search(r'Consistency check failed:.*\((pytorch_model[^()]*\.bin)\)', str(exc))
            if match is None or attempt == 3:
                raise
            print(f'Reiniciando arquivo inconsistente: {match[1]}', flush=True)
            hf_hub_download(repo_id=repo, revision=revision, filename=match[1], force_download=True)
# O forward oficial usa x_prenorm; manter a implementação original do DINOv2.
torch.hub.load(str(ROOT/'work/vendor/dinov2'), 'dinov2_vitl14', source='local', pretrained=True)
digests = {}
folders = {name:Path(path) for name,path in paths.items()}
folders['dino'] = Path(torch.hub.get_dir())/'checkpoints'
for expected in json.loads((ROOT/'configs/checkpoints-mof.json').read_text())['files']:
    path = folders[expected['role']]/expected['name']
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    digest = h.hexdigest()
    # O nome do blob LFS no cache oficial é seu SHA-256.
    blob = path.resolve().name
    if re.fullmatch(r'[0-9a-f]{64}',blob) and digest != blob:
        raise RuntimeError(f'Hash LFS incorreto: {path.name}')
    digests[str(path)] = digest
record = {'paths': paths, 'sha256': digests, 'config': cfg}
verify_weights(record, check_hashes=False)  # Todos os bytes acabaram de ser lidos acima.
destination = weights_manifest_path()
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix('.json.tmp')
temporary.write_text(json.dumps(record, indent=2)+'\n')
temporary.replace(destination)
print('Pesos preparados e hashes registrados.', flush=True)
