"""Inferência retomável: só inputs.jsonl e PNGs entram no modelo."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

from src.data.pilot import ROOT, sha, write_once, json_text
from src.evaluation.score import read_jsonl
from src.models.artifacts import weights_manifest_path


def prompt_for(q):
    return ('Responda em português usando apenas o gráfico fornecido. '
            'Dê somente a resposta final, sem explicação. '
            f"Formato: {q['formato']}. "
            'Para números, use algarismos, vírgula ou ponto decimal e nenhum separador de milhares. '
            'Para listas, separe os nomes por vírgulas na ordem solicitada.\n'
            f"Pergunta: {q['pergunta']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke', action='store_true', help='Uma pergunta por tipo, apenas teste de execução.')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('run-id deve ser um nome simples')
    cfg_path = ROOT/'configs/baseline_mof.json'
    cfg = json.loads(cfg_path.read_text())
    base = ROOT/'data/manifests/pilot50_v1'
    split = json.loads((base/'split.json').read_text())
    if sha(base/'inputs.jsonl') != split['inputs_sha256']:
        raise ValueError('Entradas alteradas desde o congelamento')
    rows = read_jsonl(base/'inputs.jsonl')
    if args.smoke:
        chosen = {}
        for r in rows:
            chosen.setdefault(r['tipo'], r)
        rows = list(chosen.values())
    allowed = {'id','grafico','tipo','estilo','nivel','pergunta','formato','image','image_sha256'}
    if any(set(r)!=allowed for r in rows):
        raise ValueError('Campos inesperados nas entradas de inferência')
    for image, digest in {(r['image'],r['image_sha256']) for r in rows}:
        if sha(ROOT/image) != digest:
            raise ValueError(f'PNG alterado: {image}')
    output = ROOT/'results'/args.run_id
    output.mkdir(parents=True, exist_ok=True)
    sources = ['src/models/mof.py','src/models/artifacts.py','src/evaluation/run.py']
    stamp = {'config':cfg, 'config_sha256':sha(cfg_path),'split_sha256':sha(base/'split.json'),
             'inputs_sha256':sha(base/'inputs.jsonl'), 'smoke':args.smoke, 'expected_questions':len(rows),
             'source_sha256':{p:sha(ROOT/p) for p in sources},
             'prompt_sha256':hashlib.sha256(''.join(prompt_for(r) for r in rows).encode()).hexdigest(),
             'weights_manifest_sha256':sha(weights_manifest_path())}
    write_once(output/'run.json',json_text(stamp))
    write_once(output/'weights.json',weights_manifest_path().read_text())
    pred_path = output/'predictions.jsonl'
    previous = read_jsonl(pred_path) if pred_path.exists() else []
    if len({p['id'] for p in previous})!=len(previous) or set(p['id'] for p in previous)-set(r['id'] for r in rows):
        raise ValueError('IDs duplicados ou inesperados na retomada')
    done = {p['id'] for p in previous if p['status']=='ok'}
    if len(done)==len(rows):
        write_once(output/'complete.json',json_text({'questions':len(done),'predictions_sha256':sha(pred_path)}))
        print('Todas as respostas já existem; nenhuma inferência repetida.')
        return
    if len(done)!=len(previous):
        raise RuntimeError('Há falhas registradas: preservar esta execução e usar um novo run-id.')
    import torch
    import numpy as np
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory<32*1024**3:
        raise RuntimeError('Este protocolo FP16 requer GPU CUDA com pelo menos 32 GiB; preferir A100 80 GB.')
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from src.models.mof import MoF
    model = MoF(cfg)
    metadata = model.metadata | {'packages':{p:importlib.metadata.version(p) for p in
                   ['torch','torchvision','transformers','accelerate','numpy','Pillow']}}
    write_once(output/'runtime.json',json_text(metadata))
    with pred_path.open('a',encoding='utf-8') as stream:
        for q in rows:
            if q['id'] in done:
                continue
            prompt = prompt_for(q)
            start = time.monotonic()
            response = model.answer(ROOT/q['image'],prompt)
            record = {'id':q['id'], 'grafico':q['grafico'], 'prompt':prompt, 'response':response,
                      'status':'ok', 'seconds':time.monotonic()-start}
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush(); os.fsync(stream.fileno())
            done.add(q['id'])
            print(f"{len(done)}/{len(rows)} {q['id']}: {response}",flush=True)
    write_once(output/'complete.json',json_text({'questions':len(done),'predictions_sha256':sha(pred_path)}))


if __name__ == '__main__':
    main()
