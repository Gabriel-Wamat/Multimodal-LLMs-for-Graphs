"""Congela metade dos gráficos por tipo/estilo, separando entradas dos alvos."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261102


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ValueError(f'Artefato congelado difere: {path}')
    else:
        path.write_text(text, encoding='utf-8')


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def select_half(images, seed=SEED):
    strata = defaultdict(list)
    for row in images:
        strata[(row['id'][:3], row['style'])].append(row['id'])
    selected = []
    for group in sorted(strata):
        ids = strata[group]
        if len(ids) % 2:
            raise ValueError(f'Estrato ímpar: {group}')
        ranked = sorted(ids, key=lambda i: hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest())
        selected.extend(ranked[:len(ids)//2])
    return sorted(selected)


def main():
    manifest_path = ROOT / 'data/manifests/imagens.json'
    labels_path = ROOT / 'dados_projeto_multi_modais/dados/gabarito.json'
    manifest = json.loads(manifest_path.read_text())
    questions = json.loads(labels_path.read_text())
    images = {r['id']: r for r in manifest['images']}
    if len(images) != 720 or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Inventário inesperado ou perguntas duplicadas')
    selected = set(select_half(list(images.values())))
    reserved = sorted(set(images)-selected)
    inputs, labels = [], []
    for q in sorted(questions, key=lambda q: q['id']):
        if q['grafico'] not in selected:
            continue
        image = images[q['grafico']]
        inputs.append({k: q[k] for k in ('id', 'grafico', 'tipo', 'estilo', 'nivel', 'pergunta', 'formato')}
                      | {'image': image['png'], 'image_sha256': image['png_sha256']})
        labels.append(q)
    for graph in selected:
        r = images[graph]
        if sha(ROOT/r['png']) != r['png_sha256'] or sha(ROOT/r['csv']) != r['csv_sha256']:
            raise ValueError(f'Dados alterados: {graph}')
    base = ROOT/'data/manifests/pilot50_v1'
    for name, rows in [('inputs.jsonl', inputs), ('labels.jsonl', labels)]:
        write_once(base/name, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))
    split = {
        'version': 'pilot50-v1', 'seed': SEED, 'unit': 'graph',
        'selection': 'sha256(seed:id), first half within chart_type x style',
        'purpose': 'exploratory_baseline_and_future_paired_comparison',
        'selected_graphs': sorted(selected), 'reserved_graphs': reserved,
        'graphs_evaluated': len(selected), 'questions_evaluated': len(inputs),
        'by_type': dict(Counter(q['tipo'] for q in inputs)),
        'by_style': dict(Counter(q['estilo'] for q in inputs)),
        'source_manifest_sha256': sha(manifest_path), 'source_labels_sha256': sha(labels_path),
        'inputs_sha256': sha(base/'inputs.jsonl'), 'labels_sha256': sha(base/'labels.jsonl'),
        'eligibility': 'No local training; source ancestry unknown. Remaining graphs are not certified unseen.',
    }
    write_once(base/'split.json', json_text(split))
    print(f"Amostra congelada: {len(selected)} gráficos, {len(inputs)} perguntas; {len(reserved)} gráficos reservados.")


if __name__ == '__main__':
    main()
