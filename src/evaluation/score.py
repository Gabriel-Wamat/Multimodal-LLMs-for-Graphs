"""Correção determinística v1: limites do gabarito e correspondência normalizada."""
import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

from src.data.pilot import ROOT, sha, json_text


def norm(value):
    text = unicodedata.normalize('NFKD', str(value).strip().casefold())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return ' '.join(text.replace('−', '-').split()).strip(' ."\'')


def number(value):
    text = norm(value).removesuffix('%').strip()
    if re.fullmatch(r'[+-]?\d{1,3}(?:\.\d{3})+,\d+', text):
        text = text.replace('.', '')
    # Sem extrair números de explicações, nem aceitar separadores ambíguos.
    if not re.fullmatch(r'[+-]?\d+(?:[.,]\d+)?', text):
        return None
    result = float(text.replace(',', '.'))
    return result if math.isfinite(result) else None


def canonical(value, fmt):
    if fmt in ('número', 'número inteiro', 'ano') or fmt.startswith('−0,9'):
        return number(value)
    if fmt == 'lista':
        return tuple(norm(s) for s in re.split(r'\s*[,;>]\s*', str(value)))
    if fmt == 'faixa':
        text = norm(value).replace('–', ' a ')
        match = re.fullmatch(r'([+-]?\d+(?:[.,]\d+)?)\s*(?:a|-)\s*([+-]?\d+(?:[.,]\d+)?)', text)
        return tuple(number(s) for s in match.groups()) if match else None
    return norm(value)


def grade(question, response):
    if response is None or not str(response).strip():
        return False, 'empty'
    if question['correcao'] == '±5%':
        value = number(response)
        if value is None:
            return False, 'invalid_numeric_format'
        return question['min'] <= value <= question['max'], 'numeric_interval'
    if question['correcao'] != 'exata':
        raise ValueError(f"Correção desconhecida: {question['correcao']}")
    value = canonical(response, question['formato'])
    target = canonical(question['resposta'], question['formato'])
    if target is None:
        raise ValueError(f"Alvo não interpretável: {question['id']}")
    return value is not None and value == target, 'exact_normalized'


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(labels, predictions):
    truth = {q['id']: q for q in labels}
    pred = {p['id']: p for p in predictions}
    if len(truth) != len(labels) or len(pred) != len(predictions):
        raise ValueError('IDs duplicados')
    if set(pred)-set(truth):
        raise ValueError('Previsões fora da amostra')
    details = []
    for q in labels:
        p = pred.get(q['id'])
        ok, reason = grade(q, p.get('response')) if p and p.get('status') == 'ok' else (False, 'missing_or_failed')
        details.append({'id': q['id'], 'grafico': q['grafico'], 'tipo': q['tipo'],
                        'estilo': q['estilo'], 'nivel': q['nivel'], 'formato': q['formato'],
                        'correcao': q['correcao'], 'correct': bool(ok), 'reason': reason,
                        'response': p.get('response') if p else None, 'target': q['resposta'],
                        'min': q['min'], 'max': q['max']})
    def stats(rows):
        count = len(rows)
        hits = sum(r['correct'] for r in rows)
        return {'n': count, 'correct': hits, 'accuracy': hits/count if count else None}
    groups = defaultdict(list)
    for r in details:
        groups[r['grafico']].append(r)
    summary = {'scorer_version': 'sih-score-v1', **stats(details),
               'prediction_records': len(predictions),
               'successful_responses': sum(p.get('status') == 'ok' for p in predictions),
               'missing_or_failed': sum(r['reason'] == 'missing_or_failed' for r in details),
               'all_records_successful': len(pred)==len(truth) and all(p.get('status')=='ok' for p in predictions),
               'graph_macro_accuracy': sum(stats(r)['accuracy'] for r in groups.values())/len(groups),
               'fully_correct_graphs': sum(all(r['correct'] for r in rows) for rows in groups.values()),
               'graph_count': len(groups), 'reason_counts': dict(Counter(r['reason'] for r in details))}
    for key in ('tipo', 'estilo', 'nivel', 'formato', 'correcao'):
        summary['by_'+key] = {v: stats([r for r in details if r[key]==v]) for v in sorted({r[key] for r in details})}
    # Reamostra gráficos inteiros dentro de tipo x estilo, preservando dependência entre perguntas.
    import numpy as np
    strata = defaultdict(list)
    for rows in groups.values():
        strata[(rows[0]['tipo'], rows[0]['estilo'])].append((sum(r['correct'] for r in rows), len(rows)))
    rng = np.random.default_rng(20261102)
    hits, totals = np.zeros(2000), np.zeros(2000)
    for rows in strata.values():
        a = np.asarray(rows)
        sample = a[rng.integers(0, len(a), size=(2000, len(a)))]
        hits += sample[:,:,0].sum(axis=1)
        totals += sample[:,:,1].sum(axis=1)
    summary['accuracy_ci95_graph_bootstrap'] = np.quantile(hits/totals, [.025,.975]).tolist()
    summary['ci_scope'] = 'Exploratory, conditional on graph independence; stratified by type/style.'
    return summary, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = ROOT/'data/manifests/pilot50_v1'
    split = json.loads((base/'split.json').read_text())
    if sha(base/'labels.jsonl') != split['labels_sha256']:
        raise ValueError('Gabarito alterado desde o congelamento')
    summary, details = evaluate(read_jsonl(base/'labels.jsonl'), read_jsonl(args.predictions))
    summary.update(split_sha256=sha(base/'split.json'), predictions_sha256=sha(args.predictions),
                   scorer_sha256=sha(Path(__file__)))
    run_file = args.predictions.parent/'run.json'
    if run_file.exists():
        summary['run_manifest_sha256'] = sha(run_file)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'metrics.json').write_text(json_text(summary))
    (args.output/'answers_scored.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in details))
    print(json_text(summary))


if __name__ == '__main__':
    main()
