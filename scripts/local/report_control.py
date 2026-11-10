"""Audit and summarize a completed full control run, preserving raw predictions."""
import argparse
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.pilot import json_text, write_once
from src.evaluation.control.score import normalize_binary
from src.evaluation.score import grade, read_jsonl
from src.models.control.artifacts import digest
from scripts.local.report_campaign import generation_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_id):
        raise ValueError('Invalid run ID')
    folder = ROOT / 'results' / args.run_id
    run = json.loads((folder/'run.json').read_text())
    complete = json.loads((folder/'complete.json').read_text())
    metrics = json.loads((folder/'scored/metrics.json').read_text())
    if run['smoke'] or metrics['smoke']:
        raise ValueError('Technical smoke runs have their own validation report')
    split_path = ROOT / run['config']['split']
    labels_path = ROOT / run['config']['labels']
    split = json.loads(split_path.read_text())
    inputs_path = ROOT / run['config']['inputs']
    source = {'run_id': args.run_id, 'run_sha256': digest(folder/'run.json'),
              'predictions_sha256': digest(folder/'predictions.jsonl')}
    if (source['predictions_sha256'] != complete['predictions_sha256']
            or source['predictions_sha256'] != metrics['predictions_sha256']
            or source['run_sha256'] != metrics['run_sha256']
            or digest(split_path) != run['split_sha256']
            or digest(labels_path) != split['labels_sha256']
            or digest(inputs_path) != run['inputs_sha256']
            or metrics['model'] != run['model']):
        raise ValueError('Run, predictions or fixed pilot provenance differs')
    for name in ['src/evaluation/score.py', 'src/evaluation/control/score.py']:
        if digest(ROOT/name) != run['source_sha256'][name]:
            raise ValueError('Current recount scorer differs from frozen source')
    if metrics['scorer_sha256'] != digest(ROOT/'src/evaluation/control/score.py'):
        raise ValueError('Stored scorer provenance differs')
    archive = folder/'source'
    for name, expected in run['source_sha256'].items():
        if digest(archive/name) != expected:
            raise ValueError('Archived source changed: '+name)
    labels = read_jsonl(labels_path)
    truth = {r['id']: r for r in labels}
    raw = read_jsonl(folder/'predictions.jsonl')
    expected = {(qid, arm) for qid in truth for arm in run['arms']}
    observed = {(r['id'], r['arm']) for r in raw}
    if (len(raw) != len(observed) or observed != expected
            or set(run['item_ids']) != set(truth)
            or complete['records'] != len(raw) or complete['items'] != len(truth)
            or len(truth) != split['questions_evaluated']
            or any(r['status'] != 'ok' for r in raw)):
        raise ValueError('Question/arm membership incomplete or duplicated')
    caps = generation_limits([source])
    lines = [f'# Controle completo — {run["model"].upper()}', '',
             f'{len(raw):,} respostas; {len(truth):,} perguntas do piloto fixo por braço. '
             'Resultado exploratório antes do alinhamento dos novos projetores.', '',
             '| Braço | Acertos / perguntas | Principal (%) | Estrita (%) | Respostas no teto | Horas observadas | Pico GPU GiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    recount = {}
    for arm in run['arms']:
        rows = [r for r in raw if r['arm'] == arm]
        normalized = normalize_binary(labels, rows)
        hits = sum(grade(truth[r['id']], r['response'])[0] for r in normalized)
        strict = sum(grade(truth[r['id']], r['response'])[0] for r in rows)
        saved = metrics['metrics'][arm]
        for key, count in [('primary_binary_aliases_v2', hits), ('strict_v1', strict)]:
            value = saved[key]
            if (value['n'] != len(truth) or value['correct'] != count
                    or not value['all_records_successful']
                    or not math.isclose(value['accuracy'], count/len(truth), abs_tol=1e-15)):
                raise ValueError('Recount differs: '+arm+'/'+key)
        recount[arm] = {'primary_correct': hits, 'strict_correct': strict, 'n': len(truth)}
        cost = metrics['costs'][arm]
        lines.append(f'| {arm} | {hits}/{len(truth)} | {100*hits/len(truth):.2f} | '
                     f'{100*strict/len(truth):.2f} | {caps[arm]["at_limit"]}/{caps[arm]["n"]} | '
                     f'{cost["total_seconds_observed"]/3600:.2f} | {cost["peak_gpu_bytes"]/2**30:.2f} |')
    lines += ['', '## Regra e interpretação', '',
              'A métrica principal utiliza o corretor por formato do piloto: comparação normalizada, '
              'tolerância numérica do gabarito e equivalentes exatos Yes/No → Sim/Não apenas nas perguntas binárias. '
              'A coluna estrita preserva a regra original. A recontagem verifica aplicação do mesmo corretor; '
              'não constitui revisão semântica dos alvos e respostas, nem BERTScore.', '',
              'Os braços com novos projetores ainda não foram alinhados. Resultados de fusão não reproduzem '
              'o I-MoF 13B treinado do artigo. No LLaVA, CLIP+CLIP e duplicação nativa são controles equivalentes '
              'de tokens; no Qwen, o controle nativo usa o encoder próprio. '
              'Atingir o teto de geração não comprova erro de conteúdo ou truncamento, isoladamente.', '',
              'Horas são a soma dos tempos observados nas respostas; não incluem toda a espera, retomada '
              'ou custo de parede. Cache, carregamento e condições do dispositivo afetam os tempos. '
              'Não são um benchmark de throughput aquecido.', '',
              'Os contrastes e intervalos armazenados em `scored/metrics.json` reamostram gráficos '
              'dentro de tipo/estilo. A comparação final antes/depois reamostra famílias inteiras. '
              'O piloto foi previamente explorado; a origem independente dos dados não foi certificada.', '',
              'A avaliação dos checkpoints treinados e a comparação final da campanha devem ser consultadas '
              'separadamente. Este relatório comprova somente o controle completo deste modelo.', '',
              f'Reprodução: `.venv/bin/python scripts/local/report_control.py {args.run_id}`. '
              'Hashes e recontagem: `REPORT_SOURCES.json`.']
    paths = [folder/'run.json', folder/'complete.json', folder/'predictions.jsonl',
             folder/'scored/metrics.json', split_path, inputs_path, labels_path]
    provenance = {'sources': {str(p.relative_to(ROOT)): digest(p) for p in paths},
                  'renderer_sha256': digest(Path(__file__)), 'recount': recount,
                  'generation_caps': caps, 'scope': 'full control for one model; not whole campaign completion'}
    write_once(folder/'REPORT_SOURCES.json', json_text(provenance))
    write_once(folder/'REPORT.md', '\n'.join(lines)+'\n')
    print(folder/'REPORT.md')


if __name__ == '__main__':
    main()
