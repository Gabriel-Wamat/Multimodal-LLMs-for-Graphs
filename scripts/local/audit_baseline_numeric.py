"""Audit the 70 numeric-format rejections of the original baseline; no rescoring."""
import csv
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.pilot import json_text, write_once
from src.evaluation.score import grade, read_jsonl
from src.models.control.artifacts import digest

RUN = 'mof13b-pilot50-v1'


def csv_target(question, records):
    graph = question['grafico']
    text = question['pergunta']
    if graph.startswith('BAR-'):
        match = re.fullmatch(r'Qual é a diferença entre (.+?) e (.+)\?', text)
        if not match: raise ValueError('Unsupported bar question')
        values = {r['categoria']:float(r['valor']) for r in records}
        a,b = match.groups()
        return abs(values[a]-values[b]), {'operation':'absolute difference','categories':[a,b],'values':[values[a],values[b]]}
    if graph.startswith('BOX-'):
        match = re.fullmatch(r'Qual é o intervalo interquartil do (.+)\?', text)
        if not match: raise ValueError('Unsupported box question')
        group = match.group(1)
        values = [float(r['valor']) for r in records if r['grupo']==group]
        q1,q3 = np.quantile(values,[0.25,0.75],method='linear').tolist()
        return q3-q1, {'operation':'Q3 minus Q1, linear quantiles','group':group,'n':len(values),'q1':q1,'q3':q3}
    if graph.startswith('HIS-'):
        match = re.fullmatch(r'Qual é a frequência da faixa ([\d.,]+)–([\d.,]+)\?', text)
        if not match: raise ValueError('Unsupported histogram question')
        lo,hi = [float(v.replace(',','.')) for v in match.groups()]
        found = [r for r in records if float(r['faixa_inicio'])==lo and float(r['faixa_fim'])==hi]
        if len(found)!=1: raise ValueError('Nonunique histogram interval')
        return float(found[0]['contagem']), {'operation':'count of requested interval','interval':[lo,hi]}
    raise ValueError('Unsupported question family')


def response_shape(question, response):
    text = response.strip()
    if question['grafico'].startswith('BOX-'):
        value = json.loads(text)
        if not isinstance(value,list) or len(value)!=4 or not all(type(x) in (int,float) and 0<=x<=1 for x in value):
            raise ValueError('Box response shape changed; manual review required')
        return 'four_values_between_zero_and_one', [float(x) for x in value], None
    if question['grafico'].startswith('HIS-'):
        match = re.fullmatch(r'([\d.,]+)\s*[-–]\s*([\d.,]+)',text)
        if not match: raise ValueError('Histogram response shape changed')
        values = [float(v.replace(',','.')) for v in match.groups()]
        return 'interval_instead_of_frequency', values, None
    values = [float(v.replace(',','.')) for v in re.findall(r'[+-]?\d+(?:[.,]\d+)?',text)]
    if len(values) not in [1,2]: raise ValueError('Bar response needs new review')
    implied = abs(values[0]-values[1]) if len(values)==2 else values[0]
    return 'bar_description_with_incompatible_values', values, implied


def main():
    folder = ROOT/'results'/RUN
    base = ROOT/'data/manifests/pilot50_v1'
    split = json.loads((base/'split.json').read_text())
    run = json.loads((folder/'run.json').read_text())
    complete = json.loads((folder/'complete.json').read_text())
    metric = json.loads((folder/'scored/metrics.json').read_text())
    if (digest(folder/'predictions.jsonl') != complete['predictions_sha256']
            or metric['predictions_sha256'] != complete['predictions_sha256']
            or digest(base/'labels.jsonl') != split['labels_sha256']
            or digest(base/'inputs.jsonl') != run['inputs_sha256']):
        raise ValueError('Baseline/input provenance differs')
    labels = read_jsonl(base/'labels.jsonl');truth = {r['id']:r for r in labels}
    inputs = {r['id']:r for r in read_jsonl(base/'inputs.jsonl')}
    if set(inputs)!=set(truth):raise ValueError('Input/label membership differs')
    raw = read_jsonl(folder/'predictions.jsonl')
    if len({r['id'] for r in raw})!=len(raw) or {r['id'] for r in raw}!=set(truth) or any(r['status']!='ok' for r in raw):
        raise ValueError('Baseline coverage differs')
    chosen = [r for r in raw if grade(truth[r['id']],r['response'])[1]=='invalid_numeric_format']
    if len(chosen)!=metric['reason_counts']['invalid_numeric_format']:
        raise ValueError('Rejected cohort changed')
    images = {r['id']:r for r in json.loads((ROOT/'data/manifests/imagens.json').read_text())['images']}
    rows=[];source_hashes={}
    for pred in chosen:
        q=truth[pred['id']];entry=inputs[q['id']];image=images[q['grafico']]
        csv_path=ROOT/image['csv'];png_path=ROOT/image['png']
        if digest(csv_path)!=image['csv_sha256'] or digest(png_path)!=image['png_sha256'] or entry['image_sha256']!=image['png_sha256'] or entry['grafico']!=q['grafico'] or (ROOT/entry['image']).resolve()!=png_path.resolve():
            raise ValueError('CSV or model input image changed')
        with csv_path.open(encoding='utf-8-sig',newline='') as stream: records=list(csv.DictReader(stream))
        value, calculation=csv_target(q,records)
        if not math.isclose(value,float(q['resposta']),abs_tol=1e-8) or not q['min']<=value<=q['max']:
            raise ValueError('CSV target differs: '+q['id'])
        shape,literals,implied=response_shape(q,pred['response'])
        compatible=[x for x in literals if q['min']<=x<=q['max']]
        rows.append({'id':q['id'],'graph':q['grafico'],'question':q['pergunta'],'response':pred['response'],
                     'target':q['resposta'],'min':q['min'],'max':q['max'],'csv_recomputed_target':value,
                     'csv_calculation':calculation,'response_shape':shape,'explicit_numeric_literals':literals,
                     'literals_in_target_interval':compatible,'implied_bar_value':implied,
                     'implied_bar_value_in_interval':q['min']<=implied<=q['max'] if implied is not None else None,
                     'csv':image['csv'],'csv_sha256':image['csv_sha256'],'image':image['png'],'image_sha256':image['png_sha256']})
        source_hashes[image['csv']]=image['csv_sha256'];source_hashes[image['png']]=image['png_sha256']
    paths=[folder/'run.json',folder/'complete.json',folder/'predictions.jsonl',folder/'scored/metrics.json',base/'labels.jsonl',base/'inputs.jsonl',ROOT/'data/manifests/imagens.json',ROOT/'src/evaluation/score.py']
    source_hashes.update({str(p.relative_to(ROOT)):digest(p) for p in paths})
    counts=dict(Counter(r['response_shape'] for r in rows))
    report={'scope':'posthoc descriptive audit of the complete numeric-format rejection cohort; not a replacement metric',
            'run_id':RUN,'cohort_rule':'all responses rejected as invalid_numeric_format by the frozen scorer',
            'cohort_size':len(rows),'csv_targets_verified':len(rows),'shape_counts':counts,
            'responses_with_explicit_literal_in_target_interval':sum(bool(r['literals_in_target_interval']) for r in rows),
            'numpy_version':np.__version__,'items':rows,'sources':source_hashes,'auditor_sha256':digest(Path(__file__)),
            'limitations':['Error-selected cohort, not an unbiased sample of all questions.',
                           'No interpretation of normalized vectors as geometry, rescaling, number words or unit conversion.',
                           'No full semantic review of 3173 answers or every answer format; no new accuracy estimate.',
                           'Response-shape rules were defined after inspecting this cohort; original grades remain unchanged.']}
    output=folder/'diagnostics/numeric_format_audit_v1.json'
    write_once(output,json_text(report))
    lines=['# Auditoria dos erros numéricos de formato — baseline original','',
           f'Coorte completa dos **{len(rows)}** casos rejeitados como `invalid_numeric_format`. '
           'Os alvos dos 70 casos foram recalculados dos CSVs e conferidos com os limites do gabarito; fontes e PNGs tiveram hashes verificados.','',
           '| Resposta observada | Casos |','|---|---:|']
    names={'four_values_between_zero_and_one':'Vetor de quatro valores entre 0 e 1 no lugar de IQR escalar',
           'interval_instead_of_frequency':'Faixa numérica no lugar da frequência solicitada',
           'bar_description_with_incompatible_values':'Descrição de barras com valores incompatíveis'}
    lines += [f'| {names[k]} | {v} |' for k,v in counts.items()]
    lines += ['',f'**{report["responses_with_explicit_literal_in_target_interval"]}/{len(rows)}** respostas contêm um número literal no intervalo aceito para o alvo. '
              'A contagem é um diagnóstico de conteúdo literal, não uma nova métrica semântica. Interpretar os vetores como caixas geométricas ou converter escalas exigiria outro contrato de avaliação.','',
              '## Exemplos conferidos','']
    for qid in ['BAR-005-N3-1','BOX-001-N3-1','HIS-106-N1-1']:
        r=next(r for r in rows if r['id']==qid)
        lines += [f'- `{qid}`: {r["question"]} Alvo do CSV: {r["csv_recomputed_target"]:g}; resposta: `{r["response"]}`.']
    lines += ['', '## Limite da conclusão','',
              'Esta coorte não sustenta a explicação de que os 70 erros sejam apenas um número correto envolvido por texto. '
              'Ela também não prova erro perceptivo, independência dos dados ou validade semântica do corretor em todos os outros casos. '
              'Separadores ambíguos, números por extenso, unidades e demais formatos não foram revisados integralmente.','',
              'O diagnóstico anterior Yes/No → Sim/Não continua separado: 119/3.173 (3,75%) na regra original e '
              '311/3.173 (9,80%) nessa análise posterior de idioma. Nenhuma previsão ou pontuação primária foi alterada.','',
              'Reprodução: `.venv/bin/python scripts/local/audit_baseline_numeric.py`. '
              'Itens, cálculos e hashes: `diagnostics/numeric_format_audit_v1.json`.']
    write_once(folder/'NUMERIC_FORMAT_AUDIT.md','\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['cohort_size','csv_targets_verified','shape_counts','responses_with_explicit_literal_in_target_interval']},ensure_ascii=False))


if __name__=='__main__':main()
