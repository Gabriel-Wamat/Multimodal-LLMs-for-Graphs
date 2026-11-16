"""Summarize completed paired analyses without recomputing or changing their metrics."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def interval(record, scale=100):
    estimate = record.get('estimate', record.get('rho'))
    if estimate is None:
        return 'não definido'
    ci = record.get('ci95')
    text = f'{scale * estimate:.2f}'
    return text + (f' [{scale*ci[0]:.2f}; {scale*ci[1]:.2f}]' if ci else ' [IC indisponível]')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', action='store_true', help='Export aggregate hypothesis results for the notebook')
    args = parser.parse_args()
    lines = ['# Avaliação pareada completa — análise exploratória', '',
             '720 pares A/B por modelo, 1.440 imagens e três condições; 4.320 respostas por modelo. '
             'Acurácia por par exige ambas as respostas corretas. Intervalos de 95% reamostram famílias inteiras.', '',
             '## Resultados por condição', '',
             '| Modelo | Condição | Acerto por pergunta (%) | Acerto por par (%) [IC95%] | Respostas inválidas |',
             '|---|---|---:|---:|---:|']
    sources = {}
    analyses = {}
    costs = {}
    for model in ['llava', 'qwen']:
        folder = ROOT / 'results' / ('paired-mitigations-full-v1-' + model)
        complete = read(folder/'complete.json')
        pred_hash = digest(folder/'predictions.jsonl')
        if pred_hash != complete['predictions_sha256'] or complete['records'] != 4320:
            raise ValueError('Incomplete or changed predictions: ' + model)
        hypothesis = read(folder/'scored/hypotheses.json')
        if hypothesis['provenance']['predictions'] != pred_hash:
            raise ValueError('Analysis prediction provenance changed')
        analyses[model] = hypothesis
        sources[model] = {name:digest(folder/name) for name in ['run.json','complete.json','scored/metrics.json','scored/hypotheses.json','predictions.jsonl']}
        metrics = read(folder/'scored/metrics.json')
        costs[model] = metrics['costs']
        for condition, value in metrics['metrics'].items():
            lines.append(f"| {model} | {condition} | {100*value['question_accuracy']:.2f} | {interval(value['paired_accuracy'])} | {value['invalid_responses']} |")
    lines += ['', '## H2 — alterações amplas menos difíceis que sutis', '',
              'Diferença de acurácia por par: ampla menos sutil, em pontos percentuais. Origens correspondentes são comparadas.', '',
              '| Modelo | Condição | Diferença (pp) [IC95%] |', '|---|---|---:|']
    for model,h in analyses.items():
        for c,v in h['H2_and_similarity_error'].items():
            lines.append(f"| {model} | {c} | {interval(v['H2']['all'])} |")
    lines += ['', 'No Qwen original, a diferença positiva é compatível com maior dificuldade nas alterações sutis. '
              'O LLaVA tem desempenho constante nulo por par; esse piso limita a interpretação de H2.', '',
              '## H3 — efeito das mitigações', '',
              'Diferença de acurácia por par: mitigação menos imagem original, em pontos percentuais.', '',
              '| Modelo | Mitigação | Diferença (pp) [IC95%] |', '|---|---|---:|']
    for model,h in analyses.items():
        for c,v in h['H3'].items():
            lines.append(f"| {model} | {c} | {interval(v['paired_delta'])} |")
    lines += ['', 'As duas mitigações reduziram o desempenho do Qwen neste protocolo. Isso não demonstra que toda estratégia de recorte ou transcrição seja prejudicial. '
              'As abstenções e respostas inválidas dos recortes integram a métrica; os testes de McNemar por atributo/intensidade permanecem nos JSONs originais, com p-valores exploratórios sem ajuste múltiplo.', '',
              '## Similaridade e erro', '', '| Modelo | Condição | Encoder | Spearman [IC95%] |', '|---|---|---|---:|']
    for model,h in analyses.items():
        for c,v in h['H2_and_similarity_error'].items():
            for encoder,r in v['similarity_error']['all'].items():
                lines.append(f'| {model} | {c} | {encoder} | {interval(r,1)} |')
    lines += ['', 'Correlação não estabelece causalidade. No LLaVA, o erro por par é constante, portanto a correlação não é definida. '
              'No Qwen com recortes, muitos reamostramentos têm correlação indefinida; o IC foi omitido, não substituído por zero.', '',
              '## H1 — representações', '',
              'As mesmas imagens foram codificadas por CLIP e DINOv3. Limiares foram calibrados nos pares de desenvolvimento; todos os 720 pares de avaliação foram preservados. '
              'Alta similaridade global não comprova incapacidade de separar uma diferença relevante. Consulte `results/representations-pairs_v1-v1/similarity_report.json` e `plots/similarity.png`. '
              'O filtro histórico com DINOv2 não é uma reprodução equivalente usando DINOv3.', '',
              '## Limites e trabalho restante', '',
              'Resultados exploratórios em imagens sintéticas derivadas dos CSVs, com famílias conhecidas agrupadas. '
              'A independência de origem não foi certificada. A revisão visual da amostra fixa deve ser consultada em experiments/five-stage-campaign/READING_REVIEW_SUMMARY.md; erro na resposta final não comprova erro de leitura. '
              'A avaliação legada do piloto original e esta tarefa binária pareada têm regras e denominadores diferentes. '
              'Treinamentos e comparação antes/depois ainda não estão concluídos.', '',
              'As fontes e hashes deste resumo estão em `sources.json`. Reproduza com `.venv/bin/python scripts/local/report_paired_results.py`.']
    out = ROOT/'results/paired-comparison-v1'
    out.mkdir(exist_ok=True)
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    if args.public:
        public = {}
        for model, hypothesis in analyses.items():
            public[model] = {
                'sources': sources[model],
                'costs': costs[model],
                'H2_and_similarity_error': {
                    condition: {'H2': value['H2'], 'spearman': value['similarity_error']['all']}
                    for condition, value in hypothesis['H2_and_similarity_error'].items()
                },
                'H3': {
                    condition: {
                        'paired_delta': value['paired_delta'],
                        'multiplicity': value['multiplicity'],
                        'mcnemar': {
                            group: {key: record[key] for key in ('lost', 'gained', 'n', 'p_exact_two_sided')}
                            for group, record in value['mcnemar_by_attribute_intensity'].items()
                        },
                    } for condition, value in hypothesis['H3'].items()
                },
            }
        target = ROOT / 'results/published/paired-hypotheses-v1.json'
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(public, ensure_ascii=False, indent=2) + '\n')
    print(out/'REPORT.md')

if __name__ == '__main__':
    main()
