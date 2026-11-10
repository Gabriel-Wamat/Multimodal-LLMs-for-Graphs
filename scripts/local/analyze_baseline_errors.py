"""Diagnóstico pós-hoc de idioma, sem alterar previsões ou métricas primárias."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.pilot import sha, write_once, json_text
from src.evaluation.score import read_jsonl, norm


def main():
    folder = ROOT/'results/mof13b-pilot50-v1'
    metrics = json.loads((folder/'scored/metrics.json').read_text())
    if not metrics['all_records_successful']:
        raise ValueError('Execução incompleta')
    if sha(folder/'predictions.jsonl') != metrics['predictions_sha256']:
        raise ValueError('Previsões alteradas')
    base = ROOT/'data/manifests/pilot50_v1'
    split = json.loads((base/'split.json').read_text())
    if sha(base/'labels.jsonl') != split['labels_sha256']:
        raise ValueError('Gabarito alterado')
    labels = {q['id']: q for q in read_jsonl(base/'labels.jsonl')}
    predictions = read_jsonl(folder/'predictions.jsonl')
    scored = {q['id']: q for q in read_jsonl(folder/'scored/answers_scored.jsonl')}
    translations = {'yes': 'sim', 'no': 'nao'}
    candidates = [p for p in predictions if labels[p['id']]['formato']=='sim/não'
                  and norm(p['response']) in translations]
    added = [p['id'] for p in candidates if not scored[p['id']]['correct'] and
             translations[norm(p['response'])]==norm(labels[p['id']]['resposta'])]
    result = {
        'purpose': 'posthoc_language_sensitivity_not_primary_metric',
        'rule': 'Only exact yes/no -> sim/não for binary questions; all other grades unchanged.',
        'predictions_sha256': metrics['predictions_sha256'],
        'primary_correct': metrics['correct'], 'primary_accuracy': metrics['accuracy'],
        'binary_english_responses': len(candidates), 'additional_correct': len(added),
        'diagnostic_correct': metrics['correct']+len(added), 'n': metrics['n'],
        'diagnostic_accuracy': (metrics['correct']+len(added))/metrics['n'],
        'newly_correct_ids': added,
    }
    out = folder/'diagnostics'
    out.mkdir(exist_ok=True)
    write_once(out/'binary_aliases_posthoc.json', json_text(result))
    pct = f"{100*result['diagnostic_accuracy']:.2f}%".replace('.', ',')
    report = f'''# Diagnóstico posterior à avaliação: idioma das respostas

A métrica primária permanece **3,75% (119/3.173)**, segundo o protocolo fixado antes
da inferência. Nas 360 perguntas de formato sim/não, o modelo respondeu em inglês:
84 vezes Yes e 276 vezes No. O corretor estrito registrou todas como incorretas.

Ao aceitar apenas os equivalentes exatos Yes→Sim e No→Não nessas perguntas,
{len(added)} respostas adicionais correspondem ao gabarito. A acurácia diagnóstica
passa a **{pct} ({result['diagnostic_correct']}/{result['n']})**. Nenhuma inferência
foi repetida e nenhum outro formato recebeu tratamento adicional.

Esta análise foi definida **após observar as respostas**. Não substitui a métrica
primária nem constitui um novo resultado previamente especificado. Mostra que o
resultado estrito inclui descumprimento do idioma solicitado, além de erros de
leitura e raciocínio. Não atribuir toda a diferença à percepção visual.

Nas perguntas numéricas com tolerância ±5%, houve 12 acertos em 1.440 (0,83%).
Logo, o problema observado não se limita às respostas binárias em inglês. Não foi
realizada revisão semântica completa de todas as respostas e rótulos.

Para a comparação futura, preservar os resultados primários. Se forem adotados
sinônimos bilíngues, versionar essa regra antes de avaliar o candidato e aplicá-la
igualmente às previsões dos dois modelos, reportando a mudança de protocolo.

Reprodução: `.venv/bin/python scripts/local/analyze_baseline_errors.py`.
Detalhes e IDs afetados: `diagnostics/binary_aliases_posthoc.json`.
'''
    write_once(folder/'DIAGNOSTICS.md', report)
    print(json.dumps({k:v for k,v in result.items() if k!='newly_correct_ids'},indent=2))


if __name__=='__main__':
    main()
