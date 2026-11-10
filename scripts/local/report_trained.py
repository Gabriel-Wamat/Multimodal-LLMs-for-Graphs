"""Audit one completed trained run against its matched unaligned control."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.pilot import json_text, write_once
from src.evaluation.compare_alignment import load_run, contrast
from src.evaluation.score import read_jsonl
from src.models.control.artifacts import digest
from scripts.local.report_campaign import generation_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.run_id):
        raise ValueError('Invalid run ID')
    folder = ROOT/'results'/args.run_id
    run = json.loads((folder/'run.json').read_text())
    training = ROOT/'experiments/training'/run['training_run']
    train_run = json.loads((training/'run.json').read_text())
    train_complete = json.loads((training/'complete.json').read_text())
    history = json.loads((training/'history.json').read_text())
    chosen = max(history, key=lambda r: (r['development']['accuracy'], -r['development']['loss'], -r['epoch']))
    if (run['selection_used_evaluation'] is not False
            or digest(training/'run.json') != run['training_record_sha256']
            or train_complete['epochs'] != train_run['config']['epochs']
            or [r['epoch'] for r in history if r['phase']=='epoch_complete'] != list(range(1,train_run['config']['epochs']+1))
            or chosen['epoch'] != run['selected_epoch']
            or train_complete['best']['epoch'] != run['selected_epoch']
            or train_complete['best_sha256'] != run['checkpoint_sha256']
            or digest(ROOT/'src/evaluation/trained.py') != run['source_sha256']):
        raise ValueError('Training/evaluation provenance or development selection differs')
    before_id = 'mof-dinov3-stage0-pilot50-v1-'+run['model']
    before_folder = ROOT/'results'/before_id
    before_run = json.loads((before_folder/'run.json').read_text())
    for record in [run['scorers'], before_run['source_sha256']]:
        for name in ['src/evaluation/score.py','src/evaluation/control/score.py']:
            if digest(ROOT/name) != record[name]:
                raise ValueError('Frozen recount scorer differs')
    weight_identity = lambda weights: {key:{name:value[name] for name in ['config','files']} for key,value in weights.items()}
    if weight_identity(run['weights']) != weight_identity(before_run['weights']):
        raise ValueError('Frozen backbone identity differs')
    configs = [{k:v for k,v in record.items() if k != 'inference_time_limit_hours'} for record in [run['config'], before_run['config']]]
    if configs[0] != configs[1]:
        raise ValueError('Inference configuration differs beyond process time limit')
    base = ROOT/'data/manifests/pilot50_v1'
    split = json.loads((base/'split.json').read_text())
    if digest(base/'labels.jsonl') != split['labels_sha256'] or digest(base/'inputs.jsonl') != split['inputs_sha256']:
        raise ValueError('Pilot provenance changed')
    labels = read_jsonl(base/'labels.jsonl')
    bdetails, bm, bsource = load_run(before_id, labels, split['inputs_sha256'], run['model'])
    adetails, am, asource = load_run(args.run_id, labels, split['inputs_sha256'], run['model'])
    arm = run['arm']
    before, after = bm[arm], am[arm]
    stored = json.loads((folder/'scored/metrics.json').read_text())
    stored_before = json.loads((before_folder/'scored/metrics.json').read_text())['metrics'][arm]
    for calculated, recorded in [(before,stored_before),(after,stored)]:
        for key, saved in [('primary','primary_binary_aliases_v2'),('strict','strict_v1')]:
            for field in ['n','correct','accuracy']:
                if not math.isclose(calculated[key][field], recorded[saved][field], abs_tol=1e-15):
                    raise ValueError('Saved score differs from frozen recount')
    family_path = ROOT/'data/manifests/train_dev_v2/split.json'
    families = {g:f for f,gs in json.loads(family_path.read_text())['family_members'].items() for g in gs}
    effect = contrast({'before':bdetails[arm],'after':adetails[arm]}, {'before':-1,'after':1}, families)
    b = {row['id']:row['correct'] for row in bdetails[arm]}
    a = {row['id']:row['correct'] for row in adetails[arm]}
    changed = {'gained':sorted(q for q in b if not b[q] and a[q]), 'lost':sorted(q for q in b if b[q] and not a[q])}
    caps = generation_limits([bsource, asource])
    ci = [100*v for v in effect['ci95']] if effect['ci95'] else None
    lines = [f'# Avaliação após alinhamento — {run["model"].upper()} / {arm}', '',
             f'Mesmas {after["primary"]["n"]} perguntas e pesos de base; somente os projetores foram ajustados. '
             f'Checkpoint selecionado no desenvolvimento: época {run["selected_epoch"]}.', '',
             '| Etapa | Acertos / perguntas | Acurácia principal (%) | Estrita (%) | Respostas no teto |',
             '|---|---:|---:|---:|---:|']
    for name, value, cap in [('Sem ajuste',before,caps[arm]),('Após ajuste',after,caps['trained_'+arm])]:
        lines.append(f'| {name} | {value["primary"]["correct"]}/{value["primary"]["n"]} | {100*value["primary"]["accuracy"]:.2f} | {100*value["strict"]["accuracy"]:.2f} | {cap["at_limit"]}/{cap["n"]} |')
    lines += ['', f'Ganho: **{effect["percentage_points"]:+.2f} pontos percentuais**; '
              + (f'IC95% por família [{ci[0]:.2f}; {ci[1]:.2f}] pp.' if ci else 'IC indisponível.'),
              f'{len(changed["gained"])} perguntas passaram a ser corretas; {len(changed["lost"])} passaram a ser incorretas.', '',
              '## Diagnóstico por formato', '', '| Formato | Perguntas | Antes (%) | Depois (%) | Diferença pp |', '|---|---:|---:|---:|---:|']
    for fmt, value in after['primary']['by_formato'].items():
        previous = before['primary']['by_formato'][fmt]
        if value['n'] != previous['n']:raise ValueError('Subgroup denominators differ')
        lines.append(f'| {fmt} | {value["n"]} | {100*previous["accuracy"]:.2f} | {100*value["accuracy"]:.2f} | {100*(value["accuracy"]-previous["accuracy"]):+.2f} |')
    lines += ['', '## Método e limites', '',
              'Acurácia pelo corretor congelado: comparação normalizada por formato, tolerância numérica do gabarito '
              'e aliases exatos Yes/No nas perguntas binárias. Recontagem não equivale a revisão semântica; não é BERTScore. '
              'O intervalo usa 2.000 reamostragens de famílias inteiras, nos mesmos IDs. A unidade é a pergunta, não o par A/B.', '',
              'Piloto previamente explorado, origem independente não certificada e uma única inicialização. '
              'Subgrupos são descritivos. O ganho não demonstra causalmente melhora de percepção nem superioridade entre sementes. '
              'A comparação completa dos dois modelos e braços ainda depende das execuções restantes.', '',
              'Custos em `MATCHED_COMPARISON.json` são tempos observados por executores com fronteiras de cronômetro diferentes: '
              'a avaliação treinada inclui processamento da imagem/entrada; o controle inicia o cronômetro depois desse processamento. '
              'Não calcular speedup entre esses tempos. Cache, carregamento e estados dos dispositivos também afetam as medições.', '',
              f'Reprodução: `.venv/bin/python scripts/local/report_trained.py {args.run_id}`. '
              'Proveniência, subgrupos e IDs que mudaram: `MATCHED_COMPARISON.json`.']
    paths = [folder/'run.json',folder/'complete.json',folder/'predictions.jsonl',folder/'scored/metrics.json',
             before_folder/'run.json',before_folder/'complete.json',before_folder/'predictions.jsonl',before_folder/'scored/metrics.json',
             training/'run.json',training/'complete.json',training/'history.json',base/'labels.jsonl',base/'inputs.jsonl',family_path,
             ROOT/'src/evaluation/compare_alignment.py',ROOT/'src/evaluation/paired/metrics.py',
             ROOT/'src/evaluation/score.py',ROOT/'src/evaluation/control/score.py',ROOT/'scripts/local/report_campaign.py']
    report = {'scope':'one completed matched training arm; whole campaign remains incomplete',
              'model':run['model'],'arm':arm,'selected_epoch':run['selected_epoch'],'before':before,'after':after,
              'contrast':effect,'changed_question_ids':changed,'generation_caps':{key:caps[key] for key in [arm,'trained_'+arm]},
              'uncertainty':{'seed':20261102,'repetitions':2000,'unit':'whole family','numpy_version':importlib.metadata.version('numpy')},
              'sources':{str(p.relative_to(ROOT)):digest(p) for p in paths},'renderer_sha256':digest(Path(__file__))}
    write_once(folder/'MATCHED_COMPARISON.json',json_text(report))
    write_once(folder/'REPORT.md','\n'.join(lines)+'\n')
    print(folder/'REPORT.md')


if __name__ == '__main__':
    main()
