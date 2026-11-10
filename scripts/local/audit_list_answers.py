"""Describe list-answer structure in completed runs without changing official scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.pilot import json_text, write_once
from src.evaluation.score import canonical, grade, read_jsonl
from src.models.control.artifacts import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_ids', nargs='+')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    labels_path = ROOT/'data/manifests/pilot50_v1/labels.jsonl'
    split_path = labels_path.with_name('split.json')
    if digest(labels_path) != json.loads(split_path.read_text())['labels_sha256']:
        raise ValueError('Frozen labels changed')
    labels = read_jsonl(labels_path)
    lists = {r['id']:r for r in labels if r['formato']=='lista'}
    sources = {str(p.relative_to(ROOT)):digest(p) for p in [labels_path, split_path]}
    report = {'scope':'Post-hoc structural description, not a replacement metric or semantic review',
              'list_questions':len(lists), 'groups':{}, 'details':[]}
    for run_id in args.run_ids:
        if Path(run_id).name != run_id or run_id in ('.','..'):
            raise ValueError('Invalid run ID')
        folder = ROOT/'results'/run_id
        run = json.loads((folder/'run.json').read_text())
        if run.get('split_sha256') is not None:
            if run['split_sha256'] != digest(split_path):
                raise ValueError('Run split differs from frozen pilot')
        elif run.get('labels_sha256') != digest(labels_path):
            raise ValueError('Run labels differ from frozen pilot')
        complete = json.loads((folder/'complete.json').read_text())
        pred_path = folder/'predictions.jsonl'
        if digest(pred_path) != complete['predictions_sha256']:
            raise ValueError('Completed predictions changed')
        rows = read_jsonl(pred_path)
        arms = sorted({r.get('arm','baseline') for r in rows})
        for p in [folder/'run.json', folder/'complete.json', pred_path]:
            sources[str(p.relative_to(ROOT))] = digest(p)
        for arm in arms:
            selected = [r for r in rows if r.get('arm','baseline')==arm]
            if (len(selected)!=len(labels) or len({r['id'] for r in selected})!=len(labels)
                    or {r['id'] for r in selected}!={r['id'] for r in labels}
                    or any(r['status']!='ok' for r in selected)):
                raise ValueError('Run/arm is not complete fixed pilot')
            counts = Counter()
            key = run_id+'/'+arm
            for r in selected:
                if r['id'] not in lists:
                    continue
                q = lists[r['id']]
                target = canonical(q['resposta'],'lista')
                answer = canonical(r['response'],'lista')
                assert len(set(target))==len(target)
                official = grade(q,r['response'])[0]
                permutation = len(answer)==len(target) and Counter(answer)==Counter(target)
                unknown = [s for s in answer if s not in target]
                recognized = [s for s in answer if s in target]
                missing = sorted(set(target)-set(recognized))
                duplicates = [s for s,n in Counter(recognized).items() if n>1]
                counts['n']+=1
                counts['exact_correct']+=int(official)
                counts['complete_permutation_wrong_order']+=int(permutation and not official)
                counts['unrecognized_fragments']+=int(bool(unknown))
                counts['missing_target_labels']+=int(bool(missing))
                counts['duplicate_recognized_labels']+=int(bool(duplicates))
                counts['no_recognized_labels']+=int(not recognized)
                report['details'].append({'group':key,'id':r['id'],'target':q['resposta'],
                    'response':r['response'],'exact_correct':official,
                    'complete_permutation':permutation,'unrecognized_fragments':unknown,
                    'missing_target_labels':missing,'duplicate_recognized_labels':duplicates,
                    'generated_tokens':r.get('generated_tokens')})
            report['groups'][key]=dict(counts)
        if run.get('smoke'):
            raise ValueError('Smoke runs excluded')
    code = ROOT/'src/evaluation/score.py'
    report['sources']=sources
    report['analysis_sources']={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),code]}
    report['limitations']=[
        'Exact list parser splits commas, semicolons and >, using the frozen scorer normalization.',
        'Unknown fragments can be extra labels, prose or formatting; not automatically factual errors.',
        'Missing labels refer to this exact parser; formatting can prevent label recognition.',
        'Categories overlap; these counts are not a disjoint error taxonomy.',
        'No source image/CSV or semantic review performed in this diagnostic.',
        'Post-hoc explored pilot; do not use to tune ongoing training or select checkpoints.',
        'Official predictions, scores, prompts and active snapshots are unchanged.',
    ]
    out=args.output if args.output.is_absolute() else ROOT/args.output
    write_once(out/'list_structure.json',json_text(report))
    lines=['# Diagnóstico estrutural de respostas em lista','',report['scope'],'',
           '| Execução / braço | N | Exatas | Permutação completa, ordem errada | Fragmentos não reconhecidos | Rótulos ausentes | Rótulos repetidos | Nenhum rótulo reconhecido |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    fields=['n','exact_correct','complete_permutation_wrong_order','unrecognized_fragments','missing_target_labels','duplicate_recognized_labels','no_recognized_labels']
    for key,c in report['groups'].items():
        lines.append('| '+key+' | '+' | '.join(str(c.get(f,0)) for f in fields)+' |')
    lines+=['','## Limites','']+['- '+s for s in report['limitations']]
    lines+=['','Detalhes por resposta, hashes das fontes e código: `list_structure.json`.']
    write_once(out/'REPORT.md','\n'.join(lines)+'\n')
    for p in [Path(__file__),code]:
        write_once(out/'source'/p.relative_to(ROOT),p.read_text())
    print(out/'REPORT.md')


if __name__=='__main__':
    main()
