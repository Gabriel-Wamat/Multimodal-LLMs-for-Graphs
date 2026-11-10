"""Summarize audited visual reviews and final-answer transitions, without inventing reviews."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.evaluation.paired.conditions import binary  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=ROOT/'experiments/five-stage-campaign'
    audit_path=base/'reading_review_audit_current_v1.json'
    audit=read(audit_path)
    if audit['status']!='pass':
        raise ValueError('Run the provenance audit first')
    label_path=ROOT/'data/manifests/pairs_v1/labels.jsonl'
    labels={v['id']:v for v in map(json.loads,label_path.read_text().splitlines())}
    all_complete=all(value['complete'] for value in audit['models'].values())
    report={'status':'complete_fixed_sample_review' if all_complete else 'partial_review','scope':'Only audited, actually visually reviewed sample items; no population-prevalence estimate or causal attribution','audit_sha256':digest(audit_path),'labels_sha256':digest(label_path),'models':{}}
    md=['# Revisão visual — síntese da amostra fixa','',('A amostra fixa foi integralmente revisada nos dois modelos: 144 imagens por modelo, cobrindo seis tipos, seis estilos e duas intensidades. A revisão foi realizada pelo assistente por inspeção visual, sem anotação independente adicional. As contagens descrevem esta amostra estratificada; não estimam prevalência no dataset.' if all_complete else 'A amostra fixa permanece parcialmente revisada. Coberturas diferentes entre modelos impedem comparar frequências brutas como prevalência.'),'','## Efeito da transcrição nos itens revisados','','Acerto da resposta final usa o mesmo corretor binário da avaliação pareada. Uma mudança de acerto não prova que o erro de leitura causou o resultado. A unidade desta tabela é a imagem/pergunta, não o par.','','| Modelo | Revisadas / amostra | Ajudou | Prejudicou | Manteve acerto | Manteve erro |','|---|---:|---:|---:|---:|---:|']
    for model,coverage in audit['models'].items():
        folder=ROOT/'results'/('paired-mitigations-full-v1-'+model)
        path=folder/'predictions.jsonl'
        if digest(path)!=coverage['final_predictions_sha256']:
            raise ValueError('Predictions changed')
        predictions={(row['id'],row['condition']):row for row in map(json.loads,path.read_text().splitlines())}
        reviewed={}
        for name,sha in coverage['sources'].items():
            p=base/name
            if digest(p)!=sha:raise ValueError('Review source changed: '+name)
            for row in read(p)['review']:reviewed[row['id']]=row
        transitions={name:[] for name in ['ajudou','prejudicou','manteve_acerto','manteve_erro']}
        categories=Counter()
        details=[]
        for key,row in sorted(reviewed.items()):
            answers={c:predictions[(key,c)]['response'] for c in ['original','crops_2x2','transcription']}
            correct={c:binary(a)==labels[key]['answer'] for c,a in answers.items()}
            effect={(False,True):'ajudou',(True,False):'prejudicou',(True,True):'manteve_acerto',(False,False):'manteve_erro'}[(correct['original'],correct['transcription'])]
            transitions[effect].append(key);categories.update(row['categories'])
            details.append({'id':key,'target':labels[key]['answer'],'answers':answers,'correct':correct,'transition':effect,'review_categories':row['categories']})
        if len(reviewed)!=coverage['reviewed_images']:raise ValueError('Coverage changed')
        report['models'][model]={'reviewed_images':len(reviewed),'sample_images':coverage['sample_images'],'by_type':coverage['by_type'],'transition_counts':{k:len(v) for k,v in transitions.items()},'transition_ids':transitions,'nonexclusive_category_mentions':dict(categories),'items':details}
        md.append('| '+model+' | '+str(len(reviewed))+'/'+str(coverage['sample_images'])+' | '+' | '.join(str(len(v)) for v in transitions.values())+' |')
    md+=['','## Exemplos revisados por efeito','','IDs ilustrativos escolhidos por ordem lexicográfica entre os casos já revisados. Não modificam a amostra ou o denominador.']
    for model,result in report['models'].items():
        md+=['','### '+model,'']
        for effect,ids in result['transition_ids'].items():
            md.append('- '+effect+': '+(', '.join('`'+key+'`' for key in ids[:3]) if ids else 'nenhum caso observado entre os itens revisados'))
        md+=['','Cobertura por tipo: '+', '.join(f'{k}={v}' for k,v in result['by_type'].items())+'.']
    md+=['','## Interpretação','','As categorias são não exclusivas. `unsupported_structure` significa elemento gráfico inexistente; `trend_direction` significa direção visual incompatível; `uncertain` preserva casos ambíguos. `no_observable_error` é limitado ao aspecto examinado, sem certificar toda a resposta.','','A descrição pode ter acertos parciais e erros simultaneamente. A ausência de valores exatos em dispersão ou boxplot não basta para classificar alucinação. Os registros individuais preservam citações literais, observações, PNG/CSV e previsões finais.','','Reproduza após novas revisões:','','```sh','.venv/bin/python scripts/local/audit_reading_review.py','.venv/bin/python scripts/local/report_reading_review.py','```']
    (base/'reading_review_summary_v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (base/'READING_REVIEW_SUMMARY.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:v['transition_counts'] for k,v in report['models'].items()},indent=2))


if __name__=='__main__':
    main()
