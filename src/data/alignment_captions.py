"""Training-only CSV descriptions for the image-text alignment stage."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from src.data.pilot import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(rows, kind):
    def number(x):return format(float(x),'.4g').replace('.',',')
    if not rows:raise ValueError('Empty source CSV')
    if kind=='BAR':
        return 'Gráfico de barras. Eixo vertical: valor. '+ '; '.join(r['categoria']+': '+number(r['valor']) for r in rows)+'.'
    if kind=='PIZ':
        return 'Gráfico de pizza. As participações percentuais são: '+ '; '.join(r['categoria']+': '+number(r['percentual'])+'%' for r in rows)+'.'
    if kind=='HIS':
        return 'Histograma. Eixo horizontal: faixa; eixo vertical: contagem. '+ '; '.join(number(r['faixa_inicio'])+' a '+number(r['faixa_fim'])+': '+r['contagem'] for r in rows)+'.'
    if kind=='LIN':
        columns=[k for k in rows[0] if k!='ano']
        pieces=[]
        for key in columns:
            values=np.array([float(r[key]) for r in rows])
            pieces.append(key+': valor inicial '+number(values[0])+', final '+number(values[-1])+', mínimo '+number(values.min())+', máximo '+number(values.max())+'.')
        return 'Gráfico de linhas de '+rows[0]['ano']+' a '+rows[-1]['ano']+'. Eixos: ano e valor. '+' '.join(pieces)
    if kind=='DIS':
        x=np.array([float(r['x']) for r in rows]);y=np.array([float(r['y']) for r in rows])
        corr=float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else 0.0
        return f'Gráfico de dispersão com {len(rows)} pontos. Eixos X e Y. X varia de {number(x.min())} a {number(x.max())}; Y varia de {number(y.min())} a {number(y.max())}. A correlação linear dos dados é aproximadamente {number(corr)}.'
    if kind=='BOX':
        groups={}
        for r in rows:groups.setdefault(r['grupo'],[]).append(float(r['valor']))
        parts=[]
        for group,values in groups.items():
            q1,med,q3=np.percentile(values,[25,50,75]);low=q1-1.5*(q3-q1);high=q3+1.5*(q3-q1)
            inliers=[v for v in values if low<=v<=high]
            parts.append(f'{group}: Q1 {number(q1)}, mediana {number(med)}, Q3 {number(q3)}, bigodes de {number(min(inliers))} a {number(max(inliers))}, {len(values)-len(inliers)} valores fora dos bigodes.')
        return 'Boxplot. Eixo vertical: valor. '+' '.join(parts)
    raise ValueError('Unsupported chart type')


def prepare(destination):
    train_path=ROOT/'data/manifests/train_dev_v2/train.jsonl'
    metadata=json.loads((train_path.parent/'manifest.json').read_text())
    if digest(train_path)!=metadata['counts']['train']['sha256']:raise ValueError('Training manifest changed')
    split=json.loads((train_path.parent/'split.json').read_text())
    unique={}
    for row in map(json.loads,train_path.read_text().splitlines()):unique.setdefault(row['grafico'],row)
    assert set(unique)==set(split['train'])
    protected=set(split['development'])|set(split['evaluation_exploratory'])|set(split['quarantined_near_evaluation'])
    assert not set(unique)&protected
    output=[]
    for graph,row in sorted(unique.items()):
        if digest(ROOT/row['image'])!=row['image_sha256']:raise ValueError('Image changed')
        source=ROOT/'dados_projeto_multi_modais/dados'/f'{graph}.csv'
        rows=list(csv.DictReader(source.open(encoding='utf-8')))
        caption=describe(rows,graph.split('-')[0])
        output.append({'id':graph+'-caption','grafico':graph,'tipo':row['tipo'],'family_id':row['family_id'],'image':row['image'],'image_sha256':row['image_sha256'],'pergunta':'Descreva o gráfico, suas variáveis e as principais características numéricas.','resposta_txt':caption,'source_csv':str(source.relative_to(ROOT)),'source_csv_sha256':digest(source),'split':'train','supervision':'training_csv_description_not_qa_labels'})
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/'train.jsonl';text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output)
    if target.exists() and target.read_text()!=text:raise ValueError('Refuse to replace caption version')
    target.write_text(text)
    record={'graphs':len(output),'questions_used_for_supervision':0,'source_training_sha256':digest(train_path),'caption_manifest_sha256':digest(target),'protected_graphs':len(protected),'source_sha256':digest(Path(__file__))}
    (destination/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/manifests/alignment_captions_v1');a=p.parse_args();print(json.dumps(prepare(a.output),indent=2))
