"""Validate reviewed sample coverage and exact final-prediction/image/CSV provenance."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    base = ROOT/'experiments/five-stage-campaign'
    sample = read(base/'reading_review_sample_v1.json')
    inputs = {row['id']:row for row in map(json.loads,(ROOT/'data/manifests/pairs_v1/inputs.jsonl').read_text().splitlines())}
    labels = {row['id']:row for row in map(json.loads,(ROOT/'data/manifests/pairs_v1/labels.jsonl').read_text().splitlines())}
    allowed = {key for pair in sample['selected'] for key in pair['items']}
    report = {'status':'pass','scope':'Provenance and coverage only; does not certify correctness of visual judgments or population prevalence','sample_sha256':digest(base/'reading_review_sample_v1.json'),'models':{}}
    for model in sample['models']:
        folder=ROOT/'results'/('paired-mitigations-full-v1-'+model)
        pred=folder/'predictions.jsonl'
        complete=read(folder/'complete.json')
        if digest(pred)!=complete['predictions_sha256']:
            raise ValueError('Changed final predictions: '+model)
        rows=[json.loads(line) for line in pred.read_text().splitlines()]
        final={(row['id'],row['condition']):row for row in rows}
        if len(final)!=len(rows) or len(rows)!=complete['records']:
            raise ValueError('Incomplete or duplicate predictions')
        seen=set();categories=Counter();files={};strata=Counter()
        for path in sorted(base.glob('reading_review_'+model+'_*_v1.json')):
            doc=read(path)
            if 'review' not in doc: continue
            files[path.name]=digest(path)
            for item in doc['review']:
                key=item['id']
                if key not in allowed or key in seen or item['model']!=model or item['review_status']!='reviewed':
                    raise ValueError('Invalid/duplicate review: '+key)
                seen.add(key)
                if item['prediction_record']!=final[(key,'transcription')]:
                    raise ValueError('Review differs from final transcription: '+key)
                transcript=final[(key,'transcription')]['calls'][0]['response']
                if item['transcript']!=transcript or not item['evidence']:
                    raise ValueError('Missing transcript/evidence: '+key)
                for evidence in item['evidence']:
                    if evidence['quote'] not in transcript or not evidence['observation'].strip():
                        raise ValueError('Invalid literal evidence: '+key)
                row=inputs[key];label=labels[key]
                if item['image']!=row['image'] or item['image_sha256']!=row['image_sha256'] or digest(ROOT/row['image'])!=row['image_sha256']:
                    raise ValueError('Image provenance mismatch: '+key)
                if item['csv_sha256']!=label['csv_sha256'] or digest(ROOT/label['derived_csv'])!=label['csv_sha256']:
                    raise ValueError('CSV provenance mismatch: '+key)
                categories.update(item['categories']);strata.update([key.split('-')[0]])
        report['models'][model]={'reviewed_images':len(seen),'sample_images':len(allowed),'remaining_images':len(allowed-seen),'complete':seen==allowed,'by_type':dict(strata),'category_mentions_not_exclusive':dict(categories),'ids':sorted(seen),'sources':files,'final_predictions_sha256':digest(pred)}
    (base/'reading_review_audit_current_v1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({model:{k:v for k,v in value.items() if k in ['reviewed_images','remaining_images','complete']} for model,value in report['models'].items()},indent=2))

if __name__=='__main__':
    main()
