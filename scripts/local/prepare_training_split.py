"""Reserve train/dev families outside the fixed exploratory pilot; never overwrite."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def main():
    raw = ROOT / 'dados_projeto_multi_modais/dados'
    pilot_path = ROOT / 'data/manifests/pilot50_v1/split.json'
    pilot = set(json.loads(pilot_path.read_text())['selected_graphs'])
    params = json.loads((raw/'parametros.json').read_text())
    groups = defaultdict(list)
    hashes = {}
    for p in sorted(raw.glob('*.csv')):
        with p.open(newline='',encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        canonical = json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False)
        h = hashlib.sha256(canonical.encode()).hexdigest()
        groups[h].append(p.stem)
        hashes[p.stem] = hashlib.sha256(p.read_bytes()).hexdigest()
    excluded = []
    strata = defaultdict(list)
    for h, ids in groups.items():
        if pilot.intersection(ids):
            excluded.extend(set(ids)-pilot)
        else:
            strata[(ids[0].split('-')[0],params[ids[0]]['estilo'])].append((h,ids))
    train, dev = [], []
    for key, families in sorted(strata.items()):
        families.sort(key=lambda x:hashlib.sha256(('train-dev-v1:'+x[0]).encode()).hexdigest())
        ndev = max(1,round(len(families)*.2))
        for i,(_,ids) in enumerate(families):
            (dev if i<ndev else train).extend(ids)
    assert not (set(train)&set(dev) or set(train)&pilot or set(dev)&pilot)
    record = {'version':'train-dev-v1','train':sorted(train),'development':sorted(dev),
              'evaluation_exploratory':sorted(pilot),'excluded_duplicate_of_evaluation':sorted(excluded),
              'source_sha256':hashes,'pilot_split_sha256':hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
              'family_rule':'exact canonical CSV content; transformed descendants must inherit split',
              'limitation':'Exact duplicate check only; near-duplicate and provenance audit still pending. Pilot already explored, not independent confirmation.',
              'eligible_for_training':False,'next_gate':'audit near duplicates and provenance before enabling training'}
    p=ROOT/'data/manifests/train_dev_v1/split.json'
    text=json.dumps(record,ensure_ascii=False,indent=2)+'\n'
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        assert p.read_text()==text,'Existing split differs'
    else:p.write_text(text)
    print(json.dumps({k:len(record[k]) for k in ['train','development','evaluation_exploratory','excluded_duplicate_of_evaluation']}))
if __name__=='__main__':main()
