"""Stage-1 image/text alignment followed by full decoder+projector SFT on graph QA."""
import argparse
import fcntl
import hashlib
import json
import math
import random
import time
from pathlib import Path
import torch
from src.data.pilot import ROOT
from src.models.control.artifacts import digest,installed,write_json
from src.models.control.hardware import validate_gpu
from src.training.graph_sft import GraphSFTModel
from src.training.master_optimizer import MasterAdamW
from src.training.sft_checkpoint import save,restore,save_weights
from src.training.run import validate


def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','llava'],required=True);p.add_argument('--arm',choices=['clip_single','clip_double','clip_dino'],required=True);p.add_argument('--run-id',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--checks-prefix',default='full-sft-check-v1');a=p.parse_args()
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',a.run_id):raise ValueError('Invalid run ID')
    cfg=json.loads((ROOT/'configs/graph_full_sft_v1.json').read_text());base=json.loads((ROOT/'configs/control_mof_stage0.json').read_text())
    validate_gpu(a.model,cfg,training=True)
    gate_path=ROOT/'experiments/full_sft_checks'/f'{a.checks_prefix}-{a.model}-{a.arm}'/'complete.json'
    gate=json.loads(gate_path.read_text())
    if gate['status']!='full_decoder_projectors_update_and_resume_pass' or gate['config_sha256']!=digest(ROOT/'configs/graph_full_sft_v1.json'):raise ValueError('Full-SFT gate missing or stale')
    if any(digest(ROOT/name)!=sha for name,sha in gate['source_sha256'].items()):raise ValueError('Full-SFT source changed after GPU gate')
    phases={s:rows(ROOT/cfg[f'stage{s}']['data']) for s in [1,2]};dev=rows(ROOT/cfg['development'])
    split=json.loads((ROOT/'data/manifests/train_dev_v2/split.json').read_text())
    for data in phases.values():
        assert {r['grafico'] for r in data}==set(split['train'])
        assert not {r['family_id'] for r in data}&{r['family_id'] for r in dev}
    assert {r['grafico'] for r in dev}==set(split['development'])
    for data in [*phases.values(),dev]:
        for image,sha in {r['image']:r['image_sha256'] for r in data}.items():
            if digest(ROOT/image)!=sha:raise ValueError('Image changed')
    caption_metadata=json.loads((ROOT/'data/manifests/alignment_captions_v1/manifest.json').read_text())
    assert digest(ROOT/cfg['stage1']['data'])==caption_metadata['caption_manifest_sha256']
    assert digest(ROOT/cfg['stage2']['data'])==caption_metadata['source_training_sha256']
    metadata=json.loads((ROOT/'data/manifests/train_dev_v2/manifest.json').read_text())
    assert digest(ROOT/cfg['stage2']['data'])==metadata['counts']['train']['sha256']
    assert digest(ROOT/cfg['development'])==metadata['counts']['development']['sha256']
    assert digest(ROOT/'data/manifests/train_dev_v2/split.json')==metadata['split_sha256']
    source_dirs=['src/training','src/models/control'];sources={str(f.relative_to(ROOT)):digest(f) for name in source_dirs for f in (ROOT/name).glob('*.py')}
    for f in ['src/evaluation/run.py','src/evaluation/score.py','src/evaluation/control/score.py']:sources[f]=digest(ROOT/f)
    record={'version':cfg['version'],'model':a.model,'arm':a.arm,'config':cfg,'base_config':base,'data_sha256':{name:digest(ROOT/name) for name in [cfg['stage1']['data'],cfg['stage2']['data'],cfg['development'],'data/manifests/train_dev_v2/split.json']},'weights':{k:installed(k,verify=True) for k in [a.model,'clip','dinov3']},'source_sha256':sources,'scope':'two-stage graph-domain adaptation, full decoder trainable in stage 2'}
    signature=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    folder=ROOT/'experiments/full_sft'/a.run_id
    if folder.exists() and not a.resume:raise ValueError('Run exists; explicit resume required')
    folder.mkdir(parents=True,exist_ok=True);lock=(folder/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'run.json').exists():
        if json.loads((folder/'run.json').read_text())!=record:raise ValueError('Source/config drift')
    else:write_json(folder/'run.json',record)
    for name in sources:
        target=folder/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and digest(target)!=sources[name]:raise ValueError('Frozen source changed')
        if not target.exists():target.write_bytes((ROOT/name).read_bytes())
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);random.seed(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=GraphSFTModel(a.model,base,a.arm,cfg['max_length'])
    state={'stage':1,'epoch':0,'cursor':0,'updates':0,'stage_updates':0,'loss_sum':0.,'loss_count':0,'train_seconds':0.,'history':[],'best':None,'initial_sft_validation_done':False}
    def optimizer_factory(m):
        import psutil
        count=sum(v.numel() for _,v in m.named_trainable());needed=count*16+4*1024**3
        if psutil.virtual_memory().available<needed:raise RuntimeError('Insufficient host RAM for full FP32 master/AdamW state')
        return MasterAdamW(m.named_trainable(),cfg[f'stage{m.stage}']['learning_rate'],cfg['weight_decay'])
    def schedule(opt,stage):
        n=cfg[f'stage{stage}']['epochs']*math.ceil(len(phases[stage])/cfg['gradient_accumulation']);warm=max(1,round(n*cfg['warmup_fraction']))
        def rate(step):
            if step<warm:return (step+1)/warm
            return .5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,n-warm))))
        return torch.optim.lr_scheduler.LambdaLR(opt.optimizer,rate)
    if a.resume:opt,scheduler,state=restore(folder/'last.pt',model,lambda m:MasterAdamW(m.named_trainable(),cfg[f'stage{m.stage}']['learning_rate'],cfg['weight_decay']),schedule,signature)
    else:opt=optimizer_factory(model);scheduler=schedule(opt,1)
    last_save=time.monotonic()
    def persist():
        nonlocal last_save
        checkpoint_started=time.perf_counter()
        event={'phase':'checkpoint_saving','stage':state['stage'],'epoch':state['epoch'],'updates':state['updates'],'updated_at_unix':time.time()}
        write_json(folder/'progress.json',event);print(json.dumps(event),flush=True)
        save(folder/'last.pt',model,opt,scheduler,state,signature);last_save=time.monotonic()
        event.update(phase='checkpoint_saved',seconds=time.perf_counter()-checkpoint_started,bytes=(folder/'last.pt').stat().st_size,updated_at_unix=time.time())
        write_json(folder/'progress.json',event);print(json.dumps(event),flush=True)
        write_json(folder/'checkpoint.json',{'stage':state['stage'],'epoch':state['epoch'],'updates':state['updates'],'bytes':(folder/'last.pt').stat().st_size,'created_at_unix':time.time()})
    def assess(epoch):
        result=validate(model,dev,folder,epoch);score=[result['accuracy'],-result['loss']]
        if state['best'] is None or tuple(score)>tuple(state['best']['score']):
            state['best']={'epoch':epoch,'score':score,'updates':state['updates']}
            save_weights(folder/'best-model.pt',model)
            write_json(folder/'best.json',{**state['best'],'sha256':digest(folder/'best-model.pt'),'bytes':(folder/'best-model.pt').stat().st_size})
        return result
    while state['stage']<=2:
        stage=state['stage'];data=phases[stage]
        if stage==2 and not state['initial_sft_validation_done']:
            result=assess(0);state['history'].append({'stage':2,'epoch':0,'phase':'initial_sft','development':result});state['initial_sft_validation_done']=True;persist()
        while state['epoch']<cfg[f'stage{stage}']['epochs']:
            epoch=state['epoch'];order=torch.randperm(len(data),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist();torch.cuda.reset_peak_memory_stats()
            while state['cursor']<len(order):
                start=time.perf_counter();indices=order[state['cursor']:state['cursor']+cfg['gradient_accumulation']];opt.zero_grad()
                for index in indices:
                    loss,_=model.loss(data[index]);state['loss_sum']+=float(loss.detach());state['loss_count']+=1;(loss/len(indices)).backward();del loss
                norm=opt.step(cfg['gradient_clip_norm']);scheduler.step();model.assert_frozen()
                state['cursor']+=len(indices);state['updates']+=1;state['stage_updates']+=1;state['train_seconds']+=time.perf_counter()-start
                if state['stage_updates']%5==0 or state['cursor']==len(order):
                    progress={'phase':'train','stage':stage,'epoch':epoch+1,'cursor':state['cursor'],'total':len(order),'updates':state['updates'],'stage_updates':state['stage_updates'],'mean_loss':state['loss_sum']/state['loss_count'],'lr':scheduler.get_last_lr(),'gradient_norm':norm,'trainable_parameters':sum(p.numel() for _,p in model.named_trainable()),'updated_at_unix':time.time()};write_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True)
                if time.monotonic()-last_save>=cfg['checkpoint_minutes']*60:persist()
            result=assess(epoch+1) if stage==2 else None
            entry={'stage':stage,'epoch':epoch+1,'phase':'epoch_complete','train_loss':state['loss_sum']/state['loss_count'],'train_seconds':state['train_seconds'],'development':result,'updates':state['updates'],'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'trainable_parameters':sum(p.numel() for _,p in model.named_trainable())}
            state['history'].append(entry);state.update(epoch=epoch+1,cursor=0,loss_sum=0.,loss_count=0,train_seconds=0.)
            write_json(folder/'history.json',state['history']);persist();print(json.dumps(entry),flush=True)
        if stage==1:
            save_weights(folder/'stage1-model.pt',model);del opt,scheduler
            model.set_stage(2);state.update(stage=2,epoch=0,cursor=0,stage_updates=0);opt=optimizer_factory(model);scheduler=schedule(opt,2);persist()
        else:break
    write_json(folder/'complete.json',{'stage1_epochs':cfg['stage1']['epochs'],'stage2_epochs':cfg['stage2']['epochs'],'updates':state['updates'],'best':state['best'],'signature':signature,'held_out_evaluation':'pending'})


if __name__=='__main__':main()
