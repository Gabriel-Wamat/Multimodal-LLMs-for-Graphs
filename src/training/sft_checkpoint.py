"""Atomic full-decoder training checkpoints, not projector-only snapshots."""
import os
import random
from pathlib import Path
import torch


def save(path,model,optimizer,scheduler,state,signature):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'format':'graph-full-sft-v1','model':model.weight_state(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'state':state,'signature':signature,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'python_rng':random.getstate()}
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        torch.save(payload,f);f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def restore(path,model,optimizer_factory,scheduler_factory,signature):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['format']!='graph-full-sft-v1' or payload['signature']!=signature:raise ValueError('Full-SFT checkpoint protocol/source mismatch')
    model.load_weights(payload['model'])
    optimizer=optimizer_factory(model)
    scheduler=scheduler_factory(optimizer,payload['state']['stage'])
    # Scheduler construction changes LR; restore optimizer groups afterwards.
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng']:torch.cuda.set_rng_state_all(payload['cuda_rng'])
    random.setstate(payload['python_rng'])
    return optimizer,scheduler,payload['state']


def save_weights(path,model):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        torch.save(model.weight_state(),f);f.flush();os.fsync(f.fileno())
    temporary.replace(path)
