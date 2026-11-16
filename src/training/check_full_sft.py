"""Real two-stage gates: decoder gradients/updates and exact full-state continuation."""
import argparse
import gc
import json
import tempfile
import shutil
import time
from pathlib import Path
import torch
from src.data.pilot import ROOT
from src.models.control.artifacts import digest,write_json
from src.models.control.hardware import validate_gpu
from src.training.graph_sft import GraphSFTModel
from src.training.master_optimizer import MasterAdamW
from src.training.sft_checkpoint import save,restore


def samples(model):
    output={}
    for name,p in model.named_trainable():
        if 'layers.0.self_attn.q_proj.weight' in name or name.startswith('projectors.') or name.startswith('head.'):
            output[name]=p.detach().reshape(-1)[:4096].float().cpu().clone()
    return output


def step(model,opt,batch):
    opt.zero_grad();losses=[]
    for row in batch:
        loss,_=model.loss(row);losses.append(float(loss.detach()));(loss/len(batch)).backward();del loss
    norm=opt.step();model.assert_frozen()
    return sum(losses)/len(losses),norm


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen','llava'],required=True);p.add_argument('--arm',choices=['clip_single','clip_double','clip_dino'],required=True);p.add_argument('--run-id',required=True);a=p.parse_args()
    cfg=json.loads((ROOT/'configs/graph_full_sft_v1.json').read_text());base=json.loads((ROOT/'configs/control_mof_stage0.json').read_text());validate_gpu(a.model,cfg,training=True)
    folder=ROOT/'experiments/full_sft_checks'/a.run_id
    if folder.exists():raise ValueError('Use a new gate run ID')
    folder.mkdir(parents=True)
    torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    chosen=[]
    for stage in [1,2]:
        kinds={}
        for row in map(json.loads,(ROOT/cfg[f'stage{stage}']['data']).read_text().splitlines()):kinds.setdefault(row['tipo'],row)
        if len(kinds)!=6:raise RuntimeError('All six chart types required')
        chosen.append(list(kinds.values()))
    m=GraphSFTModel(a.model,base,a.arm,cfg['max_length']);decoder_before=next((v.detach().flatten()[:4096].cpu().clone() for n,v in m.decoder.named_parameters() if 'layers.0.self_attn.q_proj.weight' in n))
    # Check every alignment target length before starting production.
    tokenizer=m.vlm.processor.tokenizer
    for row in map(json.loads,(ROOT/cfg['stage1']['data']).read_text().splitlines()):
        if len(tokenizer.encode(row['resposta_txt']))>700:raise RuntimeError('Caption exceeds alignment budget; revise manifest explicitly')
    opt=MasterAdamW(m.named_trainable(),cfg['stage1']['learning_rate'])
    l1,n1=step(m,opt,chosen[0]);decoder_after=next((v.detach().flatten()[:4096].cpu().clone() for n,v in m.decoder.named_parameters() if 'layers.0.self_attn.q_proj.weight' in n))
    assert torch.equal(decoder_before,decoder_after),'Stage-1 decoder changed'
    del opt;gc.collect();m.set_stage(2)
    trainable=sum(p.numel() for _,p in m.named_trainable());decoder_count=sum(p.numel() for p in m.decoder.parameters() if p.requires_grad)
    assert decoder_count>(1_000_000_000 if a.model=='qwen' else 6_000_000_000),'Full decoder not enabled'
    before=samples(m);opt=MasterAdamW(m.named_trainable(),cfg['stage2']['learning_rate']);scheduler=torch.optim.lr_scheduler.LambdaLR(opt.optimizer,lambda _:1.0)
    l2,n2=step(m,opt,chosen[1]);scheduler.step();after=samples(m)
    assert any(not torch.equal(before[n],v) for n,v in after.items() if 'q_proj.weight' in n),'Decoder GPU weights did not update'
    for branch in ['clip']+(['dino'] if a.arm=='clip_dino' else []):
        assert any(not torch.equal(before[n],v) for n,v in after.items() if n.startswith('projectors.'+branch+'.')),'Projector did not update'
    signature=digest(ROOT/'configs/graph_full_sft_v1.json');start=time.perf_counter()
    scratch=Path(tempfile.gettempdir())
    if shutil.disk_usage(scratch).free<trainable*16+8*1024**3:
        scratch=ROOT/'.cache/full-sft-gate-scratch';scratch.mkdir(parents=True,exist_ok=True)
    write_json(folder/'progress.json',{'phase':'full_checkpoint_resume_gate','trainable_parameters':trainable,'estimated_checkpoint_bytes':trainable*14,'scratch':str(scratch),'started_at_unix':time.time()})
    with tempfile.TemporaryDirectory(prefix='graph-sft-resume-',dir=scratch) as tmp:
        path=Path(tmp)/'full.pt';save(path,m,opt,scheduler,{'stage':2,'updates':1},signature);size=path.stat().st_size
        expected_loss,_=step(m,opt,[chosen[1][0]]);scheduler.step();expected=samples(m);expected_fingerprint=opt.fingerprint()
        del opt,scheduler;gc.collect()
        opt,scheduler,state=restore(path,m,lambda model:MasterAdamW(model.named_trainable(),cfg['stage2']['learning_rate']),lambda opt,stage:torch.optim.lr_scheduler.LambdaLR(opt.optimizer,lambda _:1.0),signature)
        resumed_loss,_=step(m,opt,[chosen[1][0]]);scheduler.step();actual=samples(m)
        assert abs(expected_loss-resumed_loss)<1e-6,(expected_loss,resumed_loss)
        assert all(torch.equal(expected[n],v) for n,v in actual.items()),'GPU continuation mismatch'
        assert expected_fingerprint==opt.fingerprint(),'Full master/AdamW state continuation mismatch'
        full_hash=digest(path)
    source={str((ROOT/f).relative_to(ROOT)):digest(ROOT/f) for f in ['src/training/graph_sft.py','src/training/master_optimizer.py','src/training/sft_checkpoint.py','src/training/alignment.py','src/models/control/encoders.py','src/models/control/fusion.py']}
    result={'status':'full_decoder_projectors_update_and_resume_pass','model':a.model,'arm':a.arm,'decoder_trainable_parameters':decoder_count,'decoder_all_parameters_trainable':all(p.requires_grad for p in m.decoder.parameters()),'language_head_trainable':all(p.requires_grad for p in m.head.parameters()),'total_trainable_parameters':trainable,'stage1_loss':l1,'stage2_loss':l2,'gradient_norm_stage1':n1,'gradient_norm_stage2':n2,'resume_loss_before':expected_loss,'resume_loss_after':resumed_loss,'full_checkpoint_bytes':size,'full_master_and_adamw_continuation_sha256':expected_fingerprint,'full_checkpoint_sha256':full_hash,'checkpoint_scope':'full decoder, head, projectors, CPU master weights, AdamW moments, scheduler, RNG and update cursor; temporary GPU-gate checkpoint removed after successful restore','resume_gate_seconds':time.perf_counter()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(0),'config_sha256':signature,'source_sha256':source,'items_per_stage':6,'limitation':'Diagnostic updates only; not production training or held-out accuracy'}
    write_json(folder/'complete.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
