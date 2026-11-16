"""AdamW with FP32 CPU master weights and moments; GPU parameters keep model dtype."""
import hashlib
import json
import torch


class MasterAdamW:
    def __init__(self,named_parameters,lr,weight_decay=0.0):
        self.sources=dict(named_parameters)
        if not self.sources:raise ValueError('No trainable parameters')
        self.masters={name:torch.nn.Parameter(p.detach().to(device='cpu',dtype=torch.float32).clone()) for name,p in self.sources.items()}
        self.optimizer=torch.optim.AdamW(self.masters.values(),lr=lr,weight_decay=weight_decay,foreach=False)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.sources.values():p.grad=None

    def step(self,max_norm=1.0):
        for name,p in self.sources.items():
            if p.grad is None:raise RuntimeError('Missing gradient: '+name)
            self.masters[name].grad=p.grad.detach().to(device='cpu',dtype=torch.float32)
        norm=torch.nn.utils.clip_grad_norm_(list(self.masters.values()),max_norm,error_if_nonfinite=True)
        self.optimizer.step()
        with torch.no_grad():
            for name,p in self.sources.items():p.copy_(self.masters[name].to(device=p.device,dtype=p.dtype))
        return float(norm)

    def fingerprint(self):
        """Hash every master weight and AdamW moment without cloning the whole model."""
        h=hashlib.sha256()
        for name,p in self.masters.items():
            tensors={'master':p.detach(),**self.optimizer.state.get(p,{})}
            for key,value in sorted(tensors.items()):
                if torch.is_tensor(value):
                    value=value.detach().cpu().contiguous()
                    h.update(json.dumps([name,key,list(value.shape),str(value.dtype)]).encode())
                    h.update(memoryview(value.numpy()).cast('B'))
                else:h.update(json.dumps([name,key,value]).encode())
        return h.hexdigest()

    def state_dict(self):
        return {'names':list(self.sources),'masters':{n:p.detach() for n,p in self.masters.items()},'optimizer':self.optimizer.state_dict()}

    def load_state_dict(self,state):
        if state['names']!=list(self.sources):raise ValueError('Parameter inventory changed')
        with torch.no_grad():
            for name,p in self.masters.items():
                p.data=state['masters'][name].to(device='cpu',dtype=torch.float32)
                self.sources[name].copy_(p.to(device=self.sources[name].device,dtype=self.sources[name].dtype))
        self.optimizer.load_state_dict(state['optimizer'])
