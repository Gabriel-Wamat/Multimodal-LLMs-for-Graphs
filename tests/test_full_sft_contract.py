"""CPU contracts for FP32 master AdamW and full-decoder continuation."""
import importlib.util
import tempfile
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch profile required')
class FullSFTContracts(unittest.TestCase):
    def test_master_optimizer_matches_native_adamw_and_resume(self):
        import torch
        from src.training.master_optimizer import MasterAdamW
        from src.training.sft_checkpoint import save,restore
        class Model:
            def __init__(self):self.module=torch.nn.Linear(3,2);self.stage=2
            def weight_state(self):return {'stage':2,'weights':self.module.state_dict()}
            def load_weights(self,state):self.module.load_state_dict(state['weights'])
        torch.manual_seed(7);m=Model();ref=torch.nn.Linear(3,2);ref.load_state_dict(m.module.state_dict())
        opt=MasterAdamW(m.module.named_parameters(),.01)
        native=torch.optim.AdamW(ref.parameters(),lr=.01,weight_decay=0.,foreach=False)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt.optimizer,lambda step:1./(step+1))
        x=torch.tensor([[1.,2.,3.],[-1.,.5,2.]])
        def take(model,opt):
            opt.zero_grad();loss=model.module(x).square().mean();loss.backward();opt.step();return loss.detach()
        take(m,opt);scheduler.step();native.zero_grad();ref(x).square().mean().backward();torch.nn.utils.clip_grad_norm_(list(ref.parameters()),1.);native.step()
        for a,b in zip(m.module.parameters(),ref.parameters()):self.assertTrue(torch.equal(a,b))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'full.pt';save(path,m,opt,scheduler,{'stage':2,'updates':1},'signature')
            expected_loss=take(m,opt);expected={n:p.clone() for n,p in m.module.named_parameters()}
            restored=Model();new,schedule,state=restore(path,restored,lambda model:MasterAdamW(model.module.named_parameters(),.01),lambda opt,stage:torch.optim.lr_scheduler.LambdaLR(opt.optimizer,lambda step:1./(step+1)),'signature')
            self.assertTrue(torch.equal(expected_loss,take(restored,new)))
            for n,p in restored.module.named_parameters():self.assertTrue(torch.equal(expected[n],p))
            self.assertEqual(opt.fingerprint(),new.fingerprint())
            with self.assertRaises(ValueError):restore(path,restored,lambda model:None,lambda opt,stage:None,'different')

    def test_nonfinite_gradients_are_rejected(self):
        import torch
        from src.training.master_optimizer import MasterAdamW
        p=torch.nn.Parameter(torch.ones(2));o=MasterAdamW([('p',p)],.01);p.grad=torch.full_like(p,float('nan'))
        with self.assertRaises(RuntimeError):o.step()
