"""Two-stage graph adaptation: projector alignment then full decoder instruction tuning."""
from contextlib import contextmanager
import torch
from PIL import Image
from src.data.pilot import ROOT
from src.evaluation.run import prompt_for
from src.models.control.fusion import expand_image_tokens,interleave
from src.models.control.encoders import square_rgb
from src.training.alignment import AlignmentModel,supervised_inputs


class GraphSFTModel(AlignmentModel):
    def __init__(self,key,config,arm,max_length=2048):
        if arm not in ['clip_single','clip_double','clip_dino']:raise ValueError(arm)
        super().__init__(key,config,'clip_dino' if arm=='clip_dino' else 'clip_double')
        self.arm=arm;self.max_length=max_length;self.stage=1
        self.decoder=self.vlm.core.language_model
        self.head=self.vlm.model.get_output_embeddings()
        self.components=torch.nn.ModuleDict({'decoder':self.decoder,'head':self.head,'projectors':self.projectors})
        self.set_stage(1)

    def set_stage(self,stage):
        if stage not in [1,2]:raise ValueError(stage)
        self.stage=stage
        self.vlm.model.eval().requires_grad_(False)
        self.projectors.float().train().requires_grad_(True)
        if stage==2:
            self.decoder.train().requires_grad_(True)
            self.head.train().requires_grad_(True)
            self.decoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:
            self.decoder.eval().requires_grad_(False)
            self.head.eval().requires_grad_(False)
        active={id(p) for _,p in self.named_trainable()}
        self.frozen_versions={n:p._version for n,p in self.vlm.model.named_parameters() if id(p) not in active}
        if stage==2 and not any(p.requires_grad for p in self.decoder.parameters()):raise RuntimeError('Decoder is frozen in instruction tuning')

    def named_trainable(self):
        return [(n,p) for n,p in self.components.named_parameters() if p.requires_grad]

    @contextmanager
    def visual_intervention(self,original_grid,external):
        original=self.vlm.core.get_image_features;calls=[]
        def features(*args,**kwargs):
            clip=self.projectors['clip'](external[0])
            if self.arm=='clip_single':out=clip
            else:out=interleave(clip,self.projectors['dino'](external[1]) if self.arm=='clip_dino' else clip)
            expected=576 if self.arm=='clip_single' else 1152
            if out.shape[0]!=expected:raise RuntimeError('Visual token count mismatch')
            calls.append(1)
            return [out]
        self.vlm.core.get_image_features=features
        try:yield
        finally:self.vlm.core.get_image_features=original
        if len(calls)!=1:raise RuntimeError('Visual intervention call count mismatch')

    def inputs_for(self,row):
        path=ROOT/row['image']
        with Image.open(path) as im:image=square_rgb(im)
        prompt=row['pergunta'] if row.get('supervision') else prompt_for(row)
        inputs,token=self.vlm.inputs(image,prompt)
        grid=inputs.get('image_grid_thw')
        if self.arm!='clip_single':
            inputs=expand_image_tokens(inputs,token)
            if self.key=='qwen':inputs['image_grid_thw']=grid.clone();inputs['image_grid_thw'][:,2]*=2
        external=tuple(x.detach().clone().to(torch.bfloat16) for x in self.encoders.features(path))
        return inputs,grid,external

    def loss(self,row):
        inputs,grid,external=self.inputs_for(row)
        tokenizer=self.vlm.processor.tokenizer
        ids=tokenizer.encode(str(row['resposta_txt']).strip(),add_special_tokens=False)+[tokenizer.eos_token_id]
        if not str(row['resposta_txt']).strip():raise ValueError('Empty target')
        inputs=supervised_inputs(inputs,ids)
        if inputs['input_ids'].shape[1]>self.max_length:raise ValueError('Sequence exceeds fixed length; no truncation')
        with self.visual_intervention(grid,external),torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.vlm.model(**inputs,use_cache=False)
        if not torch.isfinite(output.loss):raise RuntimeError('Nonfinite loss')
        self.assert_frozen()
        return output.loss,len(ids)

    @torch.no_grad()
    def answer(self,row):
        inputs,grid,external=self.inputs_for(row)
        decoder_training=self.decoder.training
        self.decoder.eval()
        try:
            with self.visual_intervention(grid,external),torch.autocast('cuda',dtype=torch.bfloat16):
                out=self.vlm.model.generate(**inputs,max_new_tokens=96,do_sample=False,num_beams=1,use_cache=True)
        finally:self.decoder.train(decoder_training)
        self.assert_frozen()
        return self.vlm.processor.batch_decode(out[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()

    def weight_state(self):
        if self.stage==1:return {'stage':1,'weights':self.projectors.state_dict()}
        return {'stage':2,'weights':self.components.state_dict()}

    def load_weights(self,state):
        self.set_stage(state['stage'])
        (self.projectors if state['stage']==1 else self.components).load_state_dict(state['weights'],strict=True)
        # Decoder version counters changed during load; only genuinely frozen tensors are checked.
        self.set_stage(state['stage'])
