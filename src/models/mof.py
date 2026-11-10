"""Adapter para o I-MoF oficial, preservando x_prenorm e interleaving dos autores."""
import json
import subprocess
import sys

from src.data.pilot import ROOT
from src.models.artifacts import weights_manifest_path


class MoF:
    def __init__(self, config):
        if config['dtype'] != 'float16' or config['quantization'] is not None or config['do_sample']:
            raise ValueError('Este adapter implementa somente FP16 sem quantização e geração greedy.')
        import torch
        from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel
        vendor = ROOT/'work/vendor/MMVP'
        for repo, revision in [(vendor,config['code_revision']), (ROOT/'work/vendor/dinov2',config['dino_revision'])]:
            actual = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
            if actual != revision:
                raise RuntimeError(f'Revisão divergente: {repo}')
        sys.path.insert(0, str(vendor/'LLaVA'))
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
        from llava.model.multimodal_encoder.dino_encoder import DINOVisionTower
        from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
        import llava.model.llava_arch as llava_arch
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, process_images
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

        # Mesmo cálculo do upstream: só fixa caminhos locais e remove o print de todos os pesos.
        def load_clip(tower):
            tower.image_processor = CLIPImageProcessor.from_pretrained(tower.vision_tower_name, local_files_only=True)
            tower.vision_tower = CLIPVisionModel.from_pretrained(tower.vision_tower_name, local_files_only=True)
            tower.vision_tower.requires_grad_(False)
            tower.is_loaded = True

        def load_dino(tower):
            tower.image_processor = CLIPImageProcessor.from_pretrained(tower.vision_tower_name, local_files_only=True)
            tower.clip_vision_tower = CLIPVisionModel.from_pretrained(tower.vision_tower_name, local_files_only=True)
            tower.vision_tower = torch.hub.load(str(ROOT/'work/vendor/dinov2'), 'dinov2_vitl14', source='local')
            tower.vision_tower.requires_grad_(False)
            tower.is_loaded = True

        CLIPVisionTower.load_model = load_clip
        DINOVisionTower.load_model = load_dino
        weights = json.loads(weights_manifest_path().read_text())
        if weights['config'] != config:
            raise RuntimeError('Configuração dos pesos diverge da execução')
        def build_local_tower(tower_config, load_model='clip', **kwargs):
            name = tower_config.mm_vision_tower
            if name != weights['paths']['clip'] or load_model not in ('clip', 'dino'):
                raise ValueError('Encoder fora do snapshot fixado')
            tower_class = CLIPVisionTower if load_model == 'clip' else DINOVisionTower
            return tower_class(name, args=tower_config, **kwargs)

        # O builder legado só aceita nomes iniciados por openai/laion, não caminhos.
        # Mantém as mesmas classes e encaminha exclusivamente o snapshot verificado.
        llava_arch.build_vision_tower = build_local_tower
        path = weights['paths']['model']
        model_config = LlavaConfig.from_pretrained(path, local_files_only=True)
        model_config.mm_vision_tower = weights['paths']['clip']
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
        self.model, loading = LlavaLlamaForCausalLM.from_pretrained(path, config=model_config,
            torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'': 0},
            local_files_only=True, output_loading_info=True)
        for kind in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'):
            bad = [k for k in loading.get(kind,[]) if not isinstance(k,str) or not
                   (k.startswith('model.vision_tower.') or k.startswith('model.dino_tower.'))]
            if bad:
                raise RuntimeError(f'Falha no carregamento de pesos: {kind}: {bad[:10]}')
        for tower in [self.model.get_vision_tower(), self.model.get_dino_vision_tower()]:
            tower.load_model()
            tower.to(device='cuda:0',dtype=torch.float16)
        self.model.eval()
        self.processor = self.model.get_vision_tower().image_processor
        self.config = config
        self.conv_templates, self.sep_style = conv_templates, SeparatorStyle
        self.tokenizer_image_token, self.process_images = tokenizer_image_token, process_images
        self.image_token_index, self.image_token = IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        self.metadata = {'loading': loading, 'gpu': torch.cuda.get_device_name(0),
                         'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                         'processor': self.processor.to_dict()}
        if self.processor.crop_size != {'height':config['resolution'],'width':config['resolution']}:
            raise RuntimeError('Resolução do processor divergente')

    def answer(self, image_path, prompt):
        import torch
        from PIL import Image
        conv = self.conv_templates[self.config['conversation']].copy()
        conv.append_message(conv.roles[0], self.image_token+'\n'+prompt)
        conv.append_message(conv.roles[1], None)
        ids = self.tokenizer_image_token(conv.get_prompt(), self.tokenizer,
              self.image_token_index, return_tensors='pt').unsqueeze(0).cuda()
        with Image.open(image_path) as image:
            image_tensor = self.process_images([image.convert('RGB')], self.processor, self.model.config)
        image_tensor = image_tensor.to(device='cuda',dtype=torch.float16)
        with torch.inference_mode():
            output = self.model.generate(ids, images=image_tensor, do_sample=False,
                num_beams=1, max_new_tokens=self.config['max_new_tokens'], use_cache=True)
        if not torch.equal(ids, output[:,:ids.shape[1]]):
            raise RuntimeError('Saída não contém o prefixo esperado; revisar decodificação')
        text = self.tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True).strip()
        stop = conv.sep if conv.sep_style != self.sep_style.TWO else conv.sep2
        if stop and text.endswith(stop):
            text = text[:-len(stop)].strip()
        return text
