"""Differentiable projector alignment; frozen vision backbones and language decoder."""

from contextlib import contextmanager
import torch
from PIL import Image
from src.models.control.vlm import ControlledVLM
from src.models.control.encoders import FrozenEncoders, square_rgb
from src.models.control.fusion import expand_image_tokens, interleave
from src.evaluation.run import prompt_for
from src.data.pilot import ROOT


def supervised_inputs(inputs, answer_ids):
    if inputs["input_ids"].shape[0] != 1 or not answer_ids:
        raise ValueError("Nonempty single-example target required")
    target = torch.tensor(
        [answer_ids], dtype=inputs["input_ids"].dtype, device=inputs["input_ids"].device
    )
    result = dict(inputs)
    prefix = inputs["input_ids"]
    result["input_ids"] = torch.cat([prefix, target], dim=1)
    result["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones_like(target)], dim=1
    )
    result["labels"] = torch.cat([torch.full_like(prefix, -100), target], dim=1)
    result.pop("token_type_ids", None)
    return result


class AlignmentModel:
    def __init__(self, key, config, arm):
        if arm not in ("clip_double", "clip_dino"):
            raise ValueError("Controlled training arms only")
        from src.models.control.hardware import validate_gpu

        validate_gpu(key, config, training=True)
        self.key = key
        self.arm = arm
        self.vlm = ControlledVLM(key, config)
        self.encoders = FrozenEncoders(config, self.vlm.device)
        clip = (
            self.vlm.core.multi_modal_projector
            if key == "llava"
            else self.vlm.adapters.clip
        )
        self.projectors = torch.nn.ModuleDict({"clip": clip})
        if arm == "clip_dino":
            self.projectors["dino"] = self.vlm.adapters.dino
        self.projectors.float().requires_grad_(True)
        self.projectors.train()
        trainable = {id(p) for p in self.projectors.parameters()}
        self.frozen_versions = {
            n: p._version
            for n, p in self.vlm.model.named_parameters()
            if id(p) not in trainable
        }
        assert all(
            not p.requires_grad
            for n, p in self.vlm.model.named_parameters()
            if n in self.frozen_versions
        )
        self.initial_adapter_sha256 = self.vlm.adapter_sha

    def assert_frozen(self):
        for n, p in self.vlm.model.named_parameters():
            if n in self.frozen_versions:
                if p.requires_grad or p._version != self.frozen_versions[n]:
                    raise RuntimeError("Frozen backbone changed: " + n)

    @contextmanager
    def visual_intervention(self, original_grid, external):
        original = self.vlm.core.get_image_features
        count = []

        def features(*args, **kwargs):
            if self.key == "llava":
                clip = original(*args, **kwargs)[0]
            else:
                # Qwen native visual branch is deliberately absent in CLIP-only/fusion training.
                clip = self.projectors["clip"](external[0])
            if clip.shape[0] != 576:
                raise ValueError("Unexpected visual grid")
            dino = (
                self.projectors["dino"](external[1])
                if self.arm == "clip_dino"
                else clip
            )
            output = interleave(clip, dino)
            if torch.is_grad_enabled() and not output.requires_grad:
                raise RuntimeError("Projector gradient disconnected")
            count.append(1)
            return [output]

        self.vlm.core.get_image_features = features
        try:
            yield
        finally:
            self.vlm.core.get_image_features = original
        if len(count) != 1:
            raise RuntimeError("Unexpected number of vision calls")

    def loss(self, row):
        path = ROOT / row["image"]
        with Image.open(path) as im:
            image = square_rgb(im)
        inputs, token = self.vlm.inputs(image, prompt_for(row))
        grid = inputs.get("image_grid_thw")
        inputs = expand_image_tokens(inputs, token)
        if self.key == "qwen":
            inputs["image_grid_thw"] = grid.clone()
            inputs["image_grid_thw"][:, 2] *= 2
        target = str(row["resposta_txt"]).strip()
        if not target:
            raise ValueError("Empty supervised target")
        tokenizer = self.vlm.processor.tokenizer
        answer_ids = tokenizer.encode(target, add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        if len(answer_ids) > 96:
            raise ValueError("Answer exceeds predefined length; no silent truncation")
        inputs = supervised_inputs(inputs, answer_ids)
        external = None
        if self.key == "qwen" or self.arm == "clip_dino":
            features = self.encoders.features(path)
            # Cache misses return inference tensors: clone OUTSIDE inference_mode before autograd.
            external = tuple(x.detach().clone().to(torch.bfloat16) for x in features)
        with (
            self.visual_intervention(grid, external),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            output = self.vlm.model(**inputs, use_cache=False)
        if not torch.isfinite(output.loss):
            raise RuntimeError("Nonfinite loss")
        self.assert_frozen()
        return output.loss, len(answer_ids)

    @torch.no_grad()
    def answer(self, row):
        path = ROOT / row["image"]
        with Image.open(path) as im:
            image = square_rgb(im)
        inputs, token = self.vlm.inputs(image, prompt_for(row))
        grid = inputs.get("image_grid_thw")
        inputs = expand_image_tokens(inputs, token)
        if self.key == "qwen":
            inputs["image_grid_thw"] = grid.clone()
            inputs["image_grid_thw"][:, 2] *= 2
        external = None
        if self.key == "qwen" or self.arm == "clip_dino":
            external = tuple(
                x.clone().to(torch.bfloat16) for x in self.encoders.features(path)
            )
        with (
            self.visual_intervention(grid, external),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            output = self.vlm.model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        self.assert_frozen()
        return self.vlm.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()

    def save_projectors(self, path):
        from safetensors.torch import save_file

        save_file(
            {
                k: v.detach().cpu().contiguous()
                for k, v in self.projectors.state_dict().items()
            },
            str(path),
        )

    def load_projectors(self, path):
        from safetensors.torch import load_file

        self.projectors.load_state_dict(load_file(str(path)), strict=True)
