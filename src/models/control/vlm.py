"""LLaVA/Qwen congelados e MoF intercalado no estágio anterior ao alinhamento."""

import time

import torch
from PIL import Image
from src.data.pilot import ROOT

from .artifacts import REGISTRY, digest, installed
from .encoders import FrozenEncoders, square_rgb
from .fusion import (
    ARMS,
    StageZeroAdapters,
    expand_image_tokens,
    interleave,
    shuffle_patches,
)


class ControlledVLM:
    def __init__(self, key, config):
        from transformers import (
            AutoProcessor,
            LlavaForConditionalGeneration,
            Qwen2VLForConditionalGeneration,
        )

        if key not in ("llava", "qwen"):
            raise ValueError(key)
        from .hardware import validate_gpu

        validate_gpu(key, config)
        self.key = key
        self.cfg = config
        self.device = "cuda:0"
        self.record = installed(key)
        klass = (
            LlavaForConditionalGeneration
            if key == "llava"
            else Qwen2VLForConditionalGeneration
        )
        self.processor = AutoProcessor.from_pretrained(
            self.record["path"], local_files_only=True, use_fast=True
        )
        self.model = (
            klass.from_pretrained(
                self.record["path"],
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                device_map={"": 0},
            )
            .eval()
            .requires_grad_(False)
        )
        self.core = self.model.model
        hidden = (
            self.model.config.text_config.hidden_size
            if key == "llava"
            else self.model.config.hidden_size
        )
        if hidden != (4096 if key == "llava" else 1536):
            raise ValueError("Dimensão do decoder inesperada")
        self.adapters = StageZeroAdapters(
            key, hidden, config["adapter_seed"], config["adapter_init_std"]
        )
        meta = {
            "model": key,
            "model_config": self.record["config"],
            "registry_sha256": digest(REGISTRY),
            "hidden_size": hidden,
            "seed": config["adapter_seed"],
            "std": config["adapter_init_std"],
            "type": config["adapter_type"],
            "llava_clip_reuses_native": key == "llava",
        }
        folder = (
            ROOT
            / "experiments"
            / config["experiment"]
            / "initializers"
            / key
            / str(config["adapter_seed"])
        )
        self.adapter_sha = self.adapters.persist(folder, meta)
        self.adapters.to(device=self.device, dtype=torch.bfloat16)
        self.encoders = None
        self.parameter_versions = {
            n: p._version for n, p in self.model.named_parameters()
        }
        self.adapter_versions = {
            n: p._version for n, p in self.adapters.named_parameters()
        }

    def check_frozen(self):
        if any(
            p.requires_grad or p._version != self.parameter_versions[n]
            for n, p in self.model.named_parameters()
        ):
            raise RuntimeError("Pesos originais alterados")
        if any(
            p.requires_grad or p._version != self.adapter_versions[n]
            for n, p in self.adapters.named_parameters()
        ):
            raise RuntimeError("Adaptadores alterados durante inferência")

    def inputs(self, image, prompt):
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        kwargs = {}
        if self.key == "qwen":
            size = self.cfg["qwen_resolution"]
            image = image.resize((size, size), Image.Resampling.BICUBIC)
            kwargs = {"min_pixels": size * size, "max_pixels": size * size}
        data = self.processor(
            text=[text], images=[image], return_tensors="pt", **kwargs
        )
        data = {
            k: v.to(self.device) if hasattr(v, "to") else v for k, v in data.items()
        }
        if self.key == "qwen" and data["image_grid_thw"].tolist() != [[1, 48, 48]]:
            raise ValueError(f"Grade Qwen inesperada: {data['image_grid_thw']}")
        token = (
            self.model.config.image_token_index
            if self.key == "llava"
            else self.model.config.image_token_id
        )
        if (data["input_ids"] == token).sum().item() != 576:
            raise ValueError("Número de tokens nativos inesperado")
        return data, token

    def answer(self, path, prompt, arm, image_id, verify_native=False):
        if arm not in ARMS:
            raise ValueError(arm)
        self.check_frozen()
        with Image.open(path) as image:
            image = square_rgb(image)
        inputs, image_token = self.inputs(image, prompt)
        torch.cuda.synchronize()
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        external = None
        feature_meta = None
        if arm in ("dino_double", "clip_dino", "clip_dino_shuffled") or (
            self.key == "qwen" and arm == "clip_double"
        ):
            if self.encoders is None:
                self.encoders = FrozenEncoders(self.cfg, self.device)
            external = self.encoders.features(path)
            feature_meta = self.encoders.last_metadata
        original = self.core.get_image_features
        original_grid = inputs.get("image_grid_thw")
        trace = []
        if arm != "native":
            inputs = expand_image_tokens(inputs, image_token)
            if self.key == "qwen":
                inputs["image_grid_thw"] = original_grid.clone()
                inputs["image_grid_thw"][:, 2] *= 2

        def intervention(*args, **kwargs):
            # Todos os braços calculam o caminho nativo; Qwen usa a grade física original aqui.
            if self.key == "qwen":
                pixel_values = args[0] if args else kwargs["pixel_values"]
                native = original(pixel_values, original_grid)
            else:
                native = original(*args, **kwargs)
            if len(native) != 1 or native[0].shape[0] != 576:
                raise ValueError("Somente uma imagem de 576 patches é suportada")
            visual = native[0]
            if arm == "native":
                output = visual
            elif arm == "native_double":
                output = interleave(visual, visual)
            else:
                clip = visual if self.key == "llava" else None
                if self.key == "qwen" and arm != "dino_double":
                    clip = self.adapters.clip(external[0].to(visual.dtype))
                dino = None
                if arm != "clip_double":
                    dino = self.adapters.dino(external[1].to(visual.dtype))
                if arm == "clip_double":
                    output = interleave(clip, clip)
                elif arm == "dino_double":
                    output = interleave(dino, dino)
                else:
                    if arm == "clip_dino_shuffled":
                        dino = shuffle_patches(dino, image_id, self.cfg["adapter_seed"])
                    output = interleave(clip, dino)
            if not torch.isfinite(output).all():
                raise ValueError("Tokens projetados não finitos")
            trace.append(
                {
                    "native_tokens": len(visual),
                    "output_tokens": len(output),
                    "hidden_size": output.shape[1],
                    "native_norm_mean": visual.float().norm(dim=-1).mean().item(),
                    "output_norm_mean": output.float().norm(dim=-1).mean().item(),
                }
            )
            return [output]

        self.core.get_image_features = intervention
        try:
            with torch.inference_mode():
                torch.manual_seed(self.cfg["adapter_seed"])
                torch.cuda.manual_seed_all(self.cfg["adapter_seed"])
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg["max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
        finally:
            self.core.get_image_features = original
        parity = None
        if verify_native:
            if arm != "native":
                raise ValueError("Verificação nativa exige braço native")
            with torch.inference_mode():
                torch.manual_seed(self.cfg["adapter_seed"])
                torch.cuda.manual_seed_all(self.cfg["adapter_seed"])
                reference = self.model.generate(
                    **inputs,
                    max_new_tokens=self.cfg["max_new_tokens"],
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            parity = torch.equal(reference, output)
            if not parity:
                raise RuntimeError("Controle nativo divergiu do modelo sem intervenção")
        if len(trace) != 1:
            raise RuntimeError(f"Intervenção visual chamada {len(trace)} vezes")
        if not torch.equal(
            output[:, : inputs["input_ids"].shape[1]], inputs["input_ids"]
        ):
            raise RuntimeError("Prefixo de saída divergente")
        response = self.processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0].strip()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.check_frozen()
        return {
            "response": response,
            "seconds": elapsed,
            "input_tokens": inputs["input_ids"].shape[1],
            "generated_tokens": output.shape[1] - inputs["input_ids"].shape[1],
            "native_reference_parity": parity,
            "latency_includes_reference_check": verify_native,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "visual_trace": trace[0],
            "feature_cache": feature_meta,
            "adapter_sha256": self.adapter_sha,
            "quantization": None,
            "precision": "bfloat16",
            "training_steps": 0,
        }
