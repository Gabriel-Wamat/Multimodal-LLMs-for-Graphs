"""CLIP e DINOv3 congelados, com cache rastreável por imagem/preprocessamento."""

import json
import time
from pathlib import Path

from PIL import Image
from src.data.pilot import ROOT

from .artifacts import digest, installed, write_json


def square_rgb(image):
    image = image.convert("RGB")
    size = max(image.size)
    canvas = Image.new("RGB", (size, size), (122, 116, 104))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def load_dinov3(path, config):
    """Load the identified HF safetensors with an explicit architecture and strict keys."""
    import torch
    from safetensors.torch import load_file
    from transformers import DINOv3ViTConfig, DINOv3ViTModel

    architecture = DINOv3ViTConfig(**config["architecture_config"])
    architecture._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = DINOv3ViTModel(architecture)
    model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    return model.eval().requires_grad_(False)


def dinov3_pre_norm(model, pixels):
    """Capture final-norm input; HF hidden_states[-1] may be replaced by normalized output."""
    import torch

    captured = []
    handle = model.norm.register_forward_pre_hook(
        lambda module, args: captured.append(args[0])
    )
    try:
        output = model(pixels)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("Esperada uma normalização final DINOv3")
    raw = captured[0]
    torch.testing.assert_close(
        model.norm(raw), output.last_hidden_state, rtol=1e-5, atol=1e-5
    )
    return raw


class FrozenEncoders:
    def __init__(self, config, device="cuda:0"):
        self.cfg = config
        self.device = device
        self.clip = None
        self.dino = None
        self.clip_record = installed("clip")
        self.dino_record = installed("dinov3")
        import hashlib

        stamp = {
            "clip": self.clip_record["config"],
            "dino": self.dino_record["config"],
            "clip_files": self.clip_record["files"],
            "dino_files": self.dino_record["files"],
            "clip_size": config["clip_resolution"],
            "dino_size": config["dino_resolution"],
            "layer": config["clip_layer"],
            "dino_features": config["dino_features"],
            "source": digest(Path(__file__)),
        }
        self.cache_id = hashlib.sha256(
            json.dumps(stamp, sort_keys=True).encode()
        ).hexdigest()
        self.folder = ROOT / ".cache/control/features" / self.cache_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.last_metadata = {}

    def load(self):
        if self.clip is not None:
            return
        import torch
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.clip_processor = CLIPImageProcessor.from_pretrained(
            self.clip_record["path"], local_files_only=True
        )
        self.clip = (
            CLIPVisionModel.from_pretrained(
                self.clip_record["path"],
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.dino = load_dinov3(
            self.dino_record["path"], self.dino_record["config"]
        ).to(self.device)

    def features(self, path):
        import torch
        from safetensors.torch import load_file, save_file
        from torchvision.transforms.functional import pil_to_tensor

        image_sha = digest(path)
        cache = self.folder / (image_sha + ".safetensors")
        meta = cache.with_suffix(".json")
        if cache.exists() and meta.exists():
            record = json.loads(meta.read_text())
            if (
                record["cache_id"] != self.cache_id
                or record["image_sha256"] != image_sha
                or digest(cache) != record["sha256"]
            ):
                raise ValueError("Cache de features divergente")
            tensors = load_file(str(cache), device=self.device)
            self.last_metadata = record | {"cache_hit": True}
            return tensors["clip"], tensors["dino"]
        self.load()
        with Image.open(path) as im:
            im = square_rgb(im)
        torch.cuda.synchronize()
        start = time.perf_counter()
        clip_pixels = self.clip_processor(images=im, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)
        dino_image = im.resize(
            (self.cfg["dino_resolution"],) * 2, Image.Resampling.BICUBIC
        )
        pixels = pil_to_tensor(dino_image).float().div(255).unsqueeze(0).to(self.device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        with torch.inference_mode():
            clip = (
                self.clip(clip_pixels, output_hidden_states=True)
                .hidden_states[self.cfg["clip_layer"]][0, 1:]
                .float()
            )
            raw = dinov3_pre_norm(self.dino, (pixels - mean) / std)
            n = self.cfg["grid_side"] ** 2
            if raw.shape[1] != n + 5:
                raise ValueError("Tokens DINOv3 inesperados")
            dino = raw[0, 5:].float()
        if clip.shape != (576, 1024) or dino.shape != (576, 768):
            raise ValueError("Grade/dimensões dos encoders divergentes")
        if not torch.isfinite(clip).all() or not torch.isfinite(dino).all():
            raise ValueError("Features não finitas")
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tmp = cache.with_suffix(".tmp")
        save_file(
            {"clip": clip.cpu().contiguous(), "dino": dino.cpu().contiguous()}, str(tmp)
        )
        tmp.replace(cache)
        record = {
            "cache_id": self.cache_id,
            "image_sha256": image_sha,
            "sha256": digest(cache),
            "encoder_seconds": elapsed,
            "clip_shape": list(clip.shape),
            "dino_shape": list(dino.shape),
        }
        write_json(meta, record)
        self.last_metadata = record | {"cache_hit": False}
        return clip, dino
