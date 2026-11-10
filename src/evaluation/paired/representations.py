"""Global image embeddings for all fixed pairs; never used to select the dataset."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision.transforms.functional import pil_to_tensor
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from src.data.pilot import ROOT, json_text, write_once
from src.models.control.artifacts import digest, installed, write_json
from src.models.control.encoders import load_dinov3, square_rgb


def normalized(vector):
    vector = vector.float().flatten()
    norm = vector.norm()
    if not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid global representation")
    return vector / norm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    dataset = "pairs_v1_smoke" if args.smoke else "pairs_v1"
    inputs = ROOT / "data/manifests" / dataset / "inputs.jsonl"
    rows = [json.loads(line) for line in inputs.read_text().splitlines()]
    folder = ROOT / "results" / ("representations-" + dataset + "-v1")
    folder.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (folder / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    weights = {key: installed(key, verify=True) for key in ["clip", "dinov3"]}
    protocol = {
        "inputs_sha256": digest(inputs),
        "weights": weights,
        "source_sha256": digest(Path(__file__)),
        "encoder_helpers_sha256": digest(ROOT / "src/models/control/encoders.py"),
        "clip": "final pooled CLS after post-layernorm and pretrained visual projection; L2",
        "dinov3": "final layer-normalized CLS, excluding registers/patch pooling; L2",
        "preprocessing": "square pad RGB(122,116,104), CLIP saved processor 336; DINO bicubic 384 with ImageNet normalization",
        "precision": "float32 eager, TF32 disabled",
        "dataset_selection": "none; every preregistered pair retained",
        "paper_reference_only": "CLIP >0.95 and DINOv2 <0.6; DINOv3 here is a different encoder; not an exact replication",
    }
    write_once(folder / "run.json", json_text(protocol))
    write_once(folder / "source.py", Path(__file__).read_text())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    clip, info = CLIPVisionModelWithProjection.from_pretrained(
        weights["clip"]["path"],
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    if info["missing_keys"] or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError("CLIP projection weights are incomplete: " + str(info))
    clip = clip.cuda().eval().requires_grad_(False)
    processor = CLIPImageProcessor.from_pretrained(
        weights["clip"]["path"], local_files_only=True
    )
    dino = load_dinov3(weights["dinov3"]["path"], weights["dinov3"]["config"]).cuda()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)
    cache = folder / "embeddings"
    cache.mkdir(exist_ok=True)
    vectors = {}
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            path = ROOT / row["image"]
            if digest(path) != row["image_sha256"]:
                raise ValueError("Image changed: " + row["id"])
            target = cache / (row["image_sha256"] + ".safetensors")
            metadata = target.with_suffix(".json")
            if target.exists() and metadata.exists():
                if digest(target) != json.loads(metadata.read_text())["sha256"]:
                    raise ValueError("Corrupt embedding cache")
                value = load_file(str(target))
            else:
                with Image.open(path) as image:
                    image = square_rgb(image)
                pixels = processor(images=image, return_tensors="pt")[
                    "pixel_values"
                ].cuda()
                if pixels.shape[-2:] != (336, 336):
                    raise ValueError("Unexpected CLIP resolution")
                cv = normalized(clip(pixels).image_embeds[0]).cpu()
                pixels = pil_to_tensor(
                    image.resize((384, 384), Image.Resampling.BICUBIC)
                )
                pixels = pixels.cuda().float().div(255).unsqueeze(0)
                dv = normalized(
                    dino((pixels - mean) / std).last_hidden_state[0, 0]
                ).cpu()
                value = {"clip": cv, "dinov3": dv}
                temporary = target.with_suffix(".tmp")
                save_file(value, str(temporary))
                temporary.replace(target)
                write_json(metadata, {"sha256": digest(target)})
            for key, vector in value.items():
                torch.testing.assert_close(
                    vector.norm(), torch.tensor(1.0), rtol=1e-5, atol=1e-5
                )
            vectors[row["id"]] = value
            if (index + 1) % 100 == 0:
                print(json.dumps({"images": index + 1, "total": len(rows)}), flush=True)
    groups = defaultdict(dict)
    for row in rows:
        if row["member"] in groups[row["pair_id"]]:
            raise ValueError("Duplicate pair member")
        groups[row["pair_id"]][row["member"]] = row
    scores = []
    for pair_id, pair in sorted(groups.items()):
        if set(pair) != {"A", "B"}:
            raise ValueError("Incomplete pair")
        a, b = pair["A"], pair["B"]
        score = {
            k: a[k]
            for k in ["origin", "split", "tipo", "estilo", "attribute", "intensity"]
        }
        score["pair_id"] = pair_id
        for key in ["clip", "dinov3"]:
            score[key] = float(
                torch.dot(vectors[a["id"]][key], vectors[b["id"]][key]).clamp(-1, 1)
            )
        scores.append(score)
    write_json(folder / "scores.json", scores)
    write_json(
        folder / "complete.json",
        {
            "images": len(rows),
            "pairs": len(scores),
            "seconds": time.perf_counter() - started,
            "scores_sha256": digest(folder / "scores.json"),
            "run_sha256": digest(folder / "run.json"),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        },
    )


if __name__ == "__main__":
    main()
