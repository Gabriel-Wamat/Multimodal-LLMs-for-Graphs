"""Fixed visual-only conditions; ground truth and CSVs never enter model prompts."""

import hashlib
import unicodedata
from pathlib import Path
from PIL import Image


def binary(text):
    value = unicodedata.normalize("NFKD", text.strip().casefold())
    value = "".join(c for c in value if not unicodedata.combining(c)).strip(" .\n")
    return {"sim": "Sim", "yes": "Sim", "nao": "Não", "no": "Não"}.get(value)


def aggregate(responses):
    parsed = [binary(s) for s in responses]
    if len(parsed) != 4:
        raise ValueError("Four independent crop responses required")
    for answer in ["Sim", "Não"]:
        if parsed.count(answer) >= 3:
            return answer
    return None


def crop_images(path, cache):
    path = Path(path)
    cache = Path(cache)
    key = hashlib.sha256(path.read_bytes()).hexdigest()
    folder = cache / key
    folder.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        w, h = image.size
        mx, my = w // 2, h // 2
        boxes = [(0, 0, mx, my), (mx, 0, w, my), (0, my, mx, h), (mx, my, w, h)]
        out = []
        for index, box in enumerate(boxes):
            p = folder / f"quadrant-{index}.png"
            if not p.exists():
                image.crop(box).resize((w, h), Image.Resampling.BICUBIC).save(p)
            out.append(p)
    return out


def prompt(row):
    return row["pergunta"] + "\nAlternativas: Sim; Não. Responda somente Sim ou Não."


TRANSCRIBE = (
    "Transcreva o que consegue ler neste gráfico: eixos, unidades, legendas, categorias e valores aproximados. "
    "Para boxplots, registre medianas e quartis; para dispersão, descreva os pontos e a tendência. "
    "Se um valor não estiver legível, escreva ilegível. Não invente valores e não responda a uma pergunta."
)


def evaluate_condition(model, row, path, condition, cache):
    calls = []

    def call(image, text, stage, max_tokens=96):
        previous = model.cfg["max_new_tokens"]
        model.cfg["max_new_tokens"] = max_tokens
        try:
            result = model.answer(image, text, "native", row["id"])
        finally:
            model.cfg["max_new_tokens"] = previous
        calls.append(
            {
                "stage": stage,
                "image": str(image),
                "prompt": text,
                "max_new_tokens": max_tokens,
                **result,
            }
        )
        return result["response"]

    if condition == "original":
        raw = call(path, prompt(row), "answer")
        answer = binary(raw)
    elif condition == "crops_2x2":
        responses = [
            call(
                p,
                prompt(row) + f"\nEste é o recorte {i + 1} de uma grade 2×2.",
                f"crop_{i}",
            )
            for i, p in enumerate(crop_images(path, cache))
        ]
        answer = aggregate(responses)
        raw = answer or ""
    elif condition == "transcription":
        transcript = call(path, TRANSCRIBE, "transcribe", 384)
        text = (
            prompt(row)
            + "\nTranscrição visual prévia (pode conter erros):\n"
            + transcript
        )
        raw = call(path, text, "answer_after_transcription")
        answer = binary(raw)
    else:
        raise ValueError(condition)
    return {
        "response": raw,
        "parsed_answer": answer,
        "calls": calls,
        "seconds": sum(c["seconds"] for c in calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "generated_tokens": sum(c["generated_tokens"] for c in calls),
        "peak_gpu_bytes": max(c["peak_gpu_bytes"] for c in calls),
        "abstention_or_invalid": answer is None,
    }
