"""Entrada portátil para os modelos do controle MoF/DINOv3."""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-control/bin/python"


def call(args, **kwargs):
    return subprocess.run([str(a) for a in args], cwd=ROOT, check=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "install", help="Instala ambiente Linux/Python3.10 separado; não baixa pesos."
    )
    dl = sub.add_parser(
        "download", help="Baixa pesos públicos e importa o DINOv3 local."
    )
    dl.add_argument("--dinov3-source", required=True)
    ck = sub.add_parser("check")
    ck.add_argument("--hashes", action="store_true")
    for command in ("smoke", "run"):
        s = sub.add_parser(command)
        s.add_argument("--model", choices=["llava", "qwen"], required=True)
        s.add_argument("--run-id", required=True)
        s.add_argument("--seed", type=int)
        s.add_argument("--resume", action="store_true")
    score = sub.add_parser("score")
    score.add_argument("run_id")
    args = p.parse_args()
    if args.command == "install":
        if platform.system() != "Linux" or sys.version_info[:2] != (3, 10):
            raise ValueError("Use Linux/Python3.10; no Apuana execute via SLURM.")
        os.environ["PIP_CACHE_DIR"] = str(ROOT / ".cache/pip")
        call([sys.executable, "-m", "venv", ROOT / ".venv-control"])
        call([PYTHON, "-m", "pip", "install", "-r", "configs/requirements-control.txt"])
        call([PYTHON, "-m", "pip", "check"])
        folder = ROOT / ".cache/control"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "environment.txt").open("w") as stream:
            call([PYTHON, "-m", "pip", "freeze"], stdout=stream)
    elif args.command == "download":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        call(
            [
                PYTHON,
                "-m",
                "src.models.control.download",
                "--dinov3-source",
                args.dinov3_source,
            ]
        )
    elif args.command == "check":
        sys.path.insert(0, str(ROOT))
        from src.models.control.artifacts import installed

        for key in ("llava", "qwen", "clip", "dinov3"):
            record = installed(key, verify=args.hashes)
            print(key, record["status"])
    elif args.command == "score":
        call([PYTHON, "-m", "src.evaluation.control.score", args.run_id])
    else:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TORCH_HOME"] = str(ROOT / ".cache/control/torch")
        cmd = [
            PYTHON,
            "-m",
            "src.evaluation.control.run",
            "--model",
            args.model,
            "--run-id",
            args.run_id,
        ]
        if args.command == "smoke":
            cmd += ["--smoke"]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        if args.resume:
            cmd += ["--resume"]
        call(cmd)
        call([PYTHON, "-m", "src.evaluation.control.score", args.run_id])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        sys.exit(1)
