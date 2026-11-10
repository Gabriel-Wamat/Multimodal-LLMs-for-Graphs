"""Submete os controles por modelo; pesos instalados ou preparação encadeada."""

import argparse
import json
import re
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--dinov3-source")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--node", help="Nó GPU opcional; por padrão o SLURM escolhe.")
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--afterok", help="IDs de jobs prévios, separados por dois-pontos")
    args = p.parse_args()
    if args.afterok and not re.fullmatch(r"[0-9]+(:[0-9]+)*", args.afterok):
        raise ValueError("Dependência inválida")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,60}", args.run_id):
        raise ValueError("ID inválido")
    if args.prepare and not args.dinov3_source:
        raise ValueError("--prepare exige --dinov3-source")
    for model in ["llava", "qwen"]:
        if (ROOT / "results" / f"{args.run_id}-{model}").exists():
            raise ValueError("Use um novo run-id")
    path = ROOT / "experiments/submissions" / f"{args.run_id}.json"
    if path.exists() and not args.dry_run:
        raise ValueError("Submissão já registrada")
    record = {
        "run_id": args.run_id,
        "smoke": args.smoke,
        "seed": args.seed,
        "jobs": [],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "submitting",
    }
    if not args.dry_run:
        (ROOT / "logs").mkdir(exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            json.dump(record, f, indent=2)

    def save():
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2) + "\n")
        tmp.replace(path)

    stages = []
    if args.prepare:
        stages = [
            ("setup", "setup_control.sh", [], False),
            (
                "weights",
                "download_control.sh",
                ["--dinov3-source", args.dinov3_source],
                False,
            ),
        ]
    for model in ["llava", "qwen"]:
        extra = [
            "--model",
            model,
            "--run-id",
            f"{args.run_id}-{model}",
            "--seed",
            str(args.seed),
        ]
        if args.smoke:
            extra += ["--smoke"]
        stages.append((model, "run_control.sh", extra, True))
    if args.smoke:
        stages.append(
            ("audit", "validate_control.sh", ["--run-id", args.run_id], False)
        )
    previous = args.afterok
    try:
        for stage, script, extra, gpu in stages:
            cmd = ["sbatch", "--parsable", f"--chdir={ROOT}"]
            if previous:
                cmd += [f"--dependency=afterok:{previous}", "--kill-on-invalid-dep=yes"]
            if gpu:
                if args.node:
                    cmd += [f"--nodelist={args.node}"]
                if args.smoke:
                    cmd += ["--time=00:30:00"]
            cmd += [f"scripts/apuana/{script}", *extra]
            print(shlex.join(cmd), flush=True)
            if args.dry_run:
                previous = f"<{stage}>"
            else:
                output = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
                previous = output.split(";")[0]
                if not previous.isdigit():
                    raise ValueError("Resposta inválida de sbatch")
                record["jobs"].append(
                    {"stage": stage, "job_id": previous, "command": cmd}
                )
                save()
    except Exception:
        if not args.dry_run:
            record["status"] = "submission_failed"
            save()
        raise
    if not args.dry_run:
        record["status"] = "submitted"
        save()
        print(path)


if __name__ == "__main__":
    main()
