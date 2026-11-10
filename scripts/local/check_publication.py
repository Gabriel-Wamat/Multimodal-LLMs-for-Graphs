"""Check the reviewed Git file set for private artifacts, notebook outputs and broken links."""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PERSONAL_PATH = re.compile(r"[/\\](?:Users|home)[/\\][A-Za-z0-9_.-]+", re.I)
PRIVATE_IP = re.compile(r"\b172\.2[01]\.\d{1,3}\.\d{1,3}\b")
TOKEN = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----)")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def check_file(name, content, root=ROOT):
    issues = []
    p = Path(name)
    if p.suffix in {'.pt', '.pth', '.bin', '.safetensors', '.ckpt'}:
        issues.append('model checkpoint')
    if len(content) > 2 * 1024 * 1024:
        issues.append('file exceeds 2 MiB publication limit')
    if name.startswith(('dados_projeto_multi_modais/', 'data/manifests/', 'data/processed/', 'experiments/')) and p.name != 'README.md':
        issues.append('private dataset/runtime artifact')
    if name.startswith('results/') and name != 'results/README.md' and not name.startswith('results/published/'):
        issues.append('private result archive')
    if name.startswith(('.cache/', '.venv', 'work/', 'logs/')):
        issues.append('generated local directory')
    if p.name.startswith('.env') and p.name != '.env.example':
        issues.append('environment secrets file')
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        if p.suffix not in {'.png', '.jpg'}:
            issues.append('unexpected binary file')
        return issues
    if PERSONAL_PATH.search(text) or PRIVATE_IP.search(text):
        issues.append('personal path or private infrastructure address')
    if TOKEN.search(text):
        issues.append('possible credential/private key')
    if p.suffix == '.ipynb':
        notebook = json.loads(text)
        if any(c.get('outputs') or c.get('execution_count') is not None for c in notebook['cells'] if c['cell_type']=='code'):
            issues.append('notebook contains execution outputs/state')
    if p.suffix == '.md':
        for target in LINK.findall(text):
            target = target.strip('<>').split('#',1)[0]
            if not target or re.match(r'^[a-z]+:',target):
                continue
            if not (root / p.parent / target).exists():
                issues.append('missing relative link: ' + target)
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true', help='Read staged bytes instead of the worktree')
    args = parser.parse_args()
    cmd = ['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'] if args.staged else ['git','ls-files','-z']
    names = subprocess.check_output(cmd,cwd=ROOT).decode().split('\0')
    names = [n for n in names if n]
    if not names:
        raise SystemExit('No reviewed Git files found; stage explicit paths first')
    errors = []
    for name in names:
        data = subprocess.check_output(['git','show',':'+name],cwd=ROOT) if args.staged else (ROOT/name).read_bytes()
        errors += [name+': '+issue for issue in check_file(name,data)]
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Publication checks passed: {len(names)} reviewed files')


if __name__=='__main__':
    main()
