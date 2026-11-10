"""Reconstrói PNGs dos CSVs; não lê respostas para desenhar os gráficos."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import json
import os
import platform
import inspect
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATASET = ROOT / 'dados_projeto_multi_modais'
CONFIG = json.loads((ROOT / 'configs/data.json').read_text(encoding='utf-8'))
OUTPUT = ROOT / CONFIG['images']
MANIFEST = ROOT / CONFIG['manifest']
WORKBOOK = ROOT / CONFIG['workbook']
TYPES = {'BAR': 'Barras', 'LIN': 'Linhas', 'DIS': 'Dispersão',
         'PIZ': 'Pizza', 'HIS': 'Histograma', 'BOX': 'Boxplot'}
# Convenções novas de renderização: nomes dos estilos vêm de parametros.json.
THEMES = {
    'clássico': ('#ffffff', '#202020', 'tab10', True),
    'editorial': ('#f7f4ee', '#273444', 'tab10', True),
    'cinza': ('#ffffff', '#202020', 'Greys', True),
    'minimalista': ('#ffffff', '#202020', 'tab10', False),
    'escuro': ('#19212d', '#f0f3f7', 'Set2', True),
    'pastel': ('#fffdf8', '#303741', 'Set2', True),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def render(path, style):
    kind = path.stem.split('-')[0]
    rows = read_csv(path)
    bg, fg, cmap, grid = THEMES[style]
    colors = plt.get_cmap(cmap)(np.linspace(.35, .85, 14)) if cmap == 'Greys' else plt.get_cmap(cmap).colors
    config = {'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': fg,
              'axes.labelcolor': fg, 'axes.edgecolor': fg, 'xtick.color': fg,
              'ytick.color': fg, 'axes.facecolor': bg, 'figure.facecolor': bg,
              'legend.facecolor': bg, 'legend.edgecolor': fg}
    with plt.rc_context(config):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150, layout='constrained')
        ax.set_title(TYPES[kind], color=fg, fontsize=16, pad=15)
        if kind != 'PIZ':
            ax.set_axisbelow(True)
            ax.grid(grid, axis='y', color=fg, alpha=.15) if grid else ax.grid(False)
            ax.spines[['top', 'right']].set_visible(False)
        if kind == 'BAR':
            values = np.array([float(r['valor']) for r in rows])
            assert np.isfinite(values).all() and (values >= 0).all()
            ax.bar([r['categoria'] for r in rows], values, color=colors[0])
            ax.set(ylabel='Valor', ylim=(0, max(values) * 1.12))
            plt.setp(ax.get_xticklabels(), rotation=40, ha='right')
        elif kind == 'LIN':
            years = [int(r['ano']) for r in rows]
            assert years == sorted(set(years))
            for i, series in enumerate(k for k in rows[0] if k != 'ano'):
                ax.plot(years, [float(r[series]) for r in rows], marker='o',
                        markersize=3, linewidth=1.8, color=colors[i], label=series)
            ax.set(xlabel='Ano', ylabel='Valor', xticks=years)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax.legend(labelcolor=fg)
        elif kind == 'DIS':
            x = np.array([float(r['x']) for r in rows])
            y = np.array([float(r['y']) for r in rows])
            ax.scatter(x, y, s=32, color=colors[0], alpha=.85, edgecolors=fg, linewidths=.35)
            ax.set(xlabel='X', ylabel='Y')
            ax.margins(.08)
        elif kind == 'PIZ':
            values = np.array([float(r['percentual']) for r in rows])
            assert (values > 0).all() and abs(sum(values) - 100) < .15
            # Geometria normalizada; sem transcrever gabaritos nas imagens.
            wedges, _ = ax.pie(values, colors=colors, startangle=90,
                               wedgeprops={'edgecolor': bg, 'linewidth': 1})
            ax.legend(wedges, [r['categoria'] for r in rows], loc='center left',
                      bbox_to_anchor=(1, .5), labelcolor=fg)
        elif kind == 'HIS':
            left = np.array([float(r['faixa_inicio']) for r in rows])
            right = np.array([float(r['faixa_fim']) for r in rows])
            counts = np.array([int(r['contagem']) for r in rows])
            assert (right > left).all() and (counts >= 0).all()
            assert np.allclose(right[:-1], left[1:])
            ax.bar(left, counts, width=right-left, align='edge', color=colors[0], edgecolor=bg)
            ax.set(xlabel='Faixa', ylabel='Contagem', xticks=np.r_[left, right[-1]])
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
            ax.set_ylim(bottom=0)
        elif kind == 'BOX':
            groups = defaultdict(list)
            for row in rows:
                groups[row['grupo']].append(float(row['valor']))
            box = ax.boxplot(list(groups.values()), tick_labels=list(groups), whis=1.5,
                             patch_artist=True, medianprops={'color': fg, 'linewidth': 2},
                             whiskerprops={'color': fg}, capprops={'color': fg},
                             boxprops={'edgecolor': fg},
                             flierprops={'marker': 'o', 'markersize': 4,
                                         'markeredgecolor': fg, 'markerfacecolor': 'none'})
            for i, patch in enumerate(box['boxes']):
                patch.set_facecolor(colors[i % len(colors)])
            ax.set_ylabel('Valor')
        return fig


def render_spec_sha256():
    """Identifica as convenções de desenho, independentemente dos caminhos."""
    spec = inspect.getsource(render) + json.dumps([THEMES, TYPES], sort_keys=True)
    return hashlib.sha256(spec.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    params_file = DATASET / 'dados' / 'parametros.json'
    params = json.loads(params_file.read_text())
    sources = sorted((DATASET / 'dados').glob('*.csv'))
    assert len(sources) == 720 and {p.stem for p in sources} == set(params)
    out = OUTPUT
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old = {r['id']: r for r in previous.get('images', [])}
    inputs = sources + [params_file, DATASET / 'dados/gabarito.json', DATASET / 'gabarito.xlsx']
    before = {str(p.relative_to(DATASET)): sha(p) for p in inputs}
    if previous:
        if previous.get('input_sha256') != before:
            raise RuntimeError('As fontes diferem das registradas no manifesto.')
        if previous.get('render_spec_sha256') != render_spec_sha256():
            raise RuntimeError('As convenções de renderização diferem do manifesto.')
    if not WORKBOOK.exists() and not args.validate_only:
        WORKBOOK.write_bytes((DATASET / 'gabarito.xlsx').read_bytes())
    if sha(WORKBOOK) != sha(DATASET / 'gabarito.xlsx'):
        raise RuntimeError('A cópia navegável do gabarito difere do original.')
    records = []
    for i, source in enumerate(sources, 1):
        target = out / f'{source.stem}.png'
        style = params[source.stem]['estilo']
        if target.exists():
            record = old.get(source.stem, {})
            if (record.get('png_sha256') != sha(target)
                    or record.get('csv_sha256') != sha(source)
                    or record.get('style') != style):
                raise RuntimeError(f'Arquivo existente não corresponde ao manifesto: {target.name}')
        if not args.validate_only:
            if not target.exists():
                fig = render(source, style)
                fig.savefig(target, dpi=150, metadata={'Description': 'Reconstrução CSV — versão 1'})
                plt.close(fig)
        with Image.open(target) as image:
            assert image.format == 'PNG' and image.size == (1500, 900)
            image.load()
        records.append({'id': source.stem, 'style': style,
                        'csv': str(source.relative_to(ROOT)), 'png': str(target.relative_to(ROOT)),
                        'csv_sha256': sha(source), 'png_sha256': sha(target)})
        if i % 120 == 0:
            print(f'{i}/720 imagens verificadas', flush=True)
    # Checa os caminhos reais usados pelos hyperlinks da planilha, sem alterá-la.
    links = []
    expected_links = {f'graficos/{r["id"]}.png' for r in records}
    with zipfile.ZipFile(WORKBOOK) as book:
        for name in book.namelist():
            if name.startswith('xl/worksheets/_rels/') and name.endswith('.rels'):
                for rel in ET.fromstring(book.read(name)):
                    if rel.get('Type', '').endswith('/hyperlink'):
                        target = rel.get('Target')
                        assert target in expected_links, target
                        assert (WORKBOOK.parent / target).is_file()
                        links.append(target)
    assert before == {str(p.relative_to(DATASET)): sha(p) for p in inputs}
    report = {'version': 'csv-reconstruction-v1', 'origin': 'reconstructed_not_original',
              'dimensions': [1500, 900], 'dpi': 150, 'python': platform.python_version(),
              'matplotlib': matplotlib.__version__, 'numpy': np.__version__,
              'renderer_sha256': sha(Path(__file__)), 'input_sha256': before,
              'render_spec_sha256': render_spec_sha256(),
              'path_base': 'repository_root', 'input_hash_path_base': 'dados_projeto_multi_modais',
              'hyperlinks_validated': len(links), 'unique_hyperlink_targets': len(set(links)),
              'counts': dict(Counter(r['id'][:3] for r in records)), 'images': records}
    if not args.validate_only:
        if previous:
            # Preserva o ambiente e a autoria da geração já realizada.
            for key in ('python', 'matplotlib', 'numpy', 'renderer_sha256', 'provenance'):
                if key in previous:
                    report[key] = previous[key]
        manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'OK: {len(records)} PNGs e {len(links)} links válidos; fontes preservadas.')


if __name__ == '__main__':
    main()
