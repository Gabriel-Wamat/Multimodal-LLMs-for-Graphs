"""Gera um relatório somente a partir de métricas completas e previsões verificadas."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.pilot import sha, write_once


def plot_accuracy(metrics, folder):
    for name, relative in [('MPLCONFIGDIR', 'work/matplotlib'), ('XDG_CACHE_HOME', 'work/cache')]:
        location = ROOT/relative
        location.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(location))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = list(metrics['by_tipo'].items())
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    values = [100 * stats['accuracy'] for _, stats in groups]
    ax.barh([name for name, _ in groups], values, color='#2166ac', height=.65)
    for i, (_, stats) in enumerate(groups):
        ax.text(values[i] + 1, i,
                f"{values[i]:.2f}%  ({stats['correct']}/{stats['n']})".replace('.', ','),
                va='center', fontsize=10)
    ax.axvline(100 * metrics['accuracy'], color='#c44e52', linestyle='--',
               label=f"Geral: {100 * metrics['accuracy']:.2f}%".replace('.', ','))
    ax.set(xlim=(0, 100), xlabel='Acurácia por pergunta (%)',
           title='I-MoF 13B • 360 gráficos / 3.173 perguntas')
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False)
    for extension in ('png', 'svg'):
        fig.savefig(folder/f'accuracy_by_type.{extension}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default='mof13b-pilot50-v1')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError('Identificador inválido')
    folder = ROOT/'results'/args.run_id
    metrics = json.loads((folder/'scored/metrics.json').read_text())
    run = json.loads((folder/'run.json').read_text())
    complete = json.loads((folder/'complete.json').read_text())
    prediction_sha = sha(folder/'predictions.jsonl')
    if (not metrics['all_records_successful'] or metrics['n'] != run['expected_questions']
            or complete['questions'] != metrics['n']
            or prediction_sha != metrics['predictions_sha256']
            or prediction_sha != complete['predictions_sha256']):
        raise ValueError('Cobertura ou integridade insuficiente para relatório completo')
    base = ROOT/'data/manifests/pilot50_v1'
    if (metrics['run_manifest_sha256'] != sha(folder/'run.json')
            or metrics['split_sha256'] != run['split_sha256']
            or run['split_sha256'] != sha(base/'split.json')
            or run['inputs_sha256'] != sha(base/'inputs.jsonl')
            or metrics['scorer_sha256'] != sha(ROOT/'src/evaluation/score.py')
            or run['smoke']):
        raise ValueError('Proveniência da execução divergente')
    def pct(value):
        return f'{100*value:.2f}%'.replace('.', ',')
    lo,hi = metrics['accuracy_ci95_graph_bootstrap']
    lines = [
        '# Baseline I-MoF 13B — piloto de 50% dos gráficos', '',
        f"**Acurácia por pergunta: {pct(metrics['accuracy'])} "
        f"({metrics['correct']} acertos em {metrics['n']} perguntas).**", '',
        f'IC95% exploratório por bootstrap de gráficos: {pct(lo)} a {pct(hi)}.', '',
        f"Foram avaliados {metrics['graph_count']} gráficos. "
        f"Perguntas ausentes ou com falha: {metrics['missing_or_failed']}.", '',
        f"Média de acurácia por gráfico: {pct(metrics['graph_macro_accuracy'])}. "
        f"Gráficos com todas as respostas corretas: {metrics['fully_correct_graphs']}.", '',
        '![Acurácia por tipo de gráfico](accuracy_by_type.png)', '',
    ]
    for title,key in [('Por tipo de gráfico','by_tipo'),('Por nível','by_nivel'),
                      ('Por estilo','by_estilo'),('Por regra de correção','by_correcao')]:
        lines += [f'## {title}', '', '| Grupo | Perguntas | Acertos | Acurácia |',
                  '|---|---:|---:|---:|']
        for group,stats in metrics[key].items():
            lines.append(f"| {group} | {stats['n']} | {stats['correct']} | {pct(stats['accuracy'])} |")
        lines.append('')
    cfg = run['config']
    lines += [
        '## Condições da medição', '',
        f"- Checkpoint: `{cfg['model_id']}`; revisão `{cfg['model_revision']}`.",
        '- 360 gráficos, 60 por tipo e 10 por combinação tipo × estilo; seed 20261102.',
        '- Imagem e pergunta em português, conversas independentes, sem CSV ou gabarito no prompt.',
        '- FP16, sem quantização; CLIP/DINOv2 e imagens processadas em 336×336 com padding.',
        '- Correção exata normalizada ou intervalos inclusivos do gabarito (±5%).',
        '- Inferência greedy; mesma seleção, perguntas e corretor devem ser usados na comparação futura.',
        '', '## Interpretação e limites', '',
        'Este é um baseline exploratório nos gráficos reconstruídos do projeto. O paper '
        '[Eyes Wide Shut?](https://arxiv.org/html/2401.06209v2) relata 31,3% para a variante '
        'I-MoF de 336×336 no MMVP original, que usa pares de imagens e outra regra de acerto. '
        'Esse número não é diretamente comparável à acurácia acima.', '',
        'A medição inclui leitura visual, raciocínio e cumprimento do formato de resposta em português. '
        'Respostas fora do formato podem ser consideradas incorretas pelo corretor determinístico. '
        'Não há garantia de legibilidade de todo detalhe na resolução nativa do encoder nem revisão '
        'semântica completa do gabarito. O bootstrap assume independência entre gráficos dentro dos estratos.', '',
        'Não usar os gráficos desta comparação no treinamento do candidato. Caso este piloto oriente '
        'ajustes, tratar a comparação como desenvolvimento e obter avaliação final adicional. A '
        'ancestralidade das fontes não foi certificada.', '',
        '## Artefatos', '',
        '- `predictions.jsonl`: respostas brutas, prompts, IDs e duração.',
        '- `scored/answers_scored.jsonl`: acertos, alvos e regra aplicada por pergunta.',
        '- `scored/metrics.json`: métricas completas, cobertura e hashes.',
        '- `run.json`, `runtime.json` e `complete.json`: configuração, ambiente e integridade.',
        f'- SHA-256 das previsões: `{prediction_sha}`.', '',
    ]
    path = folder/'REPORT.md'
    write_once(path,'\n'.join(lines))
    plot_accuracy(metrics, folder)
    print(path)


if __name__=='__main__':
    main()
