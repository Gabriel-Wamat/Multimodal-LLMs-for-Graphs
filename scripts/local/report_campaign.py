"""Render quantitative campaign results only after all required runs are complete."""

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.control.artifacts import digest
from src.data.pilot import write_once, json_text


def required():
    paths = [ROOT / "results/alignment-comparison-v1/report.json",
             ROOT / "data/manifests/train_dev_v2/split.json"]
    baseline = ROOT / "results/mof13b-pilot50-v1"
    paths += [baseline / name for name in ("complete.json", "run.json", "scored/metrics.json",
                                           "diagnostics/binary_aliases_posthoc.json", "diagnostics/numeric_format_audit_v1.json",
                                           "NUMERIC_FORMAT_AUDIT.md")]
    representations = ROOT / "results/representations-pairs_v1-v1"
    paths += [representations / name for name in (
        "complete.json", "scores.json", "calibration.json", "similarity_report.json",
        "plots/manifest.json", "plots/similarity.png", "plots/similarity.svg",
    )]
    for model in ["llava", "qwen"]:
        folder = ROOT / "results" / ("paired-mitigations-full-v1-" + model)
        paths += [
            folder / "complete.json",
            folder / "scored/metrics.json",
            folder / "scored/hypotheses.json",
        ]
        for arm in ["clip_double", "clip_dino"]:
            paths += [
                ROOT / "results" / f"aligned-pilot50-v1-{model}-{arm}" / "complete.json"
            ]
    for model in ["llava", "qwen"]:
        for arm in ["clip_double", "clip_dino"]:
            training = ROOT / "experiments/training" / f"alignment-v1-{model}-{arm}"
            paths += [training / name for name in ("run.json", "complete.json", "history.json")]
    review = ROOT / "experiments/five-stage-campaign"
    paths += [review / "reading_review_audit_current_v1.json",
              review / "reading_review_summary_v1.json",
              review / "reading_review_sample_v1.json"]
    paths += [ROOT / "results/list-structure-audit-v1" / name
              for name in ("list_structure.json", "REPORT.md")]
    return paths


def check_inputs():
    missing = [str(p.relative_to(ROOT)) for p in required() if not p.is_file()]
    if missing:
        return {"ready": False, "missing": missing}
    review = ROOT / "experiments/five-stage-campaign"
    audit_path = review / "reading_review_audit_current_v1.json"
    audit = json.loads(audit_path.read_text())
    summary = json.loads((review / "reading_review_summary_v1.json").read_text())
    if audit["status"] != "pass" or summary["audit_sha256"] != digest(audit_path):
        raise ValueError("Reading review summary/audit differs")
    if audit["sample_sha256"] != digest(review / "reading_review_sample_v1.json"):
        raise ValueError("Reading review sample changed")
    for model in ["llava", "qwen"]:
        coverage = audit["models"][model]
        if not coverage["complete"] or coverage["reviewed_images"] != 144:
            raise ValueError("Fixed reading review incomplete: " + model)
        prediction = ROOT / "results" / ("paired-mitigations-full-v1-" + model) / "predictions.jsonl"
        if coverage["final_predictions_sha256"] != digest(prediction):
            raise ValueError("Reviewed final predictions changed: " + model)
        for name, sha in coverage["sources"].items():
            if digest(review / name) != sha:
                raise ValueError("Reading review source changed: " + name)
    baseline_discussion()
    list_discussion()
    comparison = json.loads(required()[0].read_text())
    families_sha = digest(ROOT / "data/manifests/train_dev_v2/split.json")
    if comparison["families_sha256"] != families_sha:
        raise ValueError("Comparison family grouping changed")
    representations = ROOT / "results/representations-pairs_v1-v1"
    complete = json.loads((representations / "complete.json").read_text())
    plot = json.loads((representations / "plots/manifest.json").read_text())
    if digest(representations / "scores.json") != complete["scores_sha256"]:
        raise ValueError("Representation scores changed")
    if plot["scores_sha256"] != complete["scores_sha256"] or plot["evaluation_pairs"] != 720:
        raise ValueError("Representation plot provenance differs")
    for name, key in (("calibration.json", "calibration_sha256"),
                      ("similarity_report.json", "report_sha256")):
        if digest(representations / name) != plot[key]:
            raise ValueError("Representation analysis changed")
    for name in ("similarity.png", "similarity.svg"):
        if digest(representations / "plots" / name) != plot["files"][name]:
            raise ValueError("Representation figure changed")
    for model in ["llava", "qwen"]:
        for source in comparison["models"][model]["sources"]:
            folder = ROOT / "results" / source["run_id"]
            if (
                digest(folder / "predictions.jsonl") != source["predictions_sha256"]
                or digest(folder / "run.json") != source["run_sha256"]
            ):
                raise ValueError("Comparison source changed: " + source["run_id"])
        for source in comparison["models"][model]["sources"]:
            if source.get("training") is None:
                continue
            evaluated = json.loads((ROOT / "results" / source["run_id"] / "run.json").read_text())
            training = ROOT / "experiments/training" / evaluated["training_run"]
            complete_train = json.loads((training / "complete.json").read_text())
            train_run = json.loads((training / "run.json").read_text())
            history = json.loads((training / "history.json").read_text())
            epochs = [row["epoch"] for row in history if row["phase"] == "epoch_complete"]
            if (digest(training / "history.json") != source["training"]["history_sha256"]
                    or digest(training / "complete.json") != evaluated["training_complete_sha256"]
                    or digest(training / "run.json") != evaluated["training_record_sha256"]
                    or epochs != list(range(1, train_run["config"]["epochs"] + 1))
                    or complete_train["epochs"] != train_run["config"]["epochs"]
                    or complete_train["best"]["epoch"] != evaluated["selected_epoch"]
                    or complete_train["best_sha256"] != evaluated["checkpoint_sha256"]
                    or evaluated["selection_used_evaluation"] is not False):
                raise ValueError("Training provenance or held-out selection differs: " + source["run_id"])
        folder = ROOT / "results" / ("paired-mitigations-full-v1-" + model)
        metrics = json.loads((folder / "scored/metrics.json").read_text())
        hypotheses = json.loads((folder / "scored/hypotheses.json").read_text())
        actual = digest(folder / "predictions.jsonl")
        if (
            actual != metrics["predictions_sha256"]
            or actual != hypotheses["provenance"]["predictions"]
            or hypotheses["provenance"]["families"] != families_sha
            or hypotheses["provenance"]["representations"] != complete["scores_sha256"]
            or metrics["smoke"]
        ):
            raise ValueError("Paired report provenance differs")
    return {"ready": True, "missing": []}





def subgroup_breakdown(metrics, arms, names):
    """Describe subgroup accuracies on the same saved items; no subgroup fitting."""
    sections = ['<details><summary>Diagnósticos por tipo, estilo e formato de resposta</summary>',
                '<p>Acurácia por pergunta (%), com o mesmo corretor e denominador por grupo. '
                'Tabelas descritivas, sem inferência de significância entre subgrupos ou revisão semântica.</p>']
    for key, title in [("by_tipo", "Tipo de gráfico"), ("by_estilo", "Estilo"),
                       ("by_formato", "Formato de resposta")]:
        reference = metrics[arms[0]]["primary"][key]
        rows = []
        for group, initial in reference.items():
            values = [metrics[arm]["primary"][key][group] for arm in arms]
            if any(value["n"] != initial["n"] for value in values):
                raise ValueError("Subgroup denominators differ: " + key + "/" + group)
            rows.append([group, initial["n"], *[f"{100*v['accuracy']:.2f}" for v in values]])
        sections += ["<h3>" + title + "</h3>", html_table(["Grupo", "Perguntas", *names], rows)]
    sections.append("</details>")
    return "".join(sections)


def generation_limits(sources):
    """Count saved outputs reaching the configured cap, without re-running inference."""
    result = {}
    for source in sources:
        folder = ROOT / "results" / source["run_id"]
        run = json.loads((folder / "run.json").read_text())
        complete = json.loads((folder / "complete.json").read_text())
        path = folder / "predictions.jsonl"
        if (digest(path) != source["predictions_sha256"]
                or complete["predictions_sha256"] != source["predictions_sha256"]
                or digest(folder / "run.json") != source["run_sha256"]):
            raise ValueError("Generation limit source changed")
        cap = run["config"]["max_new_tokens"]
        if type(cap) is not int or cap < 1:
            raise ValueError("Invalid generation cap")
        arms = run.get("arms", [run.get("arm")])
        counts = {arm: {"n": 0, "at_limit": 0, "cap": cap} for arm in arms}
        seen = set()
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    raise ValueError("Partial generation record")
                row = json.loads(line)
                key = (row["id"], row["arm"])
                tokens = row["generated_tokens"]
                if (key in seen or row["arm"] not in counts
                        or type(tokens) is not int or not 0 <= tokens <= cap):
                    raise ValueError("Invalid or duplicate generation record")
                seen.add(key)
                count = counts[row["arm"]]
                count["n"] += 1
                count["at_limit"] += tokens == cap
        prefix = "trained_" if "training_run" in run else ""
        for arm, count in counts.items():
            if count["n"] != len(run["item_ids"]) or prefix + arm in result:
                raise ValueError("Generation counts differ from run membership")
            result[prefix + arm] = count
    return result


def html_table(headers, rows):
    def cells(tag, values):
        return "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values)
    return "<table><tr>" + cells("th", headers) + "</tr>" + "".join(
        "<tr>" + cells("td", row) + "</tr>" for row in rows
    ) + "</table>"


def estimate_text(record, scale=100):
    value = record.get("estimate", record.get("rho"))
    if value is None:
        return "não definido (variável constante)"
    bounds = record.get("ci95")
    interval = (f" [{scale*bounds[0]:.2f}; {scale*bounds[1]:.2f}]"
                if bounds else " [IC indisponível]")
    return f"{scale*value:.2f}" + interval



def baseline_discussion():
    folder = ROOT / "results/mof13b-pilot50-v1"
    complete = json.loads((folder / "complete.json").read_text())
    metrics = json.loads((folder / "scored/metrics.json").read_text())
    diagnostic = json.loads((folder / "diagnostics/binary_aliases_posthoc.json").read_text())
    actual = digest(folder / "predictions.jsonl")
    if (any(record["predictions_sha256"] != actual for record in [complete, metrics, diagnostic])
            or metrics["n"] != complete["questions"]
            or diagnostic["n"] != metrics["n"]
            or diagnostic["primary_correct"] != metrics["correct"]
            or not metrics["all_records_successful"]):
        raise ValueError("Original baseline provenance differs")
    numeric = json.loads((folder / "diagnostics/numeric_format_audit_v1.json").read_text())
    if numeric["cohort_size"] != metrics["reason_counts"]["invalid_numeric_format"]:
        raise ValueError("Numeric audit cohort differs")
    for name, sha in numeric["sources"].items():
        if digest(ROOT / name) != sha:
            raise ValueError("Numeric audit source changed: " + name)
    if digest(ROOT / "scripts/local/audit_baseline_numeric.py") != numeric["auditor_sha256"]:
        raise ValueError("Numeric auditor changed")
    return ("<h2>Baseline original I-MoF 13B</h2>"
            + html_table(["Regra", "Acertos / perguntas", "Acurácia %"], [
                ["Protocolo original estrito", f"{metrics['correct']} / {metrics['n']}",
                 f"{100*metrics['accuracy']:.2f}"],
                ["Diagnóstico posterior: Yes/No → Sim/Não", f"{diagnostic['diagnostic_correct']} / {diagnostic['n']}",
                 f"{100*diagnostic['diagnostic_accuracy']:.2f}"]])
            + "<p>360 gráficos reconstruídos de CSVs; 3.173 perguntas do piloto fixo. O diagnóstico de idioma foi definido após observar as respostas e preserva os demais critérios. Não substitui o resultado original nem constitui revisão semântica completa. Acurácia de conteúdo exige auditoria do corretor por tipo de resposta; BERTScore não foi calculado.</p>"
            + "<p>O checkpoint I-MoF 13B usa arquitetura e encoders diferentes dos controles LLaVA-7B/Qwen-2B desta campanha. O benchmark MMVP do artigo e as tarefas sintéticas aqui avaliadas também diferem; suas porcentagens não são comparação direta.</p>"
            + f"<p>Auditoria posterior dos {numeric['cohort_size']} erros de formato numérico: {numeric['csv_targets_verified']} alvos conferidos nos CSVs; {numeric['responses_with_explicit_literal_in_target_interval']} respostas contêm número literal no intervalo do alvo. Essa coorte não representa todas as respostas nem uma nova acurácia semântica.</p>"
            + '<p><a href="../mof13b-pilot50-v1/NUMERIC_FORMAT_AUDIT.md">Auditoria de formato numérico</a></p>'
            + '<p><a href="../mof13b-pilot50-v1/REPORT.md">Relatório do baseline</a> · <a href="../mof13b-pilot50-v1/DIAGNOSTICS.md">Diagnóstico do corretor</a></p>')


def list_discussion():
    """Show fixed completed-run diagnostics without interpreting unknown text as errors."""
    folder = ROOT / "results/list-structure-audit-v1"
    audit = json.loads((folder / "list_structure.json").read_text())
    if audit['list_questions'] != 120:
        raise ValueError('List diagnostic cohort differs')
    for name, expected in audit['sources'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('List diagnostic source changed: ' + name)
    for name, expected in audit['analysis_sources'].items():
        if digest(folder / 'source' / name) != expected:
            raise ValueError('Archived list analysis changed: ' + name)
    rows = []
    for name, counts in audit['groups'].items():
        if counts['n'] != audit['list_questions']:
            raise ValueError('Incomplete list diagnostic group')
        rows.append([name, counts['n'], counts['exact_correct'],
                     counts['complete_permutation_wrong_order'],
                     counts['unrecognized_fragments'], counts['missing_target_labels'],
                     counts['duplicate_recognized_labels']])
    return ('<h2>Diagnóstico posterior das respostas em lista</h2>'
            + '<p>120 perguntas de ordenação por execução/braço abaixo. Esta análise cobre o baseline '
            'original, os seis controles LLaVA e o primeiro LLaVA CLIP+CLIP treinado; não descreve '
            'os demais checkpoints. Uma permutação completa contém todos os rótulos uma vez, '
            'mas ainda precisa ordenar corretamente. O critério oficial permanece inalterado.</p>'
            + html_table(['Execução / braço', 'N', 'Exatas', 'Completas, ordem errada',
                          'Fragmentos desconhecidos', 'Rótulos ausentes', 'Rótulos repetidos'], rows)
            + '<p>Contagens podem se sobrepor. Ausências e fragmentos desconhecidos dependem do '
            'parser exato: texto ou formatação podem impedir reconhecimento. Isto não é revisão '
            'semântica, inspeção visual nem nova acurácia de conteúdo. Diagnóstico posterior em '
            'piloto explorado; não foi usado para ajustar os treinos em andamento.</p>'
            + '<p><a href="../list-structure-audit-v1/REPORT.md">Detalhes e fontes do diagnóstico</a></p>')


def paired_discussion():
    """Render only saved analyses; never re-fit thresholds from evaluated answers."""
    h2, h3, correlations, review_rows = [], [], [], []
    summary = json.loads((ROOT / "experiments/five-stage-campaign/reading_review_summary_v1.json").read_text())
    for model in ["llava", "qwen"]:
        folder = ROOT / "results" / ("paired-mitigations-full-v1-" + model)
        analysis = json.loads((folder / "scored/hypotheses.json").read_text())
        for condition, values in analysis["H2_and_similarity_error"].items():
            h2.append([model, condition, estimate_text(values["H2"]["all"])])
            for encoder, result in values["similarity_error"]["all"].items():
                correlations.append([model, condition, encoder, estimate_text(result, 1)])
        for condition, result in analysis["H3"].items():
            h3.append([model, condition, estimate_text(result["paired_delta"])])
        reviewed = summary["models"][model]
        counts = reviewed["transition_counts"]
        review_rows.append([model, reviewed["reviewed_images"], counts["ajudou"],
                            counts["prejudicou"], counts["manteve_acerto"], counts["manteve_erro"]])
    return "".join([
        "<h2>H2: alterações amplas versus sutis</h2><p>Diferença de acurácia por par, ampla menos sutil, em pontos percentuais; IC95% por família.</p>",
        html_table(["Modelo", "Condição", "Diferença pp [IC95%]"], h2),
        "<p>O piso de acurácia pareada do LLaVA limita a interpretação. A diferença positiva do Qwen original é compatível com maior dificuldade nas alterações sutis neste conjunto.</p>",
        "<h2>H3: efeito das mitigações</h2><p>Mitigação menos original, nos mesmos pares. Resultados específicos deste protocolo.</p>",
        html_table(["Modelo", "Mitigação", "Diferença pp [IC95%]"], h3),
        "<p>Os testes de McNemar por atributo e intensidade estão nos arquivos de hipóteses de cada modelo. Usam um par por família em cada estrato; p-valores exploratórios sem ajuste múltiplo. Abstenções integram os erros.</p>",
        "<h2>Similaridade global e erro</h2>",
        html_table(["Modelo", "Condição", "Encoder", "Spearman [IC95%]"], correlations),
        "<p>Correlação não demonstra causalidade. IC indisponível preserva reamostragens indefinidas; não representa correlação zero. A comparação histórica com DINOv2 não é replicada ao substituir o encoder por DINOv3.</p>",
        "<h2>Revisão visual da amostra fixa</h2><p>144 imagens por modelo, 24 por tipo de gráfico. Inspeção realizada pelo assistente, sem segunda anotação independente. Transições abaixo são por imagem, original → transcrição.</p>",
        html_table(["Modelo", "Imagens", "Passou a acertar", "Passou a errar", "Manteve acerto", "Manteve erro"], review_rows),
        "<p>Foram observadas omissões, valores e estruturas não sustentados pela imagem, confusão de eixos e categorias. As categorias podem se sobrepor; a amostra não estima prevalência no dataset nem comprova que erros de leitura causaram cada resposta incorreta.</p>",
        '<p><a href="../../experiments/five-stage-campaign/READING_REVIEW_SUMMARY.md">Evidências da revisão visual</a> · <a href="../paired-comparison-v1/REPORT.md">Análises pareadas e limites completos</a></p>',
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    status = check_inputs()
    if args.check:
        print(json_text(status))
        return
    if not status["ready"]:
        raise SystemExit("Campaign incomplete: " + ", ".join(status["missing"]))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    comparison = json.loads(required()[0].read_text())
    destination = ROOT / "results/campaign-report-v1"
    destination.mkdir(parents=True, exist_ok=True)
    provenance = {str(p.relative_to(ROOT)): digest(p) for p in required()}
    provenance["renderer_sha256"] = digest(Path(__file__))
    write_once(destination / "inputs.json", json_text(provenance))
    sections = [
        baseline_discussion(),
        list_discussion(),
        '<h2>Representações CLIP e DINOv3</h2>'
        '<p>720 pares de avaliação; limiares relativos definidos nos 142 pares de desenvolvimento. '
        'Todos os pares são preservados. Similaridade global alta não demonstra ausência de informação '
        'visual nem, isoladamente, explica os erros dos modelos. Linhas tracejadas: quantil 0,75.</p>'
        '<img src="../representations-pairs_v1-v1/plots/similarity.png" '
        'alt="Similaridade por par e sensibilidade às escolhas de limiar">'
    ]
    arms = [
        "native",
        "clip_double",
        "clip_dino",
        "trained_clip_double",
        "trained_clip_dino",
    ]
    names = [
        "Nativo",
        "CLIP sem ajuste",
        "CLIP+DINO sem ajuste",
        "CLIP ajustado",
        "CLIP+DINO ajustado",
    ]
    for model in ["llava", "qwen"]:
        results = comparison["models"][model]
        values = [100 * results["metrics"][arm]["primary"]["accuracy"] for arm in arms]
        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        ax.barh(
            names, values, color=["#555555", "#5b9bd5", "#80bfa8", "#245a86", "#27765b"]
        )
        ax.set(
            xlim=(0, 100),
            xlabel="Acurácia por pergunta (%)",
            title=model.upper() + " — mesmo piloto exploratório",
        )
        ax.invert_yaxis()
        for i, value in enumerate(values):
            ax.text(min(value + 1, 96), i, f"{value:.2f}%", va="center")
        ax.spines[["top", "right"]].set_visible(False)
        for ext in ["png", "svg"]:
            fig.savefig(destination / f"{model}-accuracy.{ext}", dpi=160)
        plt.close(fig)
        rows = "".join(
            "<tr><td>"
            + html.escape(name)
            + "</td><td>"
            + f"{r['percentage_points']:+.2f}"
            + "</td><td>"
            + html.escape(
                str(
                    [round(100 * v, 2) for v in r["ci95"]]
                    if r["ci95"]
                    else "indisponível"
                )
            )
            + "</td></tr>"
            for name, r in results["contrasts"].items()
        )
        sections.append(
            f'<h2>{model.upper()}: antes e depois do ajuste</h2><img src="{model}-accuracy.png"><table><tr><th>Contraste</th><th>Δ p.p.</th><th>IC 95% por família (p.p.)</th></tr>{rows}</table>'
        )
        limits = generation_limits(results["sources"])
        all_arms = "".join(
            f'<tr><td>{html.escape(arm)}</td><td>{100*m["primary"]["accuracy"]:.2f}</td>'
            f'<td>{100*m["strict"]["accuracy"]:.2f}</td><td>{m["primary"]["n"]}</td>'
            f'<td>{m["seconds"]/3600:.2f}</td><td>{m["generated_tokens"]}</td>'
            f'<td>{limits[arm]["at_limit"]}/{limits[arm]["n"]} '
            f'({100*limits[arm]["at_limit"]/limits[arm]["n"]:.1f}%; teto {limits[arm]["cap"]})</td></tr>'
            for arm, m in results["metrics"].items()
        )
        sections.append('<h3>Todos os grupos de controle e ajuste</h3><table><tr><th>Grupo</th><th>Acurácia principal %</th><th>Estrita %</th><th>Perguntas</th><th>Horas observadas</th><th>Tokens gerados</th><th>Respostas no teto</th></tr>' + all_arms + '</table>')
        sections.append("<p>Atingir o teto significa emitir o número máximo de tokens configurado; não comprova, isoladamente, erro de conteúdo ou truncamento. Os contadores descrevem apenas as respostas salvas, sem nova inferência.</p>")
        sections.append("<p><strong>Fronteiras dos cronômetros:</strong> o controle começa a medir após o processamento da imagem/entrada; a avaliação treinada inclui esse processamento. Os tempos são observações dos respectivos executores, não medidas equivalentes de ponta a ponta. Não calcular speedup antes/depois a partir desses totais.</p>")
        sections.append(subgroup_breakdown(results["metrics"], arms, names))
        training_rows = []
        for source in results["sources"]:
            training = source.get("training")
            if training is None:
                continue
            run = json.loads((ROOT / "results" / source["run_id"] / "run.json").read_text())
            folder = ROOT / "experiments/training" / run["training_run"]
            history = json.loads((folder / "history.json").read_text())
            dev_seconds = sum(row["development"]["seconds"] for row in history)
            peak = max(row.get("peak_gpu_bytes", 0) for row in history)
            training_rows.append([run["arm"], training["epoch_count"], training["selected_epoch"],
                                  training["trainable_parameters"], f"{training['training_seconds']/3600:.2f}",
                                  f"{dev_seconds/3600:.2f}", f"{peak/2**30:.2f}"])
        sections.append("<h3>Orçamento e seleção do treinamento</h3>" + html_table(
            ["Grupo", "Épocas concluídas", "Época selecionada no dev", "Parâmetros treináveis",
             "Horas de treino", "Horas de validação", "Pico alocado GiB"], training_rows)
            + "<p>Tempos são medições do processo, sem espera na fila e sem garantia de custo exclusivo do dispositivo. A seleção inclui a época zero; somente os projetores foram atualizados.</p>")
        paired_folder = ROOT / "results" / ("paired-mitigations-full-v1-" + model)
        paired = json.loads((paired_folder / "scored/metrics.json").read_text())
        conditions = ["original", "crops_2x2", "transcription"]
        estimates = (
            np.array(
                [
                    paired["metrics"][c]["paired_accuracy"]["estimate"]
                    for c in conditions
                ]
            )
            * 100
        )
        bounds = (
            np.array(
                [paired["metrics"][c]["paired_accuracy"]["ci95"] for c in conditions]
            )
            * 100
        )
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        ax.bar(range(3), estimates, color=["#555555", "#5b9bd5", "#80bfa8"])
        # Percentile intervals need not contain the point estimate; draw endpoints directly.
        ax.vlines(range(3), bounds[:, 0], bounds[:, 1], color="black")
        for x, (lo, hi) in enumerate(bounds):
            ax.hlines([lo, hi], x - 0.07, x + 0.07, color="black")
        ax.set(
            xticks=range(3),
            xticklabels=["Original", "Recortes 2×2", "Transcrição"],
            ylim=(0, 100),
            ylabel="Acurácia por par (%)",
            title=model.upper() + " — dois acertos para pontuar o par",
        )
        for ext in ["png", "svg"]:
            fig.savefig(destination / f"{model}-mitigations.{ext}", dpi=160)
        plt.close(fig)
        cost = "".join(
            f"<tr><td>{html.escape(c)}</td><td>{paired['costs'][c]['seconds'] / 3600:.2f}</td><td>{paired['costs'][c]['generated_tokens']}</td><td>{paired['costs'][c]['peak_gpu_bytes'] / 2**30:.2f}</td></tr>"
            for c in conditions
        )
        sections.append(
            f'<h2>{model.upper()}: mitigações sem treinamento</h2><img src="{model}-mitigations.png"><table><tr><th>Condição</th><th>Horas observadas</th><th>Tokens gerados</th><th>Pico GPU GiB</th></tr>{cost}</table>'
        )
    sections.append(paired_discussion())
    page = (
        """<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Resultados quantitativos da campanha</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:16px;color:#222}img{width:100%}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:8px;text-align:left}h2{margin-top:48px}</style><h1>Resultados quantitativos — comparação exploratória</h1><p>Piloto previamente explorado. Mesmos IDs e regras de correção; intervalos reamostram famílias inteiras. Os grupos de fusão possuem mais parâmetros treináveis. Uma única inicialização não estabelece robustez entre sementes.</p><p>Hardware dos treinos: LLaVA em A100 80 GB; Qwen em RTX 3090 24 GB após testes reais de gradiente, atualização e recarga. Controles e avaliações permanecem no A100. A classe de GPU pode afetar trajetórias numéricas e tempos; não interpretar diferenças entre modelos como efeito isolado do encoder nem os tempos como speedup. A mudança dos jobs pendentes está em experiments/submissions/rtx3090-relocation-v1.json.</p><p>Esta página não substitui a revisão qualitativa das transcrições nem a interpretação de H1–H3. As páginas de inspeção e os relatórios estatísticos completos ficam nos diretórios de cada execução.</p>"""
        + "".join(sections)
        + "</html>"
    )
    write_once(destination / "index.html", page)
    write_once(
        destination / "rendered.json",
        json_text(
            {
                "scope": "quantitative figures only; includes saved paired analyses and audited qualitative sample; original evidence remains separate",
                "images": 5,
                "html_sha256": digest(destination / "index.html"),
            }
        ),
    )


if __name__ == "__main__":
    main()
