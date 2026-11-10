# Preparação, treinamento e avaliação

Instale os ambientes, importe os dados e gere imagens/piloto antes dos comandos abaixo. Novas execuções precisam de IDs próprios. Artefatos históricos não fazem parte do checkout público.

## Treino e desenvolvimento

Na raiz do checkout, com o ambiente de análise:

```bash
uv run --locked python scripts/local/prepare_training_split.py
uv run --locked python -m src.data.audit_training
uv run --locked python scripts/local/finalize_training_families.py
```

Essas etapas produzem, respectivamente, o split candidato, a auditoria de semelhança e `data/manifests/train_dev_v2/`. O último passo reúne famílias conectadas e põe em quarentena as relacionadas à avaliação. Revise os candidatos e a origem dos dados antes de interpretar os splits como elegíveis para uma avaliação independente.

O protocolo `configs/alignment_training_v1.json` fixa os parâmetros e os caminhos relativos dos manifestos. O treinamento valida hashes das fontes, imagens e splits e rejeita famílias protegidas.

## Diagnósticos obrigatórios

Em um host GPU preparado, usando `.venv-control`:

```bash
.venv-control/bin/python -m src.training.check_resume

for model in llava qwen; do
  for arm in clip_double clip_dino; do
    .venv-control/bin/python -m src.training.check_alignment \
      --model "$model" --arm "$arm" \
      --run-id "alignment-check-v1-${model}-${arm}"
  done
done
```

São atualizações reais em seis itens de treino, verificando loss, gradientes, congelamento dos backbones, alteração dos projetores e recarga. A retomada em CPU verifica pesos, AdamW, scheduler, RNG e cursor. Esses testes não representam treinamento concluído ou desempenho no piloto.

## Ajuste dos projetores

```bash
.venv-control/bin/python -m src.training.run \
  --model llava --arm clip_double --run-id alignment-llava-clip-01 \
  --checks-prefix alignment-check-v1

.venv-control/bin/python -m src.training.run \
  --model llava --arm clip_dino --run-id alignment-llava-fusion-01 \
  --checks-prefix alignment-check-v1
```

Use `--model qwen` e IDs novos para o Qwen. A seleção compara acurácia de desenvolvimento, menor loss em empate e primeira época se ambas forem iguais. A época zero pode vencer. `history.json` registra a validação inicial e cada época; logs registram loss acumulada, LR, gradientes e atualizações. `best.safetensors` e `last.pt` são artefatos privados da execução.

A retomada usa o mesmo comando com `--resume` e exige o mesmo código, dados e configuração. O checkpoint é salvo nas fronteiras do otimizador; uma interrupção pode exigir refazer as atualizações posteriores ao último salvamento.

## Avaliação do checkpoint

```bash
.venv-control/bin/python -m src.evaluation.trained \
  --training-run alignment-llava-fusion-01 --run-id evaluated-llava-fusion-01
```

A avaliação usa as 3.173 perguntas do piloto sem colocar alvos no prompt. O término do treino não comprova término da avaliação; confira cobertura, `complete.json`, hashes e métricas. O piloto previamente explorado permanece evidência exploratória.

## Pares A/B

```bash
uv run --locked python -m src.data.pairs --help
uv run --locked python -m src.data.pairs

.venv-control/bin/python -m src.evaluation.paired.run \
  --model llava --run-id paired-smoke-llava --smoke
.venv-control/bin/python -m src.evaluation.paired.score paired-smoke-llava
```

Confirme o resultado técnico e a inspeção visual antes de executar sem `--smoke` com outro ID. Cada modelo responde independentemente a A e B nas condições original, recortes e transcrição. Os geradores preservam hashes e não aceitam sobrescrita divergente. Para os lançadores SLURM, consulte [o guia do cluster](../scripts/apuana/README.md).

A extração de representações é o módulo `src.evaluation.paired.representations`. As análises por par e por família estão em `src.evaluation.paired.score`, `analyze` e `hypotheses`; consulte seus argumentos com `--help`. Mantenha calibração dos limiares no desenvolvimento e os mesmos pares entre modelos/condições.
