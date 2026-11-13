# Multimodal LLMs for Graphs

Pipeline de leitura de gráficos com modelos multimodais. A partir de CSVs e perguntas com gabarito, o projeto gera imagens, executa inferência e produz previsões, métricas e relatórios. Compara o I-MoF oficial, LLaVA-1.5-7B e Qwen2-VL-2B, incluindo a combinação de representações CLIP e DINOv3 e o ajuste dos projetores visuais.

A [arquitetura e os modelos](docs/architecture.md) estão documentados no núcleo. Os comandos abaixo partem da raiz do repositório.

## Instalação

Para visualizar resultados e preparar dados, use Python 3.10–3.12, sem necessidade de GPU. Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e sincronize o ambiente:

```bash
git clone https://github.com/Gabriel-Wamat/Multimodal-LLMs-for-Graphs.git
cd Multimodal-LLMs-for-Graphs
uv sync --locked --extra notebooks --extra dev
uv pip check
```

O `uv.lock` fixa as dependências desse ambiente. Como alternativa, crie uma `.venv` com uma versão compatível de Python e execute `python -m pip install -r requirements.txt`. Os ambientes de inferência e treinamento são instalados separadamente.

## Visualização

O checkout já contém um resumo dos resultados verificados. Para gerar um relatório HTML, sem dados privados ou checkpoints:

```bash
uv run --locked python -m jupyter nbconvert \
  --execute --to html --output-dir .cache/reports \
  notebooks/resultados_experimentos.ipynb
```

Abra `.cache/reports/resultados_experimentos.html` no navegador. O relatório mostra as tabelas publicadas; quando houver arquivos adicionais em `results/` e `experiments/training/`, também apresenta os detalhes e históricos disponíveis.

Para explorar interativamente, abra o [notebook central](notebooks/resultados_experimentos.ipynb) no seu editor, selecione o Python da `.venv` e execute **Run All**. O painel apenas lê os artefatos; não inicia inferência ou treinamento. Os notebooks de [exploração dos CSVs](notebooks/explorar_dados.ipynb) e [galeria de imagens](notebooks/galeria_png.ipynb) exigem os dados preparados.

## Preparação dos dados

O conjunto original precisa ser obtido separadamente; não há um endereço público de distribuição definido neste projeto. Coloque os CSVs, `parametros.json` e `gabarito.json` em `dados_projeto_multi_modais/dados/`, e a planilha em `dados_projeto_multi_modais/gabarito.xlsx`, conforme o [formato de entrada](docs/data.md).

Com os arquivos recebidos disponíveis, reconstrua os gráficos, valide as saídas e gere o piloto:

```bash
uv run --locked python scripts/local/gerar_graficos.py
uv run --locked python scripts/local/gerar_graficos.py --validate-only
uv run --locked python -m src.data.pilot
```

Os PNGs ficam em `data/processed/graficos/`; os hashes e as entradas do piloto ficam em `data/manifests/`. O protocolo atual utiliza 720 gráficos de seis tipos e seleciona 360 para o piloto, mantendo juntas as perguntas de cada gráfico. As imagens são reconstruções dos CSVs, e o gerador recusa sobrescrever arquivos divergentes.

## Inferência por CLI

Use Linux x86_64, Python 3.10 e GPU NVIDIA com driver compatível com o perfil PyTorch/CUDA escolhido. Os pesos são baixados dos repositórios dos autores e verificados por revisão e SHA-256. Eles não são distribuídos neste GitHub.

### I-MoF oficial

Em um host GPU ou alocação SLURM, instale o ambiente legado, confira dados/pesos/GPU e execute o diagnóstico antes do piloto:

```bash
SIH_PYTHON=python3.10 bash scripts/baseline.sh install
bash scripts/baseline.sh check --gpu
bash scripts/baseline.sh smoke --run-id imof-smoke-01
bash scripts/baseline.sh run --run-id imof-pilot-01
```

O baseline utiliza LLaVA-1.5 13B, CLIP e DINOv2, em `.venv-mof`. O [guia do I-MoF](scripts/BASELINE.md) descreve os requisitos de memória, a instalação em etapas e a retomada.

### LLaVA e Qwen com CLIP/DINOv3

Defina `DINOV3_CHECKPOINT` com o caminho do checkpoint oficial obtido com a permissão de acesso correspondente. Depois, prepare `.venv-control`, importe os pesos e execute:

```bash
SIH_PYTHON=python3.10 bash scripts/control.sh install
bash scripts/control.sh download --dinov3-source "$DINOV3_CHECKPOINT"
bash scripts/control.sh check --hashes
bash scripts/control.sh smoke --model llava --run-id control-smoke-llava
bash scripts/control.sh run --model llava --run-id control-pilot-llava
```

Use `--model qwen` e IDs próprios para o Qwen. Resoluções, seeds, revisões e braços estão em `configs/`; os [controles CLIP/DINOv3](docs/CONTROLE_MOF_DINOV3.md) detalham as inicializações e os requisitos de VRAM.

Cada execução grava respostas em `results/<run-id>/predictions.jsonl` e métricas em `results/<run-id>/scored/metrics.json`. Os manifestos registram configuração, fontes e hashes. Para retomar uma execução interrompida, repita o comando com o mesmo ID e `--resume`; o executor exige as mesmas entradas, pesos, código e configuração. Use outro ID para um novo experimento.

## Treinamento dos projetores

O treino ajusta somente os projetores visuais, com encoders e modelo de linguagem congelados. Prepare os splits e conclua os diagnósticos de gradiente, atualização, recarga e retomada conforme o [guia de experimentos](docs/experiments.md). Com esses pré-requisitos aprovados:

```bash
.venv-control/bin/python -m src.training.run \
  --model llava --arm clip_dino --run-id alignment-llava-fusion-01 \
  --checks-prefix alignment-check-v1

.venv-control/bin/python -m src.evaluation.trained \
  --training-run alignment-llava-fusion-01 --run-id evaluated-llava-fusion-01
```

Históricos e checkpoints ficam em `experiments/training/<training-run>/`; a avaliação fica em `results/<run-id>/`. O checkpoint é selecionado exclusivamente no desenvolvimento, incluindo a época zero. Treinamento e avaliação do piloto têm registros de conclusão separados. O mesmo guia descreve o controle `clip_double`, o Qwen e os pares A/B.

## Execução em cluster

Os [lançadores SLURM](scripts/apuana/README.md) encadeiam instalação, download, diagnóstico e avaliação. Na raiz da cópia do projeto no cluster, confira uma submissão antes de executá-la:

```bash
mkdir -p logs
python3 scripts/apuana/submit_control.py \
  --run-id control-cluster-01 --smoke --dry-run
```

O comando exibe os jobs sem submetê-los. O guia explica como incluir a preparação dos pesos, escolher recursos e acompanhar uma execução. Os templates permitem definir nó, memória, partição e tempo; não exigem endereço ou conta pessoal no código. A CLI Python também pode rodar diretamente em um host GPU preparado.

## Resultados verificados

O [resumo publicado](results/published/README.md) registra as evidências coletadas em **03/11/2026**, com hashes de origem e limites de interpretação. Os valores abaixo são acurácias por pergunta no piloto de **360 gráficos e 3.173 perguntas**; não representam o estado atual dos jobs.

| Modelo / configuração | Antes do ajuste | Após ajuste dos projetores |
|---|---:|---:|
| LLaVA-1.5-7B nativo | 8,89% | — |
| LLaVA-1.5-7B nativo duplicado | 9,90% | — |
| LLaVA-1.5-7B, CLIP duplicado | 9,90% | **27,92% (886/3.173)** |
| LLaVA-1.5-7B, CLIP + DINOv3 | 0,00% | **24,27% (770/3.173)** |
| Qwen2-VL-2B, CLIP duplicado | Sem resultado final confirmado | **24,30% (771/3.173)** |
| Qwen2-VL-2B, CLIP + DINOv3 | Sem resultado final confirmado | Sem avaliação final confirmada |

O I-MoF oficial de 13B obteve **119/3.173 (3,75%)** pelo corretor original. O diagnóstico posterior que aceita Yes/No como Sim/Não resulta em **311/3.173 (9,80%)**, preservando a pontuação original. A tarefa dos pares A/B tem outra regra de acerto e aparece separadamente no resumo.

CLIP duplicado controla o comprimento da sequência visual, mas não o número de parâmetros treináveis. O zero da fusão antes do ajuste envolve projetores sem alinhamento. As métricas são do corretor por formato; a [avaliação e suas limitações](docs/evaluation.md) explicam essas diferenças e o caráter exploratório da comparação.

## Dependências e verificações

| Ambiente | Uso | Configuração |
|---|---|---|
| `.venv` | Análise, dados e notebooks; Python 3.10–3.12 | `pyproject.toml` e `uv.lock` |
| `.venv-mof` | I-MoF oficial; Linux/Python 3.10 | `configs/requirements-mof.txt` e `constraints-mof.txt` |
| `.venv-control` | Controles e ajuste dos projetores; Linux/Python 3.10 | `configs/requirements-control.txt` |

Para conferir o código e os arquivos versionados, sem carregar os modelos completos:

```bash
uv run --locked ruff check src scripts tests
uv run --locked python -m unittest discover -s tests -v
uv run --locked python scripts/local/check_publication.py
```

A CI executa testes portáteis, testes com PyTorch em CPU e o notebook em uma cópia sem dados privados. A validação com checkpoints completos e CUDA exige uma alocação GPU.

## Organização e documentação

```text
configs/             revisões dos modelos e parâmetros dos experimentos
src/data/            renderização, seleção, auditoria e construção de pares
src/models/          checkpoints oficiais, encoders e intercalação visual
src/evaluation/      inferência, corretores, métricas e análise dos pares
src/training/        ajuste dos projetores e retomada de checkpoints
scripts/local/       preparação, relatórios e monitoramento configurável
scripts/apuana/      instalação e execução em SLURM
notebooks/           painel de resultados, exploração e inspeção visual
results/published/   resumo portátil das evidências verificadas
tests/               verificações do código e dos protocolos
```

Dados recebidos, imagens derivadas, previsões brutas, pesos, logs e registros completos de execução ficam nas áreas de trabalho ignoradas pelo Git. O repositório publica código, configurações, notebooks sem outputs e o resumo numérico.

- [Arquitetura e modelos](docs/architecture.md): componentes, representações e controles.
- [Dados](docs/data.md): arquivos recebidos, campos, renderização e splits.
- [Experimentos](docs/experiments.md): preparação, diagnósticos, treinamento e avaliação.
- [Métricas](docs/evaluation.md): corretores, acurácia pareada, incerteza e comparabilidade.
- [Reprodução e publicação](docs/reproducibility.md): ambientes, artefatos e monitoramento.

## Referência

O baseline deriva de [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209), de Tong et al., e do [código oficial MMVP](https://github.com/tsb0601/MMVP). O I-MoF oficial combina CLIP e DINOv2. A extensão com DINOv3 em LLaVA-1.5-7B e Qwen2-VL-2B é experimental e não equivale ao checkpoint de 13B dos autores.
