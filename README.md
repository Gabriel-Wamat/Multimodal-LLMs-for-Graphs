# Multimodal LLMs for Graphs

Leitura e interpretação de gráficos por modelos multimodais. O projeto reconstrói imagens a partir de CSVs, executa perguntas com gabarito e compara representações CLIP/DINO, modelos nativos e projetores visuais ajustados. Também gera pares A/B com alterações controladas para avaliar consistência visual.

A implementação e os limites científicos estão descritos em [arquitetura e modelos](docs/architecture.md). Os comandos abaixo partem da raiz do repositório.

## Instalação

O ambiente de análise funciona sem GPU, com Python 3.10–3.12. Instale o [uv](https://docs.astral.sh/uv/getting-started/installation/) e sincronize as dependências fixadas:

```bash
git clone https://github.com/Gabriel-Wamat/Multimodal-LLMs-for-Graphs.git
cd Multimodal-LLMs-for-Graphs
uv sync --locked --extra notebooks --extra dev
uv pip check
uv run --locked python -m unittest discover -s tests -v
```

Como alternativa ao uv, crie um ambiente com `python3 -m venv .venv` e instale `python -m pip install -r requirements.txt`. O `uv.lock` fixa a resolução do ambiente de análise; os ambientes de inferência têm perfis próprios em `configs/`.

## Resultados disponíveis

O [resumo publicado](results/published/README.md) contém as métricas verificadas, os tamanhos das amostras, os hashes das previsões de origem e as limitações. Acurácia por pergunta no piloto de 360 gráficos e 3.173 perguntas:

| Modelo / configuração | Antes do ajuste | Após ajuste dos projetores |
|---|---:|---:|
| LLaVA-1.5-7B nativo | 8,89% | — |
| LLaVA-1.5-7B nativo duplicado | 9,90% | — |
| LLaVA-1.5-7B, CLIP duplicado | 9,90% | **27,92% (886/3.173)** |
| LLaVA-1.5-7B, CLIP + DINOv3 | 0,00% | **24,27% (770/3.173)** |
| Qwen2-VL-2B, CLIP duplicado | Sem resultado final confirmado | **24,30% (771/3.173)** |
| Qwen2-VL-2B, CLIP + DINOv3 | Sem resultado final confirmado | Sem avaliação final confirmada |

O I-MoF oficial de 13B foi avaliado separadamente: **119/3.173 (3,75%)** pelo corretor original. O diagnóstico posterior que aceita Yes/No como Sim/Não produz **311/3.173 (9,80%)**, preservando o resultado original. O snapshot publicado corresponde às evidências coletadas em 03/11/2026; os estados posteriores não foram confirmados. Não representa acompanhamento ao vivo do cluster.

A fusão sem alinhamento usa projetores novos e não equivale ao I-MoF treinado dos autores. CLIP duplicado repete as representações de um único encoder para igualar os 1.152 tokens da fusão; não iguala o número de parâmetros treináveis. Consulte [métricas e interpretação](docs/evaluation.md) antes de comparar tarefas ou configurações.

## Visualização

O painel central funciona com o resumo publicado mesmo sem os artefatos dos experimentos:

```bash
uv run --locked python -m ipykernel install --user \
  --name multimodal-graphs --display-name "Multimodal Graphs"
```

Abra [`notebooks/resultados_experimentos.ipynb`](notebooks/resultados_experimentos.ipynb), selecione o kernel e execute **Run All**. Ao adicionar suas próprias execuções a `results/` e `experiments/training/`, as seções detalhadas passam a exibir métricas, previsões e históricos disponíveis. O notebook apenas lê os arquivos; não inicia inferência ou treinamento.

Os notebooks de [exploração](notebooks/explorar_dados.ipynb) e [galeria](notebooks/galeria_png.ipynb) exigem o conjunto de dados preparado.

## Preparação dos dados

Os CSVs recebidos, planilhas, imagens derivadas, previsões brutas e checkpoints ficam fora do Git. O conjunto original precisa ser obtido separadamente; este repositório não disponibiliza um download público desses dados. Prepare a estrutura descrita em [formato dos dados](docs/data.md):

```text
dados_projeto_multi_modais/
├── dados/
│   ├── BAR-001.csv
│   ├── ...
│   ├── parametros.json
│   └── gabarito.json
└── gabarito.xlsx
```

Com os arquivos recebidos disponíveis:

```bash
uv run --locked python scripts/local/gerar_graficos.py
uv run --locked python scripts/local/gerar_graficos.py --validate-only
uv run --locked python -m src.data.pilot
```

A renderização verifica as fontes e preserva os arquivos existentes. O piloto mantém as perguntas do mesmo gráfico juntas e seleciona metade dos gráficos por tipo/estilo. Para preparar treino, desenvolvimento, quarentena e pares, siga [o protocolo de experimentos](docs/experiments.md).

## Inferência com os checkpoints oficiais

A inferência requer Linux x86_64, Python 3.10 e GPU NVIDIA compatível com as versões PyTorch/CUDA dos perfis. Os pesos são obtidos dos repositórios dos autores e verificados por revisão e SHA-256; não são distribuídos neste GitHub.

Para o **I-MoF oficial**, em um host GPU ou alocação SLURM:

```bash
SIH_PYTHON=python3.10 bash scripts/baseline.sh install
bash scripts/baseline.sh check --gpu
bash scripts/baseline.sh smoke --run-id imof-smoke-01
bash scripts/baseline.sh run --run-id imof-pilot-01
```

O [guia do baseline](scripts/BASELINE.md) explica o ambiente legado, os downloads e a retomada. Ele fica em `.venv-mof`, separado do ambiente de análise.

Para **LLaVA, Qwen, CLIP e DINOv3**:

```bash
SIH_PYTHON=python3.10 bash scripts/control.sh install
bash scripts/control.sh download --dinov3-source "$DINOV3_CHECKPOINT"
bash scripts/control.sh check --hashes
bash scripts/control.sh smoke --model llava --run-id control-smoke-llava
bash scripts/control.sh run --model llava --run-id control-pilot-llava
```

`DINOV3_CHECKPOINT` aponta para o checkpoint oficial obtido com a permissão de acesso correspondente. Use `--model qwen` e outro `--run-id` para o Qwen. O [guia dos controles](docs/CONTROLE_MOF_DINOV3.md) descreve revisões, inicializações e saídas.

## Treinamento e execução em cluster

O treino ajusta os projetores visuais com encoders e LLM congelados. Checkpoints são selecionados exclusivamente no desenvolvimento; a época zero também pode vencer. Os testes de gradiente, atualização, recarga e retomada são pré-requisitos. A avaliação do piloto é uma etapa separada.

O [guia de experimentos](docs/experiments.md) descreve os comandos de preparação e treinamento. Os [lançadores SLURM](scripts/apuana/README.md) permitem configurar nó e recursos no momento da submissão. Nenhum endereço, conta ou nó pessoal é necessário para usar a CLI diretamente em outro host Linux.

## Organização

```text
configs/             revisões dos modelos, hashes e parâmetros dos experimentos
src/data/            renderização, seleção, auditoria e construção de pares
src/models/          checkpoints oficiais, encoders e intercalação visual
src/evaluation/      inferência, corretores, métricas e análise dos pares
src/training/        ajuste dos projetores e retomada de checkpoints
scripts/local/       preparação, relatórios e consulta remota configurável
scripts/apuana/      instalação e execução em SLURM
notebooks/           painel de resultados, exploração e inspeção visual
results/published/   resumo portátil das evidências verificadas
tests/               testes independentes de dados privados e pesos completos
```

`data/`, `experiments/` e as demais pastas de `results/` são áreas de trabalho geradas por cada execução. Caches, dados, logs, credenciais e snapshots históricos são ignorados pelo Git. Consulte [reprodução e publicação](docs/reproducibility.md).

## Verificações

```bash
uv run --locked ruff check src scripts tests
uv run --locked python -m unittest discover -s tests -v
uv run --locked python scripts/local/check_publication.py
```

A CI executa testes portáteis e uma etapa com PyTorch em CPU. Testes de modelos com pesos completos e CUDA precisam de uma alocação GPU e não são substituídos pela CI.

## Referência

O baseline deriva de [Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs](https://arxiv.org/abs/2401.06209), de Tong et al., e do [código oficial MMVP](https://github.com/tsb0601/MMVP). O I-MoF oficial combina CLIP e DINOv2. A extensão experimental deste projeto utiliza DINOv3 em LLaVA-1.5-7B e Qwen2-VL-2B e não deve ser apresentada como reprodução equivalente do checkpoint de 13B.
