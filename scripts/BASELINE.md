# Baseline I-MoF oficial

`scripts/baseline.sh` instala e executa o checkpoint `MMVP/MoF_Models` na revisão fixada em `configs/baseline_mof.json`. A interface preserva o código de intercalação dos autores, CLIP e DINOv2. Não usa DINOv3.

## Requisitos e instalação

Linux x86_64, Python 3.10, Git, acesso aos repositórios dos pesos e GPU NVIDIA. O verificador exige pelo menos 32 GiB de VRAM para o protocolo FP16 sem quantização. Reserve espaço também para o ambiente e os downloads.

```bash
SIH_PYTHON=python3.10 bash scripts/baseline.sh install
bash scripts/baseline.sh check --gpu
```

Para separar instalação e download, use `install --env-only` e depois `download`. O instalador cria `.venv-mof`, baixa as revisões fixadas do código oficial em `work/vendor/`, verifica os 17 arquivos do manifesto e grava caminhos resolvidos em `.cache/mof/weights.json`. Nenhuma credencial é gravada nas configurações públicas.

## Dados e inferência

Prepare os arquivos conforme [o guia de dados](../docs/data.md), gere imagens/piloto e valide a coleção:

```bash
bash scripts/baseline.sh check --data-only
bash scripts/baseline.sh smoke --run-id imof-smoke-01
bash scripts/baseline.sh run --run-id imof-pilot-01
```

O piloto possui 360 gráficos/3.173 perguntas. O smoke usa seis perguntas e não substitui sua avaliação completa. A instalação, a verificação integral dos pesos e a inferência devem rodar em uma máquina GPU ou alocação adequada, não no servidor de login do cluster.

O ID identifica uma execução nova. Para retomar, use `run --run-id imof-pilot-01 --resume`; os manifestos rejeitam código, dados, parâmetros ou pesos divergentes. O teto e a precisão fazem parte do protocolo. Preserve a série anterior se decidir mudá-los.

## Saídas

As previsões ficam em `results/<run-id>/predictions.jsonl`, com perguntas, respostas e tempos. `run.json` e `runtime.json` registram configuração, ambiente e pesos; `complete.json` e `scored/metrics.json` permitem conferir hashes e cobertura. Respostas e gabaritos são corrigidos por `src/evaluation/score.py`.

O resultado histórico e o diagnóstico posterior de idioma estão no [resumo publicado](../results/published/README.md). A avaliação do I-MoF oficial na tarefa dos pares A/B não foi confirmada. Os [lançadores SLURM](apuana/README.md) são opcionais; a mesma CLI funciona diretamente no host Linux preparado.
