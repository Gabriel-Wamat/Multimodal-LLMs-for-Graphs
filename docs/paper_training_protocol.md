# Treinamento do I-MoF: reprodução e adaptação ao domínio

Escolha do usuário: aplicar a receita em duas etapas aos nossos gráficos. O executor está implementado e passa por gates GPU antes de produção; ver [adaptação aos gráficos](graph_full_sft.md). A reprodução literal abaixo permanece como referência, com diferenças de datasets e hiperparâmetros explicitadas.

## Correção de escopo

A campanha `alignment-training-v1` ajusta apenas projetores. Ela é uma ablação de alinhamento, não a reprodução do treinamento completo do artigo. Seus resultados e checkpoints devem permanecer preservados em sua própria série.

O artigo *Eyes Wide Shut?* usa duas etapas. No código oficial, `pretrain.sh` ativa `tune_mm_mlp_adapter=True`, e o executor congela o modelo e libera os projetores CLIP e DINO. Em `finetune.sh`, esse modo não é ativado: o decoder e os projetores são treinados. Os encoders visuais ativos CLIP e DINOv2 permanecem congelados. LoRA não é ativado na receita publicada.

Fontes: [artigo, Apêndice A](https://arxiv.org/html/2401.06209v2#A1), [pré-treino](https://github.com/tsb0601/MMVP/blob/main/LLaVA/pretrain.sh), [instruction tuning](https://github.com/tsb0601/MMVP/blob/main/LLaVA/finetune.sh), [executor](https://github.com/tsb0601/MMVP/blob/main/LLaVA/llava/train/train.py), [CLIP](https://github.com/tsb0601/MMVP/blob/main/LLaVA/llava/model/multimodal_encoder/clip_encoder.py), [DINOv2](https://github.com/tsb0601/MMVP/blob/main/LLaVA/llava/model/multimodal_encoder/dino_encoder.py).

## Receita publicada para LLaVA-1.5

| Item | Etapa 1 — alinhamento | Etapa 2 — instruction tuning |
|---|---|---|
| Base linguística | Vicuna-13B-v1.5 | Mesma base, com projetores da etapa 1 |
| Componentes treinados | Projetores visuais | Modelo de linguagem completo e projetores |
| Encoders visuais | CLIP e DINOv2 congelados | CLIP e DINOv2 congelados |
| Dados citados | CC595k | DataMix 665k |
| Épocas | 1 | 1 |
| Batch global, Tabela 4 | 256 | 128 |
| Learning rate, Tabela 4 | 2e-3 | 2e-5 |
| Otimizador | AdamW | AdamW |
| Warmup / weight decay | 3% / 0 | 3% / 0 |
| Scheduler | Cosseno | Cosseno |
| DeepSpeed ZeRO, Tabela 4 | 2 | 3 |

O artigo distingue LLaVA e LLaVA-1.5: as três épocas de instruction tuning da coluna LLaVA não devem ser atribuídas à coluna LLaVA-1.5.

## Divergências que precisam ser resolvidas

Os scripts públicos são templates, não uma receita integralmente preenchida:

- O pré-treino público usa learning rate `1e-3`, enquanto a Tabela 4 indica `2e-3` para LLaVA-1.5.
- O fine-tuning público combina oito GPUs, batch por dispositivo 11 e acumulação 2, resultando em batch global 176, enquanto a tabela indica 128.
- Os caminhos dos dados, da base Vicuna, dos adaptadores e do arquivo DeepSpeed são placeholders.
- É preciso conferir a composição exata dos dados citados como CC595k e DataMix 665k, as versões e a disponibilidade das imagens. Não substituir por outra versão de dataset sem registrar a diferença.
- A revisão oficial já registrada no projeto é `763500597e65c3446f09047837ceda76f4e264bf`; os quatro arquivos de treinamento relevantes foram obtidos nessa revisão e tiveram seus hashes registrados. A auditoria confirmou as diferenças de learning rate e batch global acima.

Candidato de referência: Tabela 4 publicada. A decisão final e as divergências devem ficar registradas antes de submeter uma reprodução.

## Modelos e controles

Para reprodução literal, usar Vicuna/LLaVA-1.5-13B, CLIP ViT-L/14 e DINOv2 ViT-L/14. O controle original é CLIP único. CLIP duplicado é um controle adicional da quantidade de tokens.

Escolher explicitamente uma geometria publicada: a Tabela 3 usa fusão em 224² com 512 tokens e reporta 28% no MMVP; o Apêndice E.2 também reporta fusão em 336² com 1152 tokens e 31,3%. Não misturar essas variantes numa única receita sem identificação.

Qwen, LLaVA-7B e DINOv3 são extensões: podem receber a mesma estrutura de treinamento em duas etapas, mas não são uma reprodução literal do artigo.

## Aplicação aos nossos gráficos

Se a escolha for adaptar o método ao domínio, preparar alinhamento imagem-texto somente com imagens de treino; depois, treinar modelo de linguagem e projetores nas 2.475 perguntas de treino. Preservar as 632 perguntas de desenvolvimento e as 3.173 do piloto exploratório. Não usar CSVs, descrições ou respostas de desenvolvimento/piloto para gerar supervisionamento de treino.

Comparar CLIP único, CLIP duplicado e CLIP+DINO dentro da mesma base linguística, partindo de inicializações com ancestralidade controlada. Fixar os dois estágios, ordem dos exemplos, otimizador, batch global, número de atualizações e seleção de checkpoint antes dos resultados. O piloto já explorado continua sendo evidência exploratória; confirmação de generalização exige dados novos e reservados.

## Trabalho necessário no software

1. Separar o novo executor do `AlignmentModel`, que congela o decoder por contrato.
2. Implementar os dois estágios com inventário explícito dos parâmetros treináveis.
3. Verificar gradientes e atualizações do decoder na etapa 2, além dos projetores, mantendo os encoders congelados.
4. Salvar e restaurar modelo de linguagem completo, projetores, otimizador, scheduler, RNG e posição nos dados. O checkpoint atual de projetores não cobre esse contrato.
5. Adaptar a inferência para carregar o checkpoint completo escolhido pelo desenvolvimento.
6. Medir memória, throughput, comunicação distribuída e custo de checkpoints no Apuana antes de definir o orçamento. O artigo utilizou até oito A100; a alocação atual de uma A100 validou ajuste de projetores, não full fine-tuning de 13B.
7. Usar novos IDs, snapshots congelados, backups privados e registros de progresso.

## Gates de desenho

| Gate | Estado | Evidência / próxima ação |
|---|---|---|
| D0 — objetivo | pass | Usuário escolheu adaptação em duas etapas aos gráficos |
| D1 — dados | pending | Auditar datasets originais ou criar supervisão de alinhamento de treino |
| D2 — régua | pending | MMVP por par e piloto de gráficos por pergunta são séries distintas |
| D3 — braços | pending | Arquiteturas e controles dependem da escolha de escopo |
| D4 — critérios | pending | Congelar orçamento e critérios antes dos novos resultados |
| D5 — recursos | unknown | Ainda sem diagnóstico de full fine-tuning na alocação real |
| D6 — execução | pending | Configuração de desenho não constitui executor pronto nem treino iniciado |

Configuração de desenho: [`configs/paper_training_v1.json`](../configs/paper_training_v1.json).
