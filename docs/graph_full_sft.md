# Adaptação em duas etapas aos gráficos

Esta campanha aplica a estrutura de treinamento do I-MoF aos dados do projeto. Não é reprodução literal dos datasets, tamanhos de modelo ou hiperparâmetros do artigo.

## Componentes e dados

| Etapa | Supervisão | Treinável | Congelado |
|---|---|---|---|
| 1 — alinhamento imagem-texto | 277 descrições derivadas exclusivamente dos CSVs de treino | Projetores visuais ativos | Decoder, head e encoders visuais |
| 2 — instruction tuning | 2.475 perguntas e respostas de treino | Decoder completo, embeddings, head e projetores ativos | CLIP, DINOv3 e ramo visual nativo não utilizado |

As 632 perguntas de desenvolvimento selecionam o checkpoint de SFT. As 3.173 perguntas do piloto exploratório são usadas somente na avaliação final. As famílias protegidas continuam separadas. A etapa 1 não usa gabaritos de perguntas: as descrições vêm dos CSVs pertencentes ao treino e têm manifestos/hashes próprios.

Bases: Qwen2-VL-2B e LLaVA-1.5-7B. Braços em cada base: `clip_single`, `clip_double`, `clip_dino`. A representação CLIP externa é compartilhada entre os braços, com grades idênticas às do experimento de fusão; no Qwen, ela substitui explicitamente o ramo visual nativo. O CLIP único tem 576 tokens; os dois demais têm 1152.

## Configuração prospectiva

A configuração está em [`configs/graph_full_sft_v1.json`](../configs/graph_full_sft_v1.json). São três passagens pelos dados em cada etapa, microbatch 1 e acumulação 8. Esse orçamento pequeno de adaptação difere das épocas e batches do paper, que usava centenas de milhares de exemplos. Não é um número de épocas declarado ótimo.

Learning rates: `2e-3` para alinhamento e `2e-5` para SFT, AdamW, warmup de 3%, scheduler cosseno e clipping 1. O otimizador e o scheduler são reiniciados entre etapas. O decoder opera em BF16, com checkpointing de ativações não reentrante na etapa 2; projetores, pesos mestres e estados AdamW usam FP32. Os pesos mestres e momentos ficam na RAM do nó Apuana para permitir treinamento completo, sem LoRA, em uma A100 de 80 GB. Esse mecanismo de offload é uma implementação própria da adaptação e tem testes de paridade com AdamW.

As inicializações CLIP são compartilhadas dentro de cada base. Cada braço começa novamente da mesma base linguística; não reutiliza os projetores ajustados na campanha antiga. O decoder permanece comprovadamente inalterado na primeira etapa e tem todos os seus parâmetros liberados na segunda.

## Testes obrigatórios

Os contratos CPU verificam AdamW, LR/scheduler, continuidade e rejeição de gradientes não finitos. O diagnóstico GPU deve passar para cada braço:

- Forward/backward em seis exemplos, cobrindo os seis tipos de gráfico, nos dois estágios.
- Decoder inalterado na etapa 1; gradientes e atualização efetiva do decoder e projetores na etapa 2.
- Inventário de mais de um bilhão de parâmetros do decoder Qwen ou seis bilhões no LLaVA.
- Salvamento e restauração de decoder, head, projetores, pesos mestres, momentos AdamW, scheduler, RNG e cursor.
- Comparação da continuação com e sem interrupção: loss, amostras dos pesos GPU e hash de todos os pesos mestres/estados AdamW.

O executor recusa gates ausentes, configurações alteradas ou código diferente dos gates. Atualizações diagnósticas não são treinamento de produção nem acurácia do piloto.

## Execução no Apuana

Os scripts são templates; consultar limites e disponibilidade atuais e executar a partir de um snapshot congelado. Nenhum modelo ou checkpoint completo é copiado para o Mac.

```bash
# Preparação CPU
.venv-control/bin/python -m src.data.alignment_captions
.venv-control/bin/python -m unittest discover -s tests -p test_full_sft_contract.py -v

# Submeter da raiz do snapshot congelado, com IDs novos.
sbatch scripts/apuana/check_full_sft.sh qwen graph-full-check-v1

# Somente depois dos gates aprovados:
export SFT_BACKUP_ROOT="$HOME/checkpoint-backups/graph-full-sft"
sbatch scripts/apuana/run_full_sft.sh qwen graph-full-sft-v1 graph-full-check-v1
```

O launcher de produção executa os três braços sequencialmente e avalia o checkpoint escolhido de cada um. Não reenviar sobre um job vivo ou reutilizar IDs. Para uma execução realmente interrompida, usar o mesmo snapshot e o executor com `--resume`; a retomada é específica do braço interrompido.

## Checkpoints e resultados

- `experiments/full_sft_checks/<id>/`: provas dos testes GPU, contagens e hashes; checkpoints temporários do diagnóstico são removidos após a retomada verificada.
- `experiments/full_sft/<id>/run.json`: configuração, fontes e dados verificados.
- `stage1-model.pt`: projetores ao terminar alinhamento.
- `last.pt`: estado completo recuperável, publicado atomicamente a cada 30 minutos de execução e ao final de cada época. Registrar que os checkpoints são muito maiores que os da ablação.
- `best-model.pt` e `best.json`: decoder/head/projetores do checkpoint escolhido exclusivamente pelo desenvolvimento, inclusive a época zero de SFT.
- `history.json`, `progress.json`, `checkpoint.json`: losses, métricas, fase, duração e recursos.
- `results/<id>-pilot/`: previsões e métricas estritas e com aliases binários, na mesma régua congelada do piloto.

As cópias ficam em uma pasta privada fora do projeto, no Apuana. Uma cópia no mesmo NFS protege contra remoção da pasta de execução, mas não equivale a backup em outro armazenamento. O usuário definiu que os pesos/checkpoints completos desta campanha permanecerão no Apuana.

A comparação principal é CLIP+DINOv3 versus CLIP único e versus CLIP duplicado, após o mesmo protocolo completo. A comparação com a ablação antiga é descritiva: mudam também o alinhamento por descrições, a taxa de aprendizado e o escopo dos parâmetros. Não atribuir toda diferença exclusivamente ao decoder nem prometer benefício do DINOv3.

## Comparação dos controles

Com os três pilotos concluídos para uma base, execute no Apuana:

```bash
.venv-control/bin/python -m src.evaluation.compare_full_sft --model qwen --prefix graph-full-sft-v1
```

Use `--model llava` para a outra base. O relatório em `results/<prefixo>-comparison/` contém a diferença em pontos percentuais entre fusão e cada controle, intervalos pareados por gráfico e por família, erros de formato, latência e memória. O cálculo confere as previsões com o gabarito congelado e compara a configuração, a semente e as fontes entre os três treinos.

## Confirmação em gráficos inéditos

O conjunto `confirmation_v1` usa 360 fontes sintéticas novas, com dez gráficos para cada combinação de seis tipos e seis estilos. O gerador usa uma semente fixa e produz três perguntas por gráfico. A construção independe das previsões e dos CSVs usados no ajuste. A auditoria confere as imagens contra os hashes dos gráficos do projeto e seus pares derivados. O conjunto tem sua própria distribuição de perguntas e um denominador de 1.080 respostas. As comparações entre braços devem usar as mesmas perguntas dentro de cada conjunto.

No nó de CPU, instale o ambiente de análise em `.venv-analysis` pelas dependências de `requirements.txt` e execute `bash scripts/apuana/prepare_confirmation.sh`. Após concluir os três pilotos de uma base, `sbatch scripts/apuana/evaluate_confirmation.sh <prefixo> <modelo>` avalia os checkpoints escolhidos no desenvolvimento nas imagens inéditas.

## Estabilidade entre sementes

Para quantificar a variação entre treinos, o protocolo usa as sementes 20261102, 20261103 e 20261104 para a ordem dos exemplos e a inicialização dos projetores aleatórios. CLIP único, CLIP duplicado e fusão partem da mesma inicialização CLIP dentro de cada base e semente. O projetor CLIP nativo do LLaVA parte dos pesos pré-treinados. Os dados, as épocas, os critérios de seleção e a correção das respostas são os mesmos.

Crie cada réplica com `python3 -m scripts.apuana.prepare_sft_replica --source <snapshot-primário> --destination <novo-snapshot> --primary-record <run.json-primário> --seed <semente>`. Execute os diagnósticos e os três treinos por modelo a partir dessa cópia, usando identificadores distintos. Ao completar os três pilotos e as avaliações independentes, execute:

```bash
.venv-control/bin/python -m src.evaluation.seed_stability \
  --model qwen --dataset confirmation_v1 --prefixes <primário> <réplica-1> <réplica-2>
```

O relatório calcula a diferença da fusão em pontos percentuais **dentro de cada semente** antes de resumir a média, o desvio-padrão amostral e os extremos das três diferenças. Os intervalos por gráfico medem outra fonte de variação e aparecem nos relatórios de cada execução.
