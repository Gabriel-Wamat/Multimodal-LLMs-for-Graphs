# Arquitetura e modelos

O fluxo principal é `CSV → PNG → manifesto de entradas → modelo → resposta bruta → corretor → métricas`. Gabaritos são separados das entradas do modelo. A versão recebida contém seis famílias de gráficos: barras, linhas, dispersão, pizza, histograma e boxplot.

## Componentes

| Componente | Responsabilidade |
|---|---|
| `src/data/render.py` | Reconstrução dos gráficos e validação de hashes, dimensões e links da planilha |
| `src/data/pilot.py` | Seleção estratificada e determinística do piloto |
| `src/data/audit_training.py` | Triagem de semelhança numérica entre famílias e splits |
| `src/data/pairs.py` | Geração de variantes A/B, perguntas binárias e fatos de referência |
| `src/models/mof.py` | Carregamento do I-MoF oficial com CLIP e DINOv2 |
| `src/models/control/` | LLaVA, Qwen, CLIP, DINOv3, projetores e intercalação |
| `src/evaluation/` | Inferência, correção por formato, análise pareada e comparação de alinhamento |
| `src/training/` | Loss supervisionada, ajuste dos projetores, seleção e retomada |

## Modelos

| Modelo | Revisão e parâmetros |
|---|---|
| I-MoF oficial, LLaVA-1.5 13B | `configs/baseline_mof.json`; CLIP + DINOv2; FP16; imagem 336×336 |
| LLaVA-1.5-7B e Qwen2-VL-2B-Instruct | `configs/models_control.json`; revisões fixadas; BF16 |
| CLIP ViT-L/14 | 336×336, camada −2, grade de 576 patches |
| DINOv3 ViT-B/16 | 384×384, 576 patches anteriores à normalização final, sem CLS/register tokens |

Os checkpoints são resolvidos em tempo de instalação. Os caminhos reais ficam em `.cache/`, fora do Git; as configurações publicadas contêm identificadores e hashes. `configs/checkpoints-mof.json` descreve os arquivos esperados do baseline oficial.

## Controles

`native` conserva a representação visual original. `native_double` duplica seus tokens. `clip_double` intercala duas cópias dos mesmos tokens CLIP; não são dois encoders CLIP independentes. `clip_dino` intercala CLIP e DINOv3 na mesma ordem espacial. Os braços `dino_double` e `clip_dino_shuffled` são diagnósticos adicionais implementados, omitidos da tabela principal de resultados.

No LLaVA, `native_double` e `clip_double` são equivalentes antes do ajuste. No Qwen, a representação nativa é diferente do CLIP externo, que exige seu próprio projetor. A fusão e os controles duplicados alimentam 1.152 tokens visuais. A comparação iguala o comprimento da sequência, mas não o número de parâmetros treináveis ou o histórico de inicialização.

A extensão com DINOv3 é experimental. Não é o checkpoint I-MoF de 13B dos autores. No estágio sem alinhamento, os novos projetores DINO e CLIP/Qwen são inicializados por seed e não devem ser interpretados como modelos multimodais já treinados.

## Treinamento

O treinador congela os encoders e o LLM e ajusta somente os projetores ativos. Parâmetros/AdamW dos projetores ficam em FP32 e o forward usa autocast BF16. O protocolo fixa três épocas, acumulação de gradientes, warmup/cosine e seleção exclusiva pelo desenvolvimento. A época zero participa da seleção. A inferência no piloto ocorre separadamente e carrega o checkpoint selecionado.

As condições pareadas são imagem original, quatro recortes 2×2 com agregação fixa e transcrição visual antes da resposta. A transcrição vem da imagem, sem consultar CSV ou gabarito.
