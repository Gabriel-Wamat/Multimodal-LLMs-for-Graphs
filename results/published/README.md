# Resultados publicados

Snapshot das evidências coletadas em **03/11/2026**, publicado com o código. Não é o estado ao vivo do cluster. O arquivo [`summary.json`](summary.json) guarda os numeradores, denominadores, versões de correção e hashes das previsões de origem; as respostas e os pacotes privados de execução não são distribuídos aqui.

## Piloto por pergunta

360 gráficos, 3.173 perguntas, mesmos IDs entre os controles e avaliações ajustadas. O ajuste treina somente os projetores visuais.

| Modelo / configuração | Antes do ajuste | Após ajuste |
|---|---:|---:|
| LLaVA nativo | 8,89% | — |
| LLaVA nativo duplicado | 9,90% | — |
| LLaVA — CLIP duplicado | 9,90% | **27,92% (886/3.173)** |
| LLaVA — CLIP + DINOv3 | 0,00% | **24,27% (770/3.173)** |
| Qwen — CLIP duplicado | Resultado final não confirmado | **24,30% (771/3.173)** |
| Qwen — CLIP + DINOv3 | Resultado final não confirmado | Sem avaliação final confirmada |

O I-MoF oficial de 13B obteve **119/3.173 = 3,75%** pela regra original. O diagnóstico posterior de idioma, aceitando equivalentes exatos Yes/No → Sim/Não, produz **311/3.173 = 9,80%**; ele não substitui a métrica original.

O LLaVA CLIP duplicado ganhou **18,03 pontos percentuais**, com IC95% por família **[16,39; 19,66]**. O LLaVA CLIP+DINOv3 ficou **3,66 pp** abaixo do CLIP duplicado nesta execução. Isso não estabelece superioridade robusta entre seeds ou efeito isolado do encoder: os projetores têm números de parâmetros e inicializações diferentes.

## Tarefa pareada

720 pares por modelo, 1.440 imagens e três condições. Acurácia por par exige os dois acertos; não tem o mesmo denominador/regra do piloto anterior.

| Modelo | Condição | Por pergunta | Por par [IC95%] | Respostas inválidas |
|---|---|---:|---:|---:|
| LLaVA | Original | 50,00% | 0,00% [0,00; 0,00] | 0 |
| LLaVA | Recortes 2×2 | 50,00% | 0,00% [0,00; 0,00] | 0 |
| LLaVA | Transcrição | 50,00% | 0,00% [0,00; 0,00] | 0 |
| Qwen | Original | 59,38% | 20,42% [17,40; 23,61] | 0 |
| Qwen | Recortes 2×2 | 44,31% | 0,14% [0,00; 0,42] | 161 |
| Qwen | Transcrição | 52,43% | 9,44% [7,02; 11,85] | 2 |

No Qwen original, alterações amplas menos sutis resultaram em **+28,06 pp [23,05; 33,06]** de acurácia por par. As mitigações testadas reduziram o desempenho: recortes **−20,28 pp** e transcrição **−10,97 pp** em relação à imagem original. Os intervalos reamostram famílias inteiras; conclusões permanecem exploratórias.

## Interpretação e pendências

O estágio zero com DINO usa projetores ainda sem alinhamento e não equivale ao I-MoF treinado. O controle de CLIP duplicado iguala os tokens visuais, não toda a capacidade treinável. O conjunto foi explorado antes da conclusão da campanha e sua independência de origem não foi certificada.

Acurácias são do corretor determinístico por formato. BERTScore não foi calculado e não foi realizada auditoria semântica completa de todas as respostas. Resultados sem confirmação permanecem ausentes; não são tratados como zero ou jobs ativos.

O controle completo Qwen, a avaliação final Qwen CLIP+DINOv3 e a avaliação do I-MoF nos pares não foram confirmados neste snapshot. Veja [o protocolo de métricas](../../docs/evaluation.md).

## Linhagem e atualização

Os números foram exportados dos artefatos completos após verificar os hashes das previsões e a unicidade de IDs. O exportador escreve somente campos autorizados, sem caminhos de máquinas, logs, prompts ou gabaritos:

```bash
python scripts/local/export_public_results.py --snapshot-date YYYY-MM-DD
```

Esse comando exige os pacotes privados de execução em suas pastas correspondentes. Uma cópia limpa do GitHub já contém o resumo e pode visualizá-lo sem eles. A publicação do código não repetiu inferência CUDA ou treinamento completo.

## Treino em duas etapas

Ao concluir as avaliações dos três controles de uma base, execute na raiz do projeto:

```bash
.venv-control/bin/python -m scripts.local.export_full_sft_results \
  --prefix <campanha> --model qwen --dataset pilot50_v1
```

Use `confirmation_v1` para o conjunto independente. Cada exportação confere os hashes, corrige as respostas com o gabarito e registra em `results/published/full_sft/` os acertos, denominadores, intervalos de confiança, erros por formato, custo de inferência e diferenças entre braços. O notebook lê esses resumos em um checkout público.
