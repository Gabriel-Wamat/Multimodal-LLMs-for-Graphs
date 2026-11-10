# Resultados

O Git inclui somente [published/](published/), com um resumo numérico portátil e os hashes das previsões de origem. O resumo é um snapshot histórico, sem afirmar o estado atual de jobs ou completar resultados ausentes.

As demais pastas são saídas de execução ignoradas pelo Git. Cada uma deve conservar `run.json`, previsões brutas, métricas, fontes e marcadores de término. Não confundir presença de um arquivo de status com conclusão validada.

O [painel central](../notebooks/resultados_experimentos.ipynb) lê o resumo publicado e os artefatos adicionais disponíveis no checkout. Não inicia jobs. Consulte [métricas](../docs/evaluation.md) e [reprodução](../docs/reproducibility.md).
