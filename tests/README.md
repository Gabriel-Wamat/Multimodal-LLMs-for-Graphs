# Verificações

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check src scripts tests
```

Os testes portáteis cobrem corretores, seleção, pares, bootstrap, hipóteses, HTML seguro, validação dos launchers e falhas de monitoramento. Eles criam seus próprios dados artificiais. O contrato de um arquivo histórico só é executado quando o pacote privado correspondente existe.

Com o perfil `configs/requirements-control.txt`, também são executados testes de intercalação, placeholders, inicialização, extração anterior à normalização, máscaras de supervisão, instrumentação de geração e retomada exata do otimizador/RNG. Eles funcionam em CPU com tensores pequenos e não baixam checkpoints.

Os diagnósticos reais com CUDA e modelos completos são executados separadamente por `src.training.check_alignment`. Sucesso dos testes de CPU não comprova qualidade visual, acurácia do modelo ou equivalência científica entre implementações.
