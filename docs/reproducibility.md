# Reprodução e publicação

## Ambientes

`pyproject.toml` e `uv.lock` definem o ambiente de análise em Python 3.10–3.12. Use `uv sync --locked --extra notebooks --extra dev`. O baseline oficial usa `.venv-mof`/Python 3.10 e os arquivos `requirements-mof.txt` e `constraints-mof.txt`. Os controles usam `.venv-control` e `requirements-control.txt`. As duas versões de Transformers pertencem a ambientes diferentes.

A CI usa dados artificiais nos testes e não baixa os pesos completos. Verificações com CUDA, acesso a checkpoints restritos e a coleção original continuam exigindo os recursos correspondentes.

## O que é publicado

Código de dados/modelos/avaliação/treino, configurações de protocolo, hashes esperados dos checkpoints, testes, lançadores, notebooks sem outputs e o resumo numérico portátil em `results/published/`.

Não são publicados os CSVs recebidos, gabaritos completos, planilhas, PNGs de entrada, checkpoints, ambientes, tokens, configurações pessoais, logs, previsões brutas, snapshots de fonte das execuções ou registros com caminhos de máquinas. Eles permanecem nas áreas de trabalho ignoradas pelo Git. A publicação não altera o conteúdo de arquivos históricos usados para verificar hashes.

O resumo publicado contém IDs de execução, versões de corretor, denominadores e hashes de previsões de origem, sem expor seus caminhos absolutos. Esses hashes registram a linhagem, mas sem o pacote original não permitem reproduzir integralmente os números. Os dados originais não têm URL pública de distribuição definida neste projeto.

## Registros de uma nova execução

Guarde código/configurações, revisão dos modelos, hashes das entradas e checkpoints, seed, versões de dependências, hardware, IDs do scheduler, previsões e método de seleção. Os executores geram manifestos e recusam sobrescrever uma execução divergente. Alterações de fonte/configuração exigem novo ID; preserve as fontes congeladas se precisar retomar uma série anterior.

A revisão para publicação tornou os lançadores portáteis. Ela não reproduziu inferência CUDA nem treinamento completo dos modelos. Resultados científicos anteriores referem-se aos arquivos e fontes registrados naquela execução.

## Monitoramento remoto

Copie `.env.example` para `.env` se desejar guardar configurações privadas. O arquivo não é carregado automaticamente; exporte as variáveis no terminal, ou passe as flags diretamente:

```bash
python scripts/local/watch_mof.py \
  --host "$SIH_SSH_HOST" --remote-root "$SIH_REMOTE_ROOT" \
  --run-id imof-pilot-01 --job-id 12345
```

`--watch --duration 300 --interval 30` acompanha por até cinco minutos. `--collect` copia somente uma execução concluída com `rsync >=3`, sem `--delete`. SSH e rsync são ferramentas externas opcionais.

O monitor não submete, cancela ou reinicia jobs. Ausência de conexão ou de jobs na fila não comprova falha/conclusão do experimento. Confira os artefatos e seus hashes. Credenciais Hugging Face devem ficar no ambiente ou no login oficial da ferramenta; nunca em arquivos versionados.
