# Notebooks

- `resultados_experimentos.ipynb`: painel central com resumo publicado e seções detalhadas das execuções disponíveis. Funciona sem pesos e sem a coleção original.
- `explorar_dados.ipynb`: exploração dos CSVs recebidos; exige os dados conforme `docs/data.md`.
- `galeria_png.ipynb`: inspeção das imagens reconstruídas; exige a renderização.

Instale `uv sync --locked --extra notebooks --extra dev` e selecione um kernel desse ambiente. Os notebooks são publicados sem outputs, metadados de máquina ou estado de execução. Execute Run All para ler os arquivos disponíveis. O painel de resultados apenas consulta os artefatos; ele não treina, não executa inferência e não consulta SLURM.
