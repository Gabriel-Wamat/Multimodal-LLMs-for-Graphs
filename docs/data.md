# Dados e preparação

O conjunto recebido não é distribuído por este repositório. Obtenha os arquivos do responsável pelos dados e preserve os originais em `dados_projeto_multi_modais/`. Nenhuma imagem, planilha ou resposta bruta é necessária para instalar o código ou visualizar o resumo publicado.

## Estrutura recebida

```text
dados_projeto_multi_modais/
├── dados/
│   ├── BAR-001.csv ... BAR-120.csv
│   ├── LIN-001.csv ... LIN-120.csv
│   ├── DIS-001.csv ... DIS-120.csv
│   ├── PIZ-001.csv ... PIZ-120.csv
│   ├── HIS-001.csv ... HIS-120.csv
│   ├── BOX-001.csv ... BOX-120.csv
│   ├── parametros.json
│   └── gabarito.json
└── gabarito.xlsx
```

Os CSVs usam cabeçalho, UTF-8 e separador vírgula. As colunas são:

| Tipo | Campos |
|---|---|
| Barras | `categoria`, `valor` |
| Linhas | `ano`, uma ou mais colunas de séries |
| Dispersão | `x`, `y` |
| Pizza | `categoria`, `percentual` |
| Histograma | `faixa_inicio`, `faixa_fim`, `contagem` |
| Boxplot | `grupo`, `valor` |

`parametros.json` mapeia o ID do gráfico para suas propriedades, incluindo `estilo`. `gabarito.json` é uma lista de perguntas com ID, gráfico, pergunta, formato, regra de correção e resposta. O código do piloto e os corretores são a especificação executável desses campos. A planilha recebida é verificada com os mesmos hashes do conjunto.

## Renderização

```bash
uv run --locked python scripts/local/gerar_graficos.py
uv run --locked python scripts/local/gerar_graficos.py --validate-only
uv run --locked python -m src.data.pilot
```

O protocolo atual espera 720 gráficos e 6.380 perguntas. Não é um importador genérico de qualquer coleção CSV. As saídas ficam em `data/processed/graficos/`, `data/processed/gabarito.xlsx` e `data/manifests/imagens.json`, conforme `configs/data.json`. Os PNGs têm 1500×900 pixels; os encoders posteriormente aplicam seu próprio pré-processamento.

Os gráficos são reconstruídos; não são as imagens originais do material recebido. A escolha de escala, estilo e resolução participa do experimento. O gerador recusa arquivos ou fontes divergentes em um manifesto existente. Para outro conjunto ou convenção, crie uma versão explícita de configuração/manifesto em vez de sobrescrever um experimento.

## Separação e vazamento

O piloto seleciona 360 gráficos por hash/seed, equilibrando tipo e estilo; todas as perguntas de um gráfico permanecem juntas. Na campanha histórica, a auditoria de famílias resultou em 277 gráficos de treino, 71 de desenvolvimento, 360 de avaliação exploratória e 12 em quarentena.

`inputs.jsonl` contém imagem e pergunta, sem resposta. `labels.jsonl` contém o gabarito usado pelo avaliador. Famílias conhecidas e derivados A/B herdam o split. A triagem de semelhança não comprova independência de origem; o piloto já foi explorado.
