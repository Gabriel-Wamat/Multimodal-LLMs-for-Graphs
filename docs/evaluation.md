# Métricas e interpretação

## Como uma resposta recebe 1 ou 0

**Acurácia automática por pergunta** é a fração de perguntas que passam no corretor. **Erro automático** é o complemento dessa fração. Um zero pode representar valor incorreto, contradição, formato rejeitado ou falha de execução. A distinção entre erro factual e resposta semanticamente correta em outro formato exige inspecionar a pergunta, o gráfico, o gabarito e a resposta bruta.

Exemplos de aplicação da regra principal do piloto:

| Gabarito e regra | Saída do modelo | Nota | Motivo |
|---|---|---:|---|
| 85, intervalo inclusivo de 80,75 a 89,25 | `82` | 1 | Valor dentro do intervalo. |
| Mesmo gabarito | `90` | 0 | Valor fora do intervalo. |
| Mesmo gabarito | `A barra vale 85.` | 0 | O valor está correto, mas o corretor exige uma resposta numérica isolada. |
| Sim, pergunta binária | `Yes` | 1 | A regra principal aceita o equivalente em inglês. |
| Grupo F, categoria | `O maior é o Grupo F.` | 0 | A categoria está correta, mas a frase inteira difere do rótulo esperado. |
| A, B, C, em ordem pedida | `B, A, C` | 0 | A resposta inverte duas posições. |

Uma categoria com nome parecido pode designar outra entidade. Por exemplo, “Grupo 12” e “Grupo 1” devem receber notas diferentes quando a pergunta pede a categoria com a maior barra. Conferir o nome e o valor na fonte permite distinguir esse erro factual de uma resposta correta com texto adicional.

Para auditar a semântica, registre por resposta se o fato solicitado foi respondido corretamente, se houve contradição, se faltou informação ou se o próprio gabarito é ambíguo. Registre separadamente se o formato foi cumprido. Uma taxa de acerto semântico exige revisão identificada, amostra e denominador explícitos. Os percentuais automáticos abaixo usam as regras por formato.

## Perguntas do piloto

A avaliação usa o formato solicitado no gabarito. Números com tolerância são aceitos dentro dos limites inclusivos do alvo; os demais formatos usam correspondência normalizada. Listas preservam a ordem. Respostas ausentes, falhas ou não interpretáveis contam no denominador e não recebem acerto.

A regra histórica `strict_v1` normaliza texto, acentos, caixa e formatos, mas exige o idioma esperado nas perguntas binárias. A regra `primary_binary_aliases_v2` acrescenta apenas equivalentes exatos Yes/No → Sim/Não em perguntas de formato sim/não. Essas versões devem ser identificadas; não escolher a regra depois de observar o candidato para favorecer seu resultado.

O I-MoF oficial teve 119/3.173 pela regra original. O diagnóstico posterior de idioma resulta em 311/3.173. Ambas as medidas são preservadas. Os modelos novos foram avaliados também com a regra de aliases. As saídas fora do formato podem incluir conteúdo parcialmente correto; a nota do corretor não equivale a qualidade semântica geral. BERTScore não foi calculado nesta campanha.

## Pares A/B

Acurácia por pergunta considera cada imagem. Acurácia por par só vale 1 quando A e B estão corretas. Um modelo que retorna a mesma alternativa em todo par com respostas opostas pode atingir 50% por pergunta e 0% por par. As duas métricas precisam ser exibidas juntas.

O acaso de 50% por pergunta e 25% por par pressupõe duas alternativas e palpites independentes. Não é uma garantia sobre o comportamento real dos modelos.

## Incerteza e hipóteses

- Os contrastes da análise pareada e do alinhamento reamostram famílias inteiras, preservando a dependência conhecida entre perguntas/variantes. Os resumos individuais também registram intervalos por gráficos estratificados; o método aparece nos JSONs.
- H1 descreve as similaridades de CLIP e DINOv3; limiares relativos são calibrados no desenvolvimento. Alta similaridade global não demonstra ausência de informação útil.
- H2 compara alterações amplas e sutis em origens correspondentes.
- H3 compara cada mitigação com a imagem original nos mesmos pares. McNemar usa os acertos discordantes e seus p-valores são exploratórios.
- Spearman relaciona similaridade e erro. Variáveis constantes tornam a correlação indefinida; não substituir por zero. Intervalos são omitidos quando a reamostragem degenera além do limite registrado.

## Comparabilidade

O piloto de gráficos e os pares binários são tarefas diferentes. O benchmark MMVP dos autores também não é o conjunto sintético deste projeto. Não comparar suas porcentagens como uma única classificação de modelos.

O controle CLIP duplicado iguala os tokens da fusão, mas os projetores têm capacidades e inicializações distintas. Os treinos usam uma inicialização por braço; a campanha não estabelece robustez entre seeds. Os modelos/precisões e tetos de geração diferem entre o I-MoF legado e a extensão experimental.

Os tempos observados incluem efeitos de cache/carregamento e nem todos os executores iniciam o cronômetro na mesma etapa. Não calcular speedup entre cronômetros incompatíveis. Treinos LLaVA e Qwen também usaram classes diferentes de GPU; o hardware deve acompanhar cada registro de custo.

## Revisão semântica dos checkpoints completos

Para cada conjunto, o formulário seleciona uma pergunta por combinação de tipo e estilo de gráfico. A seleção usa apenas os IDs. Os modelos recebem as mesmas 36 perguntas. Respostas idênticas à mesma pergunta compartilham uma decisão. O formulário apresenta o gráfico, a pergunta, o gabarito e a resposta. A chave que identifica o modelo fica em outro arquivo.

```bash
python -m src.evaluation.semantic_review --dataset pilot50_v1 \
  --runs <piloto-controle> <piloto-fusão> --output results/revisao-semantica
```

Preencha uma cópia de `blank_review.csv` e registre o revisor e a evidência visual ou numérica de cada decisão. Use **correta** quando a resposta expressa o fato pedido, inclusive em outra formulação. Use **incorreta** para contradição, entidade errada ou valor fora dos limites. Use **parcial** para uma resposta incompleta e **indeterminada** quando os dados visíveis ou o gabarito forem ambíguos. Informe as quatro contagens e o denominador da amostra. A proporção de casos corretos nessa revisão descreve as perguntas auditadas.

Execute `python -m scripts.apuana.score_semantic_review results/revisao-semantica <planilha-preenchida.csv>` para validar a cobertura e cruzar acerto automático com o veredito semântico por modelo. A tabela inclui a contagem de respostas corretas em significado que receberam zero do corretor automático.

## O que a revisão encontrou no I-MoF oficial

Nas mesmas 36 perguntas, a regra estrita aceitou **1/36 (2,78%)**, a regra com equivalência de sim/não aceitou **5/36 (13,89%)** e a revisão semântica identificou **6/36 (16,67%)** corretas. As 30 restantes foram julgadas incorretas. Houve zero respostas parciais e zero indeterminadas.

A diferença entre cinco e seis veio de uma categoria traduzida. A resposta identificou em inglês o grupo correto do gráfico. O corretor reconhece traduções nas perguntas de sim/não, mas exige o rótulo esperado nas perguntas de categoria. Já informar o total quando a pergunta pede o valor de uma fatia conta como erro de conteúdo.

A revisão foi feita pelo Codex, com um único revisor e conhecimento da identidade do modelo. Os percentuais descrevem apenas as 36 perguntas auditadas. No piloto de 3.173 perguntas, há pontuações automáticas. As contagens agregadas e os hashes do formulário e da revisão estão no [registro da amostra](../results/published/semantic-mof13b-v1.json).

## Revisão dos modelos com ajuste apenas dos projetores

O Codex revisou as saídas dos três checkpoints nas mesmas 36 perguntas do piloto usadas na auditoria do I-MoF. O formulário ocultava qual checkpoint havia produzido cada resposta. Respostas idênticas à mesma pergunta receberam um único julgamento. Foram 77 julgamentos distintos para 108 respostas. A revisão teve um revisor.

| Checkpoint | Acertos automáticos | Acertos semânticos | Erros semânticos |
|---|---:|---:|---:|
| LLaVA, CLIP duplicado | 14/36 | 14/36 | 22/36 |
| LLaVA, CLIP + DINOv3 | 10/36 | 10/36 | 26/36 |
| Qwen, CLIP duplicado histórico | 13/36 | 13/36 | 23/36 |

As duas regras coincidiram em cada resposta desta amostra. Os três modelos inverteram a resposta a uma pergunta sobre sobreposição de caixas e acertaram a direção de uma tendência crescente. Para perguntas aproximadas, a revisão aceitou respostas numéricas dentro dos limites inclusivos do gabarito. Houve zero casos parciais e zero indeterminados. A concordância observada descreve as perguntas revisadas. O controle Qwen desta tabela pertence à campanha histórica. [Contagens e registro da revisão](../results/published/semantic-projectors-v1.json)
