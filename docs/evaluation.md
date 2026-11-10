# Métricas e interpretação

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
