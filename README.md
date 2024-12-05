# Trabalho Final da disciplina de Aprendizado de Máquina INE5664

## Implementação de uma Rede Neural Multitarefa

Este projeto apresenta a implementação de uma rede neural customizada capaz de lidar com tarefas de:

- Classificação Binária
- Regressão
- Classificação Multiclasse

Ele inclui funções de ativação, funções de perda, retropropagação e um modelo de rede neural treinável, com dados simulados ou reais.

## Conjunto de dados

### Classificação Binária

- Fonte: https://www.kaggle.com/datasets/yasserh/heart-disease-dataset/data.
- Se refere à presença de doença cardíaca ou não no paciente, 1 tem, 0 não tem.
- Classificação com base nos parâmetros de:
  - idade
  - sexo
  - nível de dor torácica
  - pressão arterial em repouso (BPS)
  - nível de colesterol
  - nível de açúcar no sangue em jejum (FBS)
  - resultados do eletrocardiograma (ECG) em repouso
  - frequência cardíaca máxima alcançada
  - angina induzida por exercício físico
  - depressão do segmento ST no eletrocardiograma

### Regressão

- Fonte: https://www.kaggle.com/datasets/yasserh/advertising-sales-dataset
- Captura a receita de vendas gerada em relação aos custos de publicidade em múltiplos canais e determina quais canais contribuem mais para o aumento das vendas.
- Classificação com base nos parâmetros de:
  - orçamento para anúncio em TV
  - orçamento para anúncio em jornal
  - orçamento para anúncio em rádio
  - receita gerada pelas vendas
 
  ### Classificação multiclasse

  - Fonte: https://www.kaggle.com/datasets/uciml/iris
  - Ele contém dados sobre flores de íris de três espécies diferentes: Setosa, Versicolor e Virginica.
  - SepalLengthCm: O comprimento da sépala (a parte externa da flor) em centímetros.
  - Classificação com base nos parâmetros de:
    - SepalWidthCm: A largura da sépala em centímetros.
    - PetalLengthCm: O comprimento da pétala (a parte interna da flor) em centímetros.
    - PetalWidthCm: A largura da pétala em centímetros.
    - Species: A espécie da flor, que é a variável alvo para classificação. As três espécies são Setosa (0), Versicolor (1) e Virginica (2).
   
## Execução

Acessar o rede_neural.py na pasta, rodar e selecionar de 1 até 3 pra escolher o modelo desejado (binário, regressão, multiclasse).
