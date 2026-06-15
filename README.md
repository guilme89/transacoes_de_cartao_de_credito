# Detecção de Anomalias em Transações com Python

Projeto de Machine Learning para identificar transações potencialmente fraudulentas em uma base real de cartão de crédito. O foco do projeto não é apenas obter alta acurácia, mas construir uma solução capaz de lidar com uma classe extremamente desbalanceada, priorizando métricas mais adequadas para fraude, como **recall**, **precision**, **F1-score**, **ROC-AUC** e **Average Precision**.

---

## Objetivo do projeto

Desenvolver um pipeline completo de detecção de anomalias em transações financeiras, passando pelas seguintes etapas:

- análise exploratória dos dados;
- tratamento e validação da base;
- criação de variáveis derivadas;
- preparação dos dados para Machine Learning;
- treinamento de modelos de classificação;
- avaliação com métricas adequadas para dados desbalanceados;
- análise de threshold para tomada de decisão operacional;
- geração de gráficos, métricas, modelos treinados e relatórios.

O projeto busca responder perguntas práticas de negócio:

- quantas fraudes o modelo consegue capturar?
- quantos alertas gerados realmente são fraudes?
- qual modelo apresenta melhor equilíbrio entre recall e precision?
- o que muda ao reduzir o threshold de decisão?
- qual configuração é mais adequada para uma operação real de prevenção a fraudes?

---

## Dataset

O projeto utiliza o arquivo `creditcard.csv`, uma base real de transações de cartão de crédito disponível no Kaggle.

**Link oficial para download:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Após baixar o dataset, extraia o arquivo e coloque o `creditcard.csv` dentro da pasta `data/`, conforme explicado abaixo.

A variável alvo é a coluna `Class`:

| Valor | Significado |
|---:|---|
| `0` | Transação normal |
| `1` | Transação fraudulenta |

Resumo da base utilizada no treinamento:

| Indicador | Valor |
|---|---:|
| Total de transações | 284.807 |
| Total de colunas | 31 |
| Transações normais | 284.315 |
| Transações fraudulentas | 492 |
| Percentual de fraude | 0,1727% |
| Valores ausentes | 0 |

As variáveis `V1` até `V28` são anonimizadas. As colunas `Time` e `Amount` foram usadas para criação de novas variáveis.

---

## Importante sobre a pasta `data/`

A pasta `data/` existe para armazenar o dataset utilizado no treinamento.

No repositório GitHub, o arquivo `creditcard.csv` **não está incluído** por boa prática de versionamento, pois é um arquivo pesado. Por isso, dentro da pasta `data/` existe apenas um arquivo `README.md`, que serve como orientação para quem for executar o projeto.

Para baixar a base, acesse o link oficial do Kaggle:

[Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Depois do download, extraia o arquivo compactado e coloque o `creditcard.csv` exatamente neste caminho:

```text
deteccao-anomalias-transacoes-python/
└── data/
    └── creditcard.csv
```

Ou seja, depois de baixar o dataset, coloque o arquivo `creditcard.csv` dentro da pasta `data/`.

O arquivo `.gitignore` foi configurado para não enviar arquivos `.csv` da pasta `data/` para o GitHub:

```text
/data/*.csv
/data/*.zip
```

Isso mantém o repositório mais leve e profissional. O código, o notebook, os relatórios, os gráficos e os modelos ficam versionados; o dataset pesado fica apenas no ambiente local de quem for executar o projeto.

---

## Por que acurácia não é suficiente?

Em problemas de fraude, a classe fraudulenta costuma ser muito rara. Neste dataset, menos de 0,2% das transações são fraude.

Um modelo poderia atingir acurácia muito alta classificando quase todas as transações como normais. Mesmo assim, ele seria pouco útil para o negócio, porque deixaria de capturar as fraudes.

Por isso, o projeto avalia os modelos com métricas mais adequadas:

| Métrica | Interpretação |
|---|---|
| Recall | Mede quantas fraudes reais foram capturadas. |
| Precision | Mede quantos alertas de fraude estavam corretos. |
| F1-score | Equilibra precision e recall. |
| ROC-AUC | Mede a capacidade geral de separação entre as classes. |
| Average Precision | Resume a curva Precision-Recall, muito útil em bases desbalanceadas. |

---

## Estrutura do projeto

```text
deteccao-anomalias-transacoes-python/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   ├── estrutura_do_projeto.md
│   └── guia_codigo_passo_a_passo.md
│    
├── models/
│   ├── logistic_baseline.pkl
│   ├── logistic_balanced.pkl
│   ├── logistic_undersampling.pkl
│   ├── random_forest_balanced.pkl
│   ├── xgboost_balanced.pkl
│   └── melhor_modelo_fraude.pkl
├── notebooks/
│   └── 01_deteccao_anomalias_transacoes.ipynb
├── outputs/
│   ├── figures/
│   └── metrics/
├── reports/
│   ├── insights_melhoria_processo.md
│   ├── relatorio_treinamento.md
│   ├── tabela_metricas_markdown.md
│   └── tabela_thresholds_markdown.md
└── src/
    └── deteccao_anomalias_transacoes.py
```

### O que cada pasta contém

| Pasta/arquivo | Função no projeto |
|---|---|
| `README.md` | Documentação principal do repositório. |
| `requirements.txt` | Lista de bibliotecas necessárias para executar o projeto. |
| `.gitignore` | Define arquivos que não devem ser enviados ao GitHub, como datasets pesados. |
| `data/` | Pasta onde o usuário deve colocar o arquivo `creditcard.csv`. |
| `docs/` | Documentações auxiliares do projeto e explicação passo a passo do código. |
| `models/` | Modelos treinados salvos em `.pkl`. |
| `notebooks/` | Notebook com a análise didática do projeto. |
| `outputs/figures/` | Gráficos gerados pelo treinamento. |
| `outputs/metrics/` | Métricas em arquivos `.csv` e `.json`. |
| `reports/` | Relatórios finais em Markdown. |
| `src/` | Código Python principal do pipeline. |

---

## Principais etapas do código

| Etapa | O que faz |
|---|---|
| Criação de diretórios | Garante que as pastas de saída, gráficos, métricas e modelos existam antes da execução. |
| Carga do dataset | Lê o arquivo `creditcard.csv` a partir da pasta `data/`. |
| Validação da base | Confere se as colunas essenciais existem e se a variável `Class` possui os valores esperados. |
| Análise inicial | Calcula volume de registros, quantidade de fraudes, percentual de fraude e valores ausentes. |
| Visualização da classe alvo | Gera gráfico com a distribuição entre transações normais e fraudulentas. |
| Feature engineering | Cria novas variáveis a partir de `Amount` e `Time` para enriquecer os padrões disponíveis ao modelo. |
| Separação treino/teste | Divide a base mantendo a proporção original de fraudes por meio de estratificação. |
| Balanceamento | Aplica estratégias como `class_weight='balanced'` e undersampling no conjunto de treino. |
| Treinamento dos modelos | Treina Regressão Logística, Random Forest e XGBoost. |
| Avaliação | Calcula métricas de classificação, matriz de confusão, ROC-AUC e Average Precision. |
| Curvas de desempenho | Salva curvas ROC e Precision-Recall para comparação visual dos modelos. |
| Análise de threshold | Testa diferentes pontos de corte para avaliar o impacto nos alertas de fraude. |
| Importância das variáveis | Identifica quais atributos tiveram maior peso nos modelos baseados em árvores. |
| Relatório final | Gera arquivos com métricas, gráficos, modelo treinado e relatório em Markdown. |

Uma explicação mais detalhada está no arquivo `docs/guia_codigo_passo_a_passo.md`.

---

## Feature Engineering

Foram criadas variáveis para melhorar a leitura dos padrões matemáticos pelo modelo:

| Variável | Objetivo |
|---|---|
| `Amount_log` | Reduz o impacto de valores extremos na coluna `Amount`. |
| `Amount_scaled` | Padroniza o valor da transação para média 0 e desvio padrão 1. |
| `Time_scaled` | Padroniza a variável de tempo. |
| `Hour` | Aproxima a hora da transação a partir da coluna `Time`. |
| `Amount_to_median_ratio` | Compara o valor da transação com a mediana da base. |
| `High_amount_flag` | Marca transações com valor acima do percentil 99. |

Essas variáveis ajudam o modelo a capturar padrões que não aparecem de forma tão direta nos dados brutos.

---

## Modelos treinados

Foram comparados modelos simples e avançados:

| Modelo | Finalidade |
|---|---|
| Logistic Regression baseline | Modelo inicial de referência. |
| Logistic Regression balanceada | Regressão Logística com ajuste para classe desbalanceada. |
| Logistic Regression com undersampling | Modelo treinado em base balanceada por redução da classe majoritária. |
| Random Forest balanceado | Modelo baseado em árvores com ajuste de peso das classes. |
| XGBoost balanceado | Modelo avançado com ajuste para desbalanceamento. |
| XGBoost com GridSearch | Versão com busca de hiperparâmetros. |

---

## Resultados do treinamento

Métricas obtidas na base de teste:

| Modelo | Precision | Recall | F1-score | ROC-AUC | Average Precision | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression balanceada | 0,0650 | 0,8784 | 0,1210 | 0,9711 | 0,7047 | 1.871 | 18 | 130 |
| Logistic Regression baseline | 0,8585 | 0,6149 | 0,7165 | 0,9584 | 0,7123 | 15 | 57 | 91 |
| Logistic Regression com undersampling | 0,0725 | 0,8716 | 0,1339 | 0,9726 | 0,4130 | 1.650 | 19 | 129 |
| Random Forest balanceado | 0,8298 | 0,7905 | 0,8097 | 0,9426 | 0,8124 | 24 | 31 | 117 |
| XGBoost balanceado | 0,0995 | 0,8649 | 0,1785 | 0,9660 | 0,6620 | 1.158 | 20 | 128 |
| XGBoost com GridSearch | 0,2687 | 0,8514 | 0,4084 | 0,9716 | 0,7278 | 343 | 22 | 126 |

Legenda:

| Sigla | Significado |
|---|---|
| FP | Falso positivo: transação normal marcada como fraude. |
| FN | Falso negativo: fraude real não capturada. |
| TP | Verdadeiro positivo: fraude real capturada. |

---

## Melhor modelo operacional

O modelo escolhido como melhor alternativa operacional foi o **Random Forest balanceado**.

Apesar de outros modelos apresentarem recall maior, eles geraram um volume muito alto de falsos positivos. Em um cenário real, isso pode sobrecarregar a equipe de análise e prejudicar a experiência dos clientes.

O Random Forest balanceado apresentou melhor equilíbrio entre captura de fraudes e qualidade dos alertas.

| Métrica | Valor |
|---|---:|
| Precision | 0,8298 |
| Recall | 0,7905 |
| F1-score | 0,8097 |
| ROC-AUC | 0,9426 |
| Average Precision | 0,8124 |
| Fraudes capturadas | 117 |
| Fraudes não capturadas | 31 |
| Falsos positivos | 24 |

---

## Análise de threshold

O threshold é o ponto de corte usado para transformar a probabilidade prevista pelo modelo em uma classificação final.

Exemplo: com threshold `0,50`, uma transação com probabilidade de fraude maior ou igual a 50% é classificada como fraude. Ao reduzir o threshold para `0,30`, o modelo passa a ser mais sensível, classificando mais transações como suspeitas.

Comparação dos thresholds no modelo **Random Forest balanceado**:

| Threshold | Precision | Recall | F1-score | Alertas | Fraudes capturadas | Falsos positivos | Fraudes não capturadas |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,30 | 0,6276 | 0,8311 | 0,7151 | 196 | 123 | 73 | 25 |
| 0,50 | 0,8298 | 0,7905 | 0,8097 | 141 | 117 | 24 | 31 |
| 0,55 | 0,8519 | 0,7770 | 0,8127 | 135 | 115 | 20 | 33 |

Interpretação:

- o threshold `0,30` capturou mais fraudes, encontrando 123 das 148 fraudes da base de teste;
- em comparação com o threshold `0,50`, o threshold `0,30` capturou 6 fraudes a mais;
- porém, os falsos positivos subiram de 24 para 73;
- o threshold `0,50` manteve melhor equilíbrio operacional;
- o threshold `0,55` reduziu falsos positivos, mas deixou mais fraudes passarem.

Conclusão operacional:

- se a empresa deseja ser mais agressiva na prevenção a fraudes, o threshold `0,30` pode ser interessante;
- se a empresa precisa equilibrar captura de fraude com menor volume de alertas, o threshold `0,50` continua sendo mais equilibrado;
- a escolha ideal depende do custo da fraude, da capacidade da equipe de análise e do impacto de bloquear transações legítimas.

---

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd deteccao-anomalias-transacoes-python
```

### 2. Criar e ativar o ambiente virtual

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/Mac:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Baixar e adicionar o dataset

Baixe o dataset no Kaggle:

[Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Depois do download, extraia o arquivo e coloque o `creditcard.csv` dentro da pasta `data/`.

O caminho final precisa ficar assim:

```text
data/creditcard.csv
```

Sem esse arquivo, o treinamento não será executado, porque o código procura a base neste caminho.

### 5. Executar o treinamento completo

```bash
python src/deteccao_anomalias_transacoes.py
```

### 6. Rodar GridSearch opcional

```bash
python src/deteccao_anomalias_transacoes.py --run-grid-search
```

Nesta versão do projeto, o GridSearch também foi executado. O melhor conjunto encontrado foi:

```text
{'max_depth': 3, 'n_estimators': 40}
```

---

## Arquivos de saída

Após a execução, o projeto gera arquivos como:

```text
outputs/metrics/dataset_profile.json
outputs/metrics/metricas_modelos.csv
outputs/metrics/threshold_analysis.csv
outputs/metrics/threshold_comparison_030_050_055.csv
outputs/metrics/importancia_variaveis_random_forest_balanced.csv
outputs/metrics/importancia_variaveis_xgboost_balanced.csv
outputs/figures/distribuicao_class.png
outputs/figures/curva_roc_*.png
outputs/figures/curva_precision_recall_*.png
outputs/figures/importancia_variaveis_*.png
outputs/figures/comparativo_threshold_030_050_055.png
models/melhor_modelo_fraude.pkl
reports/relatorio_treinamento.md
reports/insights_melhoria_processo.md
```

---

## Principais aprendizados

- Em bases extremamente desbalanceadas, acurácia isolada pode mascarar modelos ruins.
- Recall é essencial para medir a capacidade de captura de fraudes.
- Precision é importante para controlar o volume de falsos alertas.
- O ajuste de threshold muda diretamente a estratégia operacional do modelo.
- Modelos com maior recall nem sempre são os melhores para produção, pois podem gerar falsos positivos demais.
- O Random Forest balanceado apresentou o melhor equilíbrio geral neste projeto.

---

## Possíveis melhorias futuras

Este projeto já apresenta um pipeline completo de Machine Learning para detecção de fraudes. Mesmo assim, em um ambiente real, o processo poderia evoluir com novos testes, novas variáveis e melhor acompanhamento operacional.

| Melhoria | Como ajudaria o projeto |
|---|---|
| Testar novos thresholds | Avaliar pontos de corte como `0,20`, `0,25`, `0,35` e `0,40`, buscando o melhor equilíbrio entre fraudes capturadas e falsos positivos. |
| Definir threshold por custo de negócio | Escolher o ponto de corte considerando o custo financeiro de uma fraude não capturada versus o custo operacional de investigar um falso positivo. |
| Criar novas variáveis temporais | Explorar padrões baseados em horário, sequência de transações e comportamento ao longo do tempo. |
| Aprofundar a análise dos falsos negativos | Entender as fraudes que o modelo deixou passar e buscar características comuns entre elas. |
| Aprofundar a análise dos falsos positivos | Identificar por que transações normais foram marcadas como fraude e reduzir alertas desnecessários. |
| Expandir o GridSearch | Testar mais combinações de hiperparâmetros para Random Forest e XGBoost. |
| Aplicar validação cruzada estratificada | Reduzir a dependência de uma única divisão treino/teste e obter métricas mais estáveis. |
| Calibrar probabilidades | Melhorar a interpretação das probabilidades previstas antes da escolha do threshold. |
| Reforçar explicabilidade | Usar importância de variáveis e SHAP para explicar melhor por que uma transação foi considerada suspeita. |
| Criar visão operacional dos alertas | Organizar os resultados em uma fila de investigação, priorizando transações com maior probabilidade de fraude. |

Essas melhorias não mudam o objetivo central do projeto. Elas representam próximos passos para aproximar a solução de um cenário real de prevenção a fraudes, onde o modelo precisa equilibrar captura de risco, volume de alertas e capacidade operacional da equipe de análise.

---

## Observação final

Este repositório foi estruturado para demonstrar um fluxo completo de Machine Learning aplicado à detecção de fraudes: da preparação dos dados até a análise operacional dos resultados.

Para executar o projeto, lembre-se de colocar o arquivo `creditcard.csv` dentro da pasta `data/`.
