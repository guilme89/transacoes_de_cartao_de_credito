# Detecção de Anomalias em Transações com Python

Projeto de Machine Learning para identificar transações potencialmente fraudulentas em uma base real de cartão de crédito. O foco principal não é apenas obter alta acurácia, mas construir uma solução capaz de lidar com uma classe extremamente desbalanceada, priorizando métricas mais relevantes para fraude, como **recall**, **precision**, **F1-score**, **ROC-AUC** e **Average Precision**.

---

## Objetivo

Desenvolver um pipeline completo de detecção de anomalias em transações financeiras, passando pelas etapas de análise dos dados, engenharia de variáveis, treinamento de modelos, avaliação com métricas adequadas e geração de relatórios para interpretação dos resultados.

O projeto busca responder perguntas práticas como:

- Quantas fraudes o modelo consegue capturar?
- Quantos alertas gerados realmente são fraudes?
- Qual modelo apresenta melhor equilíbrio entre recall e precision?
- Como o ajuste de threshold altera a quantidade de fraudes capturadas e falsos positivos?
- Quais variáveis mais influenciam a decisão dos modelos?

---

## Dataset

O projeto utiliza o arquivo `creditcard.csv`, uma base real de transações de cartão de crédito.

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

## Por que acurácia não é suficiente?

Em problemas de fraude, a classe fraudulenta costuma ser muito rara. Neste dataset, menos de 0,2% das transações são fraude.

Isso significa que um modelo poderia acertar quase tudo apenas classificando todas as transações como normais. Mesmo assim, ele seria inútil para o negócio, porque não encontraria as fraudes.

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
│   └── tabela_metricas_markdown.md
└── src/
    └── deteccao_anomalias_transacoes.py
```

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

Uma explicação mais detalhada do código está em `docs/guia_codigo_passo_a_passo.md`.

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

O Random Forest balanceado apresentou o melhor equilíbrio entre captura de fraudes e qualidade dos alertas.

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

Além do threshold padrão de 0,50, o projeto testa diferentes pontos de corte para entender o impacto na operação. Nesta versão, foi adicionada uma análise específica do **threshold 0,30**, porque ele representa uma configuração mais agressiva para capturar fraudes.

Comparativo operacional do modelo `random_forest_balanced`:

| Threshold | Cenário | Precision | Recall | F1-score | Alertas | Fraudes capturadas | Falsos positivos | Fraudes não capturadas |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0,30 | Agressivo: maior captura de fraudes | 0,6276 | 0,8311 | 0,7151 | 196 | 123 | 73 | 25 |
| 0,50 | Padrão operacional inicial | 0,8298 | 0,7905 | 0,8097 | 141 | 117 | 24 | 31 |
| 0,55 | Maior F1-score observado | 0,8519 | 0,7770 | 0,8127 | 135 | 115 | 20 | 33 |

Leitura do threshold 0,30:

- captura **123 fraudes**, contra 117 no threshold 0,50;
- reduz as fraudes não capturadas de 31 para **25**;
- gera **196 alertas**, contra 141 no threshold 0,50;
- aumenta os falsos positivos de 24 para **73**;
- melhora o recall, mas reduz precision e F1-score.

Conclusão: o threshold 0,30 é útil quando a prioridade do negócio é **capturar mais fraudes**, mesmo aceitando uma fila maior de análise. Para uma operação mais equilibrada, o threshold 0,50 continua mais conservador e eficiente. O threshold 0,55 apresentou o maior F1-score observado, mas capturou menos fraudes que 0,30 e 0,50.

Interpretação geral:

- thresholds menores tendem a capturar mais fraudes, mas aumentam os falsos positivos;
- thresholds maiores reduzem falsos positivos, mas podem deixar mais fraudes passarem;
- a escolha ideal depende do custo financeiro da fraude e da capacidade operacional de revisar alertas.

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

### 4. Adicionar o dataset

Coloque o arquivo `creditcard.csv` dentro da pasta:

```text
data/creditcard.csv
```

### 5. Executar o treinamento

```bash
python src/deteccao_anomalias_transacoes.py
```

### 6. Executar com GridSearch opcional

```bash
python src/deteccao_anomalias_transacoes.py --run-grid-search
```

---

## Arquivos gerados

Após a execução, o projeto gera arquivos de apoio para análise:

| Caminho | Conteúdo |
|---|---|
| `outputs/metrics/dataset_profile.json` | Perfil geral do dataset. |
| `outputs/metrics/metricas_modelos.csv` | Comparação consolidada dos modelos. |
| `outputs/metrics/threshold_analysis.csv` | Resultado da análise de diferentes thresholds. |
| `outputs/metrics/threshold_comparison_030_050_055.csv` | Comparação direta entre thresholds 0,30, 0,50 e 0,55. |
| `outputs/figures/distribuicao_class.png` | Gráfico da distribuição da variável alvo. |
| `outputs/figures/curva_roc_*.png` | Curvas ROC dos modelos treinados. |
| `outputs/figures/curva_precision_recall_*.png` | Curvas Precision-Recall dos modelos treinados. |
| `outputs/figures/importancia_variaveis_*.png` | Gráficos de importância das variáveis. |
| `outputs/figures/comparativo_threshold_030_050_055.png` | Gráfico comparando impactos operacionais dos thresholds. |
| `models/melhor_modelo_fraude.pkl` | Modelo selecionado como melhor alternativa operacional. |
| `reports/relatorio_treinamento.md` | Relatório final do treinamento. |

---

## Principais aprendizados

- Em bases desbalanceadas, acurácia pode ser uma métrica enganosa.
- Recall é essencial para medir a capacidade de capturar fraudes reais.
- Precision é importante para controlar falsos positivos e evitar excesso de alertas.
- O melhor modelo não é necessariamente o que captura mais fraudes, mas o que equilibra captura e viabilidade operacional.
- A análise de threshold permite adaptar o modelo ao apetite de risco e à capacidade de análise da empresa.
- O threshold 0,30 aumenta a captura de fraudes, mas exige maior capacidade operacional para revisar falsos positivos.
- A importância das variáveis ajuda a tornar o modelo mais interpretável e confiável.

---

## Possíveis melhorias futuras

- Testar novas variáveis comportamentais baseadas em sequência de transações.
- Avaliar custos financeiros diferentes para falso positivo e falso negativo.
- Criar uma rotina de monitoramento para acompanhar queda de performance ao longo do tempo.
- Simular cenários operacionais com limite diário de alertas.
- Comparar novos modelos mantendo o mesmo conjunto de métricas.

---

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Joblib
