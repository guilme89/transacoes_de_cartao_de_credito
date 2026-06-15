# Relatório de Treinamento - Detecção de Anomalias em Transações

## 1. Dataset utilizado

- Linhas: 284,807
- Colunas: 31
- Transações normais: 284,315
- Transações fraudulentas: 492
- Percentual de fraude: 0.1727%
- Valores ausentes: 0

## 2. Melhor modelo operacional

O melhor modelo operacional foi **random_forest_balanced**, escolhido pela combinação de Average Precision, F1-score e Recall.

Métricas no threshold 0.50:

- Accuracy: 0.9994
- Precision: 0.8298
- Recall: 0.7905
- F1-score: 0.8097
- ROC-AUC: 0.9426
- Average Precision: 0.8124
- Falsos positivos: 24
- Falsos negativos: 31
- Fraudes capturadas: 117
- Alertas gerados: 141

## 3. Análise específica do threshold 0.30

O threshold 0.30 foi testado como uma configuração mais agressiva para captura de fraude.

Métricas no threshold 0.30:

- Accuracy: 0.9989
- Precision: 0.6276
- Recall: 0.8311
- F1-score: 0.7151
- Falsos positivos: 73
- Falsos negativos: 25
- Fraudes capturadas: 123
- Fraudes não capturadas: 25
- Alertas gerados: 196

Comparado ao threshold 0.50, o threshold 0.30 capturou **6 fraudes a mais** e reduziu as fraudes não capturadas de 31 para 25. O custo dessa melhora foi o aumento de falsos positivos de 24 para 73, elevando a quantidade de alertas de 141 para 196.

## 4. Comparativo de thresholds

| Threshold | Precision | Recall | F1-score | Alertas | Fraudes capturadas | Falsos positivos | Fraudes não capturadas | Leitura |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.30 | 0.6276 | 0.8311 | 0.7151 | 196 | 123 | 73 | 25 | Mais agressivo, captura mais fraudes |
| 0.50 | 0.8298 | 0.7905 | 0.8097 | 141 | 117 | 24 | 31 | Melhor equilíbrio operacional inicial |
| 0.55 | 0.8519 | 0.7770 | 0.8127 | 135 | 115 | 20 | 33 | Maior F1-score observado |

## 5. Melhor threshold observado para equilíbrio F1/Recall

- Threshold: 0.55
- Precision: 0.8519
- Recall: 0.7770
- F1-score: 0.8127
- Alertas gerados: 135
- Fraudes capturadas: 115
- Fraudes não capturadas: 33
- Falsos positivos: 20

## 6. Leitura profissional

A base é extremamente desbalanceada. Por isso, o projeto não usa acurácia como métrica principal. A análise combina Recall, Precision, F1-score, ROC-AUC e principalmente Precision-Recall.

Em uma operação real, o threshold deve ser definido de acordo com a capacidade da equipe de tratar alertas:

- threshold menor: mais fraudes capturadas, porém mais falsos positivos;
- threshold maior: menos falsos positivos, porém maior risco de deixar fraude passar.

O threshold 0.30 seria adequado para uma operação com maior apetite de investigação, em que o custo de uma fraude não capturada é muito superior ao custo de revisar alertas adicionais. Para um cenário mais equilibrado, o threshold 0.50 mantém melhor relação entre captura e eficiência operacional.

## 7. Arquivos gerados

- `outputs/metrics/dataset_profile.json`
- `outputs/metrics/metricas_modelos.csv`
- `outputs/metrics/threshold_analysis.csv`
- `outputs/metrics/threshold_comparison_030_050_055.csv`
- `outputs/metrics/importancia_variaveis_*.csv`
- `outputs/figures/distribuicao_class.png`
- `outputs/figures/curva_roc_*.png`
- `outputs/figures/curva_precision_recall_*.png`
- `outputs/figures/importancia_variaveis_*.png`
- `outputs/figures/comparativo_threshold_030_050_055.png`
- `models/melhor_modelo_fraude.pkl`
