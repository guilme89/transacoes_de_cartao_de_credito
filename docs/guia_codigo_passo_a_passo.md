# Guia do Código - Passo a Passo

Este documento explica o papel de cada bloco do arquivo `src/deteccao_anomalias_transacoes.py`.

A ideia é que o projeto não seja apenas um código que roda, mas também um material de estudo para apresentar no GitHub.

---

## 1. Imports

O código começa importando bibliotecas de manipulação de dados, machine learning, métricas, gráficos e salvamento de arquivos.

Principais bibliotecas:

- `pandas`: leitura e manipulação do dataset;
- `numpy`: cálculos numéricos e criação de variáveis;
- `matplotlib`: geração dos gráficos;
- `sklearn`: treino, separação treino/teste, padronização, métricas e modelos;
- `joblib`: salvamento do modelo treinado;
- `xgboost`: modelo avançado, quando disponível.

---

## 2. Constantes do projeto

```python
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42
```

Essas constantes centralizam os caminhos do projeto. Isso evita espalhar nomes de pastas pelo código e facilita manutenção.

`RANDOM_STATE = 42` garante reprodutibilidade: quando o código for executado novamente, a separação dos dados e os modelos tendem a gerar resultados consistentes.

---

## 3. `ensure_directories()`

Cria automaticamente as pastas onde serão salvos:

- gráficos;
- métricas;
- modelos treinados;
- relatórios.

Sem essa função, o código poderia falhar ao tentar salvar arquivos em pastas ainda inexistentes.

---

## 4. `load_dataset()`

Responsável por carregar o dataset.

Fluxo:

1. Procura o arquivo `data/creditcard.csv`.
2. Se encontrar, carrega o arquivo local.
3. Se não encontrar, tenta carregar pela URL pública usada na aula.
4. Se não conseguir, orienta o usuário a baixar o arquivo manualmente.

Isso deixa o projeto mais flexível para rodar tanto localmente quanto em ambientes como Google Colab.

---

## 5. `validate_dataset()`

Antes de treinar qualquer modelo, o código valida se as colunas essenciais existem:

- `Time`;
- `Amount`;
- `Class`.

Também verifica se a coluna `Class` contém apenas:

- `0`: transação normal;
- `1`: transação fraudulenta.

Essa etapa evita erros silenciosos e garante que a base está no formato esperado.

---

## 6. `basic_analysis()`

Gera o perfil inicial da base:

- número de linhas;
- número de colunas;
- total de valores ausentes;
- quantidade de transações normais;
- quantidade de transações fraudulentas;
- percentual de fraude;
- estatísticas básicas de `Amount`.

O resultado é salvo em:

```text
outputs/metrics/dataset_profile.json
```

Essa análise mostra o ponto central do problema: fraude é uma classe extremamente rara.

---

## 7. `plot_class_distribution()`

Gera um gráfico de barras comparando:

- transações normais;
- transações fraudulentas.

O gráfico é salvo em:

```text
outputs/figures/distribuicao_class.png
```

Esse gráfico deixa visualmente claro por que a acurácia sozinha não é uma métrica suficiente.

---

## 8. `create_features()`

Cria variáveis derivadas para ajudar o modelo a enxergar melhor os padrões.

Variáveis criadas:

### `Amount_log`

```python
data["Amount_log"] = np.log1p(data["Amount"])
```

Aplica transformação logarítmica ao valor da transação. Isso reduz o impacto de valores muito altos.

### `Amount_to_median_ratio`

```python
data["Amount_to_median_ratio"] = data["Amount"] / (amount_median + 1e-9)
```

Compara o valor da transação com a mediana da base.

### `High_amount_flag`

```python
data["High_amount_flag"] = (data["Amount"] > amount_p99).astype(int)
```

Cria uma marcação para transações acima do percentil 99 de valor.

### `Hour`

```python
data["Hour"] = ((data["Time"] // 3600) % 24).astype(int)
```

Transforma o tempo em uma aproximação da hora do dia.

### `Amount_scaled` e `Time_scaled`

```python
data[["Amount_scaled", "Time_scaled"]] = scaler.fit_transform(data[["Amount", "Time"]])
```

Padroniza `Amount` e `Time`, deixando essas variáveis com escala mais adequada para modelos como Regressão Logística.

---

## 9. `split_data()`

Separa a base em:

- `X_train`: variáveis de treino;
- `X_test`: variáveis de teste;
- `y_train`: alvo de treino;
- `y_test`: alvo de teste.

O parâmetro mais importante é:

```python
stratify=y
```

Ele preserva a proporção de fraudes no treino e no teste. Isso é essencial em bases desbalanceadas.

---

## 10. `undersample_training_data()`

Cria uma versão balanceada apenas da base de treino.

O que faz:

1. Separa fraudes da base de treino.
2. Sorteia a mesma quantidade de transações normais.
3. Junta fraude e normal em uma base balanceada.
4. Embaralha os dados.

A base de teste não é balanceada. Isso é correto, porque a avaliação precisa representar o mundo real.

---

## 11. `build_models()`

Cria os modelos usados no projeto.

Modelos:

### `logistic_baseline`

Modelo simples de referência.

### `logistic_balanced`

Regressão Logística com:

```python
class_weight="balanced"
```

Isso força o modelo a dar mais peso para a classe rara.

### `random_forest_balanced`

Modelo baseado em árvores, usando:

```python
class_weight="balanced_subsample"
```

Ajuda o modelo a lidar com desbalanceamento em cada amostra usada pelas árvores.

### `xgboost_balanced`

Modelo avançado, usando:

```python
scale_pos_weight = quantidade_normais / quantidade_fraudes
```

Esse peso ajuda o XGBoost a considerar a raridade da classe fraudulenta.

---

## 12. `predict_probabilities()`

Retorna a probabilidade de fraude para cada transação.

Essa etapa é importante porque o modelo não precisa apenas dizer `0` ou `1`. Ele pode gerar uma probabilidade, e depois o projeto decide qual threshold usar.

---

## 13. `evaluate_model()`

Avalia o modelo usando:

- accuracy;
- precision;
- recall;
- F1-score;
- ROC-AUC;
- Average Precision;
- matriz de confusão;
- quantidade de alertas gerados.

A matriz de confusão é lida assim:

```text
[[TN, FP],
 [FN, TP]]
```

Onde:

- `TN`: normal classificada como normal;
- `FP`: normal classificada como fraude;
- `FN`: fraude classificada como normal;
- `TP`: fraude classificada como fraude.

---

## 14. `save_curves()`

Salva dois gráficos para cada modelo:

### Curva ROC

Mostra a relação entre taxa de verdadeiro positivo e taxa de falso positivo.

### Curva Precision-Recall

Mais útil em classe rara, porque mostra o equilíbrio entre capturar fraudes e controlar falsos alertas.

Os gráficos são salvos em:

```text
outputs/figures/
```

---

## 15. `threshold_analysis()`

Testa thresholds entre 0,05 e 0,95 e salva a tabela em:

```text
outputs/metrics/threshold_analysis.csv
```

Exemplo:

- threshold 0,30: configuração mais agressiva, chama mais transações de fraude e aumenta recall;
- threshold 0,50: ponto de corte padrão, usado como referência operacional inicial;
- threshold 0,70: configuração mais conservadora, reduz alertas, mas pode deixar mais fraudes passarem.

No resultado treinado do `random_forest_balanced`, o threshold 0,30 capturou 123 fraudes, contra 117 no threshold 0,50. O custo foi aumentar os falsos positivos de 24 para 73.

Essa análise é muito importante para fraude, porque o ponto de corte padrão `0.50` nem sempre é o melhor para o negócio. A escolha do threshold precisa considerar o custo de uma fraude não capturada e a capacidade de revisar alertas.

---

## 16. `save_feature_importance()`

Para modelos baseados em árvores, salva a importância das variáveis.

Isso ajuda a responder:

> Quais variáveis mais influenciaram a decisão do modelo?

Arquivos gerados:

```text
outputs/metrics/importancia_variaveis_random_forest_balanced.csv
outputs/metrics/importancia_variaveis_xgboost_balanced.csv
outputs/figures/importancia_variaveis_random_forest_balanced.png
outputs/figures/importancia_variaveis_xgboost_balanced.png
```

---

## 17. `run_optional_grid_search()`

Executa uma busca simples de hiperparâmetros com XGBoost.

Ela só roda se o usuário executar:

```bash
python src/deteccao_anomalias_transacoes.py --run-grid-search
```

Foi deixada como opcional porque GridSearch pode demorar mais.

---

## 18. `choose_best_model()`

Escolhe o melhor modelo operacional usando:

1. Average Precision;
2. F1-score;
3. Recall.

Esse critério é mais realista do que escolher apenas pelo maior recall, porque um modelo com recall muito alto pode gerar falsos positivos demais.

---

## 19. `write_training_report()`

Gera o relatório final com:

- resumo do dataset;
- melhor modelo;
- métricas principais;
- análise específica do threshold 0,30;
- comparação entre thresholds 0,30, 0,50 e 0,55;
- melhor threshold observado;
- leitura profissional dos resultados;
- lista de arquivos gerados.

Arquivo final:

```text
reports/relatorio_treinamento.md
```

---

## 20. `main()`

É a função que executa tudo em ordem:

1. cria diretórios;
2. carrega dados;
3. valida dados;
4. analisa dataset;
5. cria variáveis;
6. separa treino e teste;
7. treina modelos;
8. avalia resultados;
9. salva gráficos e métricas;
10. escolhe melhor modelo;
11. analisa thresholds;
12. gera comparação entre thresholds 0,30, 0,50 e 0,55;
13. salva modelo final;
14. gera relatório.

Esse é o fluxo completo do projeto.
