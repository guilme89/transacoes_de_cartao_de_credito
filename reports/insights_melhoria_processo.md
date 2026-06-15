# Insights para Melhorar o Processo de Detecção de Anomalias

Este relatório traz uma leitura prática dos resultados do projeto de detecção de fraudes em transações de cartão de crédito.

---

## 1. Não usar acurácia como métrica principal

A base possui apenas **0,1727%** de transações fraudulentas.

Isso significa que um modelo poderia ter acurácia muito alta mesmo errando a maioria das fraudes. Por isso, a avaliação precisa olhar métricas específicas da classe rara.

Métricas recomendadas:

- Recall;
- Precision;
- F1-score;
- Average Precision;
- curva Precision-Recall;
- matriz de confusão.

---

## 2. Separar dois objetivos diferentes

Em fraude existem dois objetivos que competem entre si:

### Capturar mais fraudes

Aqui o foco é aumentar Recall.

Risco: gerar muitos falsos positivos.

### Gerar menos falsos alertas

Aqui o foco é aumentar Precision.

Risco: deixar fraudes reais passarem.

O modelo não deve ser avaliado isoladamente. Ele precisa ser avaliado dentro de um processo operacional.

---

## 3. Melhor modelo operacional encontrado

O melhor modelo operacional foi:

```text
random_forest_balanced
```

Resultado no threshold 0,50:

| Métrica | Valor |
|---|---:|
| Precision | 0,8298 |
| Recall | 0,7905 |
| F1-score | 0,8097 |
| ROC-AUC | 0,9426 |
| Average Precision | 0,8124 |
| Falsos positivos | 24 |
| Falsos negativos | 31 |
| Fraudes capturadas | 117 |

Leitura: o modelo capturou boa parte das fraudes com baixo volume de falsos positivos.

---

## 4. Cuidado com modelos que parecem bons pelo Recall

A Regressão Logística balanceada teve Recall de **0,8784**, capturando 130 fraudes.

Porém, gerou **1.871 falsos positivos**.

Isso pode ser inviável em uma operação real, pois a equipe teria que analisar muitos alertas incorretos.

Esse é um ponto importante para apresentar no projeto: **um modelo não é bom apenas porque captura mais fraudes; ele também precisa gerar alertas administráveis.**

---

## 5. Threshold é uma decisão de negócio

O threshold padrão de 0,50 apresentou bom equilíbrio no Random Forest, mas também foi testado o threshold 0,30 para simular uma política mais agressiva de detecção.

| Threshold | Precision | Recall | F1-score | Alertas | Fraudes capturadas | Falsos positivos | Fraudes não capturadas |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,30 | 0,6276 | 0,8311 | 0,7151 | 196 | 123 | 73 | 25 |
| 0,50 | 0,8298 | 0,7905 | 0,8097 | 141 | 117 | 24 | 31 |
| 0,55 | 0,8519 | 0,7770 | 0,8127 | 135 | 115 | 20 | 33 |

Leitura prática:

- o threshold 0,30 capturou 6 fraudes a mais do que o threshold 0,50;
- ao mesmo tempo, aumentou os falsos positivos de 24 para 73;
- esse ajuste pode fazer sentido quando o custo de deixar uma fraude passar é muito alto;
- se a equipe de análise tiver capacidade limitada, o threshold 0,50 ou 0,55 tende a ser mais sustentável.

Portanto, o threshold não deve ser escolhido apenas por métrica estatística. Ele precisa refletir o custo da fraude, o custo da revisão manual e a capacidade operacional da empresa.

---

## 6. Usar fila de priorização em vez de decisão binária simples

Em vez de apenas classificar como fraude ou não fraude, uma melhoria prática seria criar faixas de risco:

| Probabilidade estimada | Ação sugerida |
|---:|---|
| Acima de 0,80 | Revisão prioritária ou bloqueio preventivo |
| Entre 0,50 e 0,80 | Fila de análise antifraude |
| Entre 0,30 e 0,50 | Monitoramento ou confirmação adicional |
| Abaixo de 0,30 | Aprovação automática, salvo regra de negócio |

Essa abordagem transforma o modelo em um processo mais inteligente.

---

## 7. Feature Engineering deve continuar evoluindo

Neste projeto foram usadas variáveis simples alinhadas ao conteúdo da aula:

- `Amount_log`;
- `Amount_scaled`;
- `Time_scaled`;
- `Hour`;
- `Amount_to_median_ratio`;
- `High_amount_flag`.

Em uma aplicação real, novas variáveis poderiam melhorar a detecção:

- quantidade de transações recentes por cartão;
- distância entre transações em curto intervalo;
- valor comparado com histórico do cliente;
- horário incomum para aquele cliente;
- mudança brusca de padrão de compra.

Essas melhorias exigiriam histórico por cliente/cartão, que não está disponível no dataset anonimizado.

---

## 8. Monitorar o modelo depois da implantação

Fraude muda com o tempo. Um modelo bom hoje pode perder desempenho depois.

Indicadores para acompanhar:

- Recall mensal;
- Precision mensal;
- quantidade de alertas;
- falsos positivos por período;
- fraudes não capturadas;
- mudança na distribuição de valores;
- mudança na distribuição de horários;
- queda na curva Precision-Recall.

---

## 9. Explicabilidade aumenta confiança

A importância das variáveis ajuda a entender quais atributos mais influenciaram o modelo.

No projeto, foram gerados arquivos de importância para Random Forest e XGBoost.

Essa explicabilidade ajuda em três pontos:

1. defesa técnica do modelo;
2. melhoria do processo de análise;
3. comunicação com áreas de negócio.

---

## 10. Próximos passos recomendados

1. Testar novos thresholds conforme capacidade operacional de análise, especialmente 0,30, 0,50 e 0,55.
2. Acompanhar Precision e Recall em períodos diferentes.
3. Criar faixas de risco em vez de resposta binária simples.
4. Explorar novas variáveis de comportamento, se houver dados por cliente/cartão.
5. Usar a curva Precision-Recall como principal referência de comparação.
6. Documentar claramente falsos positivos e falsos negativos.
7. Reavaliar o modelo periodicamente para evitar perda de performance.
