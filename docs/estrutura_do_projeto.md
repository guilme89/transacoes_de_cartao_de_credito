# Estrutura do Projeto

```text
deteccao-anomalias-transacoes-python/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── reports/
└── src/
```

## `README.md`

Documento principal do GitHub. Explica objetivo, dataset, técnicas usadas, resultados do treinamento e como executar o projeto.

## `requirements.txt`

Lista de bibliotecas necessárias para rodar o projeto.

## `.gitignore`

Evita subir arquivos pesados, temporários ou sensíveis para o GitHub, como `data/*.csv`, cache do Python e ambientes virtuais.

## `data/`

Pasta onde o usuário deve colocar o arquivo `creditcard.csv`.

O CSV real é pesado e, por boa prática, não precisa ser versionado no GitHub. O projeto já inclui as métricas e os resultados gerados a partir dele.

## `docs/`

Pasta com documentação complementar:

- `descricao_entrega_bootcamp.md`: texto pronto para colar na entrega do Bootcamp;
- `estrutura_do_projeto.md`: este arquivo;
- `guia_codigo_passo_a_passo.md`: explicação detalhada do código.

## `models/`

Pasta com os modelos treinados salvos em `.pkl`.

O principal arquivo é:

```text
models/melhor_modelo_fraude.pkl
```

## `notebooks/`

Contém o notebook principal com a análise em formato didático.

## `outputs/`

Contém os resultados do treinamento:

- gráficos em `outputs/figures/`;
- métricas em `outputs/metrics/`.

## `reports/`

Contém relatórios em Markdown com leitura executiva e técnica dos resultados.

## `src/`

Contém o script Python principal:

```text
src/deteccao_anomalias_transacoes.py
```

Esse script executa o pipeline completo de ponta a ponta.
