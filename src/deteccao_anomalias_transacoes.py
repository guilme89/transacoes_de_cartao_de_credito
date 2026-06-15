"""
Projeto: Detecção de Anomalias em Transações com Python
Autor: Guilherme Ferreira Fontoura

Este arquivo é o pipeline executável do projeto.
Ele foi escrito para ser lido como material de estudo: cada função possui docstring
explicando seu papel e os principais blocos possuem comentários objetivos.

Objetivo do pipeline:
1. Carregar o dataset real creditcard.csv.
2. Fazer uma análise inicial da base e do desbalanceamento da classe Class.
3. Criar variáveis derivadas simples com Amount e Time.
4. Separar treino e teste usando stratify para preservar a proporção de fraudes.
5. Treinar modelos supervisionados vistos no desafio/aula.
6. Avaliar os modelos com métricas adequadas para classe rara.
7. Gerar gráficos, tabelas, matriz de confusão e modelo treinado.

Observação de negócio:
Em detecção de fraude, acurácia isolada é perigosa. Como fraude é rara, o modelo
pode ter acurácia alta mesmo ignorando fraudes. Por isso, o projeto olha recall,
precision, F1-score, ROC-AUC e Average Precision.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# URL pública utilizada na aula. O projeto dá preferência ao arquivo local.
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# Diretórios principais do projeto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_METRICS = PROJECT_ROOT / "outputs" / "metrics"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def ensure_directories() -> None:
    """Cria as pastas de saída caso elas ainda não existam."""
    for directory in [OUTPUT_FIGURES, OUTPUT_METRICS, MODEL_DIR, PROJECT_ROOT / "reports"]:
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """
    Carrega o dataset de transações.

    Primeiro procura por data/creditcard.csv. Se não encontrar, tenta carregar
    pela URL pública usada na aula. Isso deixa o projeto reproduzível tanto em
    ambiente local quanto no Colab.
    """
    if DATA_PATH.exists():
        print(f"Carregando dataset local: {DATA_PATH}")
        return pd.read_csv(DATA_PATH)

    print("Dataset local não encontrado. Tentando carregar pela URL pública da aula...")
    try:
        return pd.read_csv(DATA_URL)
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível carregar o dataset. Baixe creditcard.csv e salve em data/creditcard.csv."
        ) from exc


def validate_dataset(df: pd.DataFrame) -> None:
    """Confere se as colunas essenciais existem antes de iniciar o treinamento."""
    required_columns = {"Time", "Amount", "Class"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    valid_classes = set(df["Class"].dropna().unique())
    if not valid_classes.issubset({0, 1}):
        raise ValueError("A coluna Class deve conter apenas 0 para normal e 1 para fraude.")


def basic_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Gera um resumo inicial da base.

    Essa etapa responde perguntas simples:
    - Quantas linhas e colunas existem?
    - Existem valores ausentes?
    - Qual a proporção de fraude?
    """
    class_counts = df["Class"].value_counts().sort_index()
    class_percent = df["Class"].value_counts(normalize=True).sort_index()

    profile = {
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "valores_ausentes_total": int(df.isna().sum().sum()),
        "transacoes_normais": int(class_counts.get(0, 0)),
        "transacoes_fraudulentas": int(class_counts.get(1, 0)),
        "percentual_normal": float(class_percent.get(0, 0)),
        "percentual_fraude": float(class_percent.get(1, 0)),
        "amount_min": float(df["Amount"].min()),
        "amount_mediana": float(df["Amount"].median()),
        "amount_media": float(df["Amount"].mean()),
        "amount_max": float(df["Amount"].max()),
    }

    print("\n================ VISÃO GERAL DO DATASET ================")
    print(json.dumps(profile, indent=4, ensure_ascii=False))

    with open(OUTPUT_METRICS / "dataset_profile.json", "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=4, ensure_ascii=False)

    return profile


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Salva um gráfico simples mostrando o desbalanceamento entre normal e fraude."""
    counts = df["Class"].value_counts().sort_index()
    labels = ["Normal (0)", "Fraude (1)"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts.values)
    plt.title("Distribuição da variável Class")
    plt.ylabel("Quantidade de transações")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "distribuicao_class.png", dpi=150)
    plt.close()


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria variáveis derivadas com base em Amount e Time.

    Por que isso ajuda?
    O modelo não entende contexto humano diretamente. Ele aprende padrões numéricos.
    Então criamos representações que facilitem a leitura dos dados:
    - Amount_log reduz a assimetria de valores muito altos.
    - Amount_to_median_ratio compara a transação com o valor típico da base.
    - High_amount_flag marca transações acima do percentil 99.
    - Hour aproxima a hora da transação a partir da coluna Time.
    - Amount_scaled e Time_scaled colocam variáveis em escala comparável.
    """
    data = df.copy()

    amount_median = data["Amount"].median()
    amount_p99 = data["Amount"].quantile(0.99)

    data["Amount_log"] = np.log1p(data["Amount"])
    data["Amount_to_median_ratio"] = data["Amount"] / (amount_median + 1e-9)
    data["High_amount_flag"] = (data["Amount"] > amount_p99).astype(int)
    data["Hour"] = ((data["Time"] // 3600) % 24).astype(int)

    # Padronização: média próxima de 0 e desvio padrão próximo de 1.
    scaler = StandardScaler()
    data[["Amount_scaled", "Time_scaled"]] = scaler.fit_transform(data[["Amount", "Time"]])

    return data


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa X e y e cria bases de treino e teste.

    stratify=y preserva a proporção de fraude nas duas bases. Isso é essencial
    porque a classe fraudulenta é rara.
    """
    X = df.drop(columns=["Class", "Amount", "Time"])
    y = df["Class"]

    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def undersample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Balanceia apenas a base de treino por undersampling.

    A base de teste fica desbalanceada de propósito, pois ela precisa representar
    o mundo real. Isso evita vazamento de dados e avaliação artificialmente otimista.
    """
    train_df = X_train.copy()
    train_df["Class"] = y_train.values

    frauds = train_df[train_df["Class"] == 1]
    normals = train_df[train_df["Class"] == 0].sample(
        n=len(frauds),
        random_state=RANDOM_STATE,
    )

    balanced = pd.concat([frauds, normals], axis=0).sample(
        frac=1,
        random_state=RANDOM_STATE,
    )

    return balanced.drop(columns="Class"), balanced["Class"]


def build_models(y_train: pd.Series) -> Dict[str, Any]:
    """
    Monta os modelos comparados no projeto.

    Foram usados modelos alinhados ao conteúdo do desafio:
    - Regressão Logística como baseline interpretável.
    - Regressão Logística com class_weight='balanced'.
    - Random Forest com peso para classe rara.
    - XGBoost, se a biblioteca estiver instalada.
    """
    models: Dict[str, Any] = {}

    # newton-cholesky é eficiente quando temos muitas linhas e poucas dezenas de variáveis.
    logistic_default = LogisticRegression(
        max_iter=100,
        solver="newton-cholesky",
        random_state=RANDOM_STATE,
    )

    logistic_balanced = LogisticRegression(
        max_iter=100,
        solver="newton-cholesky",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    models["logistic_baseline"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", logistic_default),
        ]
    )

    models["logistic_balanced"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", logistic_balanced),
        ]
    )

    models["random_forest_balanced"] = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    try:
        from xgboost import XGBClassifier

        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models["xgboost_balanced"] = XGBClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.10,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            subsample=0.90,
            colsample_bytree=0.90,
            random_state=RANDOM_STATE,
            n_jobs=2,
        )
    except Exception as exc:
        print(f"XGBoost não disponível. Seguindo sem ele. Motivo: {exc}")

    return models


def predict_probabilities(model: Any, X_test: pd.DataFrame) -> np.ndarray:
    """Retorna a probabilidade de fraude para cada transação da base de teste."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    scores = model.decision_function(X_test)
    return (scores - scores.min()) / (scores.max() - scores.min())


def evaluate_model(
    name: str,
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Avalia um modelo com threshold configurável.

    threshold é o ponto de corte. Com 0.50, o modelo só chama fraude quando a
    probabilidade estimada é de 50% ou mais. Em fraude, testar thresholds menores
    pode aumentar recall, mas também pode aumentar falsos positivos.
    """
    y_prob = predict_probabilities(model, X_test)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "modelo": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "average_precision": average_precision_score(y_test, y_prob),
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
        "alertas_gerados": int(y_pred.sum()),
    }

    print(f"\n================ {name.upper()} ================")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Matriz de confusão [[TN, FP], [FN, TP]]:")
    print(cm)

    return metrics


def save_curves(name: str, y_test: pd.Series, y_prob: np.ndarray) -> None:
    """Gera e salva a curva ROC e a curva Precision-Recall do modelo."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Modelo aleatório")
    plt.title(f"Curva ROC - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / f"curva_roc_{name}.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"Average Precision = {avg_precision:.4f}")
    plt.title(f"Curva Precision-Recall - {name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / f"curva_precision_recall_{name}.png", dpi=150)
    plt.close()


def threshold_analysis(y_test: pd.Series, y_prob: np.ndarray) -> pd.DataFrame:
    """Cria uma tabela com o impacto de diferentes thresholds."""
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        y_pred = (y_prob >= threshold).astype(int)
        frauds_captured = int(((y_pred == 1) & (y_test.values == 1)).sum())
        alerts = int(y_pred.sum())

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "alertas_gerados": alerts,
                "fraudes_capturadas": frauds_captured,
                "falsos_positivos": alerts - frauds_captured,
                "fraudes_nao_capturadas": int(((y_pred == 0) & (y_test.values == 1)).sum()),
            }
        )

    df_thresholds = pd.DataFrame(rows)
    df_thresholds.to_csv(OUTPUT_METRICS / "threshold_analysis.csv", index=False)
    return df_thresholds


def save_threshold_comparison(thresholds: pd.DataFrame) -> pd.DataFrame:
    """Salva uma comparação direta entre thresholds importantes para a decisão operacional."""
    scenarios = {
        0.30: "Agressivo: maior captura de fraudes",
        0.50: "Padrão operacional inicial",
        0.55: "Maior F1-score observado",
    }

    rows = []
    for threshold, scenario in scenarios.items():
        row = thresholds.loc[np.isclose(thresholds["threshold"], threshold)].iloc[0].to_dict()
        row["cenario"] = scenario
        rows.append(row)

    comparison = pd.DataFrame(rows)[
        [
            "threshold",
            "cenario",
            "precision",
            "recall",
            "f1_score",
            "alertas_gerados",
            "fraudes_capturadas",
            "falsos_positivos",
            "fraudes_nao_capturadas",
        ]
    ]
    comparison.to_csv(OUTPUT_METRICS / "threshold_comparison_030_050_055.csv", index=False)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(comparison))
    width = 0.25
    plt.bar(x - width, comparison["fraudes_capturadas"], width=width, label="Fraudes capturadas")
    plt.bar(x, comparison["falsos_positivos"], width=width, label="Falsos positivos")
    plt.bar(x + width, comparison["fraudes_nao_capturadas"], width=width, label="Fraudes não capturadas")
    plt.xticks(x, [f"{value:.2f}" for value in comparison["threshold"]])
    plt.xlabel("Threshold")
    plt.ylabel("Quantidade")
    plt.title("Comparativo operacional por threshold - Random Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "comparativo_threshold_030_050_055.png", dpi=150)
    plt.close()

    return comparison


def save_feature_importance(name: str, model: Any, feature_names: pd.Index) -> None:
    """Salva a importância das variáveis para modelos baseados em árvores."""
    raw_model = model
    if isinstance(model, Pipeline):
        raw_model = model.named_steps.get("model", model)

    if not hasattr(raw_model, "feature_importances_"):
        return

    importances = pd.Series(raw_model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)
    importances.to_csv(OUTPUT_METRICS / f"importancia_variaveis_{name}.csv", header=["importance"])

    top_15 = importances.head(15).sort_values()
    plt.figure(figsize=(10, 6))
    top_15.plot(kind="barh")
    plt.title(f"Top 15 Variáveis Mais Importantes - {name}")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / f"importancia_variaveis_{name}.png", dpi=150)
    plt.close()


def run_optional_grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Executa uma busca simples de hiperparâmetros com XGBoost, se solicitado."""
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        print(f"GridSearch não executado. XGBoost indisponível: {exc}")
        return None

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    param_grid = {
        "max_depth": [3, 5],
        "n_estimators": [40, 80],
    }

    grid = GridSearchCV(
        estimator=XGBClassifier(
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
        param_grid=param_grid,
        scoring="recall",
        cv=3,
        n_jobs=1,
    )

    grid.fit(X_train, y_train)
    print("\nMelhores parâmetros do GridSearch:")
    print(grid.best_params_)

    with open(OUTPUT_METRICS / "melhores_parametros_gridsearch.json", "w", encoding="utf-8") as file:
        json.dump(grid.best_params_, file, indent=4, ensure_ascii=False)

    return grid.best_estimator_


def choose_best_model(metrics_df: pd.DataFrame) -> str:
    """
    Escolhe o modelo operacional principal.

    Critério escolhido:
    1. Prioriza Average Precision, pois a curva Precision-Recall é muito útil em classe rara.
    2. Depois olha F1-score, que equilibra precision e recall.
    3. Depois olha recall, porque perder fraude é um risco relevante.
    """
    ordered = metrics_df.sort_values(
        by=["average_precision", "f1_score", "recall"],
        ascending=False,
    )
    return str(ordered.iloc[0]["modelo"])


def write_training_report(
    profile: Dict[str, Any],
    metrics_df: pd.DataFrame,
    best_model_name: str,
    thresholds: pd.DataFrame,
) -> None:
    """Cria um relatório Markdown com a leitura executiva dos resultados."""
    best_row = metrics_df[metrics_df["modelo"] == best_model_name].iloc[0]
    best_threshold = thresholds.sort_values(by=["f1_score", "recall"], ascending=False).iloc[0]
    threshold_030 = thresholds.loc[np.isclose(thresholds["threshold"], 0.30)].iloc[0]
    threshold_050 = thresholds.loc[np.isclose(thresholds["threshold"], 0.50)].iloc[0]

    extra_frauds_030 = int(threshold_030["fraudes_capturadas"] - threshold_050["fraudes_capturadas"])
    extra_false_positives_030 = int(threshold_030["falsos_positivos"] - threshold_050["falsos_positivos"])

    report = f"""# Relatório de Treinamento - Detecção de Anomalias em Transações

## 1. Dataset utilizado

- Linhas: {profile['linhas']:,}
- Colunas: {profile['colunas']:,}
- Transações normais: {profile['transacoes_normais']:,}
- Transações fraudulentas: {profile['transacoes_fraudulentas']:,}
- Percentual de fraude: {profile['percentual_fraude']:.4%}
- Valores ausentes: {profile['valores_ausentes_total']}

## 2. Melhor modelo operacional

O melhor modelo operacional foi **{best_model_name}**, escolhido pela combinação de Average Precision, F1-score e Recall.

Métricas no threshold 0.50:

- Accuracy: {best_row['accuracy']:.4f}
- Precision: {best_row['precision']:.4f}
- Recall: {best_row['recall']:.4f}
- F1-score: {best_row['f1_score']:.4f}
- ROC-AUC: {best_row['roc_auc']:.4f}
- Average Precision: {best_row['average_precision']:.4f}
- Falsos positivos: {int(best_row['false_positive'])}
- Falsos negativos: {int(best_row['false_negative'])}
- Fraudes capturadas: {int(best_row['true_positive'])}
- Alertas gerados: {int(best_row['alertas_gerados'])}

## 3. Análise específica do threshold 0.30

O threshold 0.30 foi testado como uma configuração mais agressiva para captura de fraude.

Métricas no threshold 0.30:

- Precision: {threshold_030['precision']:.4f}
- Recall: {threshold_030['recall']:.4f}
- F1-score: {threshold_030['f1_score']:.4f}
- Falsos positivos: {int(threshold_030['falsos_positivos'])}
- Fraudes capturadas: {int(threshold_030['fraudes_capturadas'])}
- Fraudes não capturadas: {int(threshold_030['fraudes_nao_capturadas'])}
- Alertas gerados: {int(threshold_030['alertas_gerados'])}

Comparado ao threshold 0.50, o threshold 0.30 capturou **{extra_frauds_030} fraudes a mais**. O custo dessa melhora foi gerar **{extra_false_positives_030} falsos positivos adicionais**.

## 4. Melhor threshold observado para equilíbrio F1/Recall

- Threshold: {best_threshold['threshold']:.2f}
- Precision: {best_threshold['precision']:.4f}
- Recall: {best_threshold['recall']:.4f}
- F1-score: {best_threshold['f1_score']:.4f}
- Alertas gerados: {int(best_threshold['alertas_gerados'])}
- Fraudes capturadas: {int(best_threshold['fraudes_capturadas'])}
- Falsos positivos: {int(best_threshold['falsos_positivos'])}
- Fraudes não capturadas: {int(best_threshold['fraudes_nao_capturadas'])}

## 5. Leitura profissional

A base é extremamente desbalanceada. Por isso, o projeto não usa acurácia como métrica principal.
A análise combina Recall, Precision, F1-score, ROC-AUC e principalmente Precision-Recall.

Em uma operação real, o threshold deve ser definido de acordo com a capacidade da equipe de tratar alertas:

- threshold menor: mais fraudes capturadas, porém mais falsos positivos;
- threshold maior: menos falsos positivos, porém maior risco de deixar fraude passar.

O threshold 0.30 é útil quando a prioridade do negócio é capturar mais fraudes, mesmo aceitando maior volume de revisão. Para uma operação mais equilibrada, o threshold 0.50 continua mais conservador e eficiente.

## 6. Arquivos gerados

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
"""

    (PROJECT_ROOT / "reports" / "relatorio_treinamento.md").write_text(report, encoding="utf-8")

def main(run_grid_search: bool = False) -> None:
    """Executa o pipeline completo de ponta a ponta."""
    ensure_directories()

    # 1. Carregamento e validação.
    df = load_dataset()
    validate_dataset(df)

    # 2. Análise inicial.
    profile = basic_analysis(df)
    plot_class_distribution(df)

    # 3. Engenharia de atributos.
    df_features = create_features(df)

    # 4. Separação treino/teste.
    X_train, X_test, y_train, y_test = split_data(df_features)

    print("\nDistribuição da classe na base de treino:")
    print(y_train.value_counts(normalize=True))
    print("\nDistribuição da classe na base de teste:")
    print(y_test.value_counts(normalize=True))

    # 5. Criação dos modelos.
    models = build_models(y_train)

    # Modelo com undersampling: usa treino balanceado, mas teste real desbalanceado.
    X_train_under, y_train_under = undersample_training_data(X_train, y_train)
    models["logistic_undersampling"] = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=100,
                    solver="newton-cholesky",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    metrics_list = []
    trained_models: Dict[str, Any] = {}

    # 6. Treinamento e avaliação de cada modelo.
    for name, model in models.items():
        print(f"\nTreinando modelo: {name}")

        if name == "logistic_undersampling":
            model.fit(X_train_under, y_train_under)
        else:
            model.fit(X_train, y_train)

        trained_models[name] = model
        y_prob = predict_probabilities(model, X_test)
        metrics = evaluate_model(name, model, X_test, y_test, threshold=0.50)
        metrics_list.append(metrics)
        save_curves(name, y_test, y_prob)
        save_feature_importance(name, model, X_train.columns)

    # 7. Consolidação das métricas.
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(OUTPUT_METRICS / "metricas_modelos.csv", index=False)

    # 8. Escolha do melhor modelo e análise de thresholds.
    best_model_name = choose_best_model(metrics_df)
    best_model = trained_models[best_model_name]
    best_prob = predict_probabilities(best_model, X_test)

    print("\n================ MELHOR MODELO OPERACIONAL ================")
    print(f"Modelo escolhido: {best_model_name}")
    print("Critério: Average Precision, depois F1-score, depois Recall.")

    thresholds = threshold_analysis(y_test, best_prob)
    threshold_comparison = save_threshold_comparison(thresholds)
    print("\nComparação operacional entre thresholds 0.30, 0.50 e 0.55:")
    print(threshold_comparison)
    print("\nTop 10 thresholds por F1-score e recall:")
    print(thresholds.sort_values(by=["f1_score", "recall"], ascending=False).head(10))

    # 9. Persistência do melhor modelo.
    joblib.dump(best_model, MODEL_DIR / "melhor_modelo_fraude.pkl")
    print("\nModelo salvo em models/melhor_modelo_fraude.pkl")

    # 10. GridSearch opcional.
    if run_grid_search:
        tuned_model = run_optional_grid_search(X_train, y_train)
        if tuned_model is not None:
            joblib.dump(tuned_model, MODEL_DIR / "modelo_gridsearch.pkl")
            print("Modelo GridSearch salvo em models/modelo_gridsearch.pkl")

    # 11. Relatório final em Markdown.
    write_training_report(profile, metrics_df, best_model_name, thresholds)

    print("\nPipeline finalizado com sucesso.")
    print("Verifique: outputs/figures, outputs/metrics, models e reports/relatorio_treinamento.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de detecção de anomalias em transações de cartão de crédito."
    )
    parser.add_argument(
        "--run-grid-search",
        action="store_true",
        help="Executa GridSearchCV opcional com XGBoost.",
    )
    args = parser.parse_args()
    main(run_grid_search=args.run_grid_search)
