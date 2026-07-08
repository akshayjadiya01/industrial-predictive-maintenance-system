from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_preprocessing import load_dataset, prepare_training_dataframe
from src.feature_engineering import engineer_features


def train_and_save_model(data_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Train and compare multiple models, save the best one, and return evaluation artifacts."""
    data_path = Path(data_path)
    output_path = Path(output_path)
    df = prepare_training_dataframe(load_dataset(data_path))

    X, feature_names, encoder, scaler = engineer_features(df)
    y = df["Machine Failure"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42, eval_metric="logloss"),
    }

    comparison_rows: list[dict[str, Any]] = []
    best_model_name = ""
    best_model = None
    best_metrics: dict[str, float] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        metrics = {
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, zero_division=0),
            "Recall": recall_score(y_test, prediction, zero_division=0),
            "F1 Score": f1_score(y_test, prediction, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, probability),
        }
        comparison_rows.append({"Model": name, **metrics})

        if best_model is None or metrics["F1 Score"] > best_metrics.get("F1 Score", -1):
            best_model = model
            best_model_name = name
            best_metrics = metrics

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")

    best_model.fit(X, y)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best_model,
        "feature_names": feature_names,
        "model_name": best_model_name,
        "encoder": encoder,
        "scaler": scaler,
        "metrics": {
            "accuracy": best_metrics["Accuracy"],
            "precision": best_metrics["Precision"],
            "recall": best_metrics["Recall"],
            "f1_score": best_metrics["F1 Score"],
            "roc_auc": best_metrics["ROC AUC"],
            "confusion_matrix": confusion_matrix(y_test, best_model.predict(X_test)),
        },
        "comparison_table": pd.DataFrame(comparison_rows),
        "column_names": list(X.columns),
    }

    explainer = shap.Explainer(best_model, X_train)
    shap_values = explainer(X_test)
    values = np.array(shap_values.values)
    if values.ndim == 3:
        values = values.reshape(values.shape[0], values.shape[2])
    elif values.ndim != 2:
        values = np.asarray(values).reshape(len(X_test), len(feature_names))
    shap_importance = np.abs(values).mean(axis=0)
    shap_summary = pd.DataFrame({"Feature": feature_names, "Mean Abs SHAP": shap_importance}).sort_values("Mean Abs SHAP", ascending=False)

    bundle["shap_summary"] = shap_summary
    bundle["health_gauge"] = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=85,
        domain={"x": [0.1, 0.9], "y": [0.2, 0.9]},
        title={"text": "Machine Health"},
        gauge={
            "axis": {"range": [None, 100]},
            "steps": [
                {"range": [0, 30], "color": "#ef4444"},
                {"range": [31, 60], "color": "#f59e0b"},
                {"range": [61, 100], "color": "#22c55e"},
            ],
        },
    ))

    joblib.dump(bundle, output_path)
    return bundle
