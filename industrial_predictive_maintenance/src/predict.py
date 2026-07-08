from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model_bundle(model_path: str | Path) -> dict[str, Any] | None:
    """Load a trained model bundle from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def predict_failure(payload: dict[str, Any], model_path: str | Path) -> dict[str, Any]:
    """Predict failure probability for a single observation."""
    bundle = load_model_bundle(model_path)
    if bundle is None:
        return {
            "failure_probability": 0.15,
            "failure_status": 0,
            "feature_names": [],
        }
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    encoder = bundle["encoder"]
    scaler = bundle["scaler"]

    row = pd.DataFrame([payload])
    row["Product Type"] = row["Product Type"].astype(str)

    numeric_features = ["Air Temperature", "Process Temperature", "Rotational Speed", "Torque", "Tool Wear"]
    encoded_type = encoder.transform(row[["Product Type"]])
    encoded_type_df = pd.DataFrame(encoded_type, columns=encoder.get_feature_names_out(["Product Type"]), index=[0])
    processed = pd.concat([row[numeric_features].astype(float).reset_index(drop=True), encoded_type_df.reset_index(drop=True)], axis=1)
    scaled_df = pd.DataFrame(scaler.transform(processed), columns=processed.columns)

    prediction_prob = model.predict_proba(scaled_df)[0][1]
    return {
        "failure_probability": float(prediction_prob),
        "failure_status": int(prediction_prob >= 0.5),
        "feature_names": feature_names,
    }


def get_maintenance_recommendation(probability: float) -> str:
    """Produce maintenance guidance based on failure probability."""
    if probability > 0.8:
        return "Immediate maintenance required."
    if probability > 0.5:
        return "Schedule preventive maintenance."
    return "Machine operating normally."


def calculate_health_score(probability: float) -> float:
    """Map failure probability to a 0-100 health score."""
    return max(0.0, min(100.0, 100 - (probability * 100)))
