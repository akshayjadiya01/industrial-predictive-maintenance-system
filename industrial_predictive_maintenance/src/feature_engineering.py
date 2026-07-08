from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], Any, Any]:
    """Create a processed feature matrix and return the fitted encoder and scaler."""
    numeric_features = [
        "Air Temperature",
        "Process Temperature",
        "Rotational Speed",
        "Torque",
        "Tool Wear",
    ]
    categorical_features = ["Product Type"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(df[categorical_features])
    category_names = encoder.get_feature_names_out(categorical_features)
    encoded_frame = pd.DataFrame(encoded, columns=category_names, index=df.index)

    processed = pd.concat([df[numeric_features].reset_index(drop=True), encoded_frame.reset_index(drop=True)], axis=1)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(processed)
    scaled_df = pd.DataFrame(scaled, columns=processed.columns)

    feature_names = list(scaled_df.columns)
    return scaled_df, feature_names, encoder, scaler
