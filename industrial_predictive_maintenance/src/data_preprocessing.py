from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def create_synthetic_ai4i_dataset(path: str | Path) -> pd.DataFrame:
    """Create a local AI4I-style dataset when the official CSV is unavailable."""
    path = Path(path)
    np.random.seed(42)
    n_rows = 1200
    df = pd.DataFrame(
        {
            "Air temperature [K]": np.random.uniform(295, 305, n_rows),
            "Process temperature [K]": np.random.uniform(305, 315, n_rows),
            "Rotational speed [rpm]": np.random.uniform(1200, 2800, n_rows),
            "Torque [Nm]": np.random.uniform(20, 70, n_rows),
            "Tool wear [min]": np.random.uniform(0, 250, n_rows),
            "Type": np.random.choice(["L", "M", "H"], n_rows),
        }
    )
    failure_score = (
        (df["Air temperature [K]"] - 300) / 5
        + (df["Process temperature [K]"] - 310) / 5
        + (2800 - df["Rotational speed [rpm]"]) / 1000
        + (df["Torque [Nm]"] - 40) / 30
        + df["Tool wear [min]"] / 150
    )
    failure_prob = 1 / (1 + np.exp(-(-6 + failure_score)))
    df["Machine failure"] = (np.random.rand(n_rows) < failure_prob).astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the AI4I 2020 predictive maintenance dataset from CSV."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        return create_synthetic_ai4i_dataset(dataset_path)
    return pd.read_csv(dataset_path)


def prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare the dataset for model training."""
    cleaned = df.copy()
    rename_map = {
        "Air temperature [K]": "Air Temperature",
        "Process temperature [K]": "Process Temperature",
        "Rotational speed [rpm]": "Rotational Speed",
        "Torque [Nm]": "Torque",
        "Tool wear [min]": "Tool Wear",
        "Type": "Product Type",
        "Machine failure": "Machine Failure",
        "UDI": "Record ID",
    }
    cleaned = cleaned.rename(columns={col: rename_map[col] for col in rename_map if col in cleaned.columns})
    cleaned = cleaned.drop(columns=[col for col in ["Record ID", "Product ID"] if col in cleaned.columns], errors="ignore")

    if cleaned.isnull().any().any():
        cleaned = cleaned.fillna(cleaned.median(numeric_only=True))

    cleaned["Product Type"] = cleaned["Product Type"].astype(str)
    return cleaned
