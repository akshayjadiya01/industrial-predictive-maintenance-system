from pathlib import Path

from src.data_preprocessing import load_dataset, prepare_training_dataframe
from src.predict import load_model_bundle, predict_failure

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"


def test_model_artifact_exists() -> None:
    assert MODEL_PATH.exists(), "Expected trained model artifact to be present"


def test_dataset_can_be_loaded_and_prepared() -> None:
    df = load_dataset(DATA_PATH)
    prepared = prepare_training_dataframe(df)
    assert not prepared.empty
    assert "Machine Failure" in prepared.columns


def test_prediction_returns_valid_probability() -> None:
    bundle = load_model_bundle(MODEL_PATH)
    assert bundle is not None

    payload = {
        "Air Temperature": 300.0,
        "Process Temperature": 310.0,
        "Rotational Speed": 1500.0,
        "Torque": 40.0,
        "Tool Wear": 100.0,
        "Product Type": "M",
    }
    result = predict_failure(payload, MODEL_PATH)

    assert 0.0 <= result["failure_probability"] <= 1.0
    assert "failure_status" in result
