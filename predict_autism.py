import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


ARTIFACT_DIR = Path("artifacts")
ENCODER_PATH = ARTIFACT_DIR / "final_encoder.joblib"
MODEL_PATH = ARTIFACT_DIR / "final_model.joblib"
CONFIG_PATH = ARTIFACT_DIR / "final_config.json"


def load_artifacts():
    missing = [
        p.name
        for p in [ENCODER_PATH, MODEL_PATH, CONFIG_PATH]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing artifact files: " + ", ".join(missing) + ". Run build_predictive_system.py first."
        )

    encoder_obj = joblib.load(ENCODER_PATH)
    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config, encoder_obj, model


def predict_single(sample: dict) -> dict:
    config, encoder_obj, model = load_artifacts()
    model_type = config.get("model_type", "improved")

    if model_type == "baseline":
        row_df = pd.DataFrame([sample])

        for col, encoder in encoder_obj.items():
            if col in row_df.columns:
                values = row_df[col].astype(str)
                valid = set(encoder.classes_)
                safe_values = values.where(values.isin(valid), encoder.classes_[0])
                row_df[col] = encoder.transform(safe_values)

        if hasattr(model, "feature_names_in_"):
            expected_cols = list(model.feature_names_in_)
            row_df = row_df.reindex(columns=expected_cols, fill_value=0)

        pred = int(model.predict(row_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(row_df)[0][1])

        return {
            "prediction": pred,
            "label": "Autism Positive" if pred == 1 else "Autism Negative",
            "probability_class_1": probability,
            "model_used": "baseline",
        }

    metadata = config.get("metadata", {})
    expected_cols = metadata.get("categorical_columns", []) + metadata.get("numerical_columns", [])
    if not expected_cols:
        raise ValueError("Missing metadata columns for improved model in final_config.json")

    row = {col: sample.get(col) for col in expected_cols}
    sample_df = pd.DataFrame([row])

    transformed = encoder_obj.transform(sample_df)
    pred = int(model.predict(transformed)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(transformed)[0][1])

    return {
        "prediction": pred,
        "label": "Autism Positive" if pred == 1 else "Autism Negative",
        "probability_class_1": probability,
        "model_used": "improved",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict autism class from a JSON sample using saved encoder and model artifacts."
    )
    parser.add_argument(
        "--sample-json",
        required=True,
        help="Path to JSON file containing one sample row as key-value pairs",
    )
    args = parser.parse_args()

    sample_path = Path(args.sample_json)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample JSON file not found: {sample_path}")

    with open(sample_path, "r", encoding="utf-8") as f:
        sample = json.load(f)

    result = predict_single(sample)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
