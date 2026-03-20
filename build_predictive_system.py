import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


RANDOM_STATE = 42
DATA_PATH = Path("train.csv")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)
FINAL_MODEL_PATH = ARTIFACT_DIR / "final_model.joblib"
FINAL_ENCODER_PATH = ARTIFACT_DIR / "final_encoder.joblib"
FINAL_CONFIG_PATH = ARTIFACT_DIR / "final_config.json"


def _select_winner(baseline_metrics: dict, improved_metrics: dict) -> dict:
    if not baseline_metrics.get("available"):
        return {
            "selected": "improved",
            "reason": "Baseline artifacts are not available",
        }

    baseline_f1 = baseline_metrics["f1"]
    improved_f1 = improved_metrics["f1"]

    if baseline_f1 > improved_f1:
        return {
            "selected": "baseline",
            "reason": "Baseline has higher F1 score",
        }

    if improved_f1 > baseline_f1:
        return {
            "selected": "improved",
            "reason": "Improved pipeline has higher F1 score",
        }

    if baseline_metrics["accuracy"] >= improved_metrics["accuracy"]:
        return {
            "selected": "baseline",
            "reason": "F1 tie; baseline has better or equal accuracy",
        }

    return {
        "selected": "improved",
        "reason": "F1 tie; improved has better accuracy",
    }


def _publish_final_artifacts(selected: str) -> None:
    if selected == "baseline":
        model = joblib.load(Path("best_model.pkl"))
        encoders = joblib.load(Path("encoders.pkl"))

        joblib.dump(model, FINAL_MODEL_PATH)
        joblib.dump(encoders, FINAL_ENCODER_PATH)

        final_config = {
            "model_type": "baseline",
            "model_path": str(FINAL_MODEL_PATH.name),
            "encoder_path": str(FINAL_ENCODER_PATH.name),
        }

        # Keep deployment folder clean by removing non-selected improved artifacts.
        for path in [
            ARTIFACT_DIR / "autism_model.joblib",
            ARTIFACT_DIR / "autism_encoder.joblib",
            ARTIFACT_DIR / "autism_pipeline.joblib",
            ARTIFACT_DIR / "metadata.json",
        ]:
            if path.exists():
                path.unlink()
    else:
        model = joblib.load(ARTIFACT_DIR / "autism_model.joblib")
        encoder = joblib.load(ARTIFACT_DIR / "autism_encoder.joblib")
        metadata = {}
        metadata_path = ARTIFACT_DIR / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        joblib.dump(model, FINAL_MODEL_PATH)
        joblib.dump(encoder, FINAL_ENCODER_PATH)

        final_config = {
            "model_type": "improved",
            "model_path": str(FINAL_MODEL_PATH.name),
            "encoder_path": str(FINAL_ENCODER_PATH.name),
            "metadata": metadata,
        }

    with open(FINAL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)


def _load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    return pd.read_csv(path)


def _evaluate_baseline_from_existing_artifacts(df: pd.DataFrame) -> dict:
    model_path = Path("best_model.pkl")
    encoders_path = Path("encoders.pkl")

    if not model_path.exists() or not encoders_path.exists():
        return {
            "available": False,
            "reason": "best_model.pkl and/or encoders.pkl not found",
        }

    baseline_df = df.copy()
    encoders = joblib.load(encoders_path)

    for column, encoder in encoders.items():
        if column in baseline_df.columns:
            values = baseline_df[column].astype(str)
            classes = set(encoder.classes_)
            safe_values = np.where(values.isin(classes), values, encoder.classes_[0])
            baseline_df[column] = encoder.transform(safe_values)

    X = baseline_df.drop(columns=["Class/ASD"])
    y = baseline_df["Class/ASD"]
    model = joblib.load(model_path)

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        X = X.reindex(columns=expected_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    y_pred = model.predict(X_test)

    return {
        "available": True,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }


def _train_improved_model(df: pd.DataFrame) -> dict:
    work_df = df.drop(columns=["ID"]).copy() if "ID" in df.columns else df.copy()
    X = work_df.drop(columns=["Class/ASD"])
    y = work_df["Class/ASD"].astype(int)

    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols),
        ]
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "model",
                XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.7, 0.85, 1.0],
        "model__colsample_bytree": [0.7, 0.85, 1.0],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0, 0.1, 0.3],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,
        scoring="f1",
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    encoder = best_pipeline.named_steps["preprocess"]
    model = best_pipeline.named_steps["model"]

    encoder_output_dim = int(
        encoder.transform(X_train.head(1)).shape[1]
    )

    joblib.dump(best_pipeline, ARTIFACT_DIR / "autism_pipeline.joblib")
    joblib.dump(encoder, ARTIFACT_DIR / "autism_encoder.joblib")
    joblib.dump(model, ARTIFACT_DIR / "autism_model.joblib")

    metadata = {
        "target": "Class/ASD",
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols,
        "dropped_columns": ["ID"] if "ID" in df.columns else [],
        "encoder_output_dim": encoder_output_dim,
        "random_state": RANDOM_STATE,
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
    }

    with open(ARTIFACT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "best_params": search.best_params_,
        "best_cv_f1": float(search.best_score_),
    }


def main() -> None:
    df = _load_data(DATA_PATH)

    baseline_metrics = _evaluate_baseline_from_existing_artifacts(df)
    improved_metrics = _train_improved_model(df)

    summary = {
        "baseline": baseline_metrics,
        "improved": improved_metrics,
    }

    winner = _select_winner(baseline_metrics, improved_metrics)
    summary["selected_model"] = winner
    _publish_final_artifacts(winner["selected"])

    with open(ARTIFACT_DIR / "metrics_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Artifacts saved under ./artifacts")
    if baseline_metrics.get("available"):
        print(
            "Baseline -> Accuracy: "
            f"{baseline_metrics['accuracy']:.4f}, F1: {baseline_metrics['f1']:.4f}, "
            f"Recall: {baseline_metrics['recall']:.4f}"
        )
    else:
        print(f"Baseline unavailable: {baseline_metrics.get('reason')}")

    print(
        "Improved -> Accuracy: "
        f"{improved_metrics['accuracy']:.4f}, F1: {improved_metrics['f1']:.4f}, "
        f"Recall: {improved_metrics['recall']:.4f}"
    )
    print(
        f"Selected for prediction: {winner['selected']} ({winner['reason']})"
    )
    print("Final deployment artifacts: artifacts/final_model.joblib, artifacts/final_encoder.joblib")


if __name__ == "__main__":
    main()
