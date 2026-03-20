# Autism Prediction

A machine learning project to predict autism screening outcome from questionnaire and demographic features.

## Project Contents

- `Autism_Preidiction_using_machine_Learning.ipynb`: end-to-end notebook (EDA, preprocessing, training, evaluation)
- `build_predictive_system.py`: builds final deployable artifacts
- `predict_autism.py`: CLI prediction script for JSON input
- `sample_input.json`: example input sample
- `train.csv`: training data
- `artifacts/`: final model artifacts used for inference

## Quick Start

### 1. Create and activate virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
```

### 3. Build final artifacts (if needed)

Run this when `artifacts/final_model.joblib`, `artifacts/final_encoder.joblib`, or `artifacts/final_config.json` are missing or outdated.

```powershell
python .\build_predictive_system.py
```

### 4. Run prediction

```powershell
python .\predict_autism.py --sample-json .\sample_input.json
```

Example output:

```json
{
  "prediction": 0,
  "label": "Autism Negative",
  "probability_class_1": 0.0,
  "model_used": "baseline"
}
```

## Input Format

The JSON file passed to `--sample-json` should be a single object containing feature-value pairs, for example:

```json
{
  "A1_Score": 1,
  "A2_Score": 0,
  "A3_Score": 1,
  "A4_Score": 0,
  "A5_Score": 1,
  "A6_Score": 0,
  "A7_Score": 1,
  "A8_Score": 0,
  "A9_Score": 1,
  "A10_Score": 1,
  "age": 38.17274623,
  "gender": "f",
  "ethnicity": "?",
  "jaundice": "no",
  "austim": "no",
  "contry_of_res": "Austria",
  "used_app_before": "no",
  "result": 6.351165589,
  "age_desc": "18 and more",
  "relation": "Self"
}
```

## Notes

- `.venv` should not be committed to GitHub.
- If you change model-building logic, rebuild artifacts before running inference.
