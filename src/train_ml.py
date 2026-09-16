from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("outputs/ml_gateway_week_features.csv")
MODEL_PATH = Path("outputs/ml_model.joblib")
METRICS_PATH = Path("outputs/ml_training_metrics.json")
IMPORTANCE_PATH = Path("outputs/ml_feature_importance.csv")

TARGET = "target_problem"
ID_COLS = ["gateway_id", "week_start"]

# We deliberately use a temporal split.
# Training: earlier weeks
# Validation: later weeks
TRAIN_END = "2026-01-05"
VAL_START = "2026-01-12"


def main():
    print("=" * 70)
    print("LPDG ML TRAINING")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded rows: {len(df):,}")
    print(f"Loaded columns: {len(df.columns):,}")

    # Remove rows for which the future 7-day target is unavailable.
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    df["week_start"] = pd.to_datetime(df["week_start"])

    # Keep only rows available before the validation period.
    train_df = df[df["week_start"] < VAL_START].copy()
    val_df = df[df["week_start"] >= VAL_START].copy()

    print()
    print("Temporal split:")
    print(f"Training rows:   {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(
        f"Training period: {train_df['week_start'].min().date()} "
        f"to {train_df['week_start'].max().date()}"
    )
    print(
        f"Validation period: {val_df['week_start'].min().date()} "
        f"to {val_df['week_start'].max().date()}"
    )

    # Columns that should never be model inputs.
    # The target is removed separately.
    feature_cols = [
    c for c in df.columns
    if c not in ID_COLS
    and not c.startswith("target_")
    ]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET].copy()

    X_val = val_df[feature_cols].copy()
    y_val = val_df[TARGET].copy()

    # Detect categorical and numeric features.
    categorical_cols = X_train.select_dtypes(
    include=["object", "string", "category"]
    ).columns.tolist()

    numeric_cols = [
        c for c in feature_cols
        if c not in categorical_cols
    ]

    print()
    print(f"Features: {len(feature_cols)}")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    print()
    print("Training target distribution:")
    print(y_train.value_counts().sort_index())

    print()
    print("Validation target distribution:")
    print(y_val.value_counts().sort_index())

    # Numeric preprocessing.
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            )
        ]
    )

    # Categorical preprocessing.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    print()
    print("Training Random Forest...")
    pipeline.fit(X_train, y_train)

    # Probability of the gateway being a future problem.
    val_prob = pipeline.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.50).astype(int)

    metrics = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "recall": float(recall_score(y_val, val_pred, zero_division=0)),
        "f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_val, val_prob)),
        "average_precision": float(
            average_precision_score(y_val, val_prob)
        ),
    }

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")

    print()
    print("Classification report:")
    print(
        classification_report(
            y_val,
            val_pred,
            target_names=["Normal", "Problem"],
            zero_division=0,
        )
    )

    # Save model.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # Save metrics.
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Extract feature importances after preprocessing.
    feature_names = pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importances = pipeline.named_steps[
        "model"
    ].feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance_df.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    print()
    print("Top 20 features:")
    print(
        importance_df.head(20).to_string(index=False)
    )

    print()
    print("Saved:")
    print(f"  {MODEL_PATH}")
    print(f"  {METRICS_PATH}")
    print(f"  {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()