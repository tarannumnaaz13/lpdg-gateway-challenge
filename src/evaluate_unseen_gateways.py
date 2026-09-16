import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


OUT = Path("outputs")

RANDOM_STATE = 42


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(
    OUT / "ml_gateway_week_features.csv"
)

df["week_start"] = pd.to_datetime(
    df["week_start"]
)

df["gateway_id"] = (
    df["gateway_id"]
    .astype(str)
    .str.replace(":", "", regex=False)
    .str.replace("-", "", regex=False)
    .str.upper()
)


# Remove rows without a known target

df = df[df["target_problem"].notna()].copy()

df["target_problem"] = df["target_problem"].astype(int)


# ---------------------------------------------------------
# UNSEEN GATEWAY SPLIT
# ---------------------------------------------------------

gateways = sorted(
    df["gateway_id"].unique()
)

rng = np.random.RandomState(RANDOM_STATE)

rng.shuffle(gateways)

split_point = int(len(gateways) * 0.80)

train_gateways = set(
    gateways[:split_point]
)

test_gateways = set(
    gateways[split_point:]
)


# ---------------------------------------------------------
# TEMPORAL SPLIT
# ---------------------------------------------------------

train_end = pd.Timestamp("2026-01-05")

validation_start = pd.Timestamp("2026-01-12")


train = df[
    (df["gateway_id"].isin(train_gateways)) &
    (df["week_start"] <= train_end)
].copy()


test = df[
    (df["gateway_id"].isin(test_gateways)) &
    (df["week_start"] >= validation_start)
].copy()


print()
print("=" * 70)
print("UNSEEN GATEWAY EVALUATION")
print("=" * 70)

print()
print("Total gateways:", len(gateways))
print("Training gateways:", len(train_gateways))
print("Unseen test gateways:", len(test_gateways))

print()
print("Training rows:", len(train))
print("Test rows:", len(test))

print()
print("Training period:")
print(
    train["week_start"].min().date(),
    "to",
    train["week_start"].max().date()
)

print()
print("Unseen gateway test period:")
print(
    test["week_start"].min().date(),
    "to",
    test["week_start"].max().date()
)


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

id_columns = [
    "gateway_id",
    "week_start"
]

feature_columns = [
    c
    for c in df.columns
    if c not in id_columns
    and not c.startswith("target_")
]


X_train = train[feature_columns]
y_train = train["target_problem"]

X_test = test[feature_columns]
y_test = test["target_problem"]


categorical_columns = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numeric_columns = [
    c
    for c in feature_columns
    if c not in categorical_columns
]


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

print()
print("Training model on known gateways...")

pipeline.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)


accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

average_precision = average_precision_score(
    y_test,
    probabilities
)


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("=" * 70)
print("UNSEEN GATEWAY RESULTS")
print("=" * 70)

print()
print("Accuracy          :", round(accuracy, 4))
print("Precision         :", round(precision, 4))
print("Recall            :", round(recall, 4))
print("F1 Score          :", round(f1, 4))
print("ROC-AUC           :", round(roc_auc, 4))
print("Average Precision :", round(average_precision, 4))

print()
print("Test positive rate:", round(y_test.mean(), 4))

print()
print("=" * 70)
print("GATEWAY HOLDOUT CHECK")
print("=" * 70)

overlap = train_gateways.intersection(
    test_gateways
)

print(
    "Train/test gateway overlap:",
    len(overlap)
)

if len(overlap) == 0:
    print("PASS: Test gateways are completely unseen during training.")
else:
    print("FAIL: Gateway overlap detected.")


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

results = pd.DataFrame({
    "metric": [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision"
    ],
    "value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        average_precision
    ]
})

results.to_csv(
    OUT / "unseen_gateway_metrics.csv",
    index=False
)

print()
print("Saved:")
print("outputs/unseen_gateway_metrics.csv")