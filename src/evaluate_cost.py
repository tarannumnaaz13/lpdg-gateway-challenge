import pandas as pd
from pathlib import Path

OUT = Path("outputs")

FP_COST = 380
FN_COST = 600

baseline_file = OUT / "predictions_baseline.csv"
ml_file = OUT / "predictions_ml.csv"

features = pd.read_csv(OUT / "ml_gateway_week_features.csv")
features["week_start"] = pd.to_datetime(features["week_start"])

baseline = pd.read_csv(baseline_file)
ml = pd.read_csv(ml_file)

baseline["week_start"] = pd.to_datetime(baseline["week_start"])
ml["week_start"] = pd.to_datetime(ml["week_start"])

weeks = pd.date_range(
    start="2026-02-02",
    end="2026-03-16",
    freq="7D"
)


def clean_id(series):
    return (
        series.astype(str)
        .str.replace(":", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.upper()
    )


features["gateway_id"] = clean_id(features["gateway_id"])
baseline["gateway_id"] = clean_id(baseline["gateway_id"])
ml["gateway_id"] = clean_id(ml["gateway_id"])


def evaluate(predictions, name):

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_cost = 0

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    for week in weeks:

        actual = features[
            (features["week_start"] == week)
            & (features["target_problem"].notna())
        ]

        selected = set(
            predictions.loc[
                predictions["week_start"] == week,
                "gateway_id"
            ]
        )

        actual_ids = actual["gateway_id"]

        problem = actual["target_problem"] == 1
        normal = actual["target_problem"] == 0

        selected_mask = actual_ids.isin(selected)

        tp = int((selected_mask & problem).sum())
        fp = int((selected_mask & normal).sum())
        fn = int((~selected_mask & problem).sum())

        cost = (fp * FP_COST) + (fn * FN_COST)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_cost += cost

        print(
            week.date(),
            "TP:", tp,
            "FP:", fp,
            "FN:", fn,
            "Cost: EUR", cost
        )

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)

    print()
    print("Total TP:", total_tp)
    print("Total FP:", total_fp)
    print("Total FN:", total_fn)
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("TOTAL COST: EUR", total_cost)

    return total_cost


baseline_cost = evaluate(
    baseline,
    "BASELINE 3-SIGMA"
)

ml_cost = evaluate(
    ml,
    "ML MODEL"
)


print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

savings = baseline_cost - ml_cost

if baseline_cost > 0:
    savings_percent = (savings / baseline_cost) * 100
else:
    savings_percent = 0

print("Baseline cost : EUR", baseline_cost)
print("ML cost       : EUR", ml_cost)
print("ML savings    : EUR", savings)
print("Savings %     :", round(savings_percent, 2), "%")

if ml_cost < baseline_cost:
    print()
    print("RESULT: ML BEATS BASELINE")
else:
    print()
    print("RESULT: ML DOES NOT YET BEAT BASELINE")