from pathlib import Path

import joblib
import pandas as pd


DATA_PATH = Path("outputs/ml_gateway_week_features.csv")
MODEL_PATH = Path("outputs/ml_model.joblib")
OUTPUT_PATH = Path("outputs/predictions_ml.csv")

ID_COLS = ["gateway_id", "week_start"]
TARGET = "target_problem"


def main():
    print("=" * 70)
    print("LPDG ML PREDICTION")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    df["week_start"] = pd.to_datetime(df["week_start"])

    model = joblib.load(MODEL_PATH)

    # Final challenge weeks.
    scored_weeks = pd.date_range(
        "2026-02-02",
        "2026-03-23",
        freq="7D",
    )

    feature_cols = [
        c
        for c in df.columns
        if c not in ID_COLS
        and not c.startswith("target_")
    ]

    print(f"Feature columns: {len(feature_cols)}")
    print(f"Scored weeks: {len(scored_weeks)}")

    all_predictions = []

    for week in scored_weeks:
        week_df = df[df["week_start"] == week].copy()

        if week_df.empty:
            print(f"WARNING: no rows for {week.date()}")
            continue

        X = week_df[feature_cols]

        probabilities = model.predict_proba(X)[:, 1]

        week_df["score"] = probabilities

        # Highest-risk gateways first.
        week_df = week_df.sort_values(
            ["score", "gateway_id"],
            ascending=[False, True],
        ).head(15)

        week_df["rank"] = range(1, len(week_df) + 1)

        def make_reason(row):
            reasons = []

            if row.get("recent7_disconnections", 0) > 0:
                reasons.append(
                    f"recent disconnections={row['recent7_disconnections']:.0f}"
                )

            if row.get("recent7_offline_hours", 0) > 0:
                reasons.append(
                    f"recent offline={row['recent7_offline_hours']:.1f}h"
                )

            if row.get("recent7_no_conn_importance_mean", 0) > 0:
                reasons.append("recent connectivity degradation")

            if row.get("meter_success_min", 1) < 0.80:
                reasons.append(
                    f"low meter-read success={row['meter_success_min']:.2f}"
                )

            if not reasons:
                reasons.append("elevated predicted operational risk")

            return "; ".join(reasons[:2])

        week_df["reason"] = week_df.apply(make_reason, axis=1)

        output = week_df[
            ["week_start", "rank", "gateway_id", "score", "reason"]
        ].copy()

        output["week_start"] = output["week_start"].dt.strftime(
            "%Y-%m-%d"
        )

        all_predictions.append(output)

        print(
            f"{week.date()} -> "
            f"{len(output)} gateways selected"
        )

    if not all_predictions:
        raise RuntimeError("No predictions were generated.")

    result = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("PREDICTIONS COMPLETE")
    print("=" * 70)
    print(f"Rows: {len(result)}")
    print(f"Weeks: {result['week_start'].nunique()}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()