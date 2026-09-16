import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SCORED_WEEKS = pd.date_range(
    "2026-02-02",
    "2026-03-23",
    freq="7D"
)

FEATURE_LOOKBACK_DAYS = 28
TARGET_DAYS = 7

# Operational target thresholds.
# A future week is considered problematic when either:
#   1. meter-read success is poor, OR
#   2. offline/disconnection behavior is substantially elevated.
#
# These are deliberately transparent thresholds rather than
# hidden model-derived labels.
METER_SUCCESS_THRESHOLD = 0.80
OFFLINE_HOURS_THRESHOLD = 24.0
DISCONNECTION_THRESHOLD = 50.0
REBOOT_THRESHOLD = 12.0


TELEMETRY_COLUMNS = [
    "gateway_id",
    "ts_utc",
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
    "no_conn_importance",
    "reboot_importance",
    "rx_nr_pkts",
    "rx_crc_bad",
    "network_unknown",
    "rssi_good",
    "rssi_normal",
    "rssi_bad",
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def normalize_gateway_id(value):
    """Normalize gateway IDs across files.

    Example:
        06:39:EA:56:02:C1
        0639EA5602C1

    Both become:
        0639EA5602C1
    """
    if pd.isna(value):
        return np.nan

    return (
        str(value)
        .strip()
        .upper()
        .replace(":", "")
        .replace("-", "")
        .replace(" ", "")
    )


def safe_divide(a, b):
    return np.where(b != 0, a / b, 0.0)


def add_telemetry_features(df, prefix):
    """Create aggregated telemetry features for a time window."""

    grouped = df.groupby("gateway_id")

    result = grouped.agg(
        **{
            f"{prefix}_offline_hours": (
                "offline_duration_sec",
                lambda x: x.sum() / 3600.0,
            ),
            f"{prefix}_offline_mean_sec": (
                "offline_duration_sec",
                "mean",
            ),
            f"{prefix}_offline_max_sec": (
                "offline_duration_sec",
                "max",
            ),
            f"{prefix}_disconnections": (
                "disconnection_cnt",
                "sum",
            ),
            f"{prefix}_disconnection_hours": (
                "disconnection_cnt",
                lambda x: (x > 0).sum(),
            ),
            f"{prefix}_reboots": (
                "reboot_cnt",
                "sum",
            ),
            f"{prefix}_reboot_hours": (
                "reboot_cnt",
                lambda x: (x > 0).sum(),
            ),
            f"{prefix}_reboot_duration_sec": (
                "reboot_duration_sec",
                "sum",
            ),
            f"{prefix}_no_conn_importance_mean": (
                "no_conn_importance",
                "mean",
            ),
            f"{prefix}_reboot_importance_mean": (
                "reboot_importance",
                "mean",
            ),
            f"{prefix}_rx_packets": (
                "rx_nr_pkts",
                "sum",
            ),
            f"{prefix}_crc_bad": (
                "rx_crc_bad",
                "sum",
            ),
            f"{prefix}_network_unknown": (
                "network_unknown",
                "sum",
            ),
            f"{prefix}_rssi_good": (
                "rssi_good",
                "sum",
            ),
            f"{prefix}_rssi_normal": (
                "rssi_normal",
                "sum",
            ),
            f"{prefix}_rssi_bad": (
                "rssi_bad",
                "sum",
            ),
            f"{prefix}_hours": (
                "gateway_id",
                "size",
            ),
        }
    )

    result[f"{prefix}_rssi_bad_rate"] = safe_divide(
        result[f"{prefix}_rssi_bad"],
        result[f"{prefix}_hours"],
    )

    result[f"{prefix}_rssi_good_rate"] = safe_divide(
        result[f"{prefix}_rssi_good"],
        result[f"{prefix}_hours"],
    )

    result[f"{prefix}_network_unknown_rate"] = safe_divide(
        result[f"{prefix}_network_unknown"],
        result[f"{prefix}_hours"],
    )

    result[f"{prefix}_crc_bad_rate"] = safe_divide(
        result[f"{prefix}_crc_bad"],
        result[f"{prefix}_rx_packets"],
    )

    result[f"{prefix}_coverage_ratio"] = (
        result[f"{prefix}_hours"] /
        (FEATURE_LOOKBACK_DAYS * 24)
    )

    return result.reset_index()


def build_window_features(
    telemetry,
    meter,
    master,
    visits,
    week_start,
):
    """Build features using data available BEFORE week_start."""

    week_start = pd.Timestamp(week_start, tz="UTC")
    feature_start = week_start - pd.Timedelta(days=FEATURE_LOOKBACK_DAYS)

    # --------------------------------------------------------
    # Telemetry windows
    # --------------------------------------------------------

    recent_mask = (
        (telemetry["ts_utc"] >= feature_start)
        & (telemetry["ts_utc"] < week_start)
    )

    recent = telemetry.loc[recent_mask].copy()

    # Previous 7 days, useful for detecting acceleration.
    prev7_start = week_start - pd.Timedelta(days=7)

    recent7 = recent.loc[
        recent["ts_utc"] >= prev7_start
    ].copy()

    previous21 = recent.loc[
        recent["ts_utc"] < prev7_start
    ].copy()

    f28 = add_telemetry_features(recent, "recent28")
    f7 = add_telemetry_features(recent7, "recent7")
    f21 = add_telemetry_features(previous21, "previous21")

    features = f28.merge(
        f7,
        on="gateway_id",
        how="left",
    )

    features = features.merge(
        f21,
        on="gateway_id",
        how="left",
    )

    # --------------------------------------------------------
    # Trend features
    # --------------------------------------------------------

    for metric in [
        "offline_hours",
        "disconnections",
        "reboots",
        "rssi_bad_rate",
        "crc_bad_rate",
        "network_unknown_rate",
    ]:
        c7 = f"recent7_{metric}"
        c21 = f"previous21_{metric}"

        features[f"trend_{metric}"] = (
            features[c7]
            - features[c21] / 3.0
        )

        features[f"ratio_{metric}"] = (
            features[c7] /
            (features[c21] / 3.0 + 1e-6)
        ).clip(upper=20)

    # --------------------------------------------------------
    # Meter-read features
    # --------------------------------------------------------

    meter = meter.copy()

    meter["week_start"] = pd.to_datetime(
        meter["week_start"],
        errors="coerce"
    )

    # Meter data for the previous four completed weeks.
    meter_start = (
        week_start.tz_localize(None)
        - pd.Timedelta(days=FEATURE_LOOKBACK_DAYS)
    )

    meter_end = week_start.tz_localize(None)

    meter_recent = meter.loc[
        (meter["week_start"] >= meter_start)
        & (meter["week_start"] < meter_end)
    ].copy()

    if not meter_recent.empty:
        meter_recent["read_success"] = safe_divide(
            meter_recent["meters_read"],
            meter_recent["meters_expected"],
        )

        meter_features = (
            meter_recent
            .groupby("gateway_id")
            .agg(
                meter_success_mean=(
                    "read_success",
                    "mean"
                ),
                meter_success_min=(
                    "read_success",
                    "min"
                ),
                meter_success_std=(
                    "read_success",
                    "std"
                ),
                meters_expected_mean=(
                    "meters_expected",
                    "mean"
                ),
                meters_read_mean=(
                    "meters_read",
                    "mean"
                ),
                meter_weeks_observed=(
                    "week_start",
                    "nunique"
                ),
            )
            .reset_index()
        )

        features = features.merge(
            meter_features,
            on="gateway_id",
            how="left",
        )

    # --------------------------------------------------------
    # Historical field visits
    # --------------------------------------------------------

    if not visits.empty:
        visits_copy = visits.copy()

        visits_copy["visited_on"] = pd.to_datetime(
            visits_copy["visited_on"],
            errors="coerce"
        )

        visit_cutoff = week_start.tz_localize(None)

        historical_visits = visits_copy.loc[
            visits_copy["visited_on"] < visit_cutoff
        ].copy()

        if not historical_visits.empty:
            visit_features = (
                historical_visits
                .groupby("gateway_id")
                .agg(
                    historical_visits=(
                        "visit_id",
                        "count"
                    ),
                    historical_fixed_errors=(
                        "outcome",
                        lambda x: (
                            x == "Fehler behoben"
                        ).sum()
                    ),
                    historical_no_error=(
                        "outcome",
                        lambda x: (
                            x == "Kein Fehler gefunden"
                        ).sum()
                    ),
                    historical_no_access=(
                        "outcome",
                        lambda x: (
                            x == "Kein Zugang"
                        ).sum()
                    ),
                )
                .reset_index()
            )

            features = features.merge(
                visit_features,
                on="gateway_id",
                how="left",
            )

    # --------------------------------------------------------
    # Gateway master information
    # --------------------------------------------------------

    if not master.empty:
        master_features = master[
            [
                "gateway_id",
                "tenant",
                "site_type",
                "region",
                "hw_model",
                "antenna_type",
                "fw_version",
                "n_meters_installed",
            ]
        ].copy()

        features = features.merge(
            master_features,
            on="gateway_id",
            how="left",
        )

    features["week_start"] = (
        week_start.tz_convert(None)
    )

    return features


def create_future_target(
    telemetry,
    meter,
    week_start,
):
    """Create a transparent future-week operational problem label.

    A gateway is labelled problematic if during the following 7 days:

      - meter-read success is below 80%, OR
      - total offline time is >= 12 hours, OR
      - total disconnections >= 12, OR
      - total reboots >= 12.

    The individual components are retained for interpretation.
    """

    week_start = pd.Timestamp(week_start, tz="UTC")
    target_end = week_start + pd.Timedelta(days=TARGET_DAYS)

    future_mask = (
        (telemetry["ts_utc"] >= week_start)
        & (telemetry["ts_utc"] < target_end)
    )

    future = telemetry.loc[future_mask].copy()

    if future.empty:
        return pd.DataFrame(
            columns=[
                "gateway_id",
                "target_problem",
                "target_offline_hours",
                "target_disconnections",
                "target_reboots",
                "target_meter_success",
            ]
        )

    target = (
        future
        .groupby("gateway_id")
        .agg(
            target_offline_hours=(
                "offline_duration_sec",
                lambda x: x.sum() / 3600.0
            ),
            target_disconnections=(
                "disconnection_cnt",
                "sum"
            ),
            target_reboots=(
                "reboot_cnt",
                "sum"
            ),
        )
        .reset_index()
    )

    # Meter-read success during the future week.
    meter_copy = meter.copy()

    meter_copy["week_start"] = pd.to_datetime(
        meter_copy["week_start"],
        errors="coerce"
    )

    future_start_naive = week_start.tz_localize(None)
    future_end_naive = (
        target_end.tz_localize(None)
    )

    meter_future = meter_copy.loc[
        (meter_copy["week_start"] >= future_start_naive)
        & (meter_copy["week_start"] < future_end_naive)
    ].copy()

    if not meter_future.empty:
        meter_future["read_success"] = safe_divide(
            meter_future["meters_read"],
            meter_future["meters_expected"],
        )

        meter_target = (
            meter_future
            .groupby("gateway_id")
            .agg(
                target_meter_success=(
                    "read_success",
                    "mean"
                )
            )
            .reset_index()
        )

        target = target.merge(
            meter_target,
            on="gateway_id",
            how="left",
        )
    else:
        target["target_meter_success"] = np.nan

    target["target_problem"] = (
        (target["target_offline_hours"] >= OFFLINE_HOURS_THRESHOLD)
        | (
            target["target_disconnections"]
            >= DISCONNECTION_THRESHOLD
        )
        | (
            target["target_reboots"]
            >= REBOOT_THRESHOLD
        )
        | (
            target["target_meter_success"]
            < METER_SUCCESS_THRESHOLD
        )
    ).astype(int)

    return target


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build weekly ML features for LPDG gateway ranking."
    )

    parser.add_argument(
        "--data",
        default="data",
        help="Path to the challenge data directory."
    )

    parser.add_argument(
        "--out",
        default="outputs",
        help="Output directory."
    )

    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("LPDG ML FEATURE BUILDER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load telemetry
    # --------------------------------------------------------

    print("\nLoading telemetry...")

    telemetry = pd.read_parquet(
        data_dir / "telemetry",
        columns=TELEMETRY_COLUMNS,
    )

    telemetry["gateway_id"] = (
        telemetry["gateway_id"]
        .map(normalize_gateway_id)
    )

    telemetry["ts_utc"] = pd.to_datetime(
        telemetry["ts_utc"],
        utc=True,
        errors="coerce",
    )

    telemetry = telemetry.dropna(
        subset=["gateway_id", "ts_utc"]
    )

    print(
        f"Telemetry rows: {len(telemetry):,}"
    )

    print(
        f"Telemetry gateways: "
        f"{telemetry['gateway_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Load meter reads
    # --------------------------------------------------------

    print("\nLoading meter-read success...")

    meter = pd.read_csv(
        data_dir / "meter_read_success.csv"
    )

    meter["gateway_id"] = (
        meter["gateway_id"]
        .map(normalize_gateway_id)
    )

    print(
        f"Meter rows: {len(meter):,}"
    )

    # --------------------------------------------------------
    # Load gateway master
    # --------------------------------------------------------

    print("\nLoading gateway master...")

    master = pd.read_csv(
        data_dir / "gateway_master.csv",
        encoding="latin1",
    )

    master["gateway_id"] = (
        master["gateway_id"]
        .map(normalize_gateway_id)
    )

    print(
        f"Master gateways: "
        f"{master['gateway_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Load field visits
    # --------------------------------------------------------

    print("\nLoading field visits...")

    visits = pd.read_csv(
        data_dir / "field_visits.csv",
        encoding="latin1",
    )

    visits["gateway_id"] = (
        visits["gateway_id"]
        .map(normalize_gateway_id)
    )

    print(
        f"Historical visits: {len(visits):,}"
    )

    # --------------------------------------------------------
    # Load engineer review
    # --------------------------------------------------------

    review_path = (
        data_dir /
        "engineer_review_2026-02.xlsx"
    )

    review = pd.read_excel(review_path)

    review["gateway_id"] = (
        review["gateway_id"]
        .map(normalize_gateway_id)
    )

    # --------------------------------------------------------
    # Verify gateway ID overlaps
    # --------------------------------------------------------

    telemetry_ids = set(
        telemetry["gateway_id"].dropna()
    )

    master_ids = set(
        master["gateway_id"].dropna()
    )

    visit_ids = set(
        visits["gateway_id"].dropna()
    )

    review_ids = set(
        review["gateway_id"].dropna()
    )

    overlap_report = pd.DataFrame(
        {
            "dataset": [
                "telemetry",
                "gateway_master",
                "field_visits",
                "engineer_review",
            ],
            "unique_gateways": [
                len(telemetry_ids),
                len(master_ids),
                len(visit_ids),
                len(review_ids),
            ],
            "overlap_with_telemetry": [
                len(telemetry_ids),
                len(telemetry_ids & master_ids),
                len(telemetry_ids & visit_ids),
                len(telemetry_ids & review_ids),
            ],
        }
    )

    print("\nGateway ID overlap after normalization:")
    print(
        overlap_report.to_string(index=False)
    )

    overlap_report.to_csv(
        out_dir / "gateway_overlap_report.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Build weekly samples
    # --------------------------------------------------------

    # Start earlier than the scored period so the model has
    # historical training examples.
    training_weeks = pd.date_range(
        "2025-09-01",
        "2026-03-23",
        freq="7D",
    )

    all_samples = []

    print(
        f"\nBuilding {len(training_weeks)} weekly samples..."
    )

    for i, week in enumerate(training_weeks, start=1):

        print(
            f"[{i:02d}/{len(training_weeks)}] "
            f"{week.date()}",
            end="",
            flush=True,
        )

        features = build_window_features(
            telemetry=telemetry,
            meter=meter,
            master=master,
            visits=visits,
            week_start=week,
        )

        target = create_future_target(
            telemetry=telemetry,
            meter=meter,
            week_start=week,
        )

        samples = features.merge(
            target,
            on="gateway_id",
            how="left",
        )

        samples["week_start"] = week

        all_samples.append(samples)

        print(
            f" -> {len(samples)} gateways"
        )

    dataset = pd.concat(
        all_samples,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Clean feature table
    # --------------------------------------------------------

    numeric_columns = dataset.select_dtypes(
        include=[np.number]
    ).columns

    dataset[numeric_columns] = (
        dataset[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    # Missing numeric values are expected for some gateways,
    # especially new gateways with limited history.
    for column in numeric_columns:
        if column.startswith("target_"):
            continue

        dataset[column] = dataset[column].fillna(0)

    categorical_columns = [
        "tenant",
        "site_type",
        "region",
        "hw_model",
        "antenna_type",
        "fw_version",
    ]

    for column in categorical_columns:
        if column in dataset.columns:
            dataset[column] = (
                dataset[column]
                .fillna("UNKNOWN")
                .astype(str)
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        out_dir /
        "ml_gateway_week_features.csv"
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE DATASET COMPLETE")
    print("=" * 70)

    print(
        f"Rows: {len(dataset):,}"
    )

    print(
        f"Columns: {len(dataset.columns)}"
    )

    print(
        f"Gateways: "
        f"{dataset['gateway_id'].nunique()}"
    )

    print(
        f"Weeks: "
        f"{dataset['week_start'].nunique()}"
    )

    print("\nTarget distribution:")

    if "target_problem" in dataset.columns:
        print(
            dataset["target_problem"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

        print(
            "\nTarget rate: "
            f"{dataset['target_problem'].mean():.3f}"
        )

    print(
        f"\nSaved: {output_path}"
    )

    print(
        f"Saved: "
        f"{out_dir / 'gateway_overlap_report.csv'}"
    )


if __name__ == "__main__":
    main()
