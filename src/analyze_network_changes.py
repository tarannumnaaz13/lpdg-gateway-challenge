import pandas as pd
from pathlib import Path

OUT = Path("outputs")

df = pd.read_csv(
    OUT / "ml_gateway_week_features.csv"
)

df["week_start"] = pd.to_datetime(
    df["week_start"]
)

print()
print("=" * 70)
print("NETWORK CHANGE ANALYSIS")
print("=" * 70)


# ---------------------------------------------------------
# NETWORK-RELATED FEATURES
# ---------------------------------------------------------

network_features = [
    "recent7_disconnections",
    "recent28_disconnections",
    "previous21_disconnections",
    "recent7_disconnection_hours",
    "recent28_disconnection_hours",
    "previous21_disconnection_hours",
    "recent7_no_conn_importance_mean",
    "recent28_no_conn_importance_mean",
    "previous21_no_conn_importance_mean",
    "recent7_offline_hours",
    "recent28_offline_hours",
    "previous21_offline_hours",
    "recent7_coverage_ratio",
    "recent28_coverage_ratio",
    "recent7_rx_packets",
]

available = [
    c for c in network_features
    if c in df.columns
]

print()
print("Available network features:")
for feature in available:
    print("-", feature)


# ---------------------------------------------------------
# COMPARE NORMAL VS PROBLEM GATEWAYS
# ---------------------------------------------------------

labeled = df[
    df["target_problem"].notna()
].copy()

labeled["target_problem"] = (
    labeled["target_problem"].astype(int)
)

print()
print("=" * 70)
print("NORMAL VS PROBLEM")
print("=" * 70)

comparison = (
    labeled
    .groupby("target_problem")[available]
    .mean()
    .T
)

comparison.columns = [
    "Normal",
    "Problem"
]

comparison["Problem_vs_Normal"] = (
    comparison["Problem"] /
    comparison["Normal"].replace(0, float("nan"))
)

print()
print(comparison.to_string())


# ---------------------------------------------------------
# RECENT CHANGE / TREND
# ---------------------------------------------------------

trend_features = [
    "trend_disconnections",
    "trend_disconnection_hours",
    "trend_offline_hours",
    "trend_no_conn_importance",
]

trend_available = [
    c for c in trend_features
    if c in df.columns
]

print()
print("=" * 70)
print("TREND FEATURES")
print("=" * 70)

if trend_available:

    trend_summary = (
        labeled
        .groupby("target_problem")[trend_available]
        .mean()
        .T
    )

    trend_summary.columns = [
        "Normal",
        "Problem"
    ]

    print()
    print(trend_summary.to_string())

else:
    print("No trend features found.")


# ---------------------------------------------------------
# LATEST WEEKS
# ---------------------------------------------------------

latest_week = (
    labeled["week_start"].max()
)

recent = labeled[
    labeled["week_start"] >=
    latest_week - pd.Timedelta(days=28)
]

print()
print("=" * 70)
print("RECENT 4-WEEK NETWORK CONDITIONS")
print("=" * 70)

recent_summary = (
    recent
    .groupby("target_problem")[available]
    .mean()
    .T
)

recent_summary.columns = [
    "Normal",
    "Problem"
]

print()
print(recent_summary.to_string())


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

comparison.to_csv(
    OUT / "network_normal_vs_problem.csv"
)

if trend_available:
    trend_summary.to_csv(
        OUT / "network_trend_analysis.csv"
    )

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print()
print("Saved:")
print("outputs/network_normal_vs_problem.csv")

if trend_available:
    print("outputs/network_trend_analysis.csv")