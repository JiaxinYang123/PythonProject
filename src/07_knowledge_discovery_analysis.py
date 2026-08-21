from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "forecasting_panel.csv"
)
ACTIVITY_NETWORK_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "sector_activity_network_summary.csv"
)
TAIL_COMPARISON_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "sector_tail_state_comparison.csv"
)

TARGET_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Trading & Exchanges",
]

ALPHA = 0.90
EXPECTED_VALID_WEEKS = 282


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"File not found: {INPUT_FILE}")

    panel = pd.read_csv(INPUT_FILE)

    required_columns = [
        "date",
        "category",
        "activity",
        "contraction_score",
        "network_pressure",
        "connection_concentration",
        "network_degree",
    ]
    missing_columns = sorted(set(required_columns) - set(panel.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    for column in required_columns[2:]:
        panel[column] = pd.to_numeric(panel[column], errors="raise")

    target_panel = panel[
        panel["category"].isin(TARGET_CATEGORIES)
    ].copy()

    if set(target_panel["category"].unique()) != set(TARGET_CATEGORIES):
        raise ValueError("One or more forecasting categories are missing.")
    if target_panel.duplicated(["date", "category"]).any():
        raise ValueError("Duplicated category-week observations were found.")

    # The first zero-activity week has no defined contraction score.
    valid_panel = target_panel[
        target_panel["contraction_score"].notna()
    ].copy()

    valid_counts = (
        valid_panel.groupby("category").size().reindex(TARGET_CATEGORIES)
    )
    if not (valid_counts == EXPECTED_VALID_WEEKS).all():
        raise ValueError(
            f"Unexpected valid observations: {valid_counts.to_dict()}"
        )

    analysis_columns = [
        "activity",
        "network_pressure",
        "connection_concentration",
        "network_degree",
    ]
    if valid_panel[analysis_columns].isna().any().any():
        raise ValueError("Missing activity or network values in the valid sample.")
    if (valid_panel["activity"] < 0).any():
        raise ValueError("Exposure activity must be non-negative.")
    if not valid_panel["contraction_score"].between(-1, 1).all():
        raise ValueError("Contraction scores must lie within [-1, 1].")
    if not valid_panel["network_pressure"].between(-1, 1).all():
        raise ValueError("Network pressure must lie within [-1, 1].")
    if not valid_panel["connection_concentration"].between(0, 1).all():
        raise ValueError("Connection concentration must lie within [0, 1].")
    if not valid_panel["network_degree"].between(0, 6).all():
        raise ValueError("Network degree must lie within [0, 6].")

    # Equation (3.16) reports raw exposure activity in billions of dollars.
    valid_panel["activity_billion"] = valid_panel["activity"] / 1_000_000_000


    #Table: exposure activity and network characteristics


    activity_network_rows = []

    for category in TARGET_CATEGORIES:
        category_data = valid_panel[
            valid_panel["category"] == category
        ]

        activity_network_rows.append(
            {
                "category": category,
                "valid_observations": len(category_data),
                "median_activity_billion":
                    category_data["activity_billion"].median(),
                "quantile_90_activity_billion":
                    category_data["activity_billion"].quantile(ALPHA),
                "mean_network_pressure":
                    category_data["network_pressure"].mean(),
                "standard_deviation_network_pressure":
                    category_data["network_pressure"].std(ddof=1),
                "mean_connection_concentration":
                    category_data["connection_concentration"].mean(),
                "mean_network_degree":
                    category_data["network_degree"].mean(),
            }
        )

    activity_network_summary = pd.DataFrame(activity_network_rows).round(6)

    # Table: network conditions in empirical upper-tail states
    # This full-sample classification is descriptive only and is not used
    # to train, select or evaluate the forecasting models.

    tail_comparison_rows = []

    for category in TARGET_CATEGORIES:
        category_data = valid_panel[
            valid_panel["category"] == category
        ].copy()

        tail_threshold = float(
            category_data["contraction_score"].quantile(ALPHA)
        )
        tail_data = category_data[
            category_data["contraction_score"] >= tail_threshold
        ]
        non_tail_data = category_data[
            category_data["contraction_score"] < tail_threshold
        ]

        tail_median_activity = tail_data["activity_billion"].median()
        non_tail_median_activity = non_tail_data["activity_billion"].median()
        if non_tail_median_activity <= 0:
            raise ValueError(
                f"Non-tail median activity is not positive for {category}."
            )

        tail_mean_pressure = tail_data["network_pressure"].mean()
        non_tail_mean_pressure = non_tail_data["network_pressure"].mean()
        tail_mean_concentration = tail_data[
            "connection_concentration"
        ].mean()
        non_tail_mean_concentration = non_tail_data[
            "connection_concentration"
        ].mean()

        tail_comparison_rows.append(
            {
                "category": category,
                "tail_threshold": tail_threshold,
                "tail_weeks": len(tail_data),
                "non_tail_weeks": len(non_tail_data),
                "tail_median_activity_billion": tail_median_activity,
                "non_tail_median_activity_billion": non_tail_median_activity,
                "tail_to_non_tail_activity_ratio":
                    tail_median_activity / non_tail_median_activity,
                "tail_mean_network_pressure": tail_mean_pressure,
                "non_tail_mean_network_pressure": non_tail_mean_pressure,
                "network_pressure_difference":
                    tail_mean_pressure - non_tail_mean_pressure,
                "tail_mean_concentration": tail_mean_concentration,
                "non_tail_mean_concentration": non_tail_mean_concentration,
                "concentration_difference":
                    tail_mean_concentration - non_tail_mean_concentration,
            }
        )

    tail_comparison = pd.DataFrame(tail_comparison_rows).round(6)

    if not (tail_comparison["tail_weeks"] == 29).all():
        raise ValueError("Unexpected number of empirical upper-tail weeks.")

    ACTIVITY_NETWORK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    activity_network_summary.to_csv(
        ACTIVITY_NETWORK_OUTPUT,
        index=False,
    )
    tail_comparison.to_csv(
        TAIL_COMPARISON_OUTPUT,
        index=False,
    )

    print("Knowledge-discovery tables created successfully.")
    print(f"Table 3 data: {ACTIVITY_NETWORK_OUTPUT}")
    print(f"Table 4 data: {TAIL_COMPARISON_OUTPUT}")
    print("Valid observations: 282 weeks per sector; upper-tail weeks: 29.")


if __name__ == "__main__":
    main()