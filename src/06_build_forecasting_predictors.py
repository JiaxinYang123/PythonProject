from pathlib import Path

import numpy as np
import pandas as pd

#path and setting


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MATRIX_FILE = (
    PROJECT_ROOT / "data" / "processed" / "sector_exposure_matrix.csv"
)
PANEL_FILE = (
    PROJECT_ROOT / "data" / "processed" / "sector_week_panel.csv"
)
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "forecasting_panel.csv"
)

ANALYSIS_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Privacy & Security",
    "Trading & Exchanges",
    "Primary Market Tokens",
    "Other / Unknown",
]

TARGET_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Trading & Exchanges",
]

MAX_LAG = 4

TRAINING_END = pd.Timestamp("2023-12-25")
VALIDATION_START = pd.Timestamp("2024-01-01")
VALIDATION_END = pd.Timestamp("2024-12-30")
TEST_START = pd.Timestamp("2025-01-06")
TEST_END = pd.Timestamp("2025-08-18")


def require_columns(dataframe, required_columns, table_name):
    """Check input table contains the required columns."""

    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}"
        )


def assign_sample(forecast_date):
    """Assign samples using the date of the one-week-ahead outcome."""

    if pd.isna(forecast_date):
        return "unavailable"
    if forecast_date <= TRAINING_END:
        return "training"
    if VALIDATION_START <= forecast_date <= VALIDATION_END:
        return "validation"
    if TEST_START <= forecast_date <= TEST_END:
        return "test"
    return "outside_sample"

#Main analysis


def main():
    if not MATRIX_FILE.exists():
        raise FileNotFoundError(f"File not found: {MATRIX_FILE}")
    if not PANEL_FILE.exists():
        raise FileNotFoundError(f"File not found: {PANEL_FILE}")

    sector_matrix = pd.read_csv(MATRIX_FILE)
    sector_panel = pd.read_csv(PANEL_FILE)

    require_columns(
        sector_matrix,
        ["date", "source_category", "target_category", "exposure"],
        "Sector exposure matrix",
    )
    require_columns(
        sector_panel,
        ["date", "category", "log_activity", "contraction_score"],
        "Sector-week panel",
    )

    sector_matrix["date"] = pd.to_datetime(
        sector_matrix["date"], errors="raise"
    )
    sector_panel["date"] = pd.to_datetime(
        sector_panel["date"], errors="raise"
    )
    sector_matrix["exposure"] = pd.to_numeric(
        sector_matrix["exposure"], errors="raise"
    )
    sector_panel["log_activity"] = pd.to_numeric(
        sector_panel["log_activity"], errors="raise"
    )
    sector_panel["contraction_score"] = pd.to_numeric(
        sector_panel["contraction_score"], errors="raise"
    )

    if sector_matrix.duplicated(
        ["date", "source_category", "target_category"]
    ).any():
        raise ValueError("Duplicated rows in the sector exposure matrix.")

    if sector_panel.duplicated(["date", "category"]).any():
        raise ValueError("Duplicated rows in the sector-week panel.")

    if (sector_matrix["exposure"] < 0).any():
        raise ValueError("Sector exposure values must be non-negative.")

    observed_categories = set(sector_panel["category"].dropna().unique())
    if observed_categories != set(ANALYSIS_CATEGORIES):
        raise ValueError(
            "The sector-week panel does not contain the seven expected categories."
        )

    matrix_dates = set(sector_matrix["date"].unique())
    panel_dates = set(sector_panel["date"].unique())
    if matrix_dates != panel_dates:
        raise ValueError("The matrix and panel contain different dates.")

    dates = pd.DatetimeIndex(
        sorted(pd.Timestamp(date) for date in panel_dates)
    )
    weekly_gaps = dates.to_series().diff().dropna()
    if not (weekly_gaps == pd.Timedelta(days=7)).all():
        raise ValueError("The observations are not consecutive weekly snapshots.")

    expected_matrix_rows = len(dates) * len(ANALYSIS_CATEGORIES) ** 2
    expected_panel_rows = len(dates) * len(ANALYSIS_CATEGORIES)
    if len(sector_matrix) != expected_matrix_rows:
        raise ValueError("The sector exposure matrix is not complete.")
    if len(sector_panel) != expected_panel_rows:
        raise ValueError("The sector-week panel is not complete.")

    # Construct network pressure and connection concentration


    network_rows = []

    for date in dates:
        matrix_week = sector_matrix[sector_matrix["date"] == date]
        panel_week = sector_panel[sector_panel["date"] == date]

        exposure_matrix = matrix_week.pivot(
            index="source_category",
            columns="target_category",
            values="exposure",
        ).reindex(
            index=ANALYSIS_CATEGORIES,
            columns=ANALYSIS_CATEGORIES,
        )

        if exposure_matrix.isna().any().any():
            raise ValueError(f"Incomplete sector matrix for {date.date()}.")

        # Equation: B_ij,t = S_ij,t + S_ji,t, with B_ii,t = 0.
        bilateral_matrix = (exposure_matrix + exposure_matrix.T).copy()
        for category in ANALYSIS_CATEGORIES:
            bilateral_matrix.at[category, category] = 0.0

        contraction_scores = (
            panel_week.set_index("category")["contraction_score"]
            .reindex(ANALYSIS_CATEGORIES)
        )

        for category in ANALYSIS_CATEGORIES:
            bilateral_strengths = bilateral_matrix.loc[category]
            total_connection = float(bilateral_strengths.sum())
            active_neighbours = bilateral_strengths > 0
            network_degree = int(active_neighbours.sum())

            if total_connection > 0:
                # Equation (4.6): normalised bilateral connection shares.
                weights = bilateral_strengths / total_connection
                active_scores = contraction_scores[active_neighbours]

                # Equation (4.7): exposure-weighted neighbour contraction.
                if active_scores.isna().any():
                    network_pressure = np.nan
                else:
                    network_pressure = float(
                        np.sum(
                            weights[active_neighbours].to_numpy()
                            * active_scores.to_numpy()
                        )
                    )

                # Equation (4.8): HHI of the connection shares.
                connection_concentration = float(
                    np.sum(np.square(weights.to_numpy()))
                )
            else:
                network_pressure = np.nan
                connection_concentration = np.nan

            network_rows.append(
                {
                    "date": date,
                    "category": category,
                    "network_degree": network_degree,
                    "network_pressure": network_pressure,
                    "connection_concentration": connection_concentration,
                }
            )

    network_features = pd.DataFrame(network_rows)

    valid_pressure = network_features["network_pressure"].dropna()
    valid_concentration = network_features[
        "connection_concentration"
    ].dropna()
    if not valid_pressure.between(-1, 1).all():
        raise ValueError("Network pressure is outside [-1, 1].")
    if not valid_concentration.between(0, 1).all():
        raise ValueError("Connection concentration is outside [0, 1].")

    # Construct the one-week-ahead target and autoregressive predictors


    forecasting_panel = sector_panel.merge(
        network_features,
        on=["date", "category"],
        how="left",
        validate="one_to_one",
    ).sort_values(["category", "date"]).reset_index(drop=True)

    category_groups = forecasting_panel.groupby("category", sort=False)

    forecasting_panel["forecast_date"] = category_groups["date"].shift(-1)
    forecasting_panel["target_contraction_next_week"] = (
        category_groups["contraction_score"].shift(-1)
    )

    # Section : lag 1 is Y_i,t and lag 4 is Y_i,t-3.
    for lag_number in range(1, MAX_LAG + 1):
        forecasting_panel[f"contraction_lag_{lag_number}"] = (
            category_groups["contraction_score"].shift(lag_number - 1)
        )

    valid_forecast_dates = forecasting_panel["forecast_date"].notna()
    forecast_gaps = (
        forecasting_panel.loc[valid_forecast_dates, "forecast_date"]
        - forecasting_panel.loc[valid_forecast_dates, "date"]
    )
    if not (forecast_gaps == pd.Timedelta(days=7)).all():
        raise ValueError("The response is not aligned exactly one week ahead.")

    forecasting_panel["sample"] = forecasting_panel["forecast_date"].apply(
        assign_sample
    )

    # Confirm the complete-case sample sizes reported in Section 5.1.
    model_columns = [
        "target_contraction_next_week",
        "log_activity",
        "network_pressure",
        "connection_concentration",
        *[f"contraction_lag_{lag}" for lag in range(1, MAX_LAG + 1)],
    ]

    complete_cases = forecasting_panel[
        forecasting_panel["category"].isin(TARGET_CATEGORIES)
        & forecasting_panel[model_columns].notna().all(axis=1)
    ]

    sample_counts = complete_cases.groupby("sample").size().to_dict()
    expected_counts = {
        "training": 768,
        "validation": 212,
        "test": 132,
    }
    if any(
        sample_counts.get(sample, 0) != count
        for sample, count in expected_counts.items()
    ):
        raise ValueError(
            f"Unexpected complete-case sample counts: {sample_counts}"
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecasting_panel.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("Forecasting predictors constructed successfully.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Rows: {len(forecasting_panel):,}")
    print("Complete p=4 observations: 980 training-validation and 132 test.")


if __name__ == "__main__":
    main()