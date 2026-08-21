from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECASTING_FILE = PROJECT_ROOT / "data" / "processed" / "forecasting_panel.csv"
SPECIFICATION_FILE = (
    PROJECT_ROOT / "outputs" / "model_results" / "selected_model_specifications.csv"
)

TARGET_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Trading & Exchanges",
]
EXPECTED_SPECIFICATIONS = {
    "Historical quantile": 52,
    "Time-series QR": 4,
    "Network QR": 4,
}
EXPECTED_ESTIMATION_OBSERVATIONS = 980
EXPECTED_TEST_OBSERVATIONS = 132


def main():
    for input_file in [FORECASTING_FILE, SPECIFICATION_FILE]:
        if not input_file.exists():
            raise FileNotFoundError(f"Required input not found: {input_file}")

    panel = pd.read_csv(FORECASTING_FILE)
    specifications = pd.read_csv(SPECIFICATION_FILE)

    required_specification_columns = {"model", "hyperparameter_value"}
    if not required_specification_columns.issubset(specifications.columns):
        raise ValueError("The selected-specification file has missing columns.")

    selected_values = specifications.set_index("model")[
        "hyperparameter_value"
    ].astype(int).to_dict()
    if selected_values != EXPECTED_SPECIFICATIONS:
        raise ValueError(f"Unexpected selected specifications: {selected_values}")

    model_columns = [
        "date",
        "forecast_date",
        "category",
        "sample",
        "target_contraction_next_week",
        "log_activity",
        "network_pressure",
        "connection_concentration",
        *[f"contraction_lag_{lag}" for lag in range(1, 5)],
    ]
    missing_columns = sorted(set(model_columns).difference(panel.columns))
    if missing_columns:
        raise ValueError(f"Missing forecasting columns: {missing_columns}")

    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel["forecast_date"] = pd.to_datetime(panel["forecast_date"], errors="raise")
    data = panel.loc[panel["category"].isin(TARGET_CATEGORIES)].copy()
    data = data.dropna(subset=model_columns)

    estimation_data = data.loc[data["sample"].isin(["training", "validation"])]
    test_data = data.loc[data["sample"].eq("test")]

    if len(estimation_data) != EXPECTED_ESTIMATION_OBSERVATIONS:
        raise ValueError(f"Unexpected estimation sample size: {len(estimation_data)}")
    if len(test_data) != EXPECTED_TEST_OBSERVATIONS:
        raise ValueError(f"Unexpected test sample size: {len(test_data)}")
    if estimation_data["forecast_date"].max() >= test_data["forecast_date"].min():
        raise ValueError("The estimation and test periods overlap.")

    print(
        "Final selected m=52, p_TS=4, p_Net=4; "
        "980 estimation and 132 test observations."
    )


if __name__ == "__main__":
    main()