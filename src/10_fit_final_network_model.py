from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECASTING_FILE = PROJECT_ROOT / "data" / "processed" / "forecasting_panel.csv"
SPECIFICATION_FILE = (
    PROJECT_ROOT / "outputs" / "model_results" / "selected_model_specifications.csv"
)
PREDICTION_OUTPUT = (
    PROJECT_ROOT / "outputs" / "model_results" / "final_network_test_predictions.csv"
)

ALPHA = 0.90
MAX_ITERATIONS = 10_000
PARAMETER_TOLERANCE = 1e-7
EXPECTED_ESTIMATION_OBSERVATIONS = 980
EXPECTED_TEST_OBSERVATIONS = 132
EXPECTED_TEST_WEEKS = 33

TARGET_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Trading & Exchanges",
]


def pinball_loss(actual, forecast):
    """Calculate the 90th-quantile Pinball loss in Equation 4.17."""
    error = np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float)
    return np.where(
        error >= 0,
        ALPHA * error,
        (1.0 - ALPHA) * (-error),
    )


def construct_design_matrix(dataframe, continuous_predictors):
    """Construct Equation 4.14 with Asset Management."""
    design = dataframe[continuous_predictors].astype(float).copy()

    for category in TARGET_CATEGORIES[1:]:
        design[f"category_effect__{category}"] = (
            dataframe["category"].eq(category).astype(float)
        )

    design.insert(0, "const", 1.0)
    if not np.isfinite(design.to_numpy()).all():
        raise ValueError("The design matrix contains non-finite values.")
    return design


def main():
    for input_file in [FORECASTING_FILE, SPECIFICATION_FILE]:
        if not input_file.exists():
            raise FileNotFoundError(f"Required input not found: {input_file}")

    panel = pd.read_csv(FORECASTING_FILE)
    specifications = pd.read_csv(SPECIFICATION_FILE)

    if not {"model", "hyperparameter_value"}.issubset(specifications.columns):
        raise ValueError("The selected-specification file has missing columns.")

    network_specification = specifications.loc[
        specifications["model"].eq("Network QR")
    ]
    if len(network_specification) != 1:
        raise ValueError("Exactly one Network QR specification is required.")

    selected_lag_order = int(
        network_specification.iloc[0]["hyperparameter_value"]
    )
    if selected_lag_order != 4:
        raise ValueError(f"Unexpected Network QR lag order: {selected_lag_order}")

    lag_columns = [
        f"contraction_lag_{lag}" for lag in range(1, selected_lag_order + 1)
    ]
    continuous_predictors = lag_columns + [
        "log_activity",
        "network_pressure",
        "connection_concentration",
    ]
    response_column = "target_contraction_next_week"
    required_columns = [
        "date",
        "forecast_date",
        "category",
        "sample",
        response_column,
        *continuous_predictors,
    ]
    missing_columns = sorted(set(required_columns).difference(panel.columns))
    if missing_columns:
        raise ValueError(f"Missing forecasting columns: {missing_columns}")

    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel["forecast_date"] = pd.to_datetime(panel["forecast_date"], errors="raise")
    for column in [response_column, *continuous_predictors]:
        panel[column] = pd.to_numeric(panel[column], errors="raise")

    data = panel.loc[panel["category"].isin(TARGET_CATEGORIES)].copy()
    data = data.sort_values(["forecast_date", "category"]).reset_index(drop=True)
    if data.duplicated(["date", "category"]).any():
        raise ValueError("The forecasting panel contains duplicate sector-weeks.")

    estimation_data = data.loc[
        data["sample"].isin(["training", "validation"])
    ].dropna(subset=[response_column, *continuous_predictors]).copy()
    test_data = data.loc[data["sample"].eq("test")].dropna(
        subset=[response_column, *continuous_predictors]
    ).copy()

    if len(estimation_data) != EXPECTED_ESTIMATION_OBSERVATIONS:
        raise ValueError(f"Unexpected estimation sample size: {len(estimation_data)}")
    if len(test_data) != EXPECTED_TEST_OBSERVATIONS:
        raise ValueError(f"Unexpected test sample size: {len(test_data)}")
    if (
        test_data["category"].nunique() != len(TARGET_CATEGORIES)
        or test_data["forecast_date"].nunique() != EXPECTED_TEST_WEEKS
        or not test_data.groupby("category").size().eq(EXPECTED_TEST_WEEKS).all()
    ):
        raise ValueError("The test panel must contain 33 weeks for each sector.")
    if test_data.duplicated(["forecast_date", "category"]).any():
        raise ValueError("The test panel contains duplicate forecasts.")
    if estimation_data["forecast_date"].max() >= test_data["forecast_date"].min():
        raise ValueError("The estimation and test periods overlap.")

    estimation_design = construct_design_matrix(
        estimation_data, continuous_predictors
    )
    test_design = construct_design_matrix(test_data, continuous_predictors)
    response = estimation_data[response_column].astype(float)

    if np.linalg.matrix_rank(estimation_design.to_numpy()) != estimation_design.shape[1]:
        raise ValueError("The final design matrix is not full rank.")

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        result = QuantReg(response, estimation_design).fit(
            q=ALPHA,
            max_iter=MAX_ITERATIONS,
            p_tol=PARAMETER_TOLERANCE,
        )

    if captured_warnings:
        warning_text = " | ".join(str(item.message) for item in captured_warnings)
        raise RuntimeError(f"Network QR estimation warning: {warning_text}")
    if not np.isfinite(np.asarray(result.params, dtype=float)).all():
        raise RuntimeError("The fitted network coefficients are not finite.")

    predicted_quantile = np.asarray(result.predict(test_design), dtype=float)
    if not np.isfinite(predicted_quantile).all():
        raise RuntimeError("The test forecasts are not finite.")

    actual = test_data[response_column].to_numpy(dtype=float)
    losses = pinball_loss(actual, predicted_quantile)
    predictions = test_data[["date", "forecast_date", "category", response_column]].copy()
    predictions = predictions.rename(
        columns={
            "date": "forecast_origin_date",
            response_column: "actual_contraction_score",
        }
    )
    predictions["predicted_quantile_90"] = predicted_quantile
    predictions["pinball_loss"] = losses
    predictions = predictions.sort_values(
        ["forecast_date", "category"]
    ).reset_index(drop=True)

    PREDICTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTION_OUTPUT, index=False)

    print("Final network model fitted successfully.")
    print(f"Output: {PREDICTION_OUTPUT}")
    print(
        f"Estimation observations: {len(estimation_data)}; "
        f"test observations: {len(test_data)}."
    )
    print(
        f"Mean Pinball loss: {losses.mean():.6f}; "
        f"median Pinball loss: {np.median(losses):.6f}."
    )


if __name__ == "__main__":
    main()