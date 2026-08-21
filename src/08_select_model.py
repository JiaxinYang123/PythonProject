from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.regression.quantile_regression import QuantReg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "forecasting_panel.csv"
RESULT_DIRECTORY = PROJECT_ROOT / "outputs" / "model_results"
SELECTION_RESULTS_OUTPUT = RESULT_DIRECTORY / "validation_model_selection.csv"
SELECTED_SPECIFICATIONS_OUTPUT = (
    RESULT_DIRECTORY / "selected_model_specifications.csv"
)

ALPHA = 0.90
LAG_CANDIDATES = [1, 2, 4]
HISTORICAL_WINDOW_CANDIDATES = [4, 8, 13, 26, 52]
TARGET_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Trading & Exchanges",
]
REFERENCE_CATEGORY = "Asset Management"
MAX_ITERATIONS = 10_000
PARAMETER_TOLERANCE = 1e-8
EXPECTED_VALIDATION_OBSERVATIONS = 212


def pinball_loss(actual, prediction):
    """Pinball loss at the 90th quantile (Equation 4.17)."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = actual - prediction
    return np.where(error >= 0, ALPHA * error, (1 - ALPHA) * (-error))


def calculate_forecast_metrics(actual, prediction):
    """Return the two validation measures reported in Table ."""
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)

    if len(actual) != len(prediction) or not (
        np.isfinite(actual).all() and np.isfinite(prediction).all()
    ):
        raise ValueError("Validation values must be finite and equally sized.")

    losses = pinball_loss(actual, prediction)
    return {
        "validation_observations": len(actual),
        "mean_pinball_loss": float(losses.mean()),
        "exceedance_rate": float(np.mean(actual > prediction)),
    }


def construct_design_matrix(dataframe, lag_order, include_network):
    """Construct the time-series or network QR design matrix.
    This part utilized AI-assisted proofreading."""
    design = pd.DataFrame(index=dataframe.index)
    design["intercept"] = 1.0

    for category in TARGET_CATEGORIES:
        if category == REFERENCE_CATEGORY:
            continue
        dummy_name = (
            "fixed_effect_"
            + category.lower()
            .replace("&", "and")
            .replace(",", "")
            .replace("/", "_")
            .replace(" ", "_")
        )
        design[dummy_name] = dataframe["category"].eq(category).astype(float)

    # Lagged contraction terms in Equations 4.13 and 4.14.
    for lag_number in range(1, lag_order + 1):
        column = f"contraction_lag_{lag_number}"
        design[column] = pd.to_numeric(dataframe[column], errors="raise")

    # Current log exposure activity, \Equation 3.17.
    design["log_activity"] = pd.to_numeric(
        dataframe["log_activity"], errors="raise"
    )

    if include_network:
        # The additional network conditions in Equation 4.14.
        for column in ["network_pressure", "connection_concentration"]:
            design[column] = pd.to_numeric(dataframe[column], errors="raise")

    design = design.astype(float)
    if not np.isfinite(design.to_numpy()).all():
        raise ValueError("The model design matrix contains non-finite values.")
    return design


def estimate_quantile_candidate(panel, model_name, lag_order, include_network):
    """Fit on training observations and evaluate on validation observations."""
    required_columns = [
        "target_contraction_next_week",
        "log_activity",
        *[
            f"contraction_lag_{lag_number}"
            for lag_number in range(1, lag_order + 1)
        ],
    ]
    if include_network:
        required_columns += ["network_pressure", "connection_concentration"]

    training_data = panel.loc[
        panel["sample"].eq("training")
        & panel["category"].isin(TARGET_CATEGORIES)
    ].dropna(subset=required_columns)
    validation_data = panel.loc[
        panel["sample"].eq("validation")
        & panel["category"].isin(TARGET_CATEGORIES)
    ].dropna(subset=required_columns)

    if training_data.empty:
        raise ValueError(f"No training observations for {model_name}, p={lag_order}.")
    if len(validation_data) != EXPECTED_VALIDATION_OBSERVATIONS:
        raise ValueError(
            f"Unexpected validation count for {model_name}, p={lag_order}: "
            f"{len(validation_data)}."
        )

    training_design = construct_design_matrix(
        training_data, lag_order, include_network
    )
    validation_design = construct_design_matrix(
        validation_data, lag_order, include_network
    ).reindex(columns=training_design.columns)
    training_response = pd.to_numeric(
        training_data["target_contraction_next_week"], errors="raise"
    ).astype(float)
    validation_response = pd.to_numeric(
        validation_data["target_contraction_next_week"], errors="raise"
    ).astype(float)

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        fitted_model = QuantReg(training_response, training_design).fit(
            q=ALPHA,
            max_iter=MAX_ITERATIONS,
            p_tol=PARAMETER_TOLERANCE,
        )

    if captured_warnings:
        warning_text = " | ".join(str(item.message) for item in captured_warnings)
        raise RuntimeError(f"{model_name}, p={lag_order}: {warning_text}")
    if not np.isfinite(fitted_model.params.to_numpy()).all():
        raise RuntimeError(f"Non-finite coefficients for {model_name}, p={lag_order}.")

    prediction = fitted_model.predict(validation_design).to_numpy(dtype=float)
    metrics = calculate_forecast_metrics(validation_response, prediction)
    return {
        "model": model_name,
        "hyperparameter": "p",
        "hyperparameter_value": lag_order,
        "training_observations": len(training_data),
        **metrics,
    }


def evaluate_historical_candidate(panel, window_length):
    """Rolling historical quantile."""
    working_panel = panel.copy()
    prediction_column = f"historical_quantile_m_{window_length}"
    working_panel[prediction_column] = (
        working_panel.groupby("category", sort=False)["contraction_score"]
        .transform(
            lambda series: series.rolling(
                window=window_length,
                min_periods=window_length,
            ).quantile(ALPHA)
        )
    )

    validation_data = working_panel.loc[
        working_panel["sample"].eq("validation")
        & working_panel["category"].isin(TARGET_CATEGORIES)
    ].dropna(subset=["target_contraction_next_week", prediction_column])

    if len(validation_data) != EXPECTED_VALIDATION_OBSERVATIONS:
        raise ValueError(
            f"Unexpected validation count for historical m={window_length}: "
            f"{len(validation_data)}."
        )

    actual = pd.to_numeric(
        validation_data["target_contraction_next_week"], errors="raise"
    ).to_numpy(dtype=float)
    prediction = pd.to_numeric(
        validation_data[prediction_column], errors="raise"
    ).to_numpy(dtype=float)
    metrics = calculate_forecast_metrics(actual, prediction)
    return {
        "model": "Historical quantile",
        "hyperparameter": "m",
        "hyperparameter_value": window_length,
        "training_observations": np.nan,
        **metrics,
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Forecasting panel not found: {INPUT_FILE}")

    panel = pd.read_csv(INPUT_FILE)
    required_columns = {
        "date",
        "forecast_date",
        "category",
        "sample",
        "contraction_score",
        "target_contraction_next_week",
        "log_activity",
        "network_pressure",
        "connection_concentration",
        *[f"contraction_lag_{lag}" for lag in range(1, 5)],
    }
    missing_columns = sorted(required_columns.difference(panel.columns))
    if missing_columns:
        raise ValueError(f"Missing forecasting columns: {missing_columns}")

    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel["forecast_date"] = pd.to_datetime(panel["forecast_date"], errors="raise")
    panel = panel.sort_values(["category", "date"]).reset_index(drop=True)

    missing_categories = sorted(set(TARGET_CATEGORIES).difference(panel["category"]))
    if missing_categories:
        raise ValueError(f"Missing forecasting categories: {missing_categories}")
    if panel.duplicated(["date", "category"]).any():
        raise ValueError("The forecasting panel contains duplicate sector-weeks.")

    # Model selection is confined to the training and validation periods. This part use gpt to solve debug
    selection_panel = panel.loc[
        panel["sample"].isin(["training", "validation"])
    ].copy()

    result_rows = [
        evaluate_historical_candidate(selection_panel, window_length)
        for window_length in HISTORICAL_WINDOW_CANDIDATES
    ]
    result_rows += [
        estimate_quantile_candidate(
            selection_panel, "Time-series QR", lag_order, False
        )
        for lag_order in LAG_CANDIDATES
    ]
    result_rows += [
        estimate_quantile_candidate(selection_panel, "Network QR", lag_order, True)
        for lag_order in LAG_CANDIDATES
    ]

    selection_results = pd.DataFrame(result_rows)
    selected_indices = selection_results.groupby("model")[
        "mean_pinball_loss"
    ].idxmin()
    selection_results["selected"] = selection_results.index.isin(selected_indices)

    model_order = {
        "Historical quantile": 1,
        "Time-series QR": 2,
        "Network QR": 3,
    }
    selection_results["model_order"] = selection_results["model"].map(model_order)
    selection_results = (
        selection_results.sort_values(["model_order", "hyperparameter_value"])
        .drop(columns="model_order")
        .reset_index(drop=True)
    )
    selected_specifications = selection_results.loc[
        selection_results["selected"]
    ].drop(columns="selected").reset_index(drop=True)

    expected_specifications = {
        "Historical quantile": 52,
        "Time-series QR": 4,
        "Network QR": 4,
    }
    selected_values = selected_specifications.set_index("model")[
        "hyperparameter_value"
    ].astype(int).to_dict()
    if selected_values != expected_specifications:
        raise RuntimeError(
            "Selected specifications do not match the dissertation results: "
            f"{selected_values}"
        )

    RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    selection_results.to_csv(SELECTION_RESULTS_OUTPUT, index=False)
    selected_specifications.to_csv(SELECTED_SPECIFICATIONS_OUTPUT, index=False)

    print("Model selected successfully.")
    print(f"Table 5 data: {SELECTION_RESULTS_OUTPUT}")
    print(f"Selected specifications: {SELECTED_SPECIFICATIONS_OUTPUT}")
    print("Selected: historical m=52; time-series p=4; network p=4.")


if __name__ == "__main__":
    main()