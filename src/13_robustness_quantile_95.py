from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_panel.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "model_results"
)

ALPHA = 0.95

CANDIDATE_WINDOWS = [4, 8, 13, 26, 52]
CANDIDATE_LAGS = [1, 2, 4]

TARGET_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Trading & Exchanges",
]
#  BASIC FUNCTIONS


def pinball_loss(actual, forecast):
    """
    Calculate the Pinball loss at alpha = 0.95.
    """

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    error = actual - forecast

    return np.where(
        error >= 0,
        ALPHA * error,
        (1.0 - ALPHA) * (-error),
    )


def build_design_matrix(dataframe, lag_order, network_model):
    """
    Construct the time-series or network-model design matrix.
    """

    design = pd.DataFrame(index=dataframe.index)

    # Autoregressive lags
    for lag in range(1, lag_order + 1):
        design[f"contraction_lag_{lag}"] = (
            dataframe[f"contraction_lag_{lag}"]
            .astype(float)
        )

    # Activity level
    design["log_activity"] = (
        dataframe["log_activity"].astype(float)
    )

    # Network variables
    if network_model:
        design["network_pressure"] = (
            dataframe["network_pressure"].astype(float)
        )

        design["connection_concentration"] = (
            dataframe[
                "connection_concentration"
            ].astype(float)
        )

    # Category fixed effects
    for category in TARGET_CATEGORIES[1:]:
        design[f"category_effect__{category}"] = (
            dataframe["category"]
            .eq(category)
            .astype(float)
        )

    return sm.add_constant(
        design,
        has_constant="add",
    )


def fit_and_predict(
    estimation_data,
    prediction_data,
    lag_order,
    network_model,
):

    response = "target_contraction_next_week"

    required_columns = [
        response,
        "log_activity",
    ]

    required_columns += [
        f"contraction_lag_{lag}"
        for lag in range(1, lag_order + 1)
    ]

    if network_model:
        required_columns += [
            "network_pressure",
            "connection_concentration",
        ]

    estimation_complete = (
        estimation_data
        .dropna(subset=required_columns)
        .copy()
    )

    prediction_complete = (
        prediction_data
        .dropna(subset=required_columns)
        .copy()
    )

    if len(prediction_complete) != len(prediction_data):
        raise ValueError(
            "The prediction sample contains missing model variables."
        )

    X_estimation = build_design_matrix(
        estimation_complete,
        lag_order,
        network_model,
    )

    X_prediction = build_design_matrix(
        prediction_complete,
        lag_order,
        network_model,
    )

    y_estimation = (
        estimation_complete[response]
        .astype(float)
    )

    matrix_rank = np.linalg.matrix_rank(
        X_estimation.to_numpy()
    )

    if matrix_rank != X_estimation.shape[1]:
        raise ValueError(
            "The regression design matrix is not full rank."
        )

    if not np.isfinite(X_estimation.to_numpy()).all():
        raise ValueError(
            "The estimation design matrix  non-finite values."
        )

    if not np.isfinite(X_prediction.to_numpy()).all():
        raise ValueError(
            "The prediction design matrix  non-finite values."
        )

    if not np.isfinite(y_estimation.to_numpy()).all():
        raise ValueError(
            "The estimation response non-finite values."
        )

    with warnings.catch_warnings(record=True) as model_warnings:

        warnings.simplefilter("always")

        model = sm.QuantReg(
            y_estimation,
            X_estimation,
        )

        result = model.fit(
            q=ALPHA,
            max_iter=10000,
            p_tol=1e-7,
        )

    if model_warnings:
        warning_messages = "; ".join(
            str(item.message)
            for item in model_warnings
        )
        raise RuntimeError(
            "Quantile-regression fitting produced warnings: "
            f"{warning_messages}"
        )

    if not np.isfinite(result.params.to_numpy()).all():
        raise ValueError(
            "The fitted model parameters are non-finite."
        )

    predictions = np.asarray(
        result.predict(X_prediction),
        dtype=float,
    )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Non-finite predictions were produced."
        )

    return {
        "predictions": predictions,
        "estimation_observations":
            len(estimation_complete),
    }


def performance_row(
    model_name,
    forecast,
    actual,
    selected_specification,
    mean_validation_loss,
):

    forecast = np.asarray(forecast, dtype=float)
    actual = np.asarray(actual, dtype=float)

    losses = pinball_loss(
        actual,
        forecast,
    )

    exceedances = actual > forecast

    return {
        "model": model_name,
        "selected_specification": selected_specification,
        "mean_validation_loss": mean_validation_loss,
        "mean_test_loss":
            float(losses.mean()),
        "exceedance_rate":
            float(exceedances.mean()),
    }


def main():

    print("Robustness analysis at the 95th quantile")

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)

    base_columns = [
        "date",
        "forecast_date",
        "category",
        "sample",
    ]

    missing_base_columns = [
        column
        for column in base_columns
        if column not in data.columns
    ]

    if missing_base_columns:
        raise ValueError(
            "Required panel columns are missing: "
            f"{missing_base_columns}"
        )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data["forecast_date"] = pd.to_datetime(
        data["forecast_date"]
    )

    data["sample"] = (
        data["sample"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    data = data.loc[
        data["category"].isin(TARGET_CATEGORIES)
    ].copy()

    data = data.sort_values(
        ["category", "date"]
    ).reset_index(drop=True)

    required_columns = [
        "date",
        "forecast_date",
        "category",
        "sample",
        "contraction_score",
        "target_contraction_next_week",
        "log_activity",
        "network_pressure",
        "connection_concentration",
    ] + [
        f"contraction_lag_{lag}"
        for lag in range(1, max(CANDIDATE_LAGS) + 1)
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    numeric_columns = [
        "contraction_score",
        "target_contraction_next_week",
        "log_activity",
        "network_pressure",
        "connection_concentration",
    ] + [
        f"contraction_lag_{lag}"
        for lag in range(1, max(CANDIDATE_LAGS) + 1)
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )
    # Construct historical-quantile candidates before splitting data
    for window in CANDIDATE_WINDOWS:

        column = f"historical_q95_m{window}"

        data[column] = (
            data
            .groupby("category")[
                "contraction_score"
            ]
            .transform(
                lambda series: (
                    series
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .quantile(ALPHA)
                )
            )
        )
    # Temporal samples


    training = data.loc[
        data["sample"].eq("training")
    ].copy()

    validation = data.loc[
        data["sample"].eq("validation")
    ].copy()

    estimation = data.loc[
        data["sample"].isin(
            ["training", "validation"]
        )
    ].copy()

    test = data.loc[
        data["sample"].eq("test")
    ].copy()

    if len(validation) != 212:
        raise ValueError(
            "Expected 212 validation observations."
        )

    if len(test) != 132:
        raise ValueError(
            "Expected 132 test observations."
        )

    if test.duplicated(
        ["forecast_date", "category"]
    ).any():
        raise ValueError(
            "The test sample contains duplicate sector-week rows."
        )

    test_rows_by_category = test.groupby(
        "category"
    ).size()

    if not test_rows_by_category.eq(33).all():
        raise ValueError(
            "Each forecasting category must contain 33 test rows."
        )

    if (
        training["forecast_date"].max()
        >= validation["forecast_date"].min()
    ):
        raise ValueError(
            "The training and validation periods overlap."
        )

    if (
        validation["forecast_date"].max()
        >= test["forecast_date"].min()
    ):
        raise ValueError(
            "The validation and test periods overlap."
        )

    if (
        estimation["forecast_date"].max()
        >= test["forecast_date"].min()
    ):
        raise ValueError(
            "The estimation and test periods overlap."
        )

    selection_rows = []
    # Historical quantile candidates

    for window in CANDIDATE_WINDOWS:

        column = f"historical_q95_m{window}"

        candidate = validation.dropna(
            subset=[
                "target_contraction_next_week",
                column,
            ]
        )

        actual = candidate[
            "target_contraction_next_week"
        ].to_numpy()

        forecast = candidate[
            column
        ].to_numpy()

        mean_loss = pinball_loss(
            actual,
            forecast,
        ).mean()

        selection_rows.append(
            {
                "model": "Historical quantile",
                "parameter": "m",
                "value": window,
                "validation_observations":
                    len(candidate),
                "validation_mean_pinball_loss":
                    mean_loss,
            }
        )
    # Time-series and network candidates

    for model_name, network_model in [
        ("Time-series QR", False),
        ("Network QR", True),
    ]:

        for lag_order in CANDIDATE_LAGS:

            fitted = fit_and_predict(
                estimation_data=training,
                prediction_data=validation,
                lag_order=lag_order,
                network_model=network_model,
            )

            actual = validation[
                "target_contraction_next_week"
            ].to_numpy()

            forecast = fitted["predictions"]

            mean_loss = pinball_loss(
                actual,
                forecast,
            ).mean()

            selection_rows.append(
                {
                    "model": model_name,
                    "parameter": "p",
                    "value": lag_order,
                    "validation_observations":
                        len(validation),
                    "validation_mean_pinball_loss":
                        mean_loss,
                }
            )

    selection_results = pd.DataFrame(
        selection_rows
    )

    selection_results["selected"] = False

    selected_values = {}
    selected_validation_losses = {}

    for model_name in [
        "Historical quantile",
        "Time-series QR",
        "Network QR",
    ]:

        model_rows = selection_results.loc[
            selection_results["model"]
            .eq(model_name)
        ]

        selected_index = (
            model_rows
            .sort_values(
                [
                    "validation_mean_pinball_loss",
                    "value",
                ]
            )
            .index[0]
        )

        selection_results.loc[
            selected_index,
            "selected",
        ] = True

        selected_values[model_name] = int(
            selection_results.loc[
                selected_index,
                "value",
            ]
        )

        selected_validation_losses[model_name] = float(
            selection_results.loc[
                selected_index,
                "validation_mean_pinball_loss",
            ]
        )

    selected_window = selected_values[
        "Historical quantile"
    ]

    selected_ts_lag = selected_values[
        "Time-series QR"
    ]

    selected_network_lag = selected_values[
        "Network QR"
    ]

    if (
        selected_window,
        selected_ts_lag,
        selected_network_lag,
    ) != (52, 4, 4):
        raise ValueError(
            "Unexpected selected specifications; expected "
            "m=52, p_TS=4 and p_Net=4."
        )

    historical_column = (
        f"historical_q95_m{selected_window}"
    )

    if test[historical_column].isna().any():
        raise ValueError(
            "Historical test forecasts contain missing values."
        )

    final_ts = fit_and_predict(
        estimation_data=estimation,
        prediction_data=test,
        lag_order=selected_ts_lag,
        network_model=False,
    )

    final_network = fit_and_predict(
        estimation_data=estimation,
        prediction_data=test,
        lag_order=selected_network_lag,
        network_model=True,
    )

    if (
        final_ts["estimation_observations"] != 980
        or final_network["estimation_observations"] != 980
    ):
        raise ValueError(
            "The final regression models must use 980 complete "
            "estimation observations."
        )

    actual = test[
        "target_contraction_next_week"
    ].to_numpy()

    historical_forecast = test[
        historical_column
    ].to_numpy()

    time_series_forecast = (
        final_ts["predictions"]
    )

    network_forecast = (
        final_network["predictions"]
    )

    final_results = pd.DataFrame(
        [
            performance_row(
                model_name="Historical quantile",
                forecast=historical_forecast,
                actual=actual,
                selected_specification=
                    f"m={selected_window}",
                mean_validation_loss=
                    selected_validation_losses[
                        "Historical quantile"
                    ],
            ),

            performance_row(
                model_name="Time-series QR",
                forecast=time_series_forecast,
                actual=actual,
                selected_specification=
                    f"p={selected_ts_lag}",
                mean_validation_loss=
                    selected_validation_losses[
                        "Time-series QR"
                    ],
            ),

            performance_row(
                model_name="Network QR",
                forecast=network_forecast,
                actual=actual,
                selected_specification=
                    f"p={selected_network_lag}",
                mean_validation_loss=
                    selected_validation_losses[
                        "Network QR"
                    ],
            ),
        ]
    )
    # Incremental network value

    ts_loss = float(
        final_results.loc[
            final_results["model"]
            .eq("Time-series QR"),
            "mean_test_loss",
        ].iloc[0]
    )

    network_loss = float(
        final_results.loc[
            final_results["model"]
            .eq("Network QR"),
            "mean_test_loss",
        ].iloc[0]
    )

    loss_difference = (
        ts_loss - network_loss
    )

    relative_improvement = (
        100.0
        * loss_difference
        / ts_loss
    )
    # Observation-level predictions

    predictions = test[
        [
            "date",
            "forecast_date",
            "category",
            "target_contraction_next_week",
        ]
    ].copy()

    predictions = predictions.rename(
        columns={
            "date": "forecast_origin_date",
            "target_contraction_next_week":
                "actual_contraction_score",
        }
    )

    predictions["historical_quantile_95"] = (
        historical_forecast
    )

    predictions["time_series_quantile_95"] = (
        time_series_forecast
    )

    predictions["network_quantile_95"] = (
        network_forecast
    )

    predictions["historical_pinball_loss"] = (
        pinball_loss(
            actual,
            historical_forecast,
        )
    )

    predictions["time_series_pinball_loss"] = (
        pinball_loss(
            actual,
            time_series_forecast,
        )
    )

    predictions["network_pinball_loss"] = (
        pinball_loss(
            actual,
            network_forecast,
        )
    )

    selection_file = (
        OUTPUT_DIRECTORY
        / "q95_validation_selection.csv"
    )

    final_results_file = (
        OUTPUT_DIRECTORY
        / "q95_robustness_results.csv"
    )

    predictions_file = (
        OUTPUT_DIRECTORY
        / "q95_test_predictions.csv"
    )

    selection_results.to_csv(
        selection_file,
        index=False,
    )

    final_results.to_csv(
        final_results_file,
        index=False,
    )

    predictions.to_csv(
        predictions_file,
        index=False,
    )

    print("\nFinal 95th-quantile robustness results")

    print(
        final_results.to_string(
            index=False,
            float_format=lambda value:
                f"{value:.6f}",
        )
    )

    print(
        "\nNetwork improvement over the time-series model: "
        f"{relative_improvement:.3f}%"
    )

    print("\nFiles created:")
    print(selection_file)
    print(final_results_file)
    print(predictions_file)

    print("\nRobustness analysis successfully.")


if __name__ == "__main__":
    main()