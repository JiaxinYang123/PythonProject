from pathlib import Path

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm



PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECASTING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecasting_panel.csv"
)

SPECIFICATION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_results"
    / "selected_model_specifications.csv"
)

NETWORK_PREDICTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_results"
    / "final_network_test_predictions.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "model_results"
)

FIGURE_DIRECTORY = (
    PROJECT_ROOT
    / "figures"
)

ALPHA = 0.90

TARGET_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Trading & Exchanges",
]

SHORT_NAMES = {
    "Asset Management": "Asset Management",
    "Infrastructure, Services & Financial Products":
        "Infrastructure and Services",
    "Lending, Borrowing & Real World Assets":
        "Lending, Borrowing and RWA",
    "Trading & Exchanges":
        "Trading and Exchanges",
}

# PINBALL LOSS


def pinball_loss(actual, forecast, alpha):
    """
    Calculate observation-level Pinball loss from Equation 4.17.
    """

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    error = actual - forecast

    return np.where(
        error >= 0,
        alpha * error,
        (1.0 - alpha) * (-error),
    )


def calculate_performance(data, model_name, forecast_column, loss_column):
    """
    Calculate forecasting performance for one model.
    """

    actual = data["actual_contraction_score"]
    forecast = data[forecast_column]
    loss = data[loss_column]

    exceedance = actual > forecast

    return {
        "model": model_name,
        "test_observations": len(data),
        "mean_pinball_loss": float(loss.mean()),
        "median_pinball_loss": float(loss.median()),
        "exceedance_count": int(exceedance.sum()),
        "exceedance_rate": float(exceedance.mean()),
    }


def main():

    print("Final forecasting-model comparison")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    # Check required input files

    required_files = [
        FORECASTING_FILE,
        SPECIFICATION_FILE,
        NETWORK_PREDICTION_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    panel = pd.read_csv(FORECASTING_FILE)
    specifications = pd.read_csv(SPECIFICATION_FILE)
    saved_network_predictions = pd.read_csv(
        NETWORK_PREDICTION_FILE
    )

    panel["date"] = pd.to_datetime(panel["date"])
    panel["forecast_date"] = pd.to_datetime(
        panel["forecast_date"]
    )

    saved_network_predictions["forecast_date"] = pd.to_datetime(
        saved_network_predictions["forecast_date"]
    )
    # Read specifications selected in Section 5.2.2 and Table 5


    historical_specification = specifications.loc[
        specifications["model"].eq("Historical quantile")
    ].copy()

    time_series_specification = specifications.loc[
        specifications["model"].eq("Time-series QR")
    ].copy()

    network_specification = specifications.loc[
        specifications["model"].eq("Network QR")
    ].copy()

    if len(historical_specification) != 1:
        raise ValueError(
            "Exactly one selected historical specification is required."
        )

    if len(time_series_specification) != 1:
        raise ValueError(
            "Exactly one selected time-series specification is required."
        )

    if len(network_specification) != 1:
        raise ValueError(
            "Exactly one selected network specification is required."
        )

    historical_window = int(
        historical_specification.iloc[0][
            "hyperparameter_value"
        ]
    )

    time_series_lag_order = int(
        time_series_specification.iloc[0][
            "hyperparameter_value"
        ]
    )

    network_lag_order = int(
        network_specification.iloc[0][
            "hyperparameter_value"
        ]
    )

    if (
        historical_window,
        time_series_lag_order,
        network_lag_order,
    ) != (52, 4, 4):
        raise ValueError(
            "Unexpected selected specifications; expected "
            "m=52, p_TS=4 and p_Net=4."
        )
    # Retain the four main forecasting categories


    data = panel.loc[
        panel["category"].isin(TARGET_CATEGORIES)
    ].copy()

    data = data.sort_values(
        ["category", "date"]
    ).reset_index(drop=True)


    # Construct the 52-week historical quantile forecast
    # For a prediction made at week t, the historical benchmark uses
    # the most recent 52 contraction scores observed up to week t.
    # and it therefore does not use the next-week response.


    data["historical_quantile_forecast"] = (
        data
        .groupby("category", sort=False)["contraction_score"]
        .transform(
            lambda values: (
                values
                .rolling(
                    window=historical_window,
                    min_periods=historical_window,
                )
                .quantile(ALPHA)
            )
        )
    )
    # The time-series model


    lag_columns = [
        f"contraction_lag_{lag}"
        for lag in range(1, time_series_lag_order + 1)
    ]

    time_series_predictors = (
        lag_columns
        + ["log_activity"]
    )

    response_column = "target_contraction_next_week"

    required_columns = (
        [
            "date",
            "forecast_date",
            "category",
            "sample",
            response_column,
            "historical_quantile_forecast",
        ]
        + time_series_predictors
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    for column in [
        response_column,
        "historical_quantile_forecast",
    ] + time_series_predictors:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )
    # Use the final temporal samples defined in Section 4.4
    estimation_data = data.loc[
        data["sample"].isin(["training", "validation"])
    ].copy()

    test_data = data.loc[
        data["sample"].eq("test")
    ].copy()

    estimation_data = estimation_data.dropna(
        subset=[response_column] + time_series_predictors
    ).copy()

    test_data = test_data.dropna(
        subset=[
            response_column,
            "historical_quantile_forecast",
        ] + time_series_predictors
    ).copy()

    if len(estimation_data) != 980:
        raise ValueError(
            "The estimation sample does not reproduce the "
            "expected 980 observations."
        )

    if len(test_data) != 132:
        raise ValueError(
            "The test sample does not reproduce the "
            "expected 132 observations."
        )

    if test_data.duplicated(
        ["forecast_date", "category"]
    ).any():
        raise ValueError(
            "The test sample contains duplicate sector-week rows."
        )

    if (
        estimation_data["forecast_date"].max()
        >= test_data["forecast_date"].min()
    ):
        raise ValueError(
            "The estimation and test periods overlap."
        )
    # Construct the category fixed effects in Equation 4.13
    fixed_effect_columns = []

    for category in TARGET_CATEGORIES[1:]:

        column_name = f"category_effect__{category}"

        estimation_data[column_name] = (
            estimation_data["category"]
            .eq(category)
            .astype(float)
        )

        test_data[column_name] = (
            test_data["category"]
            .eq(category)
            .astype(float)
        )

        fixed_effect_columns.append(column_name)

    predictor_columns = (
        time_series_predictors
        + fixed_effect_columns
    )
    # Construct time-series design matrices
    X_estimation = estimation_data[
        predictor_columns
    ].astype(float).copy()

    X_test = test_data[
        predictor_columns
    ].astype(float).copy()

    X_estimation = sm.add_constant(
        X_estimation,
        has_constant="add",
    )

    X_test = sm.add_constant(
        X_test,
        has_constant="add",
    )

    y_estimation = estimation_data[
        response_column
    ].astype(float)

    y_test = test_data[
        response_column
    ].astype(float)

    matrix_rank = np.linalg.matrix_rank(
        X_estimation.to_numpy()
    )

    number_of_parameters = X_estimation.shape[1]

    if matrix_rank != number_of_parameters:
        raise ValueError(
            "The time-series design matrix is not full rank."
        )

    if not np.isfinite(X_estimation.to_numpy()).all():
        raise ValueError(
            "The estimation design matrix contains non-finite values."
        )

    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError(
            "The test design matrix contains non-finite values."
        )

    if not np.isfinite(y_estimation.to_numpy()).all():
        raise ValueError(
            "The estimation response contains non-finite values."
        )

    if not np.isfinite(y_test.to_numpy()).all():
        raise ValueError(
            "The test response contains non-finite values."
        )
    # Estimate the final time-series quantile-regression model

    with warnings.catch_warnings(record=True) as captured_warnings:

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

    if captured_warnings:
        warning_messages = "; ".join(
            str(item.message)
            for item in captured_warnings
        )
        raise RuntimeError(
            "Time-series model fitting produced warnings: "
            f"{warning_messages}"
        )

    if not np.isfinite(result.params.to_numpy()).all():
        raise ValueError(
            "The fitted time-series parameters are non-finite."
        )
    # Generate time-series test forecasts

    predicted_time_series = result.predict(X_test)

    if not np.isfinite(predicted_time_series).all():
        raise ValueError(
            "The time-series forecasts contain non-finite values."
        )
    # Construct the comparison table

    comparison = test_data[
        [
            "date",
            "forecast_date",
            "category",
            response_column,
            "historical_quantile_forecast",
        ]
    ].copy()

    comparison = comparison.rename(
        columns={
            "date": "forecast_origin_date",
            response_column: "actual_contraction_score",
            "historical_quantile_forecast":
                "historical_quantile_90",
        }
    )

    comparison["time_series_quantile_90"] = (
        predicted_time_series.to_numpy()
    )
    # Merge the saved network forecasts from Section 5.2.3

    network_columns = [
        "forecast_date",
        "category",
        "actual_contraction_score",
        "predicted_quantile_90",
        "pinball_loss",
    ]

    missing_network_columns = [
        column
        for column in network_columns
        if column not in saved_network_predictions.columns
    ]

    if missing_network_columns:
        raise ValueError(
            "Required network-prediction columns are missing: "
            f"{missing_network_columns}"
        )

    saved_network_predictions = (
        saved_network_predictions[network_columns]
        .rename(
            columns={
                "actual_contraction_score":
                    "saved_network_actual",
                "predicted_quantile_90":
                    "network_quantile_90",
                "pinball_loss":
                    "saved_network_pinball_loss",
            }
        )
    )

    comparison = comparison.merge(
        saved_network_predictions,
        on=["forecast_date", "category"],
        how="left",
        validate="one_to_one",
    )

    if comparison[
        [
            "saved_network_actual",
            "network_quantile_90",
            "saved_network_pinball_loss",
        ]
    ].isna().any().any():
        raise ValueError(
            "Some saved network predictions could not be matched."
        )

    maximum_actual_difference = np.max(
        np.abs(
            comparison["actual_contraction_score"]
            - comparison["saved_network_actual"]
        )
    )

    if maximum_actual_difference > 1e-10:
        raise ValueError(
            "The saved and reconstructed test outcomes do not match."
        )

    comparison = comparison.drop(
        columns=["saved_network_actual"]
    )

    # Calculate all Pinball losses explicitly
    comparison["historical_pinball_loss"] = pinball_loss(
        actual=comparison["actual_contraction_score"],
        forecast=comparison["historical_quantile_90"],
        alpha=ALPHA,
    )

    comparison["time_series_pinball_loss"] = pinball_loss(
        actual=comparison["actual_contraction_score"],
        forecast=comparison["time_series_quantile_90"],
        alpha=ALPHA,
    )

    comparison["network_pinball_loss"] = pinball_loss(
        actual=comparison["actual_contraction_score"],
        forecast=comparison["network_quantile_90"],
        alpha=ALPHA,
    )

    maximum_network_loss_difference = np.max(
        np.abs(
            comparison["network_pinball_loss"]
            - comparison["saved_network_pinball_loss"]
        )
    )

    if maximum_network_loss_difference > 1e-10:
        raise ValueError(
            "The saved and recalculated network losses do not match."
        )

    comparison = comparison.drop(
        columns=["saved_network_pinball_loss"]
    )

    # OVERALL

    model_definitions = [
        (
            "Historical quantile",
            "historical_quantile_90",
            "historical_pinball_loss",
        ),
        (
            "Time-series QR",
            "time_series_quantile_90",
            "time_series_pinball_loss",
        ),
        (
            "Network QR",
            "network_quantile_90",
            "network_pinball_loss",
        ),
    ]

    overall_rows = []

    for (
        model_name,
        forecast_column,
        loss_column,
    ) in model_definitions:

        overall_rows.append(
            calculate_performance(
                data=comparison,
                model_name=model_name,
                forecast_column=forecast_column,
                loss_column=loss_column,
            )
        )

    overall_results = pd.DataFrame(overall_rows)
    # Overall incremental value of network information
    time_series_mean_loss = float(
        overall_results.loc[
            overall_results["model"].eq("Time-series QR"),
            "mean_pinball_loss",
        ].iloc[0]
    )

    network_mean_loss = float(
        overall_results.loc[
            overall_results["model"].eq("Network QR"),
            "mean_pinball_loss",
        ].iloc[0]
    )

    overall_loss_difference = (
        time_series_mean_loss - network_mean_loss
    )

    overall_relative_improvement = (
        100.0
        * overall_loss_difference
        / time_series_mean_loss
    )
    # SECTOR-LEVEL MODEL PERFORMANCE
    sector_rows = []

    for category in TARGET_CATEGORIES:

        category_data = comparison.loc[
            comparison["category"].eq(category)
        ].copy()

        for (
            model_name,
            forecast_column,
            loss_column,
        ) in model_definitions:

            row = calculate_performance(
                data=category_data,
                model_name=model_name,
                forecast_column=forecast_column,
                loss_column=loss_column,
            )

            row["category"] = category
            sector_rows.append(row)

    sector_results = pd.DataFrame(sector_rows)

    sector_results = sector_results[
        [
            "category",
            "model",
            "test_observations",
            "mean_pinball_loss",
            "median_pinball_loss",
            "exceedance_count",
            "exceedance_rate",
        ]
    ]

    # Sector-level incremental network value
    sector_incremental_rows = []

    for category in TARGET_CATEGORIES:

        category_results = sector_results.loc[
            sector_results["category"].eq(category)
        ]

        category_time_series_loss = float(
            category_results.loc[
                category_results["model"].eq("Time-series QR"),
                "mean_pinball_loss",
            ].iloc[0]
        )

        category_network_loss = float(
            category_results.loc[
                category_results["model"].eq("Network QR"),
                "mean_pinball_loss",
            ].iloc[0]
        )

        category_loss_difference = (
            category_time_series_loss
            - category_network_loss
        )

        category_relative_improvement = (
            100.0
            * category_loss_difference
            / category_time_series_loss
        )

        sector_incremental_rows.append(
            {
                "category": category,
                "time_series_mean_pinball_loss":
                    category_time_series_loss,
                "network_mean_pinball_loss":
                    category_network_loss,
                "loss_difference_ts_minus_network":
                    category_loss_difference,
                "relative_improvement_percent":
                    category_relative_improvement,
            }
        )

    sector_incremental_value = pd.DataFrame(
        sector_incremental_rows
    )

    comparison["network_gain_over_time_series"] = (
        comparison["time_series_pinball_loss"]
        - comparison["network_pinball_loss"]
    )

    weekly_incremental_value = (
        comparison
        .groupby("forecast_date", as_index=False)
        .agg(
            mean_time_series_loss=(
                "time_series_pinball_loss",
                "mean",
            ),
            mean_network_loss=(
                "network_pinball_loss",
                "mean",
            ),
            mean_network_gain=(
                "network_gain_over_time_series",
                "mean",
            ),
        )
        .sort_values("forecast_date")
    )

    weekly_incremental_value[
        "cumulative_network_gain"
    ] = (
        weekly_incremental_value[
            "mean_network_gain"
        ].cumsum()
    )

    # SAVE TABLES

    comparison_file = (
        OUTPUT_DIRECTORY
        / "final_model_test_comparison.csv"
    )

    overall_file = (
        OUTPUT_DIRECTORY
        / "final_model_overall_comparison.csv"
    )

    sector_incremental_file = (
        OUTPUT_DIRECTORY
        / "sector_network_incremental_value.csv"
    )

    weekly_file = (
        OUTPUT_DIRECTORY
        / "weekly_network_incremental_value.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
    )

    overall_results.to_csv(
        overall_file,
        index=False,
    )

    sector_incremental_value.to_csv(
        sector_incremental_file,
        index=False,
    )

    weekly_incremental_value.to_csv(
        weekly_file,
        index=False,
    )

    # FIGURE plot

    figure, axis = plt.subplots(
        figsize=(11, 6)
    )

    x_positions = np.arange(
        len(TARGET_CATEGORIES)
    )

    bar_width = 0.36

    time_series_sector_losses = []
    network_sector_losses = []

    for category in TARGET_CATEGORIES:

        category_results = sector_results.loc[
            sector_results["category"].eq(category)
        ]

        time_series_sector_losses.append(
            float(
                category_results.loc[
                    category_results["model"].eq("Time-series QR"),
                    "mean_pinball_loss",
                ].iloc[0]
            )
        )

        network_sector_losses.append(
            float(
                category_results.loc[
                    category_results["model"].eq("Network QR"),
                    "mean_pinball_loss",
                ].iloc[0]
            )
        )

    axis.bar(
        x_positions - bar_width / 2,
        time_series_sector_losses,
        width=bar_width,
        label="Time-series QR",
        color="#5B9BD5",
    )

    axis.bar(
        x_positions + bar_width / 2,
        network_sector_losses,
        width=bar_width,
        label="Network QR",
        color="#ED7D31",
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(
        [
            SHORT_NAMES[category]
            for category in TARGET_CATEGORIES
        ],
        rotation=12,
        ha="right",
    )

    axis.set_ylabel("Mean Pinball loss")
    axis.set_title(
        "Out-of-sample pinball loss by sector"
    )
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    figure.tight_layout()

    sector_figure_file = (
        FIGURE_DIRECTORY
        / "sector_forecasting_model_comparison.png"
    )

    figure.savefig(
        sector_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)
    # FIGURE: CUMULATIVE NETWORK GAIN

    figure, axis = plt.subplots(
        figsize=(10.5, 5.5)
    )

    axis.plot(
        weekly_incremental_value["forecast_date"],
        weekly_incremental_value[
            "cumulative_network_gain"
        ],
        color="#2F5597",
        linewidth=2,
    )

    axis.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    axis.set_xlabel("Forecast date")
    axis.set_ylabel(
        "Cumulative Pinball-loss difference"
    )
    axis.set_title(
        "Cumulative loss difference: "
        "time-series minus network model"
    )
    axis.grid(alpha=0.25)

    figure.tight_layout()

    cumulative_figure_file = (
        FIGURE_DIRECTORY
        / "cumulative_network_forecasting_gain.png"
    )

    figure.savefig(
        cumulative_figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    # Console output

    print("\nOverall out-of-sample model comparison")

    print(
        overall_results[
            [
                "model",
                "test_observations",
                "mean_pinball_loss",
                "median_pinball_loss",
                "exceedance_count",
                "exceedance_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print(
        "\nNetwork improvement over the time-series model: "
        f"{overall_relative_improvement:.3f}%"
    )

    print("\nSector-level incremental value")

    print(
        sector_incremental_value.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nFiles created:")

    for file_path in [
        comparison_file,
        overall_file,
        sector_incremental_file,
        weekly_file,
        sector_figure_file,
        cumulative_figure_file,
    ]:
        print(file_path)

    print("\nFinal model comparison completed successfully.")


if __name__ == "__main__":
    main()