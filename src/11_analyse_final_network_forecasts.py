from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "model_results"
    / "final_network_test_predictions.csv"
)

TABLE_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
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
    "Trading & Exchanges": "Trading and Exchanges",
}



# pinball loss


def pinball_loss(actual, forecast, alpha):
    """
    Recalculate the Pinball loss using Equation 4.17.
    """

    error = actual - forecast

    return np.where(
        error >= 0,
        alpha * error,
        (1.0 - alpha) * (-error),
    )



def main():

    print("Analysis of final network quantile forecasts")

    TABLE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not PREDICTION_FILE.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTION_FILE}"
        )

    predictions = pd.read_csv(PREDICTION_FILE)

    predictions["forecast_origin_date"] = pd.to_datetime(
        predictions["forecast_origin_date"],
        errors="raise",
    )

    predictions["forecast_date"] = pd.to_datetime(
        predictions["forecast_date"],
        errors="raise",
    )

    required_columns = [
        "forecast_origin_date",
        "forecast_date",
        "category",
        "actual_contraction_score",
        "predicted_quantile_90",
        "pinball_loss",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    numeric_columns = [
        "actual_contraction_score",
        "predicted_quantile_90",
        "pinball_loss",
    ]
    for column in numeric_columns:
        predictions[column] = pd.to_numeric(
            predictions[column],
            errors="raise",
        )

    predictions = predictions.loc[
        predictions["category"].isin(TARGET_CATEGORIES)
    ].copy()

    predictions = predictions.sort_values(
        ["forecast_date", "category"]
    ).reset_index(drop=True)
    # Validate the prediction table


    if predictions.duplicated(
        subset=["forecast_date", "category"]
    ).any():
        raise ValueError(
            "Duplicated category-date forecasts were found."
        )

    if predictions[required_columns].isna().any().any():
        raise ValueError(
            "Missing values were found in the final prediction table."
        )

    if not np.isfinite(predictions[numeric_columns].to_numpy()).all():
        raise ValueError(
            "Non-finite values were found in the final prediction table."
        )

    categories_found = predictions["category"].nunique()
    dates_found = predictions["forecast_date"].nunique()

    observations_by_category = predictions.groupby(
        "category"
    ).size()

    if (
        len(predictions) != 132
        or categories_found != 4
        or dates_found != 33
        or not observations_by_category.eq(33).all()
    ):
        raise ValueError(
            "The prediction table must contain 33 weeks for each of four sectors."
        )


    # Recalculate the Pinball loss


    recalculated_loss = pinball_loss(
        actual=predictions[
            "actual_contraction_score"
        ].to_numpy(),
        forecast=predictions[
            "predicted_quantile_90"
        ].to_numpy(),
        alpha=ALPHA,
    )

    maximum_loss_difference = np.max(
        np.abs(predictions["pinball_loss"] - recalculated_loss)
    )

    if maximum_loss_difference > 1e-10:
        raise ValueError(
            "The saved Pinball losses do not match Equation 4.17."
        )


    # Table: Overall out-of-sample performance


    overall_results = pd.DataFrame(
        {
            "model": ["Network QR"],
            "quantile_level": [ALPHA],
            "forecasting_weeks": [dates_found],
            "test_observations": [len(predictions)],
            "mean_pinball_loss": [
                predictions["pinball_loss"].mean()
            ],
            "median_pinball_loss": [
                predictions["pinball_loss"].median()
            ],
            "minimum_actual": [
                predictions["actual_contraction_score"].min()
            ],
            "maximum_actual": [
                predictions["actual_contraction_score"].max()
            ],
            "minimum_prediction": [
                predictions["predicted_quantile_90"].min()
            ],
            "maximum_prediction": [
                predictions["predicted_quantile_90"].max()
            ],
        }
    )

    overall_file = (
        TABLE_DIRECTORY
        / "network_test_overall_performance.csv"
    )

    overall_results.to_csv(
        overall_file,
        index=False,
    )

    print("\nOverall out-of-sample performance")

    print(
        overall_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )


    # Figure: Forecasts over time


    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(15, 9),
        sharex=True,
        sharey=True,
    )

    axes = axes.flatten()

    for axis, category in zip(axes, TARGET_CATEGORIES):

        category_data = predictions.loc[
            predictions["category"].eq(category)
        ].sort_values("forecast_date")

        axis.plot(
            category_data["forecast_date"],
            category_data["actual_contraction_score"],
            color="#24588A",
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            label="Actual contraction score",
        )

        axis.plot(
            category_data["forecast_date"],
            category_data["predicted_quantile_90"],
            color="#C84A4A",
            linewidth=1.8,
            linestyle="--",
            label="Predicted 90th conditional quantile",
        )

        axis.axhline(
            y=0,
            color="black",
            linewidth=0.8,
            linestyle=":",
        )

        axis.set_title(
            SHORT_NAMES[category],
            fontsize=12,
        )

        axis.set_ylim(-1.05, 1.05)
        axis.grid(alpha=0.25)

        date_locator = mdates.AutoDateLocator(
            minticks=4,
            maxticks=7,
        )

        axis.xaxis.set_major_locator(date_locator)
        axis.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(date_locator)
        )

    axes[0].set_ylabel("Contraction score")
    axes[2].set_ylabel("Contraction score")

    axes[2].set_xlabel("Forecast date")
    axes[3].set_xlabel("Forecast date")

    handles, labels = axes[0].get_legend_handles_labels()

    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
    )

    figure.suptitle(
        "Out-of-sample network quantile forecasts",
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0.06, 1, 0.95]
    )

    figure_file = (
        FIGURE_DIRECTORY
        / "final_network_forecasts_over_time.png"
    )

    figure.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


    # Table: Sector-level out-of-sample performance


    sector_results = (
        predictions
        .groupby("category", as_index=False)
        .agg(
            test_observations=(
                "pinball_loss",
                "size",
            ),
            mean_pinball_loss=(
                "pinball_loss",
                "mean",
            ),
            median_pinball_loss=(
                "pinball_loss",
                "median",
            ),
            mean_predicted_quantile=(
                "predicted_quantile_90",
                "mean",
            ),
            minimum_predicted_quantile=(
                "predicted_quantile_90",
                "min",
            ),
            maximum_predicted_quantile=(
                "predicted_quantile_90",
                "max",
            ),
        )
    )

    sector_results = sector_results.sort_values(
        "mean_pinball_loss"
    ).reset_index(drop=True)

    sector_file = (
        TABLE_DIRECTORY
        / "network_test_sector_performance.csv"
    )

    sector_results.to_csv(
        sector_file,
        index=False,
    )

    print("\nSector-level out-of-sample performance")

    print(
        sector_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nFiles created:")
    print(overall_file)
    print(sector_file)
    print(figure_file)


if __name__ == "__main__":
    main()