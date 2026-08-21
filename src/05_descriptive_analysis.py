from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sector_week_panel.csv"
)

TABLE_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "contraction_descriptive_statistics.csv"
)

DISTRIBUTION_FIGURE = (
    PROJECT_ROOT
    / "figures"
    / "contraction_distribution.png"
)


TARGET_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Trading & Exchanges",
]


SHORT_CATEGORY_NAMES = {
    "Asset Management": "Asset Management",
    "Infrastructure, Services & Financial Products":
        "Infrastructure and Services",
    "Lending, Borrowing & Real World Assets":
        "Lending, Borrowing and RWA",
    "Trading & Exchanges": "Trading and Exchanges",
}


EXPECTED_RAW_WEEKS = 283
EXPECTED_VALID_SCORES = 282



#  INPUT CHECKS


def check_required_columns(dataframe, required_columns):
    """Checkall required columns are present."""

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing_columns)
        )


def check_target_sample(target_panel):
    """Check the fixed dissertation sample for each target sector."""

    for category in TARGET_CATEGORIES:

        category_data = target_panel.loc[
            target_panel["category"] == category
        ]

        raw_weeks = len(category_data)
        valid_scores = category_data[
            "contraction_score"
        ].notna().sum()

        zero_activity = (
            category_data["activity"] == 0
        )

        missing_scores = category_data[
            "contraction_score"
        ].isna()

        if raw_weeks != EXPECTED_RAW_WEEKS:
            raise ValueError(
                f"{category} has {raw_weeks} raw weeks; "
                f"expected {EXPECTED_RAW_WEEKS}."
            )

        if valid_scores != EXPECTED_VALID_SCORES:
            raise ValueError(
                f"{category} has {valid_scores} valid scores; "
                f"expected {EXPECTED_VALID_SCORES}."
            )

        if not zero_activity.equals(missing_scores):
            raise ValueError(
                "Zero-activity and missing-score observations do not "
                f"match for {category}."
            )


def main():

    print("=" * 72)
    print("Descriptive analysis of exposure contraction")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{INPUT_FILE}"
        )

    panel = pd.read_csv(INPUT_FILE)

    check_required_columns(
        panel,
        required_columns=[
            "date",
            "category",
            "activity",
            "contraction_score",
        ],
    )

    panel["date"] = pd.to_datetime(
        panel["date"],
        errors="raise",
    )

    panel["activity"] = pd.to_numeric(
        panel["activity"],
        errors="raise",
    )

    panel["contraction_score"] = pd.to_numeric(
        panel["contraction_score"],
        errors="raise",
    )

    panel = panel.sort_values(
        ["category", "date"]
    ).reset_index(drop=True)

    available_categories = set(
        panel["category"].dropna().unique()
    )

    missing_categories = [
        category
        for category in TARGET_CATEGORIES
        if category not in available_categories
    ]

    if missing_categories:
        raise ValueError(
            "The following target categories were not found:\n"
            + "\n".join(missing_categories)
        )

    target_panel = panel.loc[
        panel["category"].isin(TARGET_CATEGORIES)
    ].copy()

    duplicated_rows = target_panel.duplicated(
        subset=["date", "category"]
    ).sum()

    if duplicated_rows > 0:
        raise ValueError(
            f"Duplicated category-week rows found: {duplicated_rows}"
        )

    check_target_sample(target_panel)

    valid_scores = target_panel[
        "contraction_score"
    ].dropna()

    outside_range = (
        (valid_scores < -1 - 1e-12)
        | (valid_scores > 1 + 1e-12)
    ).sum()

    if outside_range > 0:
        raise ValueError(
            f"{outside_range} contraction scores fall outside [-1, 1]."
        )

    print("\nTarget sectors found: 4")
    print(
        f"Sample period: "
        f"{target_panel['date'].min().date()} to "
        f"{target_panel['date'].max().date()}"
    )
    print(
        "Valid observations per sector: "
        f"{EXPECTED_VALID_SCORES}"
    )


    # Descriptive statistics reported in Table


    descriptive_rows = []

    for category in TARGET_CATEGORIES:

        scores = target_panel.loc[
            target_panel["category"] == category,
            "contraction_score",
        ].dropna()

        descriptive_rows.append(
            {
                "category": category,
                "valid_observations": len(scores),
                "mean": scores.mean(),
                "standard_deviation": scores.std(ddof=1),
                "minimum": scores.min(),
                "median": scores.median(),
                "quantile_90": scores.quantile(0.90),
                "quantile_95": scores.quantile(0.95),
                "maximum": scores.max(),
            }
        )

    descriptive_table = pd.DataFrame(
        descriptive_rows
    )

    statistic_columns = [
        "mean",
        "standard_deviation",
        "minimum",
        "median",
        "quantile_90",
        "quantile_95",
        "maximum",
    ]

    descriptive_table[statistic_columns] = (
        descriptive_table[statistic_columns].round(6)
    )

    TABLE_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    descriptive_table.to_csv(
        TABLE_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 72)
    print("Descriptive statistics")
    print("=" * 72)
    print(descriptive_table.to_string(index=False))


    # Empirical distributions reported in Figure


    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 8),
        sharex=True,
        constrained_layout=True,
    )

    axes = axes.flatten()
    common_bins = np.linspace(-1, 1, 31)

    for axis, category in zip(
        axes,
        TARGET_CATEGORIES,
    ):

        scores = target_panel.loc[
            target_panel["category"] == category,
            "contraction_score",
        ].dropna()

        quantile_90 = scores.quantile(0.90)

        axis.hist(
            scores,
            bins=common_bins,
            color="#4c78a8",
            edgecolor="white",
            alpha=0.85,
        )

        axis.axvline(
            x=0,
            color="black",
            linewidth=0.8,
            linestyle="--",
            label="Zero",
        )

        axis.axvline(
            x=quantile_90,
            color="#c44e52",
            linewidth=1.5,
            linestyle="--",
            label="Empirical 90th percentile",
        )

        axis.set_title(
            SHORT_CATEGORY_NAMES[category],
            fontsize=11,
        )
        axis.set_xlabel("Contraction score")
        axis.set_ylabel("Number of weeks")
        axis.legend(fontsize=8, frameon=False)

    figure.suptitle(
        "Distribution of Sector Exposure-Contraction Scores",
        fontsize=14,
    )

    DISTRIBUTION_FIGURE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        DISTRIBUTION_FIGURE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


    print(TABLE_OUTPUT)
    print(DISTRIBUTION_FIGURE)

    print("\nsuccess")


if __name__ == "__main__":
    main()