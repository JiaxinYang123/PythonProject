from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from dexposure_mapping import ANALYSIS_CATEGORIES



PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sector_exposure_matrix.csv"
)

FIGURE_PATH = (
    PROJECT_ROOT
    / "figures"
    / "sector_exposure_heatmap.png"
)

SHARE_TABLE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "sector_exposure_heatmap_shares.csv"
)

TOTAL_TABLE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "sector_exposure_heatmap_totals.csv"
)

SHORT_CATEGORY_NAMES = {
    "Asset Management": "Asset\nManagement",
    "Infrastructure, Services & Financial Products":
        "Infrastructure\n& Services",
    "Lending, Borrowing & Real World Assets":
        "Lending,\nBorrowing & RWA",
    "Privacy & Security": "Privacy\n& Security",
    "Trading & Exchanges": "Trading\n& Exchanges",
    "Primary Market Tokens": "Primary Market\nTokens",
    "Other / Unknown": "Other /\nUnknown",
}
# Input validation


def load_and_validate_sector_matrix():
    """Load and validate the complete weekly 7 x 7 matrix panel."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n{INPUT_PATH}\n"
            "Run src/03_build_sector_panel.py first."
        )

    data = pd.read_csv(INPUT_PATH)

    required_columns = {
        "date",
        "source_category",
        "target_category",
        "exposure",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "The input file is missing the following columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = data[
        [
            "date",
            "source_category",
            "target_category",
            "exposure",
        ]
    ].copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
    )

    data["exposure"] = pd.to_numeric(
        data["exposure"],
        errors="raise",
    )

    if not np.isfinite(data["exposure"]).all():
        raise ValueError(
            "The exposure column contains a non-finite value."
        )

    if (data["exposure"] < 0).any():
        raise ValueError(
            "The exposure column contains a negative value."
        )

    duplicated_rows = data.duplicated(
        subset=[
            "date",
            "source_category",
            "target_category",
        ]
    ).sum()

    if duplicated_rows > 0:
        raise ValueError(
            "Duplicated date-source-target rows found: "
            f"{duplicated_rows}"
        )

    expected_categories = set(ANALYSIS_CATEGORIES)
    source_categories = set(data["source_category"].unique())
    target_categories = set(data["target_category"].unique())

    if source_categories != expected_categories:
        raise ValueError(
            "Source categories do not match ANALYSIS_CATEGORIES.\n"
            f"Expected: {sorted(expected_categories)}\n"
            f"Found: {sorted(source_categories)}"
        )

    if target_categories != expected_categories:
        raise ValueError(
            "Target categories do not match ANALYSIS_CATEGORIES.\n"
            f"Expected: {sorted(expected_categories)}\n"
            f"Found: {sorted(target_categories)}"
        )

    number_of_dates = data["date"].nunique()
    number_of_categories = len(ANALYSIS_CATEGORIES)
    expected_rows = (
        number_of_dates
        * number_of_categories
        * number_of_categories
    )

    if len(data) != expected_rows:
        raise ValueError(
            "The matrix panel is incomplete. "
            f"Expected {expected_rows:,} rows but found "
            f"{len(data):,}."
        )

    rows_per_date = data.groupby("date").size()

    if not (
        rows_per_date == number_of_categories ** 2
    ).all():
        raise ValueError(
            "At least one date does not contain a complete 7 x 7 "
            "sector matrix."
        )

    return data

def calculate_exposure_matrices(data):
    """
    Aggregate exposure over all weeks and normalise each source row.
    """

    total_matrix = (
        data.groupby(
            ["source_category", "target_category"],
            sort=False,
        )["exposure"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(
            index=ANALYSIS_CATEGORIES,
            columns=ANALYSIS_CATEGORIES,
            fill_value=0.0,
        )
    )

    row_totals = total_matrix.sum(axis=1)

    zero_rows = row_totals[row_totals <= 0]

    if not zero_rows.empty:
        raise ValueError(
            "The following source categories have zero full-sample "
            "exposure:\n"
            + "\n".join(zero_rows.index)
        )

    share_matrix = total_matrix.div(
        row_totals,
        axis=0,
    )

    maximum_row_sum_error = (
        share_matrix.sum(axis=1) - 1.0
    ).abs().max()

    if maximum_row_sum_error > 1e-12:
        raise ValueError(
            "The row-normalised shares do not sum to one. "
            f"Maximum error: {maximum_row_sum_error:.3e}"
        )

    return total_matrix, share_matrix

# 4. OUTPUT TABLES

def save_output_tables(total_matrix, share_matrix):
    """Save the underlying numerical matrices for reproducibility."""

    SHARE_TABLE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    share_matrix.to_csv(
        SHARE_TABLE_PATH,
        index_label="source_category",
    )

    total_matrix.to_csv(
        TOTAL_TABLE_PATH,
        index_label="source_category",
    )


# Heatmap


def create_heatmap(share_matrix):
    """Create and save the directed sector-exposure heatmap."""

    values = share_matrix.to_numpy(dtype=float)

    labels = [
        SHORT_CATEGORY_NAMES[category]
        for category in ANALYSIS_CATEGORIES
    ]

    plt.style.use("seaborn-v0_8-white")

    figure, axis = plt.subplots(
        figsize=(11.5, 8.5),
        constrained_layout=True,
    )

    image = axis.imshow(
        values,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=float(values.max()),
        aspect="equal",
    )

    axis.set_xticks(np.arange(len(labels)))
    axis.set_yticks(np.arange(len(labels)))
    axis.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=9,
    )
    axis.set_yticklabels(
        labels,
        fontsize=9,
    )

    axis.set_xlabel(
        "Destination sector",
        fontsize=11,
    )
    axis.set_ylabel(
        "Source sector",
        fontsize=11,
    )
    axis.set_title(
        "Full-Sample Directed Sector-Exposure Shares",
        fontsize=13,
        pad=14,
    )

    axis.set_xticks(
        np.arange(-0.5, len(labels), 1),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, len(labels), 1),
        minor=True,
    )
    axis.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=1.0,
    )
    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # Add percentage labels and change the text colour in dark cells.
    colour_threshold = values.max() * 0.55

    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]

            text_colour = (
                "white"
                if value >= colour_threshold
                else "black"
            )

            axis.text(
                column_index,
                row_index,
                f"{value:.1%}",
                ha="center",
                va="center",
                color=text_colour,
                fontsize=8.5,
            )

    colour_bar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        format=PercentFormatter(xmax=1.0),
    )

    colour_bar.set_label(
        "Share of source-sector exposure",
        fontsize=10,
    )

    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():

    print("=" * 72)
    print("FULL-SAMPLE DIRECTED SECTOR-EXPOSURE HEATMAP")
    print("=" * 72)

    print(f"Input file:\n{INPUT_PATH}")

    data = load_and_validate_sector_matrix()

    print(
        f"\nValidated {len(data):,} matrix observations "
        f"across {data['date'].nunique():,} weekly snapshots."
    )

    total_matrix, share_matrix = calculate_exposure_matrices(
        data
    )

    save_output_tables(
        total_matrix=total_matrix,
        share_matrix=share_matrix,
    )

    create_heatmap(share_matrix)

    print("\nRow-normalised exposure shares:")
    print(
        share_matrix
        .mul(100.0)
        .round(2)
        .to_string()
    )

    print("\nOutputs created:")
    print(FIGURE_PATH)
    print(SHARE_TABLE_PATH)
    print(TOTAL_TABLE_PATH)


if __name__ == "__main__":
    main()