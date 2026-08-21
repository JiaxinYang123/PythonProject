from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SECTOR_MATRIX_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sector_exposure_matrix.csv"
)

SECTOR_PANEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sector_week_panel.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "sector_network_validation.csv"
)



# load tables


sector_matrix = pd.read_csv(
    SECTOR_MATRIX_PATH
)

sector_panel = pd.read_csv(
    SECTOR_PANEL_PATH
)

validation = pd.read_csv(
    VALIDATION_PATH
)

#basic structure

print("=" * 72)
print("SECTOR-PANEL VALIDATION")
print("=" * 72)

matrix_dates = sector_matrix["date"].nunique()
panel_dates = sector_panel["date"].nunique()
validation_dates = validation["date"].nunique()
category_count = sector_panel["category"].nunique()

matrix_date_set = set(
    sector_matrix["date"].astype(str)
)

panel_date_set = set(
    sector_panel["date"].astype(str)
)

validation_date_set = set(
    validation["date"].astype(str)
)

matrix_duplicates = sector_matrix.duplicated(
    subset=[
        "date",
        "source_category",
        "target_category",
    ]
).sum()

panel_duplicates = sector_panel.duplicated(
    subset=[
        "date",
        "category",
    ]
).sum()

validation_duplicates = validation.duplicated(
    subset=["date"]
).sum()

matrix_rows_per_week = sector_matrix.groupby(
    "date"
).size()

panel_rows_per_week = sector_panel.groupby(
    "date"
).size()

print("\ntable stucture")
print(matrix_dates)
print(panel_dates)
print(validation_dates)

print(
    f"Sector-matrix rows: "
    f"{len(sector_matrix):,}"
)

print(
    f"Sector-panel rows: "
    f"{len(sector_panel):,}"
)

print(f"Matrix dates: {matrix_dates}")
print(f"Panel dates: {panel_dates}")
print(f"Validation dates: {validation_dates}")
print(f"Categories: {category_count}")

print(
    f"Duplicated matrix rows: "
    f"{matrix_duplicates}"
)

print(
    f"Duplicated panel rows: "
    f"{panel_duplicates}"
)

print(
    f"Duplicated validation rows: "
    f"{validation_duplicates}"
)

print(
    "Matrix rows per week: "
    f"min={matrix_rows_per_week.min()}, "
    f"max={matrix_rows_per_week.max()}"
)

print(
    "Panel rows per week: "
    f"min={panel_rows_per_week.min()}, "
    f"max={panel_rows_per_week.max()}"
)



# Recompute input,output from S_t


recalculated_inflow = (
    sector_matrix
    .groupby(
        [
            "date",
            "target_category",
        ],
        as_index=False,
    )["exposure"]
    .sum()
    .rename(
        columns={
            "target_category": "category",
            "exposure": "recalculated_inflow",
        }
    )
)

recalculated_outflow = (
    sector_matrix
    .groupby(
        [
            "date",
            "source_category",
        ],
        as_index=False,
    )["exposure"]
    .sum()
    .rename(
        columns={
            "source_category": "category",
            "exposure": "recalculated_outflow",
        }
    )
)

flow_check = (
    sector_panel
    .merge(
        recalculated_inflow,
        on=["date", "category"],
        how="left",
    )
    .merge(
        recalculated_outflow,
        on=["date", "category"],
        how="left",
    )
)

flow_check["inflow_difference"] = abs(
    flow_check["inflow"]
    - flow_check["recalculated_inflow"]
)

flow_check["outflow_difference"] = abs(
    flow_check["outflow"]
    - flow_check["recalculated_outflow"]
)

maximum_inflow_difference = flow_check[
    "inflow_difference"
].max()

maximum_outflow_difference = flow_check[
    "outflow_difference"
].max()

print("\nFlow reconstitution")
print("=" * 72)

print(
    "Maximum inflow difference: "
    f"${maximum_inflow_difference:,.6f}"
)

print(
    "Maximum outflow difference: "
    f"${maximum_outflow_difference:,.6f}"
)



# Recompute activity


recalculated_activity = (
    sector_panel["inflow"]
    + sector_panel["outflow"]
)

activity_difference = abs(
    sector_panel["activity"]
    - recalculated_activity
)

maximum_activity_difference = (
    activity_difference.max()
)

activity_is_nonnegative = bool(
    (sector_panel["activity"] >= 0).all()
)

recalculated_log_activity = np.log1p(
    sector_panel["activity"]
)

log_activity_difference = abs(
    sector_panel["log_activity"]
    - recalculated_log_activity
)

maximum_log_activity_difference = (
    log_activity_difference.max()
)

print("\nExposure activities")

print(
    "Maximum activity difference: "
    f"${maximum_activity_difference:,.6f}"
)

print(
    "Maximum log-activity difference: "
    f"{maximum_log_activity_difference:.12e}"
)

print(
    "All activity values non-negative: "
    f"{activity_is_nonnegative}"
)



# Contraction score


positive_activity = sector_panel[
    sector_panel["activity"] > 0
].copy()

positive_activity[
    "recalculated_contraction_score"
] = (
    positive_activity["outflow"]
    - positive_activity["inflow"]
) / positive_activity["activity"]

positive_activity[
    "contraction_difference"
] = abs(
    positive_activity["contraction_score"]
    - positive_activity[
        "recalculated_contraction_score"
    ]
)

maximum_contraction_difference = (
    positive_activity[
        "contraction_difference"
    ].max()
)

minimum_contraction_score = positive_activity[
    "contraction_score"
].min()

maximum_contraction_score = positive_activity[
    "contraction_score"
].max()

zero_activity_mask = (
    sector_panel["activity"] == 0
)

missing_contraction_mask = (
    sector_panel["contraction_score"].isna()
)

zero_activity_matches_missing = (
    zero_activity_mask.equals(
        missing_contraction_mask
    )
)

zero_activity_rows = sector_panel[
    zero_activity_mask
]

missing_contraction_rows = sector_panel[
    missing_contraction_mask
].copy()

print("\ncontraction scores")

print(
    "Maximum formula difference: "
    f"{maximum_contraction_difference:.12f}"
)

print(
    "Minimum contraction score: "
    f"{minimum_contraction_score:.6f}"
)

print(
    "Maximum contraction score: "
    f"{maximum_contraction_score:.6f}"
)

print(
    f"Zero-activity rows: "
    f"{len(zero_activity_rows)}"
)

print(
    f"Missing contraction scores: "
    f"{len(missing_contraction_rows)}"
)

print(
    "Zero activity matches missing scores: "
    f"{zero_activity_matches_missing}"
)



#  Exposure conservation


maximum_relative_difference = validation[
    "relative_exposure_difference"
].max()

print("\nExposure conservation")

print(
    "Maximum relative difference: "
    f"{maximum_relative_difference:.12e}"
)



# Zero activity by category


zero_activity_summary = (
    zero_activity_rows
    .groupby("category")
    .size()
    .sort_values(ascending=False)
)

print("\nZero activity summary")
print()

if zero_activity_summary.empty:

    print("None")

else:

    for category, count in zero_activity_summary.items():
        print(f"{category}: {count}")




structure_is_valid = (
    len(sector_matrix) == 13867
    and len(sector_panel) == 1981
    and matrix_dates == 283
    and panel_dates == 283
    and validation_dates == 283
    and category_count == 7
    and matrix_duplicates == 0
    and panel_duplicates == 0
    and validation_duplicates == 0
    and matrix_date_set == panel_date_set
    and matrix_date_set == validation_date_set
    and matrix_rows_per_week.min() == 49
    and matrix_rows_per_week.max() == 49
    and panel_rows_per_week.min() == 7
    and panel_rows_per_week.max() == 7
)

flows_are_valid = (
    maximum_inflow_difference < 0.01
    and maximum_outflow_difference < 0.01
)

activity_is_valid = (
    maximum_activity_difference < 0.01
    and maximum_log_activity_difference < 1e-12
    and activity_is_nonnegative
)

contraction_is_valid = (
    maximum_contraction_difference < 1e-12
    and minimum_contraction_score >= -1
    and maximum_contraction_score <= 1
    and zero_activity_matches_missing
)

conservation_is_valid = (
    maximum_relative_difference < 1e-12
)


print("FINAL STATUS:")


print(
    f"Table structure valid: "
    f"{structure_is_valid}"
)

print(
    f"Inflow and outflow valid: "
    f"{flows_are_valid}"
)

print(
    f"Exposure activity valid: "
    f"{activity_is_valid}"
)

print(
    f"Contraction score valid: "
    f"{contraction_is_valid}"
)

print(
    f"Exposure conservation valid: "
    f"{conservation_is_valid}"
)

all_valid = (
    structure_is_valid
    and flows_are_valid
    and activity_is_valid
    and contraction_is_valid
    and conservation_is_valid
)

if all_valid:

    print(
        "\nPASS: The sector-week panel is "
        "mathematically consistent."
    )

else:

    print(
        "\nREVIEW: At least one validation "
        "condition failed."
    )

print("=" * 72)

if not all_valid:
    raise AssertionError(
        "At least one sector-panel validation condition failed."
    )