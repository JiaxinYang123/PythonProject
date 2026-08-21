from collections import defaultdict
from pathlib import Path
import math

import pandas as pd

from dexposure_io import iterate_weekly_snapshots
from dexposure_mapping import ANALYSIS_CATEGORIES
from dexposure_mapping import convert_to_analysis_category
from dexposure_mapping import load_official_mappings
from dexposure_mapping import normalise_identifier
from dexposure_mapping import resolve_official_category




# 1. PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FULL_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "historical-network_week_2020-03-30.json"
)

ID_TO_INFO_PATH = (
    PROJECT_ROOT
    / "data"
    / "mapping"
    / "id_to_info.json"
)

TOKEN_TO_PROTOCOL_PATH = (
    PROJECT_ROOT
    / "data"
    / "mapping"
    / "token_to_protocol.json"
)

PROCESSED_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

SECTOR_MATRIX_PATH = (
    PROCESSED_DATA_DIR
    / "sector_exposure_matrix.csv"
)

SECTOR_PANEL_PATH = (
    PROCESSED_DATA_DIR
    / "sector_week_panel.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "sector_network_validation.csv"
)

MAPPING_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tables"
    / "category_mapping_summary.csv"
)



EXPECTED_SNAPSHOT_COUNT = 283
EXPECTED_FIRST_DATE = "2020-03-23"
EXPECTED_LAST_DATE = "2025-08-18"




def read_finite_value(value, description):

    if value is None:
        return 0.0

    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"Non-finite value found in {description}: {value}"
        )

    return numeric_value


def read_nonnegative_value(value, description):
    """
    Read a finite and non-negative numerical value.
    """

    numeric_value = read_finite_value(
        value=value,
        description=description,
    )

    if numeric_value < 0:
        raise ValueError(
            f"Negative link value found in {description}: {value}"
        )

    return numeric_value

def classify_identifier(
    identifier,
    id_to_info,
    token_index,
    category_cache,
):
    """
    Classify a node or link endpoint using the official DeXposure
    mapping files and the documented seven-category crosswalk.

    Classification results are cached because the same identifiers
    appear repeatedly across weekly snapshots.
    """

    normalised_identifier = normalise_identifier(
        identifier
    )

    if normalised_identifier in category_cache:
        return category_cache[normalised_identifier]

    official_result = resolve_official_category(
        node_identifier=normalised_identifier,
        id_to_info=id_to_info,
        token_index=token_index,
    )

    raw_category = official_result["category"]

    analysis_category = convert_to_analysis_category(
        raw_category
    )

    result = {
        "raw_category": raw_category,
        "analysis_category": analysis_category,
        "mapping_method": official_result["mapping_method"],
    }

    category_cache[normalised_identifier] = result

    return result


def main():

    print("=" * 72)
    print("FULL DEXPOSURE SECTOR-WEEK PANEL CONSTRUCTION")
    print("=" * 72)

    print(f"Input file:\n{FULL_DATA_PATH}")

    # Load the official mappings supplied by DeXposure.
    id_to_info, token_index = load_official_mappings(
        ID_TO_INFO_PATH,
        TOKEN_TO_PROTOCOL_PATH,
    )

    print("\nOfficial mapping files loaded.")
    print(f"Protocol mapping entries: {len(id_to_info):,}")
    print(f"Token mapping entries: {len(token_index):,}")
    print(f"Analysis categories: {len(ANALYSIS_CATEGORIES)}")

    category_cache = {}

    sector_matrix_rows = []
    sector_panel_rows = []
    validation_rows = []

    mapping_summary = defaultdict(
        int
    )

    processed_dates = []

    print("\nProcessing all weekly snapshots...")
    print("This may take several minutes.")

    for snapshot_number, (
        date,
        nodes,
        links,
    ) in enumerate(
        iterate_weekly_snapshots(FULL_DATA_PATH),
        start=1,
    ):

        processed_dates.append(date)


   # Initialise the complete 7 x 7 sector matrix


        sector_matrix = {
            (source_category, target_category): 0.0
            for source_category in ANALYSIS_CATEGORIES
            for target_category in ANALYSIS_CATEGORIES
        }

        sector_node_count = {
            category: 0
            for category in ANALYSIS_CATEGORIES
        }


   # Count official nodes by category


        for node in nodes:

            classification = classify_identifier(
                identifier=node.get("id"),
                id_to_info=id_to_info,
                token_index=token_index,
                category_cache=category_cache,
            )

            category = classification[
                "analysis_category"
            ]

            sector_node_count[category] += 1

            summary_key = (
                classification["raw_category"],
                classification["analysis_category"],
                classification["mapping_method"],
            )

            mapping_summary[summary_key] += 1


        # Aggregate official directed links by category


        total_link_exposure = 0.0

        for link in links:

            source_identifier = link.get("source")
            target_identifier = link.get("target")

            source_classification = classify_identifier(
                identifier=source_identifier,
                id_to_info=id_to_info,
                token_index=token_index,
                category_cache=category_cache,
            )

            target_classification = classify_identifier(
                identifier=target_identifier,
                id_to_info=id_to_info,
                token_index=token_index,
                category_cache=category_cache,
            )

            source_category = source_classification[
                "analysis_category"
            ]

            target_category = target_classification[
                "analysis_category"
            ]

            # Final directed DeXposure link weights must be non-negative.
            exposure = read_nonnegative_value(
                link.get("size", 0.0),
                description=f"link exposure on {date}",
            )

            sector_matrix[
                (source_category, target_category)
            ] += exposure

            total_link_exposure += exposure


        # Store all entries of the sector matrix S_t


        for source_category in ANALYSIS_CATEGORIES:

            for target_category in ANALYSIS_CATEGORIES:

                exposure = sector_matrix[
                    (source_category, target_category)
                ]

                sector_matrix_rows.append(
                    {
                        "date": date,
                        "source_category": source_category,
                        "target_category": target_category,
                        "exposure": exposure,
                    }
                )


        # sector inflow, outflow and contraction


        for category in ANALYSIS_CATEGORIES:

            inflow = sum(
                sector_matrix[
                    (source_category, category)
                ]
                for source_category in ANALYSIS_CATEGORIES
            )

            outflow = sum(
                sector_matrix[
                    (category, target_category)
                ]
                for target_category in ANALYSIS_CATEGORIES
            )

            within_sector_exposure = sector_matrix[
                (category, category)
            ]

            cross_sector_inflow = (
                inflow - within_sector_exposure
            )

            cross_sector_outflow = (
                outflow - within_sector_exposure
            )

            activity = inflow + outflow
            net_inflow = inflow - outflow

            if activity > 0:

                exposure_shift_ratio = (
                    net_inflow / activity
                )

                contraction_score = (
                    -exposure_shift_ratio
                )

                log_activity = math.log1p(
                    activity
                )

            else:

                exposure_shift_ratio = math.nan
                contraction_score = math.nan
                log_activity = 0.0

            sector_panel_rows.append(
                {
                    "date": date,
                    "category": category,
                    "node_count":
                        sector_node_count[category],
                    "inflow": inflow,
                    "outflow": outflow,
                    "within_sector_exposure":
                        within_sector_exposure,
                    "cross_sector_inflow":
                        cross_sector_inflow,
                    "cross_sector_outflow":
                        cross_sector_outflow,
                    "activity": activity,
                    "log_activity": log_activity,
                    "net_inflow": net_inflow,
                    "exposure_shift_ratio":
                        exposure_shift_ratio,
                    "contraction_score":
                        contraction_score,
                }
            )


        # Exposure-conservation validation

        aggregated_sector_exposure = sum(
            sector_matrix.values()
        )

        exposure_difference = abs(
            total_link_exposure
            - aggregated_sector_exposure
        )

        relative_difference = (
            exposure_difference
            / (total_link_exposure + 1e-12)
        )

        validation_rows.append(
            {
                "date": date,
                "node_count": len(nodes),
                "link_count": len(links),
                "total_link_exposure":
                    total_link_exposure,
                "aggregated_sector_exposure":
                    aggregated_sector_exposure,
                "absolute_exposure_difference":
                    exposure_difference,
                "relative_exposure_difference":
                    relative_difference,
            }
        )

        if snapshot_number == 1 or snapshot_number % 25 == 0:

            print(
                f"Processed {snapshot_number:3d} snapshots "
                f"| date={date} "
                f"| nodes={len(nodes):,} "
                f"| links={len(links):,}"
            )

    snapshot_count = len(processed_dates)

    if snapshot_count != EXPECTED_SNAPSHOT_COUNT:

        raise ValueError(
            "Unexpected number of snapshots: "
            f"{snapshot_count}"
        )

    if processed_dates[0] != EXPECTED_FIRST_DATE:

        raise ValueError(
            "Unexpected first snapshot date: "
            f"{processed_dates[0]}"
        )

    if processed_dates[-1] != EXPECTED_LAST_DATE:

        raise ValueError(
            "Unexpected last snapshot date: "
            f"{processed_dates[-1]}"
        )


    # CREATE OUTPUT TABLES


    sector_matrix_df = pd.DataFrame(
        sector_matrix_rows
    )

    sector_panel_df = pd.DataFrame(
        sector_panel_rows
    )

    validation_df = pd.DataFrame(
        validation_rows
    )

    mapping_summary_rows = []

    for (
        raw_category,
        analysis_category,
        mapping_method,
    ), node_observations in mapping_summary.items():

        mapping_summary_rows.append(
            {
                "raw_category":
                    raw_category,
                "analysis_category":
                    analysis_category,
                "mapping_method":
                    mapping_method,
                "node_observations":
                    node_observations,
            }
        )

    mapping_summary_df = pd.DataFrame(
        mapping_summary_rows
    ).sort_values(
        by="node_observations",
        ascending=False,
    )

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sector_matrix_df.to_csv(
        SECTOR_MATRIX_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    sector_panel_df.to_csv(
        SECTOR_PANEL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    mapping_summary_df.to_csv(
        MAPPING_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    maximum_exposure_difference = validation_df[
        "absolute_exposure_difference"
    ].max()

    total_exposure = sector_matrix_df[
        "exposure"
    ].sum()

    residual_exposure = sector_matrix_df.loc[
        (
            sector_matrix_df["source_category"]
            == "Other / Unknown"
        )
        |
        (
            sector_matrix_df["target_category"]
            == "Other / Unknown"
        ),
        "exposure",
    ].sum()

    residual_exposure_share = (
        residual_exposure / total_exposure
        if total_exposure > 0
        else math.nan
    )

    zero_activity_rows = (
        sector_panel_df["activity"] == 0
    ).sum()

    print("\n" + "=" * 72)
    print("CONSTRUCTION SUMMARY")
    print("=" * 72)

    print(f"Snapshots processed: {snapshot_count}")
    print(f"First date: {processed_dates[0]}")
    print(f"Last date: {processed_dates[-1]}")
    print(f"Analysis categories: {len(ANALYSIS_CATEGORIES)}")

    print(
        "Sector-matrix observations: "
        f"{len(sector_matrix_df):,}"
    )

    print(
        "Sector-week observations: "
        f"{len(sector_panel_df):,}"
    )

    print(
        "Maximum exposure difference: "
        f"${maximum_exposure_difference:,.6f}"
    )

    print(
        "Exposure involving Other / Unknown: "
        f"{residual_exposure_share:.2%}"
    )

    print(
        "Sector-week rows with zero activity: "
        f"{zero_activity_rows:,}"
    )

    print("\nFiles created:")

    for output_path in [
        SECTOR_MATRIX_PATH,
        SECTOR_PANEL_PATH,
        VALIDATION_PATH,
        MAPPING_SUMMARY_PATH,
    ]:
        print(f"  {output_path}")

    print("\nSector-panel construction completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()