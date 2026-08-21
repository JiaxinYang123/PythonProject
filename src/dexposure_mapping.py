import json
import math
from pathlib import Path


def normalise_identifier(value):
    """
    Convert protocol and token identifiers to a consistent string format.
    """

    if value is None:
        return None

    if isinstance(value, float):

        if math.isnan(value):
            return None

        if value.is_integer():
            return str(int(value))

    text = str(value).strip()

    if text == "":
        return None

    return text


def load_json_file(path):
    """
    Load an official DeXposure JSON mapping file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_official_mappings(
    id_to_info_path,
    token_to_protocol_path,
):
    """
    Load official DeXposure protocol and token mappings.

    No new mapping model or similarity calculation is performed.
    """

    raw_id_to_info = load_json_file(
        id_to_info_path
    )

    token_to_protocol = load_json_file(
        token_to_protocol_path
    )

    id_to_info = {
        normalise_identifier(identifier): information
        for identifier, information in raw_id_to_info.items()
    }

    token_index = {}

    for token_group, token_records in token_to_protocol.items():

        for token_name, token_information in token_records.items():

            token_key = normalise_identifier(
                token_name
            )

            mapped_identifier = normalise_identifier(
                token_information.get("id")
            )

            token_index[token_key] = {
                "token_group": token_group,
                "mapped_identifier": mapped_identifier,
            }

    return id_to_info, token_index


def resolve_official_category(
    node_identifier,
    id_to_info,
    token_index,
):
    """
    Resolve a node using only official DeXposure mapping information.

    Residual token groups are kept separate at this stage. They are not
    automatically assigned to a protocol sector.
    """

    node_key = normalise_identifier(
        node_identifier
    )

    if node_key is None:
        return {
            "category": "UNMAPPED",
            "mapping_method": "missing identifier",
        }

    # Direct protocol-ID mapping
    if node_key in id_to_info:

        category = id_to_info[node_key].get(
            "category",
            "Unknown",
        )

        return {
            "category": category,
            "mapping_method": "id_to_info",
        }

    # Token mapping supplied by DeXposure
    token_record = token_index.get(node_key)

    if token_record is not None:

        mapped_identifier = token_record[
            "mapped_identifier"
        ]

        token_group = token_record[
            "token_group"
        ]

        # A token mapped to a known protocol inherits that
        # protocol's official broad category.
        if mapped_identifier in id_to_info:

            category = id_to_info[
                mapped_identifier
            ].get(
                "category",
                "Unknown",
            )

            return {
                "category": category,
                "mapping_method": "token mapped to protocol",
            }

        # Primary tokens remain a separate official endpoint class.
        if token_group == "PRIMARY":

            return {
                "category": "Primary Market Tokens",
                "mapping_method": "primary token",
            }

        # Do not guess a protocol category for residual token groups.
        return {
            "category": f"TOKEN_{token_group}",
            "mapping_method": "residual token group",
        }

    return {
        "category": "UNMAPPED",
        "mapping_method": "not found in official mappings",
    }
# ============================================================
# ANALYSIS-LEVEL CATEGORY CROSSWALK
# ============================================================

ANALYSIS_CATEGORIES = [
    "Asset Management",
    "Infrastructure, Services & Financial Products",
    "Lending, Borrowing & Real World Assets",
    "Privacy & Security",
    "Trading & Exchanges",
    "Primary Market Tokens",
    "Other / Unknown",
]


def convert_to_analysis_category(raw_category):
    """
    Convert official DeXposure labels and residual mapping outputs into
    the seven analysis categories used in this study.

    The five principal protocol categories and Primary Market Tokens
    are retained separately. Residual and unidentified labels are
    retained in a common Other / Unknown category rather than dropped.
    """

    directly_retained = {
        "Asset Management",
        "Infrastructure, Services & Financial Products",
        "Lending, Borrowing & Real World Assets",
        "Privacy & Security",
        "Trading & Exchanges",
        "Primary Market Tokens",
    }

    if raw_category in directly_retained:
        return raw_category

    return "Other / Unknown"