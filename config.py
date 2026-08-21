from pathlib import Path


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MAPPING_DIR = DATA_DIR / "mapping"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_RESULT_DIR = OUTPUT_DIR / "model_results"
LOG_DIR = OUTPUT_DIR / "logs"

# Figure directory
FIGURE_DIR = PROJECT_ROOT / "figures"

# Required DeXposure files
FULL_NETWORK_FILE = (
    RAW_DATA_DIR / "historical-network_week_2020-03-30.json"
)

SMALL_NETWORK_FILE = (
    RAW_DATA_DIR / "historical-network_week_2025-07-01.json"
)

ID_TO_INFO_FILE = MAPPING_DIR / "id_to_info.json"
TOKEN_TO_PROTOCOL_FILE = MAPPING_DIR / "token_to_protocol.json"
REV_MAP_FILE = MAPPING_DIR / "rev_map.json"
META_DF_FILE = MAPPING_DIR / "meta_df.csv"