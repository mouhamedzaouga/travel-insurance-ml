"""Constantes partagées par tout le projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "travel_insurance_raw.csv"
CLEAN_CSV = ROOT / "data" / "processed" / "travel_insurance_clean.csv"
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_JSON = ROOT / "reports" / "results.json"

RANDOM_STATE = 42

COLUMN_NAMES = [
    "Agency", "Agency_Type", "Distribution_Channel", "Product_Name",
    "Claim_Status", "Duration", "Destination", "Net_Sales",
    "Commission", "Gender", "Age",
]
NUM_COLS = ["Duration", "Net_Sales", "Commission", "Age"]
CAT_COLS = ["Agency", "Agency_Type", "Distribution_Channel", "Product_Name",
            "Destination", "Gender"]
TARGET = "Claim"            # 1 = sinistre déclaré, 0 = pas de sinistre

# Palette cohérente pour toutes les figures
COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]
