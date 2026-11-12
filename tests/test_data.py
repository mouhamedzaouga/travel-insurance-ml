import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from travel_insurance import config as cfg, data  # noqa: E402


def _toy():
    return pd.DataFrame({
        "Agency": ["A", "A", "B", "B", "C"],
        "Agency_Type": ["Airlines"] * 5,
        "Distribution_Channel": ["Online"] * 5,
        "Product_Name": ["P"] * 5,
        "Claim_Status": ["No", "No", "Yes", "No", np.nan],
        "Duration": [10, 10, 5, -2, np.nan],
        "Destination": ["X"] * 4 + [np.nan],
        "Net_Sales": [20, 20, 30, 10, np.nan],
        "Commission": [0, 0, 5, 1, np.nan],
        "Gender": [np.nan, np.nan, "M", "F", np.nan],
        "Age": [30, 30, 118, 40, np.nan],
    })


def test_clean_rules():
    df, log = data.clean(_toy())
    assert log["n_empty_rows"] == 1
    assert log["n_duplicates"] == 1
    assert log["n_invalid_duration"] == 1
    assert log["n_invalid_age"] == 1
    assert len(df) == 2
    assert df["Age"].isna().sum() == 1            # 118 -> NaN, pas imputé
    assert set(df["Gender"]) <= {"M", "F", "Unknown"}
    assert df["Claim"].tolist() == [0, 1]


def test_real_data_invariants():
    df, log = data.clean(data.load_raw())
    assert df.duplicated(subset=[c for c in df.columns if c != "Claim"]).sum() == 0
    assert (df["Duration"] > 0).all()
    assert df["Age"].dropna().between(0, 100).all()
    assert 0.005 < df["Claim"].mean() < 0.02
    assert list(data.load_raw().columns) == cfg.COLUMN_NAMES
