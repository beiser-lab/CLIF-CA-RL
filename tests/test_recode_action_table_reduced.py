from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from recode_action_table_reduced import recode_actions_reduced


def test_recode_actions_reduced_maps_actions_as_expected() -> None:
    df = pd.DataFrame(
        [
            {"action": "start", "end_interval_dose": 0.1},
            {"action": "increase", "end_interval_dose": 0.2},
            {"action": "decrease", "end_interval_dose": 0.1},
            {"action": "discontinue", "end_interval_dose": 0.0},
            {"action": "no_change", "end_interval_dose": 0.3},
            {"action": "no_change", "end_interval_dose": 0.0},
            {"action": "on", "end_interval_dose": 0.5},
        ]
    )
    out = recode_actions_reduced(df, action_col="action", dose_col="end_interval_dose", off_threshold=0.0)
    assert out["action_reduced"].tolist() == [
        "increase",
        "increase",
        "decrease",
        "off",
        "stay",
        "off",
        "increase",
    ]
