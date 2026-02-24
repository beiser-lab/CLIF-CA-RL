from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from build_fixed_interval_action_table import build_fixed_interval_action_table


def _fixture_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_event_id": 1,
                "hospitalization_id": "h1",
                "action_dttm_utc": "2024-01-01T00:10:00Z",
                "med_category": "norepinephrine",
                "med_group": "vasoactives",
                "med_dose_unit": "mcg/kg/min",
                "action": "start",
                "action_source": "mar_start",
                "med_dose": 0.1,
                "prev_dose": None,
                "delta_dose": None,
                "mar_action_category": "start",
                "mar_action_name": "Restarted",
            },
            {
                "action_event_id": 2,
                "hospitalization_id": "h1",
                "action_dttm_utc": "2024-01-01T00:20:00Z",
                "med_category": "norepinephrine",
                "med_group": "vasoactives",
                "med_dose_unit": "mcg/kg/min",
                "action": "increase",
                "action_source": "mar_dose_change",
                "med_dose": 0.2,
                "prev_dose": 0.1,
                "delta_dose": 0.1,
                "mar_action_category": "dose_change",
                "mar_action_name": "Rate Change",
            },
            {
                "action_event_id": 3,
                "hospitalization_id": "h1",
                "action_dttm_utc": "2024-01-01T02:10:00Z",
                "med_category": "norepinephrine",
                "med_group": "vasoactives",
                "med_dose_unit": "mcg/kg/min",
                "action": "decrease",
                "action_source": "mar_dose_change",
                "med_dose": 0.15,
                "prev_dose": 0.2,
                "delta_dose": -0.05,
                "mar_action_category": "dose_change",
                "mar_action_name": "Rate Change",
            },
            {
                "action_event_id": 4,
                "hospitalization_id": "h1",
                "action_dttm_utc": "2024-01-01T02:20:00Z",
                "med_category": "norepinephrine",
                "med_group": "vasoactives",
                "med_dose_unit": "mcg/kg/min",
                "action": "discontinue",
                "action_source": "mar_stop",
                "med_dose": 0.0,
                "prev_dose": 0.15,
                "delta_dose": -0.15,
                "mar_action_category": "stop",
                "mar_action_name": "Stopped",
            },
        ]
    )


def test_fixed_interval_build_includes_no_change_and_carry_forward() -> None:
    out = build_fixed_interval_action_table(
        events=_fixture_events(),
        interval_minutes=60,
        window_mode="exposure",
        hosp=None,
        default_off_dose=0.0,
    )
    # Window should be [00:00,03:00) with 3 bins.
    assert len(out) == 3
    assert out["action"].tolist() == ["increase", "no_change", "discontinue"]

    # Bin 0 resolved by last event in interval (increase at 00:20).
    assert out.loc[0, "n_events_in_bin"] == 2
    assert out.loc[0, "prev_interval_dose"] == 0.0
    assert out.loc[0, "end_interval_dose"] == 0.2

    # Bin 1 has no events and should hold dose.
    assert out.loc[1, "n_events_in_bin"] == 0
    assert out.loc[1, "action"] == "no_change"
    assert out.loc[1, "prev_interval_dose"] == 0.2
    assert out.loc[1, "end_interval_dose"] == 0.2

    # Bin 2 resolves to discontinue and returns to off dose.
    assert out.loc[2, "n_events_in_bin"] == 2
    assert out.loc[2, "action"] == "discontinue"
    assert out.loc[2, "prev_interval_dose"] == 0.2
    assert out.loc[2, "end_interval_dose"] == 0.0
