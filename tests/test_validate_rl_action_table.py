from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from validate_rl_action_table import validate_action_table


def _get_failed_rows(output, check_name: str) -> int:
    return int(
        output.results.loc[
            output.results["check_name"] == check_name, "failed_rows"
        ].iloc[0]
    )


def _valid_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "action_event_id": 1,
                "hospitalization_id": "h1",
                "action_dttm_utc": "2024-01-01T00:00:00Z",
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
                "action_dttm_utc": "2024-01-01T00:10:00Z",
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
                "action_dttm_utc": "2024-01-01T00:20:00Z",
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
                "action_dttm_utc": "2024-01-01T00:30:00Z",
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
            {
                "action_event_id": 5,
                "hospitalization_id": "h2",
                "action_dttm_utc": "2024-01-02T00:00:00Z",
                "med_category": "vasopressin",
                "med_group": "vasoactives",
                "med_dose_unit": "Units/min",
                "action": "start",
                "action_source": "inferred_positive_after_off",
                "med_dose": 0.03,
                "prev_dose": None,
                "delta_dose": None,
                "mar_action_category": "going",
                "mar_action_name": "New Bag",
            },
            {
                "action_event_id": 6,
                "hospitalization_id": "h2",
                "action_dttm_utc": "2024-01-02T01:00:00Z",
                "med_category": "vasopressin",
                "med_group": "vasoactives",
                "med_dose_unit": "Units/min",
                "action": "discontinue",
                "action_source": "inferred_silent_end",
                "med_dose": 0.03,
                "prev_dose": 0.03,
                "delta_dose": None,
                "mar_action_category": "inferred",
                "mar_action_name": "inferred_silent_end",
            },
        ]
    )


def test_validate_action_table_passes_on_valid_fixture() -> None:
    output = validate_action_table(_valid_fixture(), positive_threshold=0.0)
    assert output.missing_columns == []
    assert output.has_errors is False
    assert int(output.results["failed_rows"].sum()) == 0


def test_validate_action_table_detects_missing_required_columns() -> None:
    df = _valid_fixture().drop(columns=["action"])
    output = validate_action_table(df, positive_threshold=0.0)
    assert output.has_errors is True
    assert "action" in output.missing_columns
    assert _get_failed_rows(output, "missing_required_columns") == 1


def test_validate_action_table_detects_invalid_increase_delta() -> None:
    df = _valid_fixture().copy()
    idx = df["action"] == "increase"
    df.loc[idx, "delta_dose"] = -0.01
    output = validate_action_table(df, positive_threshold=0.0)
    assert _get_failed_rows(output, "increase_delta_not_positive") == 1
    assert output.has_errors is True


def test_validate_action_table_detects_missing_terminal_discontinue() -> None:
    df = _valid_fixture().copy()
    df = df[~((df["hospitalization_id"] == "h2") & (df["action"] == "discontinue"))]
    output = validate_action_table(df, positive_threshold=0.0)
    assert _get_failed_rows(output, "start_groups_without_terminal_discontinue") == 1
    assert output.has_errors is True


def test_validate_action_table_can_disable_terminal_discontinue_check() -> None:
    df = _valid_fixture().copy()
    df = df[~((df["hospitalization_id"] == "h2") & (df["action"] == "discontinue"))]
    output = validate_action_table(
        df,
        positive_threshold=0.0,
        require_terminal_discontinue=False,
    )
    assert _get_failed_rows(output, "start_groups_without_terminal_discontinue") == 0
