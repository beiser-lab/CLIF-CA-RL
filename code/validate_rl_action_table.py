#!/usr/bin/env python3
"""Validate RL medication action-table outputs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_EVENT_COLUMNS = {
    "action_event_id",
    "hospitalization_id",
    "action_dttm_utc",
    "med_category",
    "med_group",
    "med_dose_unit",
    "action",
    "action_source",
    "med_dose",
    "prev_dose",
    "delta_dose",
    "mar_action_category",
    "mar_action_name",
}

VALID_ACTIONS = {"start", "increase", "decrease", "discontinue"}
KEY_DUPLICATE_COLUMNS = [
    "hospitalization_id",
    "med_category",
    "med_dose_unit",
    "action_dttm_utc",
    "action",
    "action_source",
    "med_dose",
    "delta_dose",
]
GROUP_COLUMNS = ["hospitalization_id", "med_category", "med_dose_unit"]


@dataclass
class ValidationOutput:
    results: pd.DataFrame
    missing_columns: list[str]

    @property
    def failing_checks(self) -> pd.DataFrame:
        return self.results[self.results["failed_rows"] > 0].copy()

    @property
    def has_errors(self) -> bool:
        return not self.failing_checks.empty


def _result(check_name: str, failed_rows: int, description: str) -> dict[str, object]:
    return {
        "check_name": check_name,
        "failed_rows": int(failed_rows),
        "description": description,
    }


def _lower_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def validate_action_table(
    df: pd.DataFrame,
    positive_threshold: float = 0.0,
    require_terminal_discontinue: bool = True,
) -> ValidationOutput:
    missing = sorted(REQUIRED_EVENT_COLUMNS - set(df.columns))
    checks = [
        _result(
            "missing_required_columns",
            len(missing),
            "All required action-table columns are present.",
        )
    ]

    if missing:
        return ValidationOutput(results=pd.DataFrame(checks), missing_columns=missing)

    work = df.copy()
    work["action_dttm_utc"] = pd.to_datetime(work["action_dttm_utc"], errors="coerce", utc=True)
    work["med_dose"] = pd.to_numeric(work["med_dose"], errors="coerce")
    work["prev_dose"] = pd.to_numeric(work["prev_dose"], errors="coerce")
    work["delta_dose"] = pd.to_numeric(work["delta_dose"], errors="coerce")

    action = _lower_text(work["action"])
    source = _lower_text(work["action_source"])
    mar_category = _lower_text(work["mar_action_category"])

    checks.extend(
        [
            _result(
                "invalid_action_values",
                (~action.isin(VALID_ACTIONS)).sum(),
                "Actions are limited to start/increase/decrease/discontinue.",
            ),
            _result(
                "missing_action_timestamps",
                work["action_dttm_utc"].isna().sum(),
                "All action rows have parseable timestamps.",
            ),
            _result(
                "duplicate_action_event_id",
                work["action_event_id"].duplicated().sum(),
                "action_event_id values are unique.",
            ),
            _result(
                "duplicate_key_rows",
                work.duplicated(subset=KEY_DUPLICATE_COLUMNS).sum(),
                "No exact duplicate event keys exist.",
            ),
        ]
    )

    inferred_start = source == "inferred_positive_after_off"
    checks.append(
        _result(
            "start_inferred_nonpositive_dose",
            (inferred_start & ~(work["med_dose"] > positive_threshold)).sum(),
            "Inferred starts require med_dose > threshold.",
        )
    )

    increase = action == "increase"
    checks.extend(
        [
            _result(
                "increase_delta_not_positive",
                (increase & ~(work["delta_dose"] > 0)).sum(),
                "Increase events require delta_dose > 0.",
            ),
            _result(
                "increase_prev_not_positive",
                (increase & ~(work["prev_dose"] > positive_threshold)).sum(),
                "Increase events require prev_dose > threshold.",
            ),
        ]
    )

    decrease = action == "decrease"
    checks.extend(
        [
            _result(
                "decrease_delta_not_negative",
                (decrease & ~(work["delta_dose"] < 0)).sum(),
                "Decrease events require delta_dose < 0.",
            ),
            _result(
                "decrease_prev_not_positive",
                (decrease & ~(work["prev_dose"] > positive_threshold)).sum(),
                "Decrease events require prev_dose > threshold.",
            ),
            _result(
                "decrease_current_not_positive",
                (decrease & ~(work["med_dose"] > positive_threshold)).sum(),
                "Decrease events require current med_dose > threshold.",
            ),
        ]
    )

    drop0 = source == "inferred_drop_to_zero"
    checks.extend(
        [
            _result(
                "drop0_prev_not_positive",
                (drop0 & ~(work["prev_dose"] > positive_threshold)).sum(),
                "Drop-to-zero discontinues require prev_dose > threshold.",
            ),
            _result(
                "drop0_current_positive",
                (drop0 & (work["med_dose"] > positive_threshold)).sum(),
                "Drop-to-zero discontinues require med_dose <= threshold.",
            ),
        ]
    )

    mar_stop = source == "mar_stop"
    checks.extend(
        [
            _result(
                "mar_stop_action_not_discontinue",
                (mar_stop & (action != "discontinue")).sum(),
                "mar_stop source rows should map to discontinue action.",
            ),
            _result(
                "mar_stop_mar_category_not_stop",
                (mar_stop & (mar_category != "stop")).sum(),
                "mar_stop source rows should have mar_action_category=stop.",
            ),
        ]
    )

    silent_end = source == "inferred_silent_end"
    checks.extend(
        [
            _result(
                "silent_end_action_not_discontinue",
                (silent_end & (action != "discontinue")).sum(),
                "inferred_silent_end rows should map to discontinue action.",
            ),
            _result(
                "silent_end_mar_category_not_inferred",
                (silent_end & (mar_category != "inferred")).sum(),
                "inferred_silent_end rows should have mar_action_category=inferred.",
            ),
        ]
    )

    if require_terminal_discontinue:
        starts = (
            work[action == "start"]
            .groupby(GROUP_COLUMNS)["action_dttm_utc"]
            .min()
            .rename("first_start")
        )
        discont = (
            work[action == "discontinue"]
            .groupby(GROUP_COLUMNS)["action_dttm_utc"]
            .max()
            .rename("last_discontinue")
        )
        joined = starts.to_frame().join(discont, how="left")
        missing_terminal = joined[
            joined["last_discontinue"].isna()
            | (joined["last_discontinue"] < joined["first_start"])
        ]
        checks.append(
            _result(
                "start_groups_without_terminal_discontinue",
                len(missing_terminal),
                (
                    "Each stay-med-unit with a start should have a discontinue "
                    "at or after first_start."
                ),
            )
        )
    else:
        checks.append(
            _result(
                "start_groups_without_terminal_discontinue",
                0,
                "Check disabled by caller.",
            )
        )

    return ValidationOutput(results=pd.DataFrame(checks), missing_columns=missing)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RL action tables generated from CLIF medication data."
    )
    parser.add_argument(
        "--action-table-csv",
        required=True,
        help="Path to RL action table CSV output.",
    )
    parser.add_argument(
        "--report-csv",
        default="",
        help="Optional output path for validation-check summary CSV.",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="Dose threshold treated as off (default: 0.0).",
    )
    parser.add_argument(
        "--no-require-terminal-discontinue",
        action="store_true",
        help="Disable the terminal-discontinue check for each started infusion group.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Return exit code 0 even if validation checks fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.action_table_csv)
    df = pd.read_csv(input_path)

    output = validate_action_table(
        df=df,
        positive_threshold=args.positive_threshold,
        require_terminal_discontinue=not args.no_require_terminal_discontinue,
    )

    if args.report_csv:
        report_path = Path(args.report_csv)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        output.results.to_csv(report_path, index=False)
        print(f"Wrote validation report: {report_path}")

    print(output.results.to_string(index=False))
    if output.missing_columns:
        print("\nMissing required columns:")
        for col in output.missing_columns:
            print(f"- {col}")

    failures = int((output.results["failed_rows"] > 0).sum())
    total_failed_rows = int(output.results["failed_rows"].sum())
    print(f"\nChecks with failures: {failures}")
    print(f"Total failed rows across checks: {total_failed_rows}")

    if output.has_errors and not args.allow_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
