#!/usr/bin/env python3
"""Convert event-level RL actions into fixed-interval actions with explicit no_change."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_EVENT_COLS = {
    "hospitalization_id",
    "action_dttm_utc",
    "med_category",
    "med_dose_unit",
    "action",
    "action_source",
    "med_dose",
}

OPTIONAL_EVENT_COLS = {"med_group", "action_event_id"}
VALID_EVENT_ACTIONS = {"start", "increase", "decrease", "discontinue"}
GROUP_COLS = ["hospitalization_id", "med_category", "med_dose_unit"]


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _normalize_action_col(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def _choose_med_group(s: pd.Series) -> str:
    vals = s.dropna().astype(str)
    if vals.empty:
        return ""
    mode = vals.mode()
    return mode.iloc[0] if not mode.empty else vals.iloc[0]


def _ceil_right_boundary(ts: pd.Timestamp, interval: pd.Timedelta) -> pd.Timestamp:
    out = ts.ceil(interval)
    if out <= ts:
        out = out + interval
    return out


def _build_group_intervals(
    g: pd.DataFrame,
    interval: pd.Timedelta,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    default_off_dose: float,
) -> pd.DataFrame:
    g = g.sort_values(["action_dttm_utc", "action_event_id"]).reset_index(drop=True)

    n_bins = int(np.ceil((window_end - window_start) / interval))
    if n_bins <= 0:
        return pd.DataFrame()

    starts = pd.date_range(start=window_start, periods=n_bins, freq=interval, tz="UTC")
    ends = starts + interval

    secs = interval.total_seconds()
    bin_idx = np.floor((g["action_dttm_utc"] - window_start).dt.total_seconds() / secs).astype(int)
    bin_idx = bin_idx.clip(lower=0, upper=n_bins - 1)
    g = g.assign(bin_idx=bin_idx)

    rows: list[dict[str, object]] = []
    current_dose = default_off_dose

    for i in range(n_bins):
        sub = g[g["bin_idx"] == i]
        prev_interval_dose = current_dose

        if sub.empty:
            action = "no_change"
            action_source = "none"
            resolved_event_id = pd.NA
            first_event_dttm_utc = pd.NaT
            last_event_dttm_utc = pd.NaT
            n_events_in_bin = 0
            actions_in_bin = ""
            sources_in_bin = ""
            med_dose_at_action = np.nan
        else:
            sub = sub.sort_values(["action_dttm_utc", "action_event_id"])
            resolved = sub.iloc[-1]
            action = str(resolved["action"])
            action_source = str(resolved["action_source"])
            resolved_event_id = resolved.get("action_event_id", pd.NA)
            first_event_dttm_utc = sub["action_dttm_utc"].iloc[0]
            last_event_dttm_utc = sub["action_dttm_utc"].iloc[-1]
            n_events_in_bin = len(sub)
            actions_in_bin = "|".join(sub["action"].astype(str).tolist())
            sources_in_bin = "|".join(sub["action_source"].astype(str).tolist())
            med_dose_at_action = resolved["med_dose"]

            if action == "discontinue":
                current_dose = default_off_dose
            elif pd.notna(med_dose_at_action):
                current_dose = float(med_dose_at_action)

        rows.append(
            {
                "hospitalization_id": g["hospitalization_id"].iloc[0],
                "med_category": g["med_category"].iloc[0],
                "med_group": _choose_med_group(g["med_group"]) if "med_group" in g.columns else "",
                "med_dose_unit": g["med_dose_unit"].iloc[0],
                "interval_index": i,
                "interval_start_utc": starts[i],
                "interval_end_utc": ends[i],
                "action": action,
                "action_source": action_source,
                "resolved_event_id": resolved_event_id,
                "n_events_in_bin": n_events_in_bin,
                "first_event_dttm_utc": first_event_dttm_utc,
                "last_event_dttm_utc": last_event_dttm_utc,
                "actions_in_bin": actions_in_bin,
                "action_sources_in_bin": sources_in_bin,
                "prev_interval_dose": prev_interval_dose,
                "med_dose_at_action": med_dose_at_action,
                "end_interval_dose": current_dose,
            }
        )

    return pd.DataFrame(rows)


def build_fixed_interval_action_table(
    events: pd.DataFrame,
    interval_minutes: int,
    window_mode: str = "exposure",
    hosp: pd.DataFrame | None = None,
    default_off_dose: float = 0.0,
) -> pd.DataFrame:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be > 0")

    require_columns(events, REQUIRED_EVENT_COLS, "event action table")
    work = events.copy()
    work["hospitalization_id"] = work["hospitalization_id"].astype("string")
    work["med_category"] = work["med_category"].astype("string")
    work["med_dose_unit"] = work["med_dose_unit"].astype("string")
    if "med_group" not in work.columns:
        work["med_group"] = ""
    if "action_event_id" not in work.columns:
        work["action_event_id"] = np.arange(1, len(work) + 1)

    work["action_dttm_utc"] = pd.to_datetime(work["action_dttm_utc"], errors="coerce", utc=True)
    work["med_dose"] = pd.to_numeric(work["med_dose"], errors="coerce")
    work["action"] = _normalize_action_col(work["action"])
    work["action_source"] = _normalize_action_col(work["action_source"])
    work = work.dropna(subset=["hospitalization_id", "med_category", "med_dose_unit", "action_dttm_utc"])
    work = work[work["action"].isin(VALID_EVENT_ACTIONS)].copy()
    if work.empty:
        raise ValueError("No valid action rows after filtering and timestamp parsing.")

    interval = pd.Timedelta(minutes=interval_minutes)

    hosp_lookup: pd.DataFrame | None = None
    if window_mode == "hospitalization":
        if hosp is None:
            raise ValueError("--window-mode hospitalization requires hospitalization data.")
        required_hosp = {"hospitalization_id", "admission_dttm", "discharge_dttm"}
        require_columns(hosp, required_hosp, "hospitalization table")
        hosp_lookup = hosp.copy()
        hosp_lookup["hospitalization_id"] = hosp_lookup["hospitalization_id"].astype("string")
        hosp_lookup["admission_dttm"] = pd.to_datetime(
            hosp_lookup["admission_dttm"], errors="coerce", utc=True
        )
        hosp_lookup["discharge_dttm"] = pd.to_datetime(
            hosp_lookup["discharge_dttm"], errors="coerce", utc=True
        )
        hosp_lookup = hosp_lookup[["hospitalization_id", "admission_dttm", "discharge_dttm"]]

    out = []
    for _, g in work.groupby(GROUP_COLS, sort=False):
        min_t = g["action_dttm_utc"].min()
        max_t = g["action_dttm_utc"].max()

        start = min_t.floor(interval)
        end = _ceil_right_boundary(max_t, interval)

        if window_mode == "hospitalization" and hosp_lookup is not None:
            h = hosp_lookup[hosp_lookup["hospitalization_id"] == g["hospitalization_id"].iloc[0]]
            if not h.empty:
                adm = h["admission_dttm"].iloc[0]
                dis = h["discharge_dttm"].iloc[0]
                if pd.notna(adm):
                    start = adm.floor(interval)
                if pd.notna(dis):
                    end = _ceil_right_boundary(dis, interval)
                if end <= start:
                    end = _ceil_right_boundary(max_t, interval)

        group_intervals = _build_group_intervals(
            g=g,
            interval=interval,
            window_start=start,
            window_end=end,
            default_off_dose=default_off_dose,
        )
        if not group_intervals.empty:
            out.append(group_intervals)

    if not out:
        raise ValueError("No interval rows were generated.")

    result = pd.concat(out, ignore_index=True)
    result = result.sort_values(
        ["hospitalization_id", "med_category", "med_dose_unit", "interval_start_utc"]
    ).reset_index(drop=True)
    result["fixed_action_id"] = np.arange(1, len(result) + 1)

    cols = [
        "fixed_action_id",
        "hospitalization_id",
        "med_category",
        "med_group",
        "med_dose_unit",
        "interval_index",
        "interval_start_utc",
        "interval_end_utc",
        "action",
        "action_source",
        "resolved_event_id",
        "n_events_in_bin",
        "first_event_dttm_utc",
        "last_event_dttm_utc",
        "actions_in_bin",
        "action_sources_in_bin",
        "prev_interval_dose",
        "med_dose_at_action",
        "end_interval_dose",
    ]
    return result[cols]


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["med_category", "med_dose_unit", "action"], dropna=False)
        .size()
        .reset_index(name="n_intervals")
        .sort_values(["med_category", "med_dose_unit", "action"])
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert event-level RL action table to fixed intervals with explicit no_change bins."
        )
    )
    parser.add_argument(
        "--event-action-csv",
        required=True,
        help="Input event-level action table CSV (output of build_clif_rl_action_table.py).",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output fixed-interval action table CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional summary output CSV. Defaults to <output_stem>_summary.csv.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=60,
        help="Fixed interval size in minutes (default: 60).",
    )
    parser.add_argument(
        "--window-mode",
        choices=["exposure", "hospitalization"],
        default="exposure",
        help=(
            "exposure: intervals span first->last action per stay-med-unit; "
            "hospitalization: intervals span admission->discharge."
        ),
    )
    parser.add_argument(
        "--hospitalization-parquet",
        default="",
        help=(
            "Required if --window-mode hospitalization. "
            "Path to clif_hospitalization.parquet with admission/discharge timestamps."
        ),
    )
    parser.add_argument(
        "--default-off-dose",
        type=float,
        default=0.0,
        help="Dose value used to represent off-state in interval carry-forward (default: 0.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pd.read_csv(args.event_action_csv)

    hosp_df = None
    if args.window_mode == "hospitalization":
        if not args.hospitalization_parquet:
            raise ValueError("--window-mode hospitalization requires --hospitalization-parquet")
        hosp_df = pd.read_parquet(args.hospitalization_parquet)

    output = build_fixed_interval_action_table(
        events=events,
        interval_minutes=args.interval_minutes,
        window_mode=args.window_mode,
        hosp=hosp_df,
        default_off_dose=args.default_off_dose,
    )

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    summary = build_summary(output)
    summary_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else out_path.with_name(f"{out_path.stem}_summary.csv")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote fixed-interval action table: {out_path}")
    print(f"Wrote summary: {summary_path}")
    print(f"Rows: {len(output):,}")
    print("\nAction counts:")
    print(output["action"].value_counts().to_string())


if __name__ == "__main__":
    main()
