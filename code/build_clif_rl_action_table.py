#!/usr/bin/env python3
"""Build a vasoactive/inotrope RL action table from CLIF continuous MAR data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MED_REQUIRED_COLS = {
    "hospitalization_id",
    "admin_dttm",
    "med_category",
    "med_group",
    "med_dose",
    "med_dose_unit",
    "mar_action_name",
    "mar_action_category",
}
HOSP_REQUIRED_COLS = {"hospitalization_id", "discharge_dttm"}


def parse_csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def filter_medications(
    df: pd.DataFrame, include_groups: list[str], include_categories: list[str]
) -> pd.DataFrame:
    if not include_groups and not include_categories:
        return df.copy()

    mask = pd.Series(False, index=df.index)
    if include_groups:
        group_norm = {g.lower() for g in include_groups}
        mask = mask | df["med_group_norm"].isin(group_norm)
    if include_categories:
        cat_norm = {c.lower() for c in include_categories}
        mask = mask | df["med_category_norm"].isin(cat_norm)
    return df[mask].copy()


def shape_events(frame: pd.DataFrame, action: str, source: str) -> pd.DataFrame:
    cols = [
        "hospitalization_id",
        "med_category",
        "med_group",
        "med_dose_unit",
        "admin_dttm",
        "med_dose",
        "prev_dose",
        "delta_dose",
        "mar_action_category",
        "mar_action_name",
    ]
    out = frame[cols].copy()
    out = out.rename(columns={"admin_dttm": "action_dttm_utc"})
    out["action"] = action
    out["action_source"] = source
    return out


def infer_actions(
    med_df: pd.DataFrame,
    hosp_df: pd.DataFrame | None,
    infer_silent_end: bool,
    positive_threshold: float,
) -> pd.DataFrame:
    df = med_df.copy()
    df["hospitalization_id"] = df["hospitalization_id"].astype("string")
    df["admin_dttm"] = pd.to_datetime(df["admin_dttm"], errors="coerce", utc=True)
    df = df.dropna(
        subset=[
            "hospitalization_id",
            "admin_dttm",
            "med_category",
            "med_dose_unit",
            "mar_action_category",
        ]
    )

    df["med_group"] = df["med_group"].astype("string")
    df["med_category"] = df["med_category"].astype("string")
    df["med_dose_unit"] = df["med_dose_unit"].astype("string")
    df["mar_action_name"] = df["mar_action_name"].astype("string")
    df["mar_action_category"] = df["mar_action_category"].astype("string")

    df["med_group_norm"] = normalize_series(df["med_group"])
    df["med_category_norm"] = normalize_series(df["med_category"])
    df["mar_action_category_norm"] = normalize_series(df["mar_action_category"])

    group_cols = ["hospitalization_id", "med_category", "med_dose_unit"]
    priority = {"start": 0, "dose_change": 1, "going": 2, "verify": 3, "stop": 4, "other": 5}
    df["action_priority"] = df["mar_action_category_norm"].map(priority).fillna(9).astype(int)
    order_cols = group_cols + ["admin_dttm", "action_priority", "mar_action_name"]
    df = df.sort_values(order_cols)

    dose_df = df.dropna(subset=["med_dose"]).copy()
    dose_df["prev_dose"] = dose_df.groupby(group_cols)["med_dose"].shift(1)
    dose_df["delta_dose"] = dose_df["med_dose"] - dose_df["prev_dose"]

    start_exp = shape_events(
        dose_df[dose_df["mar_action_category_norm"] == "start"], "start", "mar_start"
    )
    start_inf_mask = (
        (dose_df["med_dose"] > positive_threshold)
        & (
            dose_df["prev_dose"].isna()
            | (dose_df["prev_dose"] <= positive_threshold)
        )
        & (dose_df["mar_action_category_norm"] != "start")
    )
    start_inf = shape_events(
        dose_df[start_inf_mask], "start", "inferred_positive_after_off"
    )

    dc = dose_df[
        (dose_df["mar_action_category_norm"] == "dose_change")
        & dose_df["prev_dose"].notna()
    ].copy()
    increase = shape_events(
        dc[(dc["prev_dose"] > positive_threshold) & (dc["delta_dose"] > 0)],
        "increase",
        "mar_dose_change",
    )
    decrease = shape_events(
        dc[
            (dc["prev_dose"] > positive_threshold)
            & (dc["delta_dose"] < 0)
            & (dc["med_dose"] > positive_threshold)
        ],
        "decrease",
        "mar_dose_change",
    )
    disc_drop0 = shape_events(
        dc[(dc["prev_dose"] > positive_threshold) & (dc["med_dose"] <= positive_threshold)],
        "discontinue",
        "inferred_drop_to_zero",
    )

    stop = df[df["mar_action_category_norm"] == "stop"].copy()
    stop["prev_dose"] = dose_df["prev_dose"]
    stop["delta_dose"] = dose_df["delta_dose"]
    stop = shape_events(stop, "discontinue", "mar_stop")

    events = pd.concat(
        [start_exp, start_inf, increase, decrease, stop, disc_drop0], ignore_index=True
    )

    events = events.drop_duplicates(
        subset=[
            "hospitalization_id",
            "med_category",
            "med_dose_unit",
            "action_dttm_utc",
            "action",
            "action_source",
            "med_dose",
            "delta_dose",
        ]
    )

    if infer_silent_end:
        starts = (
            events[events["action"] == "start"]
            .groupby(group_cols)["action_dttm_utc"]
            .max()
            .rename("last_start")
        )
        discont = (
            events[events["action"] == "discontinue"]
            .groupby(group_cols)["action_dttm_utc"]
            .max()
            .rename("last_disc")
        )
        last_obs = df.groupby(group_cols)["admin_dttm"].max().rename("last_obs")
        last_pos = (
            dose_df[dose_df["med_dose"] > positive_threshold]
            .groupby(group_cols)["admin_dttm"]
            .max()
            .rename("last_positive")
        )
        last_pos_rows = (
            dose_df[dose_df["med_dose"] > positive_threshold]
            .sort_values(order_cols)
            .groupby(group_cols)
            .tail(1)
            .set_index(group_cols)
        )
        last_pos_dose = last_pos_rows["med_dose"].rename("last_positive_dose")
        last_pos_group = last_pos_rows["med_group"].rename("last_positive_group")

        status = (
            starts.to_frame()
            .join(discont, how="left")
            .join(last_obs, how="left")
            .join(last_pos, how="left")
            .join(last_pos_dose, how="left")
            .join(last_pos_group, how="left")
            .reset_index()
        )

        if hosp_df is not None and not hosp_df.empty:
            hosp = hosp_df.copy()
            hosp["hospitalization_id"] = hosp["hospitalization_id"].astype("string")
            hosp["discharge_dttm"] = pd.to_datetime(
                hosp["discharge_dttm"], errors="coerce", utc=True
            )
            status = status.merge(
                hosp[["hospitalization_id", "discharge_dttm"]],
                on="hospitalization_id",
                how="left",
            )
        else:
            status["discharge_dttm"] = pd.NaT

        open_mask = status["last_start"].notna() & (
            status["last_disc"].isna() | (status["last_start"] > status["last_disc"])
        )
        open_df = status[open_mask].copy()
        open_df = open_df[open_df["last_positive"].notna()]

        use_discharge = open_df["discharge_dttm"].notna() & (
            open_df["discharge_dttm"] >= open_df["last_start"]
        )
        open_df["action_dttm_utc"] = np.where(
            use_discharge, open_df["discharge_dttm"], open_df["last_obs"]
        )
        open_df["med_dose"] = open_df["last_positive_dose"]
        open_df["prev_dose"] = open_df["last_positive_dose"]
        open_df["delta_dose"] = np.nan
        open_df["med_group"] = open_df["last_positive_group"]
        open_df["mar_action_category"] = "inferred"
        open_df["mar_action_name"] = "inferred_silent_end"
        open_df["action"] = "discontinue"
        open_df["action_source"] = "inferred_silent_end"

        silent_cols = [
            "hospitalization_id",
            "med_category",
            "med_group",
            "med_dose_unit",
            "action_dttm_utc",
            "med_dose",
            "prev_dose",
            "delta_dose",
            "mar_action_category",
            "mar_action_name",
            "action",
            "action_source",
        ]
        silent_disc = open_df[silent_cols].dropna(subset=["action_dttm_utc"])
        events = pd.concat([events, silent_disc], ignore_index=True)

    action_order = {"start": 0, "increase": 1, "decrease": 2, "discontinue": 3}
    events["action_order"] = events["action"].map(action_order).fillna(9).astype(int)
    events = events.sort_values(group_cols + ["action_dttm_utc", "action_order", "action_source"])
    events = events.reset_index(drop=True)
    events["action_event_id"] = np.arange(1, len(events) + 1)

    out_cols = [
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
    ]
    return events[out_cols]


def build_summary(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(
            ["med_category", "med_group", "med_dose_unit", "action", "action_source"],
            dropna=False,
        )
        .size()
        .reset_index(name="n_events")
        .sort_values(
            ["med_category", "med_group", "med_dose_unit", "action", "action_source"]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an RL action table from CLIF continuous medication administrations. "
            "Default filter keeps med_group='vasoactives' plus med_category='angiotensin'."
        )
    )
    parser.add_argument(
        "--medication-parquet",
        required=True,
        help="Path to clif_medication_admin_continuous.parquet",
    )
    parser.add_argument(
        "--hospitalization-parquet",
        default="",
        help=(
            "Optional path to clif_hospitalization.parquet. "
            "Used for silent-end discontinuation timing at discharge_dttm."
        ),
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path for RL action event table CSV",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional path for per-med/action summary CSV",
    )
    parser.add_argument(
        "--include-med-groups",
        default="vasoactives",
        help="Comma-separated med_group values to include (default: vasoactives)",
    )
    parser.add_argument(
        "--include-med-categories",
        default="angiotensin",
        help="Comma-separated med_category values to include (default: angiotensin)",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="Dose threshold treated as off (default: 0.0)",
    )
    parser.add_argument(
        "--no-infer-silent-end",
        action="store_true",
        help="Disable inferred discontinuation at discharge/last observation.",
    )
    args = parser.parse_args()

    med_path = Path(args.medication_parquet)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary_csv) if args.summary_csv else out_path.with_name(
        f"{out_path.stem}_summary.csv"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    include_groups = parse_csv_list(args.include_med_groups)
    include_categories = parse_csv_list(args.include_med_categories)

    med_df = pd.read_parquet(med_path)
    require_columns(med_df, MED_REQUIRED_COLS, "medication parquet")

    med_df["med_group_norm"] = normalize_series(med_df["med_group"])
    med_df["med_category_norm"] = normalize_series(med_df["med_category"])
    med_df = filter_medications(med_df, include_groups, include_categories)
    if med_df.empty:
        raise ValueError(
            "No rows left after medication filters. "
            "Adjust --include-med-groups / --include-med-categories."
        )

    hosp_df = None
    if args.hospitalization_parquet:
        hosp_df = pd.read_parquet(args.hospitalization_parquet)
        require_columns(hosp_df, HOSP_REQUIRED_COLS, "hospitalization parquet")

    events = infer_actions(
        med_df=med_df,
        hosp_df=hosp_df,
        infer_silent_end=not args.no_infer_silent_end,
        positive_threshold=args.positive_threshold,
    )
    events.to_csv(out_path, index=False)

    summary = build_summary(events)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote action table: {out_path}")
    print(f"Wrote summary table: {summary_path}")
    print(f"Rows: {len(events):,}")
    print("\nAction counts:")
    print(events["action"].value_counts().to_string())


if __name__ == "__main__":
    main()
