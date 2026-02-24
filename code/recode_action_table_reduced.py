#!/usr/bin/env python3
"""Recode RL action tables to a reduced action set: increase/decrease/stay/off."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _norm(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


def recode_actions_reduced(
    df: pd.DataFrame,
    action_col: str = "action",
    dose_col: str = "end_interval_dose",
    off_threshold: float = 0.0,
) -> pd.DataFrame:
    if action_col not in df.columns:
        raise ValueError(f"Missing action column: {action_col}")

    out = df.copy()
    action_norm = _norm(out[action_col])

    has_dose = dose_col in out.columns
    dose = pd.to_numeric(out[dose_col], errors="coerce") if has_dose else pd.Series(np.nan, index=out.index)

    reduced = pd.Series(pd.NA, index=out.index, dtype="string")

    # Direct mappings.
    reduced[action_norm.isin({"increase"})] = "increase"
    reduced[action_norm.isin({"decrease"})] = "decrease"
    reduced[action_norm.isin({"off", "discontinue"})] = "off"
    reduced[action_norm.isin({"on", "start"})] = "increase"

    # no_change/stay behavior:
    # - if end dose is off-state (<= threshold), label as off
    # - otherwise label as stay
    no_change_mask = action_norm.isin({"no_change", "stay"})
    if has_dose:
        off_mask = no_change_mask & dose.notna() & (dose <= off_threshold)
        stay_mask = no_change_mask & ~off_mask
        reduced[off_mask] = "off"
        reduced[stay_mask] = "stay"
    else:
        reduced[no_change_mask] = "stay"

    # Any unknown labels default to stay to avoid breaking downstream fixed-interval models.
    unknown = reduced.isna()
    reduced[unknown] = "stay"

    out["action_original"] = out[action_col]
    out["action_reduced"] = reduced
    out["action_recode_rule"] = "reduced4"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recode action labels into {increase,decrease,stay,off}. "
            "Maps start/on->increase and no_change->stay (or off if end dose <= threshold)."
        )
    )
    parser.add_argument("--input-csv", required=True, help="Input action table CSV.")
    parser.add_argument("--output-csv", required=True, help="Output recoded CSV.")
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional summary CSV path. Defaults to <output_stem>_summary.csv.",
    )
    parser.add_argument(
        "--action-col",
        default="action",
        help="Name of action column to recode (default: action).",
    )
    parser.add_argument(
        "--dose-col",
        default="end_interval_dose",
        help=(
            "Dose column used to classify no_change bins as off vs stay "
            "(default: end_interval_dose)."
        ),
    )
    parser.add_argument(
        "--off-threshold",
        type=float,
        default=0.0,
        help="Dose threshold treated as off-state (default: 0.0).",
    )
    parser.add_argument(
        "--replace-action",
        action="store_true",
        help="Overwrite the input action column with action_reduced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    out = recode_actions_reduced(
        df=df,
        action_col=args.action_col,
        dose_col=args.dose_col,
        off_threshold=args.off_threshold,
    )

    if args.replace_action:
        out[args.action_col] = out["action_reduced"]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    summary = (
        out.groupby(["action_reduced"], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["action_reduced"])
    )
    summary_path = (
        Path(args.summary_csv)
        if args.summary_csv
        else output_path.with_name(f"{output_path.stem}_summary.csv")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    print(f"Wrote recoded table: {output_path}")
    print(f"Wrote summary: {summary_path}")
    print("\nReduced action counts:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
