# CLIF RL Action Table Builder

This repo contains a reusable script to convert CLIF continuous medication administration data into an RL action table for vasoactive infusions.

Script: [`code/build_clif_rl_action_table.py`](/Users/davidbeiser/Documents/CLIF-CA-RL/code/build_clif_rl_action_table.py)

## CLIF Template Layout

This repository now follows the CLIF project template-style top-level structure:

- `code/` (analysis scripts)
- `config/` (site/project configuration files)
- `outlier-thresholds/` (reference threshold files)
- `output/` (analysis outputs)
- `renv/` (R environment files if used)
- `utils/` (shared helper functions)

## What This Builds

An event-level action table with one row per inferred RL action:

- `start`
- `increase`
- `decrease`
- `discontinue`

The table is derived from `clif_medication_admin_continuous.parquet` and optionally `clif_hospitalization.parquet`.

## Default Medication Scope

By default, the script includes:

- `med_group = vasoactives`
- plus `med_category = angiotensin`

In this CLIF extract, that captures:

- Pressors: `norepinephrine`, `phenylephrine`, `vasopressin`, `angiotensin`
- Inotropes: `dobutamine`, `milrinone`, `isoproterenol`
- Mixed vasoactive agents: `epinephrine`, `dopamine`

Site mappings can differ. You can override inclusion with:

- `--include-med-groups`
- `--include-med-categories`

## Action Definitions

Actions are built separately within each:

- `hospitalization_id + med_category + med_dose_unit`
- Default dose threshold: `0.0` (set by `--positive-threshold`, default `0.0`)
  - This threshold is used for inferred start/stop boundaries and dose-change directionality.

### `start`

- Explicit: `mar_action_category = start`
- Inferred: first positive dose after prior dose is `<= threshold` or missing
  - source: `inferred_positive_after_off`

### `increase`

- `mar_action_category = dose_change`
- prior dose `> threshold`
- `delta_dose = med_dose - prev_dose > 0`

### `decrease`

- `mar_action_category = dose_change`
- prior dose `> threshold`
- `delta_dose < 0`
- current dose still `> threshold` (not a discontinuation)

### `discontinue`

Priority of signatures:

1. Explicit stop:
   - `mar_action_category = stop`
   - source: `mar_stop`
2. Implicit drop-to-zero:
   - `dose_change` row with prior dose `> threshold` and current dose `<= threshold`
   - source: `inferred_drop_to_zero`
3. Silent-end inference:
   - if an infusion has a `start` with no later discontinue signal, infer discontinue at:
     - `discharge_dttm` (if available and after last start), else
     - last observed med timestamp for that stay-med-unit
   - source: `inferred_silent_end`

## Output Columns

Main CSV columns:

- `action_event_id`
- `hospitalization_id`
- `action_dttm_utc`
- `med_category`
- `med_group`
- `med_dose_unit`
- `action`
- `action_source`
- `med_dose`
- `prev_dose`
- `delta_dose`
- `mar_action_category`
- `mar_action_name`

Summary CSV columns:

- `med_category`
- `med_group`
- `med_dose_unit`
- `action`
- `action_source`
- `n_events`

Default summary output name (if `--summary-csv` is not provided):

- `<output_csv_stem>_summary.csv`
- Example in this repo: `output/rl_action_table_vaso_inotrope_summary.csv`

## How To Run

Use an environment with `pandas`, `numpy`, and parquet support (`pyarrow` or `fastparquet`).

Example:

```bash
conda run -n cares-clif-linkage python /Users/davidbeiser/Documents/CLIF-CA-RL/code/build_clif_rl_action_table.py \
  --medication-parquet /path/to/clif_medication_admin_continuous.parquet \
  --hospitalization-parquet /path/to/clif_hospitalization.parquet \
  --output-csv /path/to/output/rl_action_table_vaso_inotrope.csv \
  --summary-csv /path/to/output/rl_action_table_vaso_inotrope_summary.csv
```

## Validation

Validate the generated action table and fail fast on invariant violations:

```bash
conda run -n cares-clif-linkage python /Users/davidbeiser/Documents/CLIF-CA-RL/code/validate_rl_action_table.py \
  --action-table-csv /path/to/output/rl_action_table_vaso_inotrope.csv \
  --report-csv /path/to/output/rl_action_table_validation_summary.csv
```

- Default threshold used by validator logic is `0.0` (same semantics as builder).
- By default, validator exits with code `1` when any checks fail.

## Fixed-Interval Actions (`no_change`)

For fixed-interval RL models, convert event actions into interval bins and explicitly include `no_change` when no event occurred in the bin:

```bash
conda run -n cares-clif-linkage python /Users/davidbeiser/Documents/CLIF-CA-RL/code/build_fixed_interval_action_table.py \
  --event-action-csv /path/to/output/rl_action_table_vaso_inotrope.csv \
  --output-csv /path/to/output/rl_action_table_vaso_inotrope_q60m.csv \
  --summary-csv /path/to/output/rl_action_table_vaso_inotrope_q60m_summary.csv \
  --interval-minutes 60 \
  --window-mode exposure
```

Key behavior:

- Bins are generated per `hospitalization_id + med_category + med_dose_unit`.
- If a bin has no events, action is `no_change`.
- If a bin has multiple events, the last event in that bin resolves the interval action.
- `end_interval_dose` is carried forward across `no_change` bins and reset to off-dose on `discontinue`.

## Important Notes For Cross-Site Use

- `mar_action_category` and `med_group` mappings can vary by site ETL.
- Keep unit-specific actions separate (`med_dose_unit` is part of the grouping key).
- If your site has sparse stop charting, keep silent-end inference enabled.
- If you want strict MAR-only labels, run with `--no-infer-silent-end`.

## Tests

Run the pytest suite:

```bash
conda run -n cares-clif-linkage pytest /Users/davidbeiser/Documents/CLIF-CA-RL/tests -q
```
