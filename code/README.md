# Code

Scripts for building RL-ready medication action tables from CLIF data.

- `build_clif_rl_action_table.py`: create event-level `start/increase/decrease/discontinue` action labels.
- `build_fixed_interval_action_table.py`: convert event-level actions into fixed-interval bins and emit explicit `no_change`.
- `validate_rl_action_table.py`: validate action-table invariants and return non-zero on validation failure.
