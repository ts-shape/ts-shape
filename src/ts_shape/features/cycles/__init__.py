"""Cycles

Utilities to detect and process cycles in timeseries.

Classes:
- CycleExtractor: Build cycles from state/step/value changes.
  - process_persistent_cycle: A True stretch (rising to falling edge) is a cycle.
  - process_trigger_cycle: Each trigger (rising edge) runs to the next one.
  - process_separate_start_end_cycle: Separate start and end signals.
  - process_step_sequence: Entering a start step opens, an end step closes.
  - process_state_change_cycle: Every sample is a boundary.
  - process_value_change_cycle: Any significant value change is a boundary.
  - validate_cycles / detect_overlapping_cycles / suggest_method.

- CycleDataProcessor: Split/merge/group by cycle windows.
  - split_by_cycle: Split values by cycle ranges.
  - merge_dataframes_by_cycle: Annotate values with cycle UUIDs.
  - group_by_cycle_uuid: Group values by cycle key.
  - split_dataframes_by_group: Further split by column groupings.
  - compute_cycle_statistics / compare_cycles / identify_golden_cycles.
  A sample belongs to a cycle when cycle_start <= systime < cycle_end.
"""
