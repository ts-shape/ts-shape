import pandas as pd  # type: ignore
from ts_shape.features.cycles.cycles_extractor import CycleExtractor
from ts_shape.features.cycles.cycle_processor import CycleDataProcessor


def make_cycle_df():
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:05:00",
                    "2023-01-01 00:10:00",
                    "2023-01-01 00:15:00",
                    "2023-01-01 00:20:00",
                ]
            ),
            "value_bool": [True, True, False, True, False],
            "value_integer": [0, 0, 1, 0, 1],
            "value_double": [0.0, 0.0, 0.0, 0.0, 0.0],
            "value_string": ["", "", "", "", ""],
        }
    )


def test_cycle_extractor_basic_flows():
    df = make_cycle_df()
    ce = CycleExtractor(df, start_uuid="start")

    persist = ce.process_persistent_cycle()
    assert {"cycle_start", "cycle_end", "cycle_uuid"}.issubset(persist.columns)
    assert len(persist) >= 1

    trigger = ce.process_trigger_cycle()
    assert {"cycle_start", "cycle_end", "cycle_uuid"}.issubset(trigger.columns)

    sep = ce.process_separate_start_end_cycle()
    assert {"cycle_start", "cycle_end", "cycle_uuid"}.issubset(sep.columns)

    steps = ce.process_step_sequence(start_step=0, end_step=1)
    assert {"cycle_start", "cycle_end", "cycle_uuid"}.issubset(steps.columns)

    state = ce.process_state_change_cycle()
    assert {"cycle_start", "cycle_end", "cycle_uuid"}.issubset(state.columns)

    value_change = ce.process_value_change_cycle()
    assert {"cycle_start", "cycle_end", "cycle_uuid"}.issubset(value_change.columns)


def test_cycle_data_processor_split_merge_group():
    cycles = pd.DataFrame(
        {
            "cycle_start": pd.to_datetime(
                ["2023-01-01 00:00:00", "2023-01-01 00:12:00"]
            ),
            "cycle_end": pd.to_datetime(["2023-01-01 00:10:00", "2023-01-01 00:20:00"]),
            "cycle_uuid": ["c1", "c2"],
        }
    )
    values = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2023-01-01 00:01:00", "2023-01-01 00:05:00", "2023-01-01 00:15:00"]
            ),
            "value_integer": [1, 2, 3],
        }
    )

    proc = CycleDataProcessor(cycles, values)
    split = proc.split_by_cycle()
    assert set(split.keys()) == {"c1", "c2"}

    merged = proc.merge_dataframes_by_cycle()
    assert "cycle_uuid" in merged.columns
    assert set(merged["cycle_uuid"].unique()) <= {"c1", "c2"}

    groups = proc.group_by_cycle_uuid(merged)
    assert len(groups) == 2

    # Split by group helper
    split_groups = proc.split_dataframes_by_group(groups, "value_integer")
    assert len(split_groups) >= 2
