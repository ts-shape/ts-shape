"""Tests for :class:`MachineStatusNormalizer`.

The assertions worth reading twice are the ones about *duration*: interval ends
follow step semantics (a state holds until the next update), data gaps become
``UNKNOWN`` rather than extending the last state, and window splitting keeps
per-hour seconds summing to the per-day figure.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.events._output import validate_event_output
from ts_shape.events.production.signals.spec import (
    MachineState,
    StatusMode,
    StatusSpec,
)
from ts_shape.events.production.signals.status import MachineStatusNormalizer

BASE = pd.Timestamp("2025-01-01 06:00:00")
HORIZON = pd.Timestamp("2025-01-01 12:00:00")


def sig(uuid: str, offsets: list[str], values: list, column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": [BASE + pd.Timedelta(o) for o in offsets],
            "uuid": uuid,
            column: values,
        }
    )


def normalizer(
    frame: pd.DataFrame, spec: StatusSpec, *, until=HORIZON
) -> MachineStatusNormalizer:
    return MachineStatusNormalizer(frame, spec, machine_id="M1", until=until)


def states(norm: MachineStatusNormalizer) -> list[str]:
    return norm.status_intervals()["state"].tolist()


def seconds(norm: MachineStatusNormalizer, state: str) -> float:
    intervals = norm.status_intervals()
    return float(intervals.loc[intervals["state"] == state, "duration_seconds"].sum())


# ---------------------------------------------------------------------------
# Boolean
# ---------------------------------------------------------------------------
class TestBoolean:
    def test_maps_true_and_false_to_the_configured_states(self):
        frame = sig("s", ["0h", "1h", "2h"], [True, False, True], "value_bool")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert states(norm) == ["producing", "idle", "producing"]

    def test_intervals_end_where_the_next_begins_not_at_the_last_repeat(self):
        # A change-triggered signal reports once; the state holds for an hour.
        frame = sig("s", ["0h", "1h"], [True, False], "value_bool")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        intervals = norm.status_intervals()
        assert intervals["duration_seconds"].tolist() == [3600.0, 18000.0]

    def test_the_last_interval_closes_at_the_horizon(self):
        frame = sig("s", ["0h"], [True], "value_bool")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert norm.status_intervals()["end"].iloc[0] == HORIZON

    def test_without_a_horizon_the_last_interval_ends_at_the_last_reading(self):
        frame = sig("s", ["0h", "1h"], [True, False], "value_bool")
        norm = normalizer(
            frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"), until=None
        )
        # The trailing zero-length interval is dropped rather than reported.
        assert states(norm) == ["producing"]

    def test_custom_false_state(self):
        frame = sig("s", ["0h", "1h"], [True, False], "value_bool")
        norm = normalizer(
            frame,
            StatusSpec(
                mode=StatusMode.BOOLEAN, uuid="s", false_state=MachineState.DOWN
            ),
        )
        assert states(norm) == ["producing", "down"]

    def test_reads_string_encoded_run_flags(self):
        frame = sig("s", ["0h", "1h"], ["RUN", "STOP"], "value_string")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert states(norm) == ["producing", "idle"]

    def test_consecutive_repeats_collapse_into_one_interval(self):
        frame = sig(
            "s", ["0h", "1h", "2h", "3h"], [True, True, True, False], "value_bool"
        )
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert states(norm) == ["producing", "idle"]
        assert seconds(norm, "producing") == 3 * 3600.0


# ---------------------------------------------------------------------------
# Code map / PackML / stacklight
# ---------------------------------------------------------------------------
class TestCodeMap:
    SPEC = StatusSpec(
        mode=StatusMode.CODE_MAP,
        uuid="s",
        state_map={
            "0": MachineState.OFF,
            "1": MachineState.IDLE,
            "3": MachineState.PRODUCING,
            "7": MachineState.DOWN,
        },
    )

    def test_maps_integer_codes(self):
        frame = sig("s", ["0h", "1h", "2h"], [3, 7, 3], "value_integer")
        assert states(normalizer(frame, self.SPEC)) == [
            "producing",
            "down",
            "producing",
        ]

    def test_unmapped_codes_become_unknown_rather_than_a_guess(self):
        frame = sig("s", ["0h", "1h", "2h"], [3, 99, 3], "value_integer")
        norm = normalizer(frame, self.SPEC)
        assert states(norm) == ["producing", "unknown", "producing"]

    def test_unmapped_values_are_reported_with_counts(self):
        frame = sig("s", ["0h", "1h", "2h", "3h"], [3, 99, 3, 99], "value_integer")
        unmapped = normalizer(frame, self.SPEC).unmapped_values()
        assert unmapped["raw_value"].tolist() == ["99"]
        assert unmapped["occurrences"].iloc[0] == 2

    def test_string_codes_are_matched_case_insensitively(self):
        spec = StatusSpec(
            mode=StatusMode.CODE_MAP,
            uuid="s",
            state_map={"auto": MachineState.PRODUCING, "stoerung": MachineState.DOWN},
        )
        frame = sig("s", ["0h", "1h"], ["AUTO", " Stoerung "], "value_string")
        assert states(normalizer(frame, spec)) == ["producing", "down"]

    def test_raw_values_survive_for_drill_down(self):
        frame = sig("s", ["0h", "1h"], [3, 7], "value_integer")
        assert normalizer(frame, self.SPEC).status_intervals()[
            "raw_value"
        ].tolist() == ["3", "7"]

    def test_planned_stop_values_override_the_map(self):
        spec = StatusSpec(
            mode=StatusMode.CODE_MAP,
            uuid="s",
            state_map={"1": MachineState.IDLE, "3": MachineState.PRODUCING},
            planned_stop_values=("1",),
        )
        frame = sig("s", ["0h", "1h"], [3, 1], "value_integer")
        assert states(normalizer(frame, spec)) == ["producing", "planned_stop"]


class TestPackML:
    def test_maps_the_standard_state_numbers(self):
        frame = sig("s", ["0h", "1h", "2h", "3h"], [6, 11, 9, 2], "value_integer")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.PACKML, uuid="s"))
        assert states(norm) == ["producing", "blocked", "down", "planned_stop"]

    def test_availability_classes_follow_the_mapping(self):
        frame = sig("s", ["0h", "1h", "2h"], [6, 9, 2], "value_integer")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.PACKML, uuid="s"))
        assert norm.status_intervals()["availability_class"].tolist() == [
            "productive",
            "loss",
            "planned",
        ]


class TestStacklight:
    def test_maps_lamp_colours(self):
        frame = sig(
            "s",
            ["0h", "1h", "2h", "3h"],
            ["green", "yellow", "red", "blue"],
            "value_string",
        )
        norm = normalizer(frame, StatusSpec(mode=StatusMode.STACKLIGHT, uuid="s"))
        assert states(norm) == ["producing", "idle", "down", "starved"]


# ---------------------------------------------------------------------------
# One-hot
# ---------------------------------------------------------------------------
class TestOneHot:
    SPEC = StatusSpec(
        mode=StatusMode.ONE_HOT,
        state_map={
            "run": MachineState.PRODUCING,
            "fault": MachineState.DOWN,
            "setup": MachineState.SETUP,
        },
    )

    def test_resolves_a_single_asserted_flag(self):
        frame = pd.concat(
            [
                sig("run", ["0h", "2h"], [True, False], "value_bool"),
                sig("fault", ["0h"], [False], "value_bool"),
                sig("setup", ["0h"], [False], "value_bool"),
            ],
            ignore_index=True,
        )
        assert states(normalizer(frame, self.SPEC)) == ["producing", "idle"]

    def test_a_fault_wins_over_a_simultaneous_run_flag(self):
        # Machines assert the drive bit and the alarm word at the same time;
        # counting that as production is the failure this prevents.
        frame = pd.concat(
            [
                sig("run", ["0h"], [True], "value_bool"),
                sig("fault", ["1h", "2h"], [True, False], "value_bool"),
            ],
            ignore_index=True,
        )
        assert states(normalizer(frame, self.SPEC)) == [
            "producing",
            "down",
            "producing",
        ]

    def test_custom_priority_is_honoured(self):
        spec = StatusSpec(
            mode=StatusMode.ONE_HOT,
            state_map={"run": MachineState.PRODUCING, "fault": MachineState.DOWN},
            priority=(MachineState.PRODUCING, MachineState.DOWN),
        )
        frame = pd.concat(
            [
                sig("run", ["0h"], [True], "value_bool"),
                sig("fault", ["1h"], [True], "value_bool"),
            ],
            ignore_index=True,
        )
        assert states(normalizer(frame, spec)) == ["producing"]

    def test_no_flag_asserted_falls_back_to_the_false_state(self):
        frame = pd.concat(
            [
                sig("run", ["0h", "1h"], [True, False], "value_bool"),
                sig("fault", ["0h"], [False], "value_bool"),
            ],
            ignore_index=True,
        )
        assert states(normalizer(frame, self.SPEC)) == ["producing", "idle"]

    def test_source_uuid_lists_every_contributing_signal(self):
        frame = sig("run", ["0h"], [True], "value_bool")
        norm = normalizer(frame, self.SPEC)
        assert norm.status_intervals()["source_uuid"].iloc[0] == "run,fault,setup"

    def test_unmapped_values_is_empty_for_one_hot(self):
        frame = sig("run", ["0h"], [True], "value_bool")
        assert normalizer(frame, self.SPEC).unmapped_values().empty


# ---------------------------------------------------------------------------
# Debounce and staleness
# ---------------------------------------------------------------------------
class TestDebounce:
    def test_a_one_second_flicker_is_absorbed(self):
        frame = sig(
            "s",
            ["0h", "1h", "1h1s", "2h"],
            [True, False, True, False],
            "value_bool",
        )
        norm = normalizer(
            frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s", min_duration="5s")
        )
        assert states(norm) == ["producing", "idle"]
        # The flicker's time stays with the surrounding producing interval.
        assert seconds(norm, "producing") == 2 * 3600.0

    def test_without_debounce_the_flicker_survives(self):
        frame = sig(
            "s", ["0h", "1h", "1h1s", "2h"], [True, False, True, False], "value_bool"
        )
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert len(norm.status_intervals()) == 4

    def test_debounce_preserves_the_total_observed_span(self):
        frame = sig(
            "s", ["0h", "1h", "1h1s", "2h"], [True, False, True, False], "value_bool"
        )
        norm = normalizer(
            frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s", min_duration="5s")
        )
        intervals = norm.status_intervals()
        assert intervals["duration_seconds"].sum() == (HORIZON - BASE).total_seconds()
        assert intervals["start"].iloc[0] == BASE
        assert intervals["end"].iloc[-1] == HORIZON

    def test_a_short_leading_interval_is_absorbed_by_its_successor(self):
        frame = sig("s", ["0h", "0h1s", "2h"], [True, False, True], "value_bool")
        norm = normalizer(
            frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s", min_duration="1min")
        )
        intervals = norm.status_intervals()
        assert intervals["start"].iloc[0] == BASE
        assert intervals["state"].iloc[0] == "idle"


class TestStaleness:
    def test_a_data_gap_becomes_unknown_instead_of_extending_the_state(self):
        frame = sig("s", ["0h", "4h"], [True, True], "value_bool")
        norm = normalizer(
            frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s", stale_after="30min")
        )
        intervals = norm.status_intervals()
        # The trailing silence up to the horizon goes stale too.
        assert intervals["state"].tolist() == [
            "producing",
            "unknown",
            "producing",
            "unknown",
        ]
        # Only the first 30 minutes of the gap count as producing.
        assert intervals["duration_seconds"].iloc[0] == 1800.0
        assert intervals["duration_seconds"].iloc[1] == pytest.approx(3.5 * 3600.0)

    def test_a_trailing_silence_before_the_horizon_also_goes_unknown(self):
        frame = sig("s", ["0h"], [True], "value_bool")
        norm = normalizer(
            frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s", stale_after="30min")
        )
        assert states(norm) == ["producing", "unknown"]

    def test_without_stale_after_a_gap_extends_the_last_state(self):
        frame = sig("s", ["0h", "4h"], [True, True], "value_bool")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert states(norm) == ["producing"]

    def test_unknown_time_is_excluded_from_availability(self):
        frame = sig("s", ["0h"], [True], "value_bool")
        norm = normalizer(
            frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s", stale_after="1h")
        )
        row = norm.state_summary().iloc[0]
        assert row["planned_busy_seconds"] == 3600.0
        assert row["availability_pct"] == 100.0
        assert row["unknown_seconds"] == 5 * 3600.0


# ---------------------------------------------------------------------------
# Summaries and derived views
# ---------------------------------------------------------------------------
class TestStateSummary:
    @pytest.fixture()
    def norm(self) -> MachineStatusNormalizer:
        frame = sig(
            "s",
            ["0h", "1h30min", "2h", "3h"],
            [6, 9, 6, 2],
            "value_integer",
        )
        return normalizer(frame, StatusSpec(mode=StatusMode.PACKML, uuid="s"))

    def test_conforms_to_the_summary_schema(self, norm):
        validate_event_output(norm.state_summary("1h"), "summary")
        validate_event_output(norm.state_summary(), "summary")

    def test_every_state_gets_a_column_so_machines_concatenate(self, norm):
        summary = norm.state_summary()
        for state in MachineState:
            assert f"{state.value}_seconds" in summary.columns

    def test_intervals_crossing_a_boundary_are_split(self, norm):
        # The down interval starts at 07:30 and ends at 08:00, so it belongs
        # half to each hour rather than wholly to one.
        hourly = norm.state_summary("1h").set_index("start")
        assert hourly.loc[BASE + pd.Timedelta("1h"), "producing_seconds"] == 1800.0
        assert hourly.loc[BASE + pd.Timedelta("1h"), "down_seconds"] == 1800.0

    def test_hourly_seconds_sum_to_the_overall_summary(self, norm):
        hourly = norm.state_summary("1h")
        overall = norm.state_summary().iloc[0]
        for state in MachineState:
            column = f"{state.value}_seconds"
            assert hourly[column].sum() == pytest.approx(overall[column])

    def test_availability_excludes_planned_stops_from_the_denominator(self, norm):
        row = norm.state_summary().iloc[0]
        # 2.5 h producing, 0.5 h down, 3 h planned stop.
        assert row["producing_seconds"] == pytest.approx(2.5 * 3600)
        assert row["planned_busy_seconds"] == pytest.approx(3 * 3600)
        assert row["availability_pct"] == pytest.approx(100 * 2.5 / 3)

    def test_availability_is_unknown_when_there_was_no_planned_busy_time(self):
        frame = sig("s", ["0h"], [2], "value_integer")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.PACKML, uuid="s"))
        assert pd.isna(norm.state_summary().iloc[0]["availability_pct"])

    def test_weekly_windows_work_with_anchored_offsets(self, norm):
        summary = norm.state_summary("1W")
        assert len(summary) == 1
        assert summary["producing_seconds"].iloc[0] == pytest.approx(2.5 * 3600)


class TestDerivedViews:
    @pytest.fixture()
    def norm(self) -> MachineStatusNormalizer:
        frame = sig("s", ["0h", "1h", "1h5min", "2h"], [6, 9, 6, 9], "value_integer")
        return normalizer(frame, StatusSpec(mode=StatusMode.PACKML, uuid="s"))

    def test_downtime_intervals_select_the_loss_states(self, norm):
        downtime = norm.downtime_intervals()
        assert set(downtime["state"]) == {"down"}
        validate_event_output(downtime, "interval")

    def test_downtime_min_duration_separates_micro_stops(self, norm):
        assert len(norm.downtime_intervals()) == 2
        assert len(norm.downtime_intervals(min_duration="10min")) == 1

    def test_transitions_report_the_pair_and_the_previous_duration(self, norm):
        transitions = norm.transitions()
        validate_event_output(transitions, "point")
        assert transitions["transition"].iloc[0] == "producing->down"
        assert transitions["previous_duration_seconds"].iloc[0] == 3600.0
        assert len(transitions) == len(norm.status_intervals()) - 1

    def test_state_at_does_an_as_of_lookup(self, norm):
        result = norm.state_at(
            [
                BASE + pd.Timedelta("30min"),
                BASE + pd.Timedelta("1h1min"),
                BASE - pd.Timedelta("1h"),
            ]
        )
        assert result.tolist() == ["producing", "down", "unknown"]

    def test_state_at_preserves_the_input_order_for_unsorted_probes(self, norm):
        result = norm.state_at(
            [BASE + pd.Timedelta("3h"), BASE + pd.Timedelta("30min")]
        )
        assert result.tolist() == ["down", "producing"]

    def test_state_at_with_no_timestamps(self, norm):
        assert norm.state_at([]).empty

    def test_diagnostics_summarise_the_signal(self, norm):
        row = norm.diagnostics().iloc[0]
        assert row["machine_id"] == "M1"
        assert row["mode"] == "packml"
        assert row["readings"] == 4
        assert row["intervals"] == len(norm.status_intervals())
        assert "producing" in row["states_seen"]


class TestEdgeCases:
    def test_missing_signal_yields_empty_canonical_output(self):
        frame = sig("other", ["0h"], [True], "value_bool")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert norm.status_intervals().empty
        validate_event_output(norm.status_intervals(), "interval")
        assert norm.state_summary().empty
        assert norm.transitions().empty
        assert norm.diagnostics()["intervals"].iloc[0] == 0

    def test_empty_input(self, empty_df):
        norm = normalizer(empty_df, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert norm.status_intervals().empty
        assert norm.state_at([BASE]).tolist() == ["unknown"]

    def test_a_single_reading_spans_to_the_horizon(self):
        frame = sig("s", ["0h"], [True], "value_bool")
        norm = normalizer(frame, StatusSpec(mode=StatusMode.BOOLEAN, uuid="s"))
        assert len(norm.status_intervals()) == 1
        assert norm.transitions().empty
