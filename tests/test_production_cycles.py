"""Tests for :class:`CycleTimeNormalizer`.

Two themes dominate: every derivation mode must produce the same canonical
per-part duration, and the artefacts that ruin cycle-time averages -- the gap
across a break, a multi-cavity cycle counted as one slow part, milliseconds read
as seconds -- must be removed *visibly*.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.events._output import validate_event_output
from ts_shape.events.production.signals.cycles import CycleTimeNormalizer
from ts_shape.events.production.signals.spec import (
    CounterKind,
    CounterMode,
    CounterSpec,
    CycleMode,
    CycleSpec,
    MachineState,
    StatusMode,
    StatusSpec,
    TimeUnit,
)
from ts_shape.events.production.signals.status import MachineStatusNormalizer

BASE = pd.Timestamp("2025-01-01 06:00:00")


def sig(uuid: str, offsets: list[str], values: list, column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": [BASE + pd.Timedelta(o) for o in offsets],
            "uuid": uuid,
            column: values,
        }
    )


def normalizer(frame: pd.DataFrame, spec: CycleSpec, **kwargs) -> CycleTimeNormalizer:
    kwargs.setdefault("machine_id", "M1")
    return CycleTimeNormalizer(frame, spec, **kwargs)


def times(norm: CycleTimeNormalizer) -> list[float]:
    return norm.cycles()["cycle_time_seconds"].round(6).tolist()


# ---------------------------------------------------------------------------
# EXPLICIT
# ---------------------------------------------------------------------------
class TestExplicit:
    def test_takes_the_published_value_in_seconds(self):
        frame = sig("c", ["0min", "1min", "2min"], [45.0, 47.0, 46.0], "value_double")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c"))
        assert times(norm) == [45.0, 47.0, 46.0]

    def test_converts_milliseconds(self):
        frame = sig("c", ["0min", "1min"], [45000.0, 47000.0], "value_double")
        norm = normalizer(
            frame,
            CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", unit=TimeUnit.MILLISECONDS),
        )
        assert times(norm) == [45.0, 47.0]

    def test_converts_minutes(self):
        frame = sig("c", ["0min"], [1.5], "value_double")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", unit=TimeUnit.MINUTES)
        )
        assert times(norm) == [90.0]

    def test_the_interval_ends_when_the_value_is_reported(self):
        frame = sig("c", ["1h"], [60.0], "value_double")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c"))
        cycle = norm.cycles().iloc[0]
        assert cycle["end"] == BASE + pd.Timedelta("1h")
        assert cycle["start"] == BASE + pd.Timedelta("59min")

    def test_cavities_give_a_comparable_per_part_time(self):
        frame = sig("c", ["0min"], [48.0], "value_double")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", cavities=4)
        )
        cycle = norm.cycles().iloc[0]
        assert cycle["cycle_time_seconds"] == 12.0
        assert cycle["cycle_time_raw_seconds"] == 48.0
        assert cycle["parts_per_cycle"] == 4.0


class TestUnitInference:
    def test_infers_seconds(self):
        frame = sig("c", ["0min", "1min"], [45.0, 47.0], "value_double")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", unit=TimeUnit.AUTO)
        )
        assert norm.resolved_unit() is TimeUnit.SECONDS

    def test_infers_milliseconds(self):
        frame = sig("c", ["0min", "1min"], [45000.0, 47000.0], "value_double")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", unit=TimeUnit.AUTO)
        )
        assert norm.resolved_unit() is TimeUnit.MILLISECONDS
        assert times(norm) == [45.0, 47.0]

    def test_inference_accounts_for_cavities(self):
        # 48000 ms across 4 cavities is 12 s per part; without dividing by the
        # cavities no unit would look plausible against a 5-30 s band.
        frame = sig("c", ["0min"], [48000.0], "value_double")
        norm = normalizer(
            frame,
            CycleSpec(
                mode=CycleMode.EXPLICIT,
                uuid="c",
                unit=TimeUnit.AUTO,
                cavities=4,
                min_seconds=5,
                max_seconds=30,
            ),
        )
        assert norm.resolved_unit() is TimeUnit.MILLISECONDS
        assert times(norm) == [12.0]

    def test_falls_back_to_seconds_when_nothing_fits(self):
        frame = sig("c", ["0min"], [1e12], "value_double")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", unit=TimeUnit.AUTO)
        )
        assert norm.resolved_unit() is TimeUnit.SECONDS

    def test_an_empty_signal_resolves_to_seconds(self):
        frame = sig("other", ["0min"], [1.0], "value_double")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", unit=TimeUnit.AUTO)
        )
        assert norm.resolved_unit() is TimeUnit.SECONDS


# ---------------------------------------------------------------------------
# Derived modes
# ---------------------------------------------------------------------------
class TestCounterDelta:
    def test_measures_the_gap_between_increments(self):
        # Readings at 20 s and 40 s are the first two counted events; 0 s is
        # only the baseline.
        frame = sig("c", ["0s", "20s", "40s"], [10, 11, 12], "value_integer")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="c"))
        assert times(norm) == [20.0]

    def test_divides_by_the_parts_in_the_increment(self):
        # A burst of 4 parts over 40 s is a 10 s cycle, not a 40 s one.
        frame = sig("c", ["0s", "20s", "60s"], [10, 11, 15], "value_integer")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="c"))
        cycles = norm.cycles()
        assert cycles["cycle_time_seconds"].tolist() == [10.0]
        assert cycles["parts_per_cycle"].tolist() == [4.0]

    def test_accepts_supplied_count_events(self):
        from ts_shape.events.production.signals.counters import (
            ProductionCounterNormalizer,
        )

        frame = sig("c", ["0s", "20s", "40s"], [True, True, True], "value_bool")
        events = ProductionCounterNormalizer(
            frame,
            counters=[
                CounterSpec(
                    uuid="c",
                    kind=CounterKind.OK,
                    mode=CounterMode.PULSE,
                    edge="any",
                )
            ],
        ).count_events()
        norm = normalizer(
            frame,
            CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="c"),
            counter_events=events,
        )
        assert times(norm) == [20.0, 20.0]

    def test_a_single_increment_yields_no_cycle(self):
        frame = sig("c", ["0s", "20s"], [10, 11], "value_integer")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="c"))
        assert norm.cycles().empty


class TestTriggerEdge:
    def test_measures_between_rising_edges(self):
        frame = sig(
            "t",
            ["0s", "10s", "30s", "40s", "60s"],
            [True, False, True, False, True],
            "value_bool",
        )
        norm = normalizer(frame, CycleSpec(mode=CycleMode.TRIGGER_EDGE, uuid="t"))
        assert times(norm) == [30.0, 30.0]

    def test_falling_edges(self):
        frame = sig(
            "t", ["0s", "10s", "30s", "40s"], [True, False, True, False], "value_bool"
        )
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.TRIGGER_EDGE, uuid="t", edge="falling")
        )
        assert times(norm) == [30.0]

    def test_edge_any_handles_pulse_logged_triggers(self):
        frame = sig("t", ["0s", "25s", "50s"], [True, True, True], "value_bool")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.TRIGGER_EDGE, uuid="t", edge="any")
        )
        assert times(norm) == [25.0, 25.0]


class TestTriggerPair:
    def test_matches_each_start_to_the_next_end(self):
        frame = pd.concat(
            [
                sig("s", ["0s", "30s"], [True, True], "value_bool"),
                sig("e", ["12s", "44s"], [True, True], "value_bool"),
            ],
            ignore_index=True,
        )
        norm = normalizer(
            frame,
            CycleSpec(mode=CycleMode.TRIGGER_PAIR, uuid="s", end_uuid="e", edge="any"),
        )
        assert times(norm) == [12.0, 14.0]

    def test_a_repeated_start_pulse_uses_the_last_one_before_the_end(self):
        frame = pd.concat(
            [
                sig("s", ["0s", "5s"], [True, True], "value_bool"),
                sig("e", ["12s"], [True], "value_bool"),
            ],
            ignore_index=True,
        )
        norm = normalizer(
            frame,
            CycleSpec(mode=CycleMode.TRIGGER_PAIR, uuid="s", end_uuid="e", edge="any"),
        )
        assert times(norm) == [7.0]

    def test_a_missing_end_signal_yields_nothing(self):
        frame = sig("s", ["0s"], [True], "value_bool")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.TRIGGER_PAIR, uuid="s", end_uuid="e")
        )
        assert norm.cycles().empty


class TestPartPresent:
    def test_measures_the_true_duration(self):
        frame = sig(
            "p",
            ["0s", "10s", "25s", "33s"],
            [True, False, True, False],
            "value_bool",
        )
        norm = normalizer(frame, CycleSpec(mode=CycleMode.PART_PRESENT, uuid="p"))
        assert times(norm) == [10.0, 8.0]

    def test_a_flag_that_never_goes_true_yields_nothing(self):
        frame = sig("p", ["0s", "10s"], [False, False], "value_bool")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.PART_PRESENT, uuid="p"))
        assert norm.cycles().empty


class TestStateInterval:
    @staticmethod
    def _setup() -> tuple[pd.DataFrame, MachineStatusNormalizer]:
        counter = sig("c", ["0s", "60s", "120s"], [0, 3, 6], "value_integer")
        run = sig("r", ["0s", "120s"], [True, False], "value_bool")
        frame = pd.concat([counter, run], ignore_index=True)
        status = MachineStatusNormalizer(
            frame,
            StatusSpec(mode=StatusMode.BOOLEAN, uuid="r"),
            machine_id="M1",
            until=BASE + pd.Timedelta("180s"),
        )
        return frame, status

    def test_divides_producing_time_by_the_parts_produced(self):
        frame, status = self._setup()
        norm = normalizer(
            frame,
            CycleSpec(mode=CycleMode.STATE_INTERVAL, uuid="c"),
            status=status,
        )
        cycles = norm.cycles()
        # 120 s producing, 6 parts booked inside it -> 20 s per part.
        assert cycles["cycle_time_seconds"].tolist() == [20.0]
        assert cycles["parts_per_cycle"].tolist() == [6.0]

    def test_requires_a_status_normalizer(self):
        frame, _ = self._setup()
        with pytest.raises(ValueError, match="must be passed as status="):
            normalizer(frame, CycleSpec(mode=CycleMode.STATE_INTERVAL))

    def test_a_producing_run_with_no_parts_is_not_a_measurement(self):
        run = sig("r", ["0s", "60s"], [True, False], "value_bool")
        counter = sig("c", ["0s"], [0], "value_integer")
        frame = pd.concat([run, counter], ignore_index=True)
        status = MachineStatusNormalizer(
            frame,
            StatusSpec(mode=StatusMode.BOOLEAN, uuid="r"),
            until=BASE + pd.Timedelta("120s"),
        )
        norm = normalizer(
            frame,
            CycleSpec(mode=CycleMode.STATE_INTERVAL, uuid="c"),
            status=status,
        )
        assert norm.cycles().empty


class TestModesAgree:
    def test_every_derivation_finds_the_same_twenty_second_cycle(self):
        offsets = ["0s", "20s", "40s", "60s"]
        counter = normalizer(
            sig("c", offsets, [0, 1, 2, 3], "value_integer"),
            CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="c"),
        )
        trigger = normalizer(
            sig("t", offsets, [True, True, True, True], "value_bool"),
            CycleSpec(mode=CycleMode.TRIGGER_EDGE, uuid="t", edge="any"),
        )
        explicit = normalizer(
            sig("e", offsets, [20.0] * 4, "value_double"),
            CycleSpec(mode=CycleMode.EXPLICIT, uuid="e"),
        )
        assert (
            set(times(counter)) == set(times(trigger)) == set(times(explicit)) == {20.0}
        )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
class TestFiltering:
    def test_the_gap_across_a_break_is_rejected_not_averaged_in(self):
        # The counter's first reading only establishes a baseline, so the first
        # measurable gap runs from 20 s to 40 s.
        frame = sig(
            "c",
            ["0s", "20s", "40s", "2h", "2h20s"],
            [0, 1, 2, 3, 4],
            "value_integer",
        )
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="c", max_seconds=60)
        )
        assert times(norm) == [20.0, 20.0]
        rejected = norm.rejected()
        assert rejected["reason"].tolist() == ["too_long"]
        validate_event_output(rejected, "interval")

    def test_sensor_bounce_is_rejected_as_too_short(self):
        frame = sig("c", ["0s", "1s", "2s", "22s"], [0, 1, 2, 3], "value_integer")
        norm = normalizer(
            frame, CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="c", min_seconds=5)
        )
        assert times(norm) == [20.0]
        assert norm.rejected()["reason"].tolist() == ["too_short"]

    def test_a_zero_length_cycle_is_rejected_as_non_positive(self):
        frame = sig("c", ["0min", "1min"], [0.0, 20.0], "value_double")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c"))
        assert norm.rejected()["reason"].tolist() == ["non_positive"]
        assert times(norm) == [20.0]

    def test_cycles_overlapping_a_gated_state_are_rejected(self):
        counter = sig(
            "c", ["0s", "20s", "40s", "60s", "80s"], [0, 1, 2, 3, 4], "value_integer"
        )
        # Down from 25 s to 35 s, entirely inside the 20-40 s cycle.
        run = sig("r", ["0s", "45s", "55s"], [True, False, True], "value_bool")
        frame = pd.concat([counter, run], ignore_index=True)
        status = MachineStatusNormalizer(
            frame,
            StatusSpec(
                mode=StatusMode.BOOLEAN, uuid="r", false_state=MachineState.DOWN
            ),
            until=BASE + pd.Timedelta("80s"),
        )
        norm = normalizer(
            frame,
            CycleSpec(
                mode=CycleMode.COUNTER_DELTA,
                uuid="c",
                gate_states=(MachineState.DOWN,),
            ),
            status=status,
        )
        # An endpoint-only check would have kept this cycle: both ends are in
        # producing time and the fault sits entirely in the middle.
        assert norm.rejected()["reason"].tolist() == ["gated_state"]
        assert len(norm.cycles()) == 2

    def test_gate_states_require_a_status_normalizer(self):
        frame = sig("c", ["0s"], [1], "value_integer")
        with pytest.raises(ValueError, match="needs a MachineStatusNormalizer"):
            normalizer(
                frame,
                CycleSpec(
                    mode=CycleMode.COUNTER_DELTA,
                    uuid="c",
                    gate_states=(MachineState.DOWN,),
                ),
            )

    def test_diagnostics_count_every_rejection_reason(self):
        frame = sig(
            "c", ["0s", "1s", "2s", "22s", "2h"], [0, 1, 2, 3, 4], "value_integer"
        )
        norm = normalizer(
            frame,
            CycleSpec(
                mode=CycleMode.COUNTER_DELTA, uuid="c", min_seconds=5, max_seconds=60
            ),
        )
        row = norm.diagnostics().iloc[0]
        assert row["candidates"] == 3
        assert row["accepted"] == 1
        assert row["rejected_too_short"] == 1
        assert row["rejected_too_long"] == 1


# ---------------------------------------------------------------------------
# Statistics and outputs
# ---------------------------------------------------------------------------
class TestStatistics:
    @pytest.fixture()
    def norm(self) -> CycleTimeNormalizer:
        frame = sig(
            "c",
            [f"{i}min" for i in range(6)],
            [10.0, 12.0, 11.0, 13.0, 12.0, 60.0],
            "value_double",
        )
        return normalizer(
            frame,
            CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", max_seconds=30),
            ideal_cycle_seconds=11.0,
        )

    def test_cycles_conform_to_the_interval_schema(self, norm):
        validate_event_output(norm.cycles(), "interval")

    def test_statistics_conform_to_the_summary_schema(self, norm):
        validate_event_output(norm.cycle_statistics(), "summary")

    def test_statistics_describe_the_accepted_cycles(self, norm):
        row = norm.cycle_statistics().iloc[0]
        assert row["cycles"] == 5
        assert row["mean_seconds"] == pytest.approx(11.6)
        assert row["median_seconds"] == 12.0
        assert row["min_seconds"] == 10.0
        assert row["max_seconds"] == 13.0

    def test_the_target_produces_deviation_and_hit_rate(self, norm):
        row = norm.cycle_statistics().iloc[0]
        assert row["target_seconds"] == 11.0
        assert row["mean_deviation_pct"] == pytest.approx(100 * (11.6 - 11) / 11)
        assert row["within_target_pct"] == pytest.approx(40.0)

    def test_grouped_statistics(self, norm):
        events = norm.cycles()
        events["part_number"] = ["A", "A", "B", "B", "B"]
        grouped = norm.cycle_statistics(group_by=("part_number",), events=events)
        assert grouped["part_number"].tolist() == ["A", "B"]
        assert grouped["cycles"].tolist() == [2, 3]

    def test_grouped_statistics_reject_a_missing_column(self, norm):
        with pytest.raises(ValueError, match="not present on the cycle records"):
            norm.cycle_statistics(group_by=("part_number",))

    def test_by_window_attributes_a_cycle_to_the_window_of_its_end(self, norm):
        windows = norm.by_window("2min")
        validate_event_output(windows, "summary")
        assert windows["cycles"].sum() == 5

    def test_deviations_flag_slow_and_fast_cycles(self, norm):
        deviations = norm.deviations(tolerance_pct=5.0)
        validate_event_output(deviations, "point")
        assert set(deviations["direction"]) <= {"slow", "fast"}
        assert (deviations["deviation_pct"].abs() > 5.0).all()

    def test_deviations_use_the_ideal_cycle_by_default(self, norm):
        assert (norm.deviations(tolerance_pct=5.0)["target_seconds"] == 11.0).all()

    def test_deviations_require_a_target(self):
        frame = sig("c", ["0min"], [10.0], "value_double")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c"))
        with pytest.raises(ValueError, match="needs a positive target"):
            norm.deviations()

    def test_no_deviations_when_everything_is_on_target(self):
        frame = sig("c", ["0min", "1min"], [10.0, 10.0], "value_double")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c"))
        assert norm.deviations(target=10.0).empty


class TestEdgeCases:
    def test_missing_signal_yields_empty_canonical_output(self):
        frame = sig("other", ["0min"], [1.0], "value_double")
        norm = normalizer(frame, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c"))
        assert norm.cycles().empty
        validate_event_output(norm.cycles(), "interval")
        assert norm.rejected().empty
        assert norm.cycle_statistics().empty
        assert norm.by_window().empty
        assert norm.diagnostics()["candidates"].iloc[0] == 0

    def test_empty_input(self, empty_df):
        norm = normalizer(empty_df, CycleSpec(mode=CycleMode.EXPLICIT, uuid="c"))
        assert norm.cycles().empty
