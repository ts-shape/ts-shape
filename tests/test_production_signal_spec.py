"""Tests for the declarative signal spec model and the value-coercion layer.

These two modules are the foundation the production normalizers stand on: the
specs decide what a fleet configuration is allowed to say, and the coercion
helpers decide what a messy value column is allowed to mean. Both are worth
pinning tightly, because a wrong answer here is silent everywhere above.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.production.signals.spec import (
    PACKML_STATE_MAP,
    STACKLIGHT_STATE_MAP,
    ContextBinding,
    ContextSpec,
    CounterKind,
    CounterMode,
    CounterSpec,
    CycleMode,
    CycleSpec,
    FleetSpec,
    MachineSpec,
    MachineState,
    ResetPolicy,
    Rollover,
    StatusMode,
    StatusSpec,
    TimeUnit,
)
from ts_shape.events.production.signals.values import (
    coerce_bool,
    coerce_label,
    coerce_numeric,
    resolve_value_column,
    signal_frame,
    union_timeline,
)


# ---------------------------------------------------------------------------
# MachineState vocabulary
# ---------------------------------------------------------------------------
class TestMachineState:
    def test_availability_classes_partition_the_vocabulary(self):
        classes = {state.availability_class for state in MachineState}
        assert classes == {"productive", "loss", "planned", "unknown"}

    def test_only_producing_is_productive(self):
        productive = [s for s in MachineState if s.is_productive]
        assert productive == [MachineState.PRODUCING]

    def test_planned_stop_and_off_are_outside_planned_busy_time(self):
        assert MachineState.PLANNED_STOP.availability_class == "planned"
        assert MachineState.OFF.availability_class == "planned"
        assert not MachineState.PLANNED_STOP.is_downtime

    def test_unknown_is_neither_productive_nor_loss(self):
        # Folding "no data" into either bucket is what makes availability lie.
        assert MachineState.UNKNOWN.availability_class == "unknown"
        assert not MachineState.UNKNOWN.is_downtime
        assert not MachineState.UNKNOWN.is_productive

    def test_downtime_states(self):
        downtime = {s.value for s in MachineState if s.is_downtime}
        assert downtime == {"idle", "setup", "down", "blocked", "starved"}


class TestDefaultStateMaps:
    def test_packml_execute_is_producing(self):
        assert PACKML_STATE_MAP["6"] is MachineState.PRODUCING

    def test_packml_covers_all_seventeen_states(self):
        assert {int(k) for k in PACKML_STATE_MAP} == set(range(1, 18))

    def test_stacklight_green_produces_red_stops(self):
        assert STACKLIGHT_STATE_MAP["green"] is MachineState.PRODUCING
        assert STACKLIGHT_STATE_MAP["red"] is MachineState.DOWN


# ---------------------------------------------------------------------------
# CounterSpec
# ---------------------------------------------------------------------------
class TestCounterSpec:
    def test_accepts_plain_strings_for_enums(self):
        spec = CounterSpec(uuid="c", kind="nok", mode="pulse", reset_policy="ignore")
        assert spec.kind is CounterKind.NOK
        assert spec.mode is CounterMode.PULSE
        assert spec.reset_policy is ResetPolicy.IGNORE

    def test_rejects_unknown_enum_value_with_the_alternatives(self):
        with pytest.raises(ValueError, match="not a valid CounterMode"):
            CounterSpec(uuid="c", mode="counting_up")

    def test_rejects_empty_uuid(self):
        with pytest.raises(ValueError, match="non-empty signal identifier"):
            CounterSpec(uuid="")

    def test_rejects_non_positive_parts_per_count(self):
        with pytest.raises(ValueError, match="parts_per_count must be > 0"):
            CounterSpec(uuid="c", parts_per_count=0)

    def test_rollover_only_applies_to_cumulative(self):
        with pytest.raises(ValueError, match="wrap-around only applies"):
            CounterSpec(uuid="c", mode=CounterMode.PULSE, rollover_at=65536)

    def test_rollover_tolerance_must_be_a_fraction(self):
        with pytest.raises(ValueError, match="rollover_tolerance must be in"):
            CounterSpec(uuid="c", rollover_at=65536, rollover_tolerance=1.5)

    def test_rollover_presets_are_numeric(self):
        spec = CounterSpec(uuid="c", rollover_at=Rollover.UINT16)
        assert spec.rollover_at == 65536.0

    def test_defect_code_defaults_ok_values_to_zero(self):
        spec = CounterSpec(uuid="c", mode=CounterMode.DEFECT_CODE)
        assert spec.ok_values == ("0",)

    def test_defect_code_rejects_an_ok_kind(self):
        with pytest.raises(ValueError, match="classifies every part"):
            CounterSpec(uuid="c", mode=CounterMode.DEFECT_CODE, kind=CounterKind.OK)

    def test_rejects_unknown_edge(self):
        with pytest.raises(ValueError, match="edge='sideways' is invalid"):
            CounterSpec(uuid="c", mode=CounterMode.PULSE, edge="sideways")

    def test_classifies_quality_only_for_verdict_modes(self):
        assert CounterSpec(uuid="c", mode=CounterMode.QUALITY_FLAG).classifies_quality
        assert CounterSpec(
            uuid="c", mode=CounterMode.DEFECT_CODE, kind=CounterKind.NOK
        ).classifies_quality
        assert not CounterSpec(uuid="c", mode=CounterMode.CUMULATIVE).classifies_quality

    def test_signal_uuids_include_the_trigger(self):
        spec = CounterSpec(
            uuid="verdict", mode=CounterMode.QUALITY_FLAG, trigger_uuid="done"
        )
        assert spec.signal_uuids == ("verdict", "done")

    def test_round_trips_through_dict(self):
        spec = CounterSpec(
            uuid="c",
            kind=CounterKind.NOK,
            mode=CounterMode.CUMULATIVE,
            rollover_at=Rollover.UINT16,
            max_plausible_delta=500,
            parts_per_count=8,
        )
        assert CounterSpec.from_dict(spec.to_dict()) == spec

    def test_from_dict_ignores_unrelated_config_columns(self):
        spec = CounterSpec.from_dict(
            {"uuid": "c", "kind": "ok", "plant": "Ingolstadt", "comment": "n/a"}
        )
        assert spec.uuid == "c"
        assert spec.kind is CounterKind.OK


# ---------------------------------------------------------------------------
# StatusSpec
# ---------------------------------------------------------------------------
class TestStatusSpec:
    def test_packml_gets_the_default_map(self):
        spec = StatusSpec(mode=StatusMode.PACKML, uuid="s")
        assert spec.state_map["6"] is MachineState.PRODUCING

    def test_stacklight_gets_the_default_map(self):
        spec = StatusSpec(mode=StatusMode.STACKLIGHT, uuid="s")
        assert spec.state_map["red"] is MachineState.DOWN

    def test_state_map_keys_are_normalised_to_lowercase_strings(self):
        spec = StatusSpec(
            mode=StatusMode.CODE_MAP,
            uuid="s",
            state_map={4: MachineState.PRODUCING, " Fault ": "down"},
        )
        assert spec.state_map["4"] is MachineState.PRODUCING
        assert spec.state_map["fault"] is MachineState.DOWN

    def test_code_map_requires_a_map(self):
        with pytest.raises(ValueError, match="requires state_map"):
            StatusSpec(mode=StatusMode.CODE_MAP, uuid="s")

    def test_one_hot_requires_a_map_and_no_uuid(self):
        with pytest.raises(ValueError, match="requires state_map mapping each signal"):
            StatusSpec(mode=StatusMode.ONE_HOT)
        with pytest.raises(ValueError, match="leave uuid unset"):
            StatusSpec(
                mode=StatusMode.ONE_HOT,
                uuid="s",
                state_map={"a": MachineState.PRODUCING},
            )

    def test_single_signal_modes_require_a_uuid(self):
        with pytest.raises(ValueError, match="requires uuid"):
            StatusSpec(mode=StatusMode.BOOLEAN)

    def test_default_state_is_unknown_so_unmapped_codes_stay_visible(self):
        spec = StatusSpec(mode=StatusMode.PACKML, uuid="s")
        assert spec.default_state is MachineState.UNKNOWN

    def test_priority_defaults_put_faults_before_production(self):
        spec = StatusSpec(mode=StatusMode.BOOLEAN, uuid="s")
        order = list(spec.priority)
        assert order.index(MachineState.DOWN) < order.index(MachineState.PRODUCING)

    def test_signal_uuids_for_one_hot_are_the_map_keys(self):
        spec = StatusSpec(
            mode=StatusMode.ONE_HOT,
            state_map={"run": MachineState.PRODUCING, "fault": MachineState.DOWN},
        )
        assert set(spec.signal_uuids) == {"run", "fault"}

    def test_round_trips_through_dict(self):
        spec = StatusSpec(
            mode=StatusMode.CODE_MAP,
            uuid="s",
            state_map={"3": MachineState.PRODUCING},
            min_duration="5s",
            stale_after="10min",
        )
        assert StatusSpec.from_dict(spec.to_dict()) == spec


# ---------------------------------------------------------------------------
# CycleSpec
# ---------------------------------------------------------------------------
class TestCycleSpec:
    def test_trigger_pair_requires_an_end_signal(self):
        with pytest.raises(ValueError, match="requires end_uuid"):
            CycleSpec(mode=CycleMode.TRIGGER_PAIR, uuid="start")

    def test_state_interval_needs_no_uuid(self):
        spec = CycleSpec(mode=CycleMode.STATE_INTERVAL)
        assert spec.signal_uuids == ()

    def test_other_modes_require_a_uuid(self):
        with pytest.raises(ValueError, match="requires uuid"):
            CycleSpec(mode=CycleMode.EXPLICIT)

    def test_auto_unit_only_applies_to_explicit_signals(self):
        with pytest.raises(ValueError, match="only applies to CycleMode.EXPLICIT"):
            CycleSpec(mode=CycleMode.TRIGGER_EDGE, uuid="t", unit=TimeUnit.AUTO)

    def test_bounds_must_be_ordered(self):
        with pytest.raises(ValueError, match="must be below"):
            CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", min_seconds=10, max_seconds=5)

    def test_rejects_non_positive_cavities(self):
        with pytest.raises(ValueError, match="cavities must be > 0"):
            CycleSpec(mode=CycleMode.EXPLICIT, uuid="c", cavities=0)

    def test_seconds_factors(self):
        assert TimeUnit.MILLISECONDS.seconds_factor == pytest.approx(1e-3)
        assert TimeUnit.MINUTES.seconds_factor == 60.0
        with pytest.raises(ValueError, match="no fixed factor"):
            TimeUnit.AUTO.seconds_factor  # noqa: B018

    def test_round_trips_through_dict(self):
        spec = CycleSpec(
            mode=CycleMode.EXPLICIT,
            uuid="c",
            unit=TimeUnit.MILLISECONDS,
            cavities=4,
            gate_states=(MachineState.DOWN,),
        )
        assert CycleSpec.from_dict(spec.to_dict()) == spec


# ---------------------------------------------------------------------------
# ContextSpec
# ---------------------------------------------------------------------------
class TestContextSpec:
    def test_requires_name_and_uuid(self):
        with pytest.raises(ValueError, match="non-empty column name"):
            ContextSpec(name="", uuid="u")
        with pytest.raises(ValueError, match="non-empty signal identifier"):
            ContextSpec(name="order_number", uuid="")

    def test_numeric_context_rejects_label_options(self):
        with pytest.raises(ValueError, match="only apply to label context"):
            ContextSpec(name="target", uuid="u", numeric=True, upper=True)

    def test_defaults_to_last_known_binding(self):
        assert ContextSpec(name="o", uuid="u").binding is ContextBinding.LAST_KNOWN

    def test_round_trips_through_dict(self):
        spec = ContextSpec(
            name="serial_number",
            uuid="u",
            binding=ContextBinding.NEXT_KNOWN,
            valid_for="30s",
            zero_is_sentinel=True,
        )
        assert ContextSpec.from_dict(spec.to_dict()) == spec


# ---------------------------------------------------------------------------
# MachineSpec / FleetSpec
# ---------------------------------------------------------------------------
def _machine(machine_id: str = "M1", **kwargs) -> MachineSpec:
    defaults: dict = {
        "counters": (CounterSpec(uuid=f"{machine_id}.total"),),
        "attributes": {"plant": "P1", "line": "L1"},
    }
    defaults.update(kwargs)
    return MachineSpec(machine_id=machine_id, **defaults)


class TestMachineSpec:
    def test_rejects_duplicate_context_names(self):
        with pytest.raises(ValueError, match="duplicate context dimension"):
            _machine(
                context=(
                    ContextSpec(name="order_number", uuid="a"),
                    ContextSpec(name="order_number", uuid="b"),
                )
            )

    def test_rejects_the_same_counter_signal_twice(self):
        with pytest.raises(ValueError, match="more than once"):
            _machine(
                counters=(
                    CounterSpec(uuid="c", kind=CounterKind.OK),
                    CounterSpec(uuid="c", kind=CounterKind.NOK),
                )
            )

    def test_allows_several_counters_of_the_same_kind_on_different_signals(self):
        spec = _machine(
            counters=(
                CounterSpec(uuid="lane1", kind=CounterKind.OK),
                CounterSpec(uuid="lane2", kind=CounterKind.OK),
            )
        )
        assert spec.counter_kinds == (CounterKind.OK,)
        assert len(spec.counters_of("ok")) == 2

    def test_rejects_non_positive_ideal_cycle(self):
        with pytest.raises(ValueError, match="ideal_cycle_seconds"):
            _machine(ideal_cycle_seconds=0)

    def test_signal_uuids_are_deduplicated_and_ordered(self):
        spec = MachineSpec(
            machine_id="M1",
            counters=(CounterSpec(uuid="a"),),
            status=StatusSpec(mode=StatusMode.BOOLEAN, uuid="b"),
            cycle=CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="a"),
            context=(ContextSpec(name="order_number", uuid="c"),),
        )
        assert spec.signal_uuids == ("a", "c", "b")

    def test_with_context_appends(self):
        spec = _machine().with_context(ContextSpec(name="order_number", uuid="o"))
        assert spec.context_names == ("order_number",)

    def test_round_trips_through_json(self):
        spec = MachineSpec(
            machine_id="M1",
            counters=(CounterSpec(uuid="a", kind=CounterKind.OK),),
            status=StatusSpec(mode=StatusMode.PACKML, uuid="b"),
            cycle=CycleSpec(mode=CycleMode.COUNTER_DELTA, uuid="a"),
            context=(ContextSpec(name="order_number", uuid="c"),),
            ideal_cycle_seconds=12.0,
            attributes={"plant": "P1"},
        )
        restored = MachineSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        assert restored == spec


class TestFleetSpec:
    def test_rejects_duplicate_machine_ids(self):
        with pytest.raises(ValueError, match="duplicate machine_id"):
            FleetSpec(machines=(_machine("M1"), _machine("M1")))

    def test_rejects_a_signal_claimed_by_two_machines(self):
        shared = CounterSpec(uuid="shared.counter")
        fleet = FleetSpec(
            machines=(
                MachineSpec(machine_id="M1", counters=(shared,)),
                MachineSpec(machine_id="M2", counters=(shared,)),
            )
        )
        with pytest.raises(ValueError, match="counted twice"):
            fleet.uuid_index()

    def test_uuid_index_maps_every_signal(self):
        fleet = FleetSpec(machines=(_machine("M1"), _machine("M2")))
        assert fleet.uuid_index() == {"M1.total": "M1", "M2.total": "M2"}

    def test_machine_lookup_and_error(self):
        fleet = FleetSpec(machines=(_machine("M1"),))
        assert fleet.machine("M1").machine_id == "M1"
        with pytest.raises(KeyError, match="No machine 'M9'"):
            fleet.machine("M9")

    def test_iteration_and_length(self):
        fleet = FleetSpec(machines=(_machine("M1"), _machine("M2")))
        assert len(fleet) == 2
        assert [m.machine_id for m in fleet] == ["M1", "M2"]

    def test_attribute_keys_are_the_union_sorted(self):
        fleet = FleetSpec(
            machines=(
                _machine("M1", attributes={"plant": "P1"}),
                _machine("M2", attributes={"line": "L2", "cell": "C1"}),
            )
        )
        assert fleet.attribute_keys == ("cell", "line", "plant")

    def test_filter_selects_a_sub_fleet(self):
        fleet = FleetSpec(
            machines=(
                _machine("M1", attributes={"plant": "P1"}),
                _machine("M2", attributes={"plant": "P2"}),
            )
        )
        assert fleet.filter(plant="P2").machine_ids == ("M2",)

    def test_round_trips_through_json(self):
        fleet = FleetSpec(machines=(_machine("M1"), _machine("M2")))
        assert FleetSpec.from_dict(json.loads(json.dumps(fleet.to_dict()))) == fleet

    def test_from_dict_accepts_a_bare_list(self):
        fleet = FleetSpec(machines=(_machine("M1"),))
        assert FleetSpec.from_dict(fleet.to_dict()["machines"]) == fleet

    def test_signal_uuids_span_the_fleet_without_duplicates(self):
        fleet = FleetSpec(machines=(_machine("M1"), _machine("M2")))
        assert fleet.signal_uuids == ("M1.total", "M2.total")


class TestFleetSpecFromRecords:
    """A tidy tag table is how signal mappings exist in a real plant."""

    @staticmethod
    def _records() -> list[dict]:
        return [
            {
                "machine_id": "PRESS-01",
                "role": "counter_total",
                "uuid": "p1.total",
                "mode": "cumulative",
                "rollover_at": 65536,
                "attr_plant": "Ingolstadt",
                "ideal_cycle_seconds": 12.0,
            },
            {
                "machine_id": "PRESS-01",
                "role": "counter_nok",
                "uuid": "p1.nok",
                "mode": "cumulative",
            },
            {
                "machine_id": "PRESS-01",
                "role": "status",
                "uuid": "p1.packml",
                "mode": "packml",
                "stale_after": "10min",
            },
            {
                "machine_id": "PRESS-01",
                "role": "cycle",
                "uuid": "p1.ct",
                "mode": "explicit",
                "unit": "ms",
                "cavities": 4,
            },
            {
                "machine_id": "PRESS-01",
                "role": "context",
                "uuid": "p1.order",
                "name": "order_number",
                "valid_for": "12h",
            },
            {
                "machine_id": "PRESS-01",
                "role": "attribute",
                "name": "line",
                "value": "L1",
            },
            {
                "machine_id": "CELL-07",
                "role": "counter_ok",
                "uuid": "c7.done",
                "mode": "pulse",
                "parts_per_count": 2,
            },
        ]

    def test_builds_the_expected_fleet(self):
        fleet = FleetSpec.from_records(self._records())
        assert fleet.machine_ids == ("PRESS-01", "CELL-07")

        press = fleet.machine("PRESS-01")
        assert press.counter_kinds == (CounterKind.TOTAL, CounterKind.NOK)
        assert press.counters[0].rollover_at == 65536.0
        assert press.status is not None
        assert press.status.mode is StatusMode.PACKML
        assert press.status.stale_after == "10min"
        assert press.cycle is not None
        assert press.cycle.unit is TimeUnit.MILLISECONDS
        assert press.cycle.cavities == 4
        assert press.context_names == ("order_number",)
        assert press.ideal_cycle_seconds == 12.0
        assert press.attributes == {"plant": "Ingolstadt", "line": "L1"}

        cell = fleet.machine("CELL-07")
        assert cell.counters[0].mode is CounterMode.PULSE
        assert cell.counters[0].parts_per_count == 2

    def test_accepts_a_dataframe_export(self):
        # A config table arrives from a spreadsheet, so unset cells are NaN.
        frame = pd.DataFrame(self._records())
        fleet = FleetSpec.from_records(frame.to_dict("records"))
        assert fleet.machine("CELL-07").counters[0].rollover_at is None

    def test_rejects_a_row_without_a_machine(self):
        with pytest.raises(ValueError, match="has no 'machine_id'"):
            FleetSpec.from_records([{"role": "counter_ok", "uuid": "x"}])

    def test_rejects_an_unknown_role(self):
        with pytest.raises(ValueError, match="Unknown role='sensor'"):
            FleetSpec.from_records(
                [{"machine_id": "M1", "role": "sensor", "uuid": "x"}]
            )

    def test_context_row_requires_a_name(self):
        with pytest.raises(ValueError, match="needs a 'name' column"):
            FleetSpec.from_records(
                [{"machine_id": "M1", "role": "context", "uuid": "x"}]
            )


# ---------------------------------------------------------------------------
# Value coercion
# ---------------------------------------------------------------------------
class TestResolveValueColumn:
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "systime": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00"]),
                "uuid": ["a", "b"],
                "value_double": [np.nan, 1.5],
                "value_integer": [7, None],
                "value_bool": [None, None],
                "value_string": [None, None],
            }
        )

    def test_picks_the_column_the_signal_actually_uses(self):
        frame = self._frame()
        assert resolve_value_column(frame, uuid="a") == "value_integer"
        assert resolve_value_column(frame, uuid="b") == "value_double"

    def test_explicit_preference_wins_when_it_has_data(self):
        frame = self._frame()
        assert (
            resolve_value_column(frame, uuid="a", preferred="value_integer")
            == "value_integer"
        )

    def test_falls_back_to_detection_when_the_preference_is_empty(self):
        frame = self._frame()
        # value_bool holds nothing for this signal, so honouring the request
        # would silently return an all-null column.
        assert (
            resolve_value_column(frame, uuid="a", preferred="value_bool")
            == "value_integer"
        )

    def test_raises_for_a_nonexistent_preference(self):
        with pytest.raises(ValueError, match="does not exist"):
            resolve_value_column(self._frame(), uuid="a", preferred="value_blob")

    def test_raises_when_no_value_column_exists(self):
        with pytest.raises(ValueError, match="None of the value columns"):
            resolve_value_column(pd.DataFrame({"systime": [], "uuid": []}))

    def test_raises_for_an_all_null_signal(self):
        frame = self._frame()
        frame.loc[frame["uuid"] == "a", ["value_integer"]] = None
        with pytest.raises(ValueError, match="no non-null values"):
            resolve_value_column(frame, uuid="a")


class TestCoerceNumeric:
    def test_passes_through_numeric(self):
        assert coerce_numeric(pd.Series([1, 2, 3])).tolist() == [1.0, 2.0, 3.0]

    def test_converts_booleans(self):
        assert coerce_numeric(pd.Series([True, False])).tolist() == [1.0, 0.0]

    def test_parses_stringified_numbers_and_decimal_commas(self):
        result = coerce_numeric(pd.Series([" 12 ", "1,5", "3.25"]))
        assert result.tolist() == [12.0, 1.5, 3.25]

    def test_leaves_thousands_separators_unparsed_rather_than_mangling_them(self):
        # "1,234.5" must not silently become 1.2345 or 12345.
        assert pd.isna(coerce_numeric(pd.Series(["1,234.5"])).iloc[0])

    def test_unparseable_becomes_nan(self):
        assert pd.isna(coerce_numeric(pd.Series(["n/a"])).iloc[0])

    def test_empty_series(self):
        assert coerce_numeric(pd.Series(dtype="object")).empty


class TestCoerceBool:
    def test_passes_through_booleans(self):
        assert coerce_bool(pd.Series([True, False])).tolist() == [True, False]

    def test_numeric_is_truthy_when_non_zero(self):
        assert coerce_bool(pd.Series([0, 1, 5, -2])).tolist() == [
            False,
            True,
            True,
            True,
        ]

    def test_recognises_industrial_string_tokens(self):
        values = pd.Series(["RUN", "stop", "On", "OFF", "OK", "nok", "1", "0"])
        assert coerce_bool(values).tolist() == [
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ]

    def test_stringified_numbers_keep_their_truthiness(self):
        assert coerce_bool(pd.Series(["3", "0.0"])).tolist() == [True, False]

    def test_unrecognised_and_null_use_the_default(self):
        assert coerce_bool(pd.Series(["wat", None])).tolist() == [False, False]
        assert coerce_bool(pd.Series(["wat", None]), default=True).tolist() == [
            True,
            True,
        ]


class TestCoerceLabel:
    def test_strips_and_uppercases(self):
        assert coerce_label(pd.Series([" a-100 "]), upper=True).iloc[0] == "A-100"

    def test_renders_integral_numbers_without_a_trailing_zero(self):
        # An order number in value_double must match the same number as text.
        assert coerce_label(pd.Series([123.0, 4711.0])).tolist() == ["123", "4711"]

    def test_keeps_genuine_decimals(self):
        assert coerce_label(pd.Series([1.5])).iloc[0] == "1.5"

    def test_maps_placeholders_to_missing(self):
        values = pd.Series(["", "n/a", "#NV", " - ", "NULL", "real"])
        result = coerce_label(values)
        assert result.isna().tolist() == [True, True, True, True, True, False]

    def test_zero_is_a_real_value_unless_asked_otherwise(self):
        assert coerce_label(pd.Series(["0"])).iloc[0] == "0"
        assert pd.isna(coerce_label(pd.Series(["0"]), zero_is_sentinel=True).iloc[0])

    def test_strips_leading_zeros_on_request(self):
        assert (
            coerce_label(pd.Series(["000123"]), strip_leading_zeros=True).iloc[0]
            == "123"
        )

    def test_an_all_zero_identifier_survives_zero_stripping(self):
        assert coerce_label(pd.Series(["000"]), strip_leading_zeros=True).iloc[0] == "0"

    def test_extra_sentinels_are_honoured(self):
        result = coerce_label(pd.Series(["KEIN AUFTRAG"]), sentinels=("kein auftrag",))
        assert result.isna().iloc[0]


class TestSignalFrame:
    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "systime": pd.to_datetime(
                    [
                        "2025-01-01 00:00",
                        "2025-01-01 00:02",
                        "2025-01-01 00:01",
                        "2025-01-01 00:02",
                    ]
                ),
                "uuid": ["a", "a", "a", "a"],
                "value_integer": [1, 2, 3, 9],
            }
        )

    def test_sorts_and_collapses_duplicate_timestamps(self):
        out = signal_frame(self._frame(), "a")
        assert out["value"].tolist() == [1, 3, 9]
        assert out["systime"].is_monotonic_increasing

    def test_duplicate_policies(self):
        assert signal_frame(self._frame(), "a", duplicate_policy="first")[
            "value"
        ].tolist() == [1, 3, 2]
        assert signal_frame(self._frame(), "a", duplicate_policy="max")[
            "value"
        ].tolist() == [1, 3, 9]
        assert signal_frame(self._frame(), "a", duplicate_policy="min")[
            "value"
        ].tolist() == [1, 3, 2]
        assert len(signal_frame(self._frame(), "a", duplicate_policy="keep")) == 4

    def test_rejects_an_unknown_duplicate_policy(self):
        with pytest.raises(ValueError, match="Unknown duplicate_policy"):
            signal_frame(self._frame(), "a", duplicate_policy="median")

    def test_missing_signal_returns_an_empty_frame_not_an_error(self):
        out = signal_frame(self._frame(), "nope")
        assert out.empty
        assert list(out.columns) == ["systime", "value"]

    def test_rows_without_a_usable_timestamp_are_dropped(self):
        frame = self._frame()
        frame.loc[0, "systime"] = pd.NaT
        assert len(signal_frame(frame, "a")) == 2

    def test_requires_the_time_column(self):
        with pytest.raises(ValueError, match="Time column 'ts' not found"):
            signal_frame(self._frame(), "a", time_column="ts")


class TestUnionTimeline:
    def test_forward_fills_each_signal_independently(self):
        a = pd.DataFrame(
            {"systime": pd.to_datetime(["2025-01-01 00:00"]), "value": [True]}
        )
        b = pd.DataFrame(
            {"systime": pd.to_datetime(["2025-01-01 00:01"]), "value": [False]}
        )
        out = union_timeline({"a": a, "b": b})
        assert out["a"].tolist() == [True, True]
        assert pd.isna(out["b"].iloc[0])
        assert out["b"].iloc[1] is False

    def test_simultaneous_updates_collapse_to_one_row(self):
        stamp = pd.to_datetime(["2025-01-01 00:00"])
        out = union_timeline(
            {
                "a": pd.DataFrame({"systime": stamp, "value": [True]}),
                "b": pd.DataFrame({"systime": stamp, "value": [True]}),
            }
        )
        assert len(out) == 1

    def test_all_empty_inputs_yield_the_declared_columns(self):
        empty = pd.DataFrame(
            {"systime": pd.Series(dtype="datetime64[ns]"), "value": []}
        )
        out = union_timeline({"a": empty, "b": empty})
        assert out.empty
        assert set(out.columns) == {"systime", "a", "b"}
