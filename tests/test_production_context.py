"""Tests for :class:`ProductionContextResolver`.

The failure modes being pinned here are the ones that quietly corrupt a
context-broken-down report: stale forward-fill, late-bound identifiers attached
to the wrong part, placeholders becoming their own reporting row, and one order
splitting into three because of padding and case.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.events._output import validate_event_output
from ts_shape.events.production.signals.context import ProductionContextResolver
from ts_shape.events.production.signals.spec import (
    ContextBinding,
    ContextSpec,
    CounterKind,
    CounterMode,
    CounterSpec,
)

BASE = pd.Timestamp("2025-01-01 06:00:00")


def sig(uuid: str, offsets: list[str], values: list, column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": [BASE + pd.Timedelta(o) for o in offsets],
            "uuid": uuid,
            column: values,
        }
    )


def events(offsets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": [BASE + pd.Timedelta(o) for o in offsets],
            "uuid": "prod:count",
            "source_uuid": "c",
            "quantity": 1.0,
        }
    )


def resolver(frame: pd.DataFrame, *specs: ContextSpec) -> ProductionContextResolver:
    return ProductionContextResolver(frame, specs=list(specs), machine_id="M1")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_requires_at_least_one_spec(self):
        with pytest.raises(ValueError, match="at least one ContextSpec"):
            ProductionContextResolver(sig("o", ["0h"], ["A"], "value_string"), specs=[])

    def test_rejects_duplicate_dimension_names(self):
        with pytest.raises(ValueError, match="Duplicate context dimension"):
            resolver(
                sig("o", ["0h"], ["A"], "value_string"),
                ContextSpec(name="order_number", uuid="a"),
                ContextSpec(name="order_number", uuid="b"),
            )

    def test_dimensions_and_spec_lookup(self):
        res = resolver(
            sig("o", ["0h"], ["A"], "value_string"),
            ContextSpec(name="order_number", uuid="o"),
        )
        assert res.dimensions == ("order_number",)
        assert res.spec_for("order_number").uuid == "o"
        with pytest.raises(KeyError, match="No context dimension 'batch'"):
            res.spec_for("batch")


# ---------------------------------------------------------------------------
# attach
# ---------------------------------------------------------------------------
class TestAttach:
    def test_attaches_the_value_in_force_at_each_event(self):
        frame = sig("o", ["0h", "2h"], ["A-100", "A-101"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        out = res.attach(events(["1h", "3h"]))
        assert out["order_number"].tolist() == ["A-100", "A-101"]

    def test_events_before_any_context_get_nothing(self):
        frame = sig("o", ["2h"], ["A-100"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        out = res.attach(events(["1h", "3h"]))
        assert pd.isna(out["order_number"].iloc[0])
        assert out["order_number"].iloc[1] == "A-100"

    def test_valid_for_stops_a_stale_value_from_spreading(self):
        # Without a limit, an order written once in January is still attached
        # in March.
        frame = sig("o", ["0h"], ["A-100"], "value_string")
        res = resolver(
            frame, ContextSpec(name="order_number", uuid="o", valid_for="1h")
        )
        out = res.attach(events(["30min", "5h"]))
        assert out["order_number"].iloc[0] == "A-100"
        assert pd.isna(out["order_number"].iloc[1])

    def test_next_known_binds_a_late_published_identifier(self):
        # The serial is written two seconds *after* the part it identifies; a
        # backward join would attach it to the following part.
        parts = events(["1h", "2h"])
        frame = sig(
            "s",
            ["1h0min2s", "2h0min2s"],
            ["SN-1", "SN-2"],
            "value_string",
        )
        res = resolver(
            frame,
            ContextSpec(
                name="serial_number",
                uuid="s",
                binding=ContextBinding.NEXT_KNOWN,
                valid_for="30s",
            ),
        )
        out = res.attach(parts)
        assert out["serial_number"].tolist() == ["SN-1", "SN-2"]

    def test_last_known_would_have_mis_attached_the_late_serial(self):
        parts = events(["1h", "2h"])
        frame = sig("s", ["1h0min2s", "2h0min2s"], ["SN-1", "SN-2"], "value_string")
        res = resolver(frame, ContextSpec(name="serial_number", uuid="s"))
        out = res.attach(parts)
        assert pd.isna(out["serial_number"].iloc[0])
        assert out["serial_number"].iloc[1] == "SN-1"

    def test_nearest_binding(self):
        frame = sig("o", ["1h", "3h"], ["A", "B"], "value_string")
        res = resolver(
            frame,
            ContextSpec(name="order_number", uuid="o", binding=ContextBinding.NEAREST),
        )
        out = res.attach(events(["2h30min"]))
        assert out["order_number"].iloc[0] == "B"

    def test_a_sentinel_ends_the_context(self):
        frame = sig("o", ["0h", "2h"], ["A-100", ""], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        out = res.attach(events(["1h", "3h"]))
        assert out["order_number"].iloc[0] == "A-100"
        assert pd.isna(out["order_number"].iloc[1])

    def test_clear_on_sentinel_false_carries_the_last_real_value(self):
        frame = sig("o", ["0h", "2h"], ["A-100", ""], "value_string")
        res = resolver(
            frame,
            ContextSpec(name="order_number", uuid="o", clear_on_sentinel=False),
        )
        out = res.attach(events(["1h", "3h"]))
        assert out["order_number"].tolist() == ["A-100", "A-100"]

    def test_normalisation_collapses_cosmetic_variants(self):
        frame = sig(
            "p", ["0h", "1h", "2h"], [" 000471 ", "000471", "471"], "value_string"
        )
        res = resolver(
            frame,
            ContextSpec(
                name="part_number", uuid="p", upper=True, strip_leading_zeros=True
            ),
        )
        out = res.attach(events(["30min", "1h30min", "2h30min"]))
        assert out["part_number"].tolist() == ["471", "471", "471"]

    def test_zero_is_a_sentinel_only_on_request(self):
        frame = sig("s", ["0h"], ["0"], "value_string")
        assert (
            resolver(frame, ContextSpec(name="serial_number", uuid="s"))
            .attach(events(["1h"]))["serial_number"]
            .iloc[0]
            == "0"
        )
        assert pd.isna(
            resolver(
                frame,
                ContextSpec(name="serial_number", uuid="s", zero_is_sentinel=True),
            )
            .attach(events(["1h"]))["serial_number"]
            .iloc[0]
        )

    def test_numeric_context_stays_numeric(self):
        frame = sig("t", ["0h"], [1500.0], "value_double")
        res = resolver(
            frame, ContextSpec(name="target_quantity", uuid="t", numeric=True)
        )
        assert res.attach(events(["1h"]))["target_quantity"].iloc[0] == 1500.0

    def test_a_numeric_identifier_matches_its_string_form(self):
        frame = sig("o", ["0h"], [4711.0], "value_double")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        assert res.attach(events(["1h"]))["order_number"].iloc[0] == "4711"

    def test_original_row_order_and_index_survive(self):
        frame = sig("o", ["0h", "2h"], ["A", "B"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        unsorted = events(["3h", "1h"])
        unsorted.index = [10, 20]
        out = res.attach(unsorted)
        assert out.index.tolist() == [10, 20]
        assert out["order_number"].tolist() == ["B", "A"]

    def test_attaches_to_interval_frames_on_start(self):
        frame = sig("o", ["0h"], ["A"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        intervals = pd.DataFrame(
            {
                "start": [BASE + pd.Timedelta("1h")],
                "end": [BASE + pd.Timedelta("2h")],
                "duration_seconds": [3600.0],
            }
        )
        assert res.attach(intervals)["order_number"].iloc[0] == "A"

    def test_a_missing_context_signal_yields_a_null_column(self):
        frame = sig("other", ["0h"], ["A"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        out = res.attach(events(["1h"]))
        assert "order_number" in out.columns
        assert out["order_number"].isna().all()

    def test_selecting_a_subset_of_dimensions(self):
        frame = pd.concat(
            [
                sig("o", ["0h"], ["A"], "value_string"),
                sig("p", ["0h"], ["P1"], "value_string"),
            ],
            ignore_index=True,
        )
        res = resolver(
            frame,
            ContextSpec(name="order_number", uuid="o"),
            ContextSpec(name="part_number", uuid="p"),
        )
        out = res.attach(events(["1h"]), dimensions=["order_number"])
        assert "order_number" in out.columns
        assert "part_number" not in out.columns

    def test_refuses_to_clobber_an_existing_column(self):
        frame = sig("o", ["0h"], ["A"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        prepared = events(["1h"]).assign(order_number="from-erp")
        with pytest.raises(ValueError, match="already exist"):
            res.attach(prepared)
        assert res.attach(prepared, overwrite=True)["order_number"].iloc[0] == "A"

    def test_requires_a_timestamp_column(self):
        frame = sig("o", ["0h"], ["A"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        with pytest.raises(ValueError, match="no timestamp column found"):
            res.attach(pd.DataFrame({"quantity": [1.0]}))

    def test_empty_events_get_the_columns_anyway(self):
        frame = sig("o", ["0h"], ["A"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        out = res.attach(events([]))
        assert "order_number" in out.columns


# ---------------------------------------------------------------------------
# timeline / changes / segments
# ---------------------------------------------------------------------------
class TestTimelineAndChanges:
    @pytest.fixture()
    def res(self) -> ProductionContextResolver:
        frame = pd.concat(
            [
                sig("o", ["0h", "0h30min", "2h"], ["A", "A", "B"], "value_string"),
                sig("p", ["0h", "2h"], ["P1", "P2"], "value_string"),
            ],
            ignore_index=True,
        )
        return resolver(
            frame,
            ContextSpec(name="order_number", uuid="o"),
            ContextSpec(name="part_number", uuid="p"),
        )

    def test_timeline_conforms_to_the_point_schema(self, res):
        validate_event_output(res.timeline(), "point")

    def test_timeline_collapses_republished_values(self, res):
        orders = res.timeline().query("dimension == 'order_number'")
        assert orders["value"].tolist() == ["A", "B"]

    def test_changes_exclude_the_first_reading_of_each_dimension(self, res):
        changes = res.changes()
        assert len(changes) == 2
        assert set(changes["transition"]) == {"A->B", "P1->P2"}

    def test_changes_can_be_filtered_to_one_dimension(self, res):
        changes = res.changes(dimensions=["order_number"])
        assert changes["dimension"].tolist() == ["order_number"]

    def test_a_context_ending_is_itself_a_change(self):
        frame = sig("o", ["0h", "1h", "2h"], ["A", "", "B"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        assert res.changes()["transition"].tolist() == ["A-><none>", "<none>->B"]

    def test_segments_bracket_each_constant_combination(self, res):
        segments = res.segments(
            ["order_number", "part_number"], until=BASE + pd.Timedelta("4h")
        )
        validate_event_output(segments, "interval")
        assert segments["order_number"].tolist() == ["A", "B"]
        assert segments["part_number"].tolist() == ["P1", "P2"]
        assert segments["duration_seconds"].iloc[0] == 2 * 3600.0

    def test_a_segment_starting_on_the_last_reading_has_no_measured_duration(self, res):
        # The context changes at 2 h, which is also the last reading, so there
        # is no evidence the new combination ran for any time.
        assert res.segments(["order_number"])["order_number"].tolist() == ["A"]

    def test_segments_close_the_last_run_at_the_given_horizon(self, res):
        segments = res.segments(["order_number"], until=BASE + pd.Timedelta("5h"))
        assert segments["end"].iloc[-1] == BASE + pd.Timedelta("5h")

    def test_segments_of_a_single_dimension(self, res):
        segments = res.segments(["order_number"], until=BASE + pd.Timedelta("4h"))
        assert segments["order_number"].tolist() == ["A", "B"]

    def test_empty_inputs_yield_canonical_empties(self):
        frame = sig("other", ["0h"], ["A"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        assert res.timeline().empty
        assert res.changes().empty
        assert res.segments().empty
        validate_event_output(res.timeline(), "point")
        validate_event_output(res.segments(), "interval")


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------
class TestCoverage:
    def test_reports_readings_values_and_placeholders(self):
        frame = sig("o", ["0h", "1h", "2h"], ["A", "", "B"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        row = res.coverage().iloc[0]
        assert row["dimension"] == "order_number"
        assert row["readings"] == 3
        assert row["distinct_values"] == 2
        assert row["placeholder_readings"] == 1
        assert row["binding"] == "last_known"

    def test_fill_pct_shows_how_much_production_can_be_attributed(self):
        # The order is set for the first hour of a two-hour span and cleared
        # after, so only half the range is attributable.
        frame = pd.concat(
            [
                sig("o", ["0h", "1h"], ["A", ""], "value_string"),
                sig("other", ["2h"], [1], "value_integer"),
            ],
            ignore_index=True,
        )
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        assert res.coverage()["fill_pct"].iloc[0] == pytest.approx(50.0)

    def test_a_constant_tag_shows_a_single_distinct_value(self):
        frame = sig("o", ["0h", "1h"], ["A", "A"], "value_string")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        assert res.coverage()["distinct_values"].iloc[0] == 1

    def test_a_missing_signal_reports_zero_readings(self):
        frame = sig("other", ["0h"], [1], "value_integer")
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        row = res.coverage().iloc[0]
        assert row["readings"] == 0
        assert row["distinct_values"] == 0
        assert pd.isna(row["first_seen"])


# ---------------------------------------------------------------------------
# Integration with counted production
# ---------------------------------------------------------------------------
class TestIntegrationWithCounters:
    def test_production_splits_across_orders(self):
        from ts_shape.events.production.signals.counters import (
            ProductionCounterNormalizer,
        )

        pulses = sig(
            "c",
            ["0h", "1h", "2h", "3h"],
            [True, True, True, True],
            "value_bool",
        )
        orders = sig("o", ["0h", "2h"], ["A-100", "A-101"], "value_string")
        frame = pd.concat([pulses, orders], ignore_index=True)

        counts = ProductionCounterNormalizer(
            frame,
            counters=[
                CounterSpec(
                    uuid="c",
                    kind=CounterKind.OK,
                    mode=CounterMode.PULSE,
                    edge="any",
                )
            ],
            machine_id="M1",
        )
        res = resolver(frame, ContextSpec(name="order_number", uuid="o"))
        enriched = res.attach(counts.count_events())
        by_order = counts.counts_by_context(["order_number"], events=enriched)
        assert by_order.set_index("order_number")["qty_ok"].to_dict() == {
            "A-100": 2.0,
            "A-101": 2.0,
        }
