"""Tests for :class:`ProductionCounterNormalizer`.

Every test here corresponds to a way a real part counter is encoded or a way a
real part counter misbehaves. The point of the class is that all of them produce
the same canonical quantities, so the assertions are mostly "this encoding
counts the same parts as that one".
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.events._output import validate_event_output
from ts_shape.events.production.signals.counters import ProductionCounterNormalizer
from ts_shape.events.production.signals.spec import (
    CounterKind,
    CounterMode,
    CounterSpec,
    ResetPolicy,
    Rollover,
)

BASE = pd.Timestamp("2025-01-01 06:00:00")


def sig(uuid: str, values: list, column: str, *, freq: str = "10min") -> pd.DataFrame:
    """One signal in the standard schema, sampled at a fixed interval."""
    return pd.DataFrame(
        {
            "systime": pd.date_range(BASE, periods=len(values), freq=freq),
            "uuid": uuid,
            column: values,
        }
    )


def normalizer(frame: pd.DataFrame, *specs: CounterSpec) -> ProductionCounterNormalizer:
    return ProductionCounterNormalizer(frame, counters=list(specs), machine_id="M1")


def total(norm: ProductionCounterNormalizer, kind: str = "total") -> float:
    events = norm.count_events()
    return float(events.loc[events["kind"] == kind, "quantity"].sum())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_requires_at_least_one_counter(self):
        with pytest.raises(ValueError, match="at least one CounterSpec"):
            ProductionCounterNormalizer(sig("c", [1], "value_integer"), counters=[])

    def test_missing_signal_yields_empty_canonical_output(self):
        norm = normalizer(sig("other", [1, 2], "value_integer"), CounterSpec(uuid="c"))
        events = norm.count_events()
        assert events.empty
        validate_event_output(events, "point")
        assert norm.diagnostics()["readings"].iloc[0] == 0

    def test_empty_input(self, empty_df):
        norm = normalizer(empty_df, CounterSpec(uuid="c"))
        assert norm.count_events().empty
        assert norm.totals().empty
        assert norm.counts_by_window("1h").empty
        assert norm.anomalies().empty


# ---------------------------------------------------------------------------
# CUMULATIVE
# ---------------------------------------------------------------------------
class TestCumulative:
    def test_differentiates_a_totalizer(self):
        norm = normalizer(
            sig("c", [100, 103, 107, 110], "value_integer"), CounterSpec(uuid="c")
        )
        assert total(norm) == 10.0

    def test_first_reading_is_only_a_baseline(self):
        # Production before the first reading cannot be attributed to it.
        norm = normalizer(sig("c", [100, 101], "value_integer"), CounterSpec(uuid="c"))
        events = norm.count_events()
        assert len(events) == 1
        assert events["quantity"].iloc[0] == 1.0

    def test_a_reset_to_zero_counts_the_new_value(self):
        norm = normalizer(
            sig("c", [10, 12, 0, 3], "value_integer"), CounterSpec(uuid="c")
        )
        # 10->12 = 2, 12->0 = reset (nothing yet), 0->3 = 3.
        assert total(norm) == 5.0
        assert norm.anomalies()["anomaly"].tolist() == ["reset"]

    def test_a_reset_to_a_non_zero_value_counts_that_value(self):
        norm = normalizer(sig("c", [500, 7], "value_integer"), CounterSpec(uuid="c"))
        assert total(norm) == 7.0

    def test_reset_policy_ignore_books_nothing(self):
        norm = normalizer(
            sig("c", [500, 7, 9], "value_integer"),
            CounterSpec(uuid="c", reset_policy=ResetPolicy.IGNORE),
        )
        assert total(norm) == 2.0

    def test_reset_policy_raise_reports_the_moment(self):
        norm = normalizer(
            sig("c", [500, 7], "value_integer"),
            CounterSpec(uuid="c", reset_policy=ResetPolicy.RAISE),
        )
        with pytest.raises(ValueError, match="decreased at 2025-01-01 06:10"):
            norm.count_events()

    def test_a_sixteen_bit_wrap_recovers_the_parts(self):
        norm = normalizer(
            sig("c", [65530, 65534, 3], "value_integer"),
            CounterSpec(uuid="c", rollover_at=Rollover.UINT16),
        )
        # 65530->65534 = 4, then the wrap 65534->3 = 5.
        assert total(norm) == 9.0
        assert norm.anomalies()["anomaly"].tolist() == ["rollover"]

    def test_a_reset_is_not_mistaken_for_a_wrap(self):
        # The single most damaging failure mode: 9 -> 0 on a wrapping counter
        # would imply 65527 parts if the wrap were taken at face value.
        norm = normalizer(
            sig("c", [5, 9, 0, 4], "value_integer"),
            CounterSpec(uuid="c", rollover_at=Rollover.UINT16),
        )
        assert total(norm) == 8.0
        assert norm.anomalies()["anomaly"].tolist() == ["reset"]

    def test_rollover_tolerance_bounds_what_counts_as_a_wrap(self):
        spec = CounterSpec(uuid="c", rollover_at=100, rollover_tolerance=0.1)
        # 95 -> 2 implies 7 parts: inside 10% of the register, so a wrap.
        assert total(normalizer(sig("c", [90, 95, 2], "value_integer"), spec)) == 12.0
        # 60 -> 2 would imply 42 parts: beyond the tolerance, so a reset.
        assert total(normalizer(sig("c", [55, 60, 2], "value_integer"), spec)) == 7.0

    def test_implausible_jumps_are_dropped_and_reported(self):
        norm = normalizer(
            sig("c", [10, 11, 999999, 1000000], "value_integer"),
            CounterSpec(uuid="c", max_plausible_delta=100),
        )
        assert total(norm) == 2.0
        anomalies = norm.anomalies()
        assert anomalies["anomaly"].tolist() == ["implausible_jump"]
        assert anomalies["severity"].iloc[0] == "error"

    def test_allow_negative_keeps_correction_steps(self):
        norm = normalizer(
            sig("c", [10, 15, 13], "value_integer"),
            CounterSpec(uuid="c", allow_negative=True),
        )
        assert total(norm) == 3.0
        assert norm.anomalies().empty

    def test_parts_per_count_scales_multi_cavity_counters(self):
        norm = normalizer(
            sig("c", [0, 1, 2], "value_integer"),
            CounterSpec(uuid="c", parts_per_count=8),
        )
        assert total(norm) == 16.0

    def test_reads_a_stringified_counter(self):
        norm = normalizer(
            sig("c", ["10", "12", "15"], "value_string"), CounterSpec(uuid="c")
        )
        assert total(norm) == 5.0

    def test_non_monotonic_input_order_is_sorted_first(self):
        frame = sig("c", [10, 12, 14], "value_integer").iloc[::-1]
        assert total(normalizer(frame, CounterSpec(uuid="c"))) == 4.0

    def test_duplicate_timestamps_collapse_to_one_reading(self):
        frame = pd.DataFrame(
            {
                "systime": pd.to_datetime(
                    ["2025-01-01 06:00", "2025-01-01 06:00", "2025-01-01 06:10"]
                ),
                "uuid": "c",
                "value_integer": [10, 12, 14],
            }
        )
        # Keeping both 06:00 rows would book a phantom 2 parts.
        assert total(normalizer(frame, CounterSpec(uuid="c"))) == 2.0


# ---------------------------------------------------------------------------
# Other encodings
# ---------------------------------------------------------------------------
class TestIncremental:
    def test_sums_per_reading_quantities(self):
        norm = normalizer(
            sig("c", [1, 1, 3, 1], "value_integer"),
            CounterSpec(uuid="c", mode=CounterMode.INCREMENTAL),
        )
        assert total(norm) == 6.0

    def test_negative_increments_are_clipped_and_flagged(self):
        norm = normalizer(
            sig("c", [2, -1, 3], "value_integer"),
            CounterSpec(uuid="c", mode=CounterMode.INCREMENTAL),
        )
        assert total(norm) == 5.0
        assert norm.anomalies()["anomaly"].tolist() == ["implausible_jump"]

    def test_negative_increments_are_kept_when_allowed(self):
        norm = normalizer(
            sig("c", [2, -1, 3], "value_integer"),
            CounterSpec(uuid="c", mode=CounterMode.INCREMENTAL, allow_negative=True),
        )
        assert total(norm) == 4.0


class TestPulse:
    def test_counts_rising_edges(self):
        norm = normalizer(
            sig("c", [False, True, False, True, True], "value_bool"),
            CounterSpec(uuid="c", mode=CounterMode.PULSE),
        )
        assert total(norm) == 2.0

    def test_a_signal_that_starts_true_counts_its_first_reading(self):
        # Change-triggered data: the first sample *is* the transition.
        norm = normalizer(
            sig("c", [True, False, True], "value_bool"),
            CounterSpec(uuid="c", mode=CounterMode.PULSE),
        )
        assert total(norm) == 2.0

    def test_falling_edges(self):
        norm = normalizer(
            sig("c", [True, False, True, False], "value_bool"),
            CounterSpec(uuid="c", mode=CounterMode.PULSE, edge="falling"),
        )
        assert total(norm) == 2.0

    def test_both_edges(self):
        norm = normalizer(
            sig("c", [True, False, True], "value_bool"),
            CounterSpec(uuid="c", mode=CounterMode.PULSE, edge="both"),
        )
        assert total(norm) == 3.0

    def test_edge_any_counts_every_logged_pulse(self):
        # Some historians write one row per pulse and never the false between,
        # where edge detection would see a single unending pulse.
        norm = normalizer(
            sig("c", [True, True, True], "value_bool"),
            CounterSpec(uuid="c", mode=CounterMode.PULSE, edge="any"),
        )
        assert total(norm) == 3.0

    def test_multi_cavity_pulse(self):
        norm = normalizer(
            sig("c", [True, False, True], "value_bool"),
            CounterSpec(uuid="c", mode=CounterMode.PULSE, parts_per_count=4),
        )
        assert total(norm) == 8.0

    def test_reads_a_numeric_pulse(self):
        norm = normalizer(
            sig("c", [0, 1, 0, 1], "value_integer"),
            CounterSpec(uuid="c", mode=CounterMode.PULSE),
        )
        assert total(norm) == 2.0


class TestQualityFlag:
    def test_classifies_each_reading_as_ok_or_nok(self):
        norm = normalizer(
            sig("v", [True, True, False, True], "value_bool"),
            CounterSpec(uuid="v", mode=CounterMode.QUALITY_FLAG),
        )
        assert total(norm, "ok") == 3.0
        assert total(norm, "nok") == 1.0

    def test_invert_reads_true_as_a_reject(self):
        norm = normalizer(
            sig("v", [True, False], "value_bool"),
            CounterSpec(uuid="v", mode=CounterMode.QUALITY_FLAG, invert=True),
        )
        assert total(norm, "nok") == 1.0
        assert total(norm, "ok") == 1.0

    def test_kind_nok_counts_only_rejects(self):
        norm = normalizer(
            sig("v", [True, False, False], "value_bool"),
            CounterSpec(uuid="v", mode=CounterMode.QUALITY_FLAG, kind=CounterKind.NOK),
        )
        assert total(norm, "nok") == 2.0
        assert total(norm, "ok") == 0.0
        assert norm.configured_kinds == frozenset({CounterKind.NOK})

    def test_a_trigger_samples_a_level_held_verdict_per_part(self):
        # The verdict stays true across many scans; only the trigger says how
        # many parts there were.
        verdict = sig("v", [True, True, True, True], "value_bool", freq="1min")
        trigger = pd.DataFrame(
            {
                "systime": [
                    BASE + pd.Timedelta(seconds=90),
                    BASE + pd.Timedelta(seconds=150),
                ],
                "uuid": "t",
                "value_bool": [True, True],
            }
        )
        norm = normalizer(
            pd.concat([verdict, trigger], ignore_index=True),
            CounterSpec(
                uuid="v",
                mode=CounterMode.QUALITY_FLAG,
                trigger_uuid="t",
                edge="any",
            ),
        )
        assert total(norm, "ok") == 2.0

    def test_configured_kinds_covers_all_three_for_a_verdict_signal(self):
        norm = normalizer(
            sig("v", [True], "value_bool"),
            CounterSpec(uuid="v", mode=CounterMode.QUALITY_FLAG),
        )
        assert norm.configured_kinds == frozenset(
            {CounterKind.OK, CounterKind.NOK, CounterKind.TOTAL}
        )


class TestDefectCode:
    def test_zero_is_good_and_anything_else_is_a_reject(self):
        norm = normalizer(
            sig("d", [0, 0, 7, 0, 12], "value_integer"),
            CounterSpec(uuid="d", mode=CounterMode.DEFECT_CODE),
        )
        assert total(norm, "ok") == 3.0
        assert total(norm, "nok") == 2.0

    def test_the_code_is_carried_as_the_defect_reason(self):
        norm = normalizer(
            sig("d", [0, 7], "value_integer"),
            CounterSpec(uuid="d", mode=CounterMode.DEFECT_CODE),
        )
        events = norm.count_events()
        rejects = events[events["kind"] == "nok"]
        assert rejects["defect_code"].tolist() == ["7"]
        assert events[events["kind"] == "ok"]["defect_code"].isna().all()

    def test_custom_ok_values_are_matched_case_insensitively(self):
        norm = normalizer(
            sig("d", ["ok", "OK", "scratch"], "value_string"),
            CounterSpec(uuid="d", mode=CounterMode.DEFECT_CODE, ok_values=("Ok",)),
        )
        assert total(norm, "ok") == 2.0
        assert total(norm, "nok") == 1.0

    def test_kind_nok_emits_only_rejects(self):
        norm = normalizer(
            sig("d", [0, 7], "value_integer"),
            CounterSpec(uuid="d", mode=CounterMode.DEFECT_CODE, kind=CounterKind.NOK),
        )
        assert norm.count_events()["kind"].tolist() == ["nok"]


class TestBatchTotal:
    def test_counts_each_change_of_an_absolute_quantity(self):
        norm = normalizer(
            sig("b", [0, 250, 250, 400], "value_integer"),
            CounterSpec(uuid="b", mode=CounterMode.BATCH_TOTAL),
        )
        # The repeated 250 is a republished value, not another 250 parts.
        assert total(norm) == 650.0

    def test_a_republished_constant_is_counted_once(self):
        norm = normalizer(
            sig("b", [100, 100, 100], "value_integer"),
            CounterSpec(uuid="b", mode=CounterMode.BATCH_TOTAL),
        )
        assert total(norm) == 100.0


# ---------------------------------------------------------------------------
# Combination and derivation
# ---------------------------------------------------------------------------
class TestDerivation:
    def test_total_is_derived_from_ok_plus_nok(self):
        frame = pd.concat(
            [
                sig("ok", [0, 8], "value_integer"),
                sig("nok", [0, 2], "value_integer"),
            ],
            ignore_index=True,
        )
        norm = normalizer(
            frame,
            CounterSpec(uuid="ok", kind=CounterKind.OK),
            CounterSpec(uuid="nok", kind=CounterKind.NOK),
        )
        row = norm.totals().iloc[0]
        assert (row["qty_ok"], row["qty_nok"], row["qty_total"]) == (8.0, 2.0, 10.0)
        assert row["good_rate_pct"] == pytest.approx(80.0)

    def test_ok_is_derived_from_total_minus_nok(self):
        frame = pd.concat(
            [
                sig("t", [0, 10], "value_integer"),
                sig("nok", [0, 3], "value_integer"),
            ],
            ignore_index=True,
        )
        norm = normalizer(
            frame,
            CounterSpec(uuid="t", kind=CounterKind.TOTAL),
            CounterSpec(uuid="nok", kind=CounterKind.NOK),
        )
        assert norm.totals()["qty_ok"].iloc[0] == 7.0

    def test_nok_is_derived_from_total_minus_ok(self):
        frame = pd.concat(
            [
                sig("t", [0, 10], "value_integer"),
                sig("ok", [0, 9], "value_integer"),
            ],
            ignore_index=True,
        )
        norm = normalizer(
            frame,
            CounterSpec(uuid="t", kind=CounterKind.TOTAL),
            CounterSpec(uuid="ok", kind=CounterKind.OK),
        )
        assert norm.totals()["qty_nok"].iloc[0] == 1.0

    def test_a_total_only_machine_reports_unknown_quality_not_perfect_quality(self):
        norm = normalizer(sig("t", [0, 10], "value_integer"), CounterSpec(uuid="t"))
        row = norm.totals().iloc[0]
        assert row["qty_total"] == 10.0
        assert pd.isna(row["qty_ok"])
        assert pd.isna(row["qty_nok"])
        assert pd.isna(row["good_rate_pct"])

    def test_assume_all_ok_is_an_explicit_opt_in(self):
        norm = normalizer(sig("t", [0, 10], "value_integer"), CounterSpec(uuid="t"))
        row = norm.totals(assume_all_ok=True).iloc[0]
        assert (row["qty_ok"], row["qty_nok"]) == (10.0, 0.0)
        assert row["good_rate_pct"] == 100.0

    def test_a_measured_kind_with_no_events_is_zero_not_unknown(self):
        # A shift with no rejects produced zero rejects; that is a measurement.
        frame = pd.concat(
            [
                sig("ok", [0, 5], "value_integer"),
                sig("nok", [0, 0], "value_integer"),
            ],
            ignore_index=True,
        )
        norm = normalizer(
            frame,
            CounterSpec(uuid="ok", kind=CounterKind.OK),
            CounterSpec(uuid="nok", kind=CounterKind.NOK),
        )
        row = norm.totals().iloc[0]
        assert row["qty_nok"] == 0.0
        assert row["good_rate_pct"] == 100.0

    def test_rework_and_scrap_stay_out_of_total(self):
        frame = pd.concat(
            [
                sig("t", [0, 10], "value_integer"),
                sig("rw", [0, 2], "value_integer"),
                sig("sc", [0, 1], "value_integer"),
            ],
            ignore_index=True,
        )
        norm = normalizer(
            frame,
            CounterSpec(uuid="t", kind=CounterKind.TOTAL),
            CounterSpec(uuid="rw", kind=CounterKind.REWORK),
            CounterSpec(uuid="sc", kind=CounterKind.SCRAP),
        )
        row = norm.totals().iloc[0]
        assert row["qty_total"] == 10.0
        assert (row["qty_rework"], row["qty_scrap"]) == (2.0, 1.0)

    def test_two_lanes_of_the_same_kind_are_summed(self):
        frame = pd.concat(
            [
                sig("lane1", [0, 4], "value_integer"),
                sig("lane2", [0, 6], "value_integer"),
            ],
            ignore_index=True,
        )
        norm = normalizer(
            frame,
            CounterSpec(uuid="lane1", kind=CounterKind.OK),
            CounterSpec(uuid="lane2", kind=CounterKind.OK),
        )
        assert norm.totals()["qty_ok"].iloc[0] == 10.0

    def test_different_encodings_agree_on_the_same_production(self):
        cumulative = normalizer(
            sig("c", [0, 1, 2, 3], "value_integer"), CounterSpec(uuid="c")
        )
        incremental = normalizer(
            sig("i", [1, 1, 1], "value_integer"),
            CounterSpec(uuid="i", mode=CounterMode.INCREMENTAL),
        )
        pulsed = normalizer(
            sig("p", [True, False, True, False, True], "value_bool"),
            CounterSpec(uuid="p", mode=CounterMode.PULSE),
        )
        assert total(cumulative) == total(incremental) == total(pulsed) == 3.0


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------
class TestOutputs:
    @pytest.fixture()
    def norm(self) -> ProductionCounterNormalizer:
        frame = pd.concat(
            [
                sig("ok", [0, 3, 6, 9, 12, 15], "value_integer", freq="30min"),
                sig("nok", [0, 0, 1, 1, 2, 2], "value_integer", freq="30min"),
            ],
            ignore_index=True,
        )
        return normalizer(
            frame,
            CounterSpec(uuid="ok", kind=CounterKind.OK, label="good parts"),
            CounterSpec(uuid="nok", kind=CounterKind.NOK),
        )

    def test_count_events_conform_to_the_point_schema(self, norm):
        validate_event_output(norm.count_events(), "point")

    def test_count_events_are_deltas_carrying_identity(self, norm):
        events = norm.count_events()
        assert events["is_delta"].all()
        assert set(events["machine_id"]) == {"M1"}
        assert set(events["uuid"]) == {"prod:count"}
        assert set(events["source_uuid"]) == {"ok", "nok"}

    def test_zero_rows_are_dropped_by_default_and_available_on_request(self, norm):
        assert len(norm.count_events()) < len(norm.count_events(include_zero=True))

    def test_windowed_output_conforms_to_the_summary_schema(self, norm):
        validate_event_output(norm.counts_by_window("1h"), "summary")

    def test_windows_sum_to_the_totals(self, norm):
        windows = norm.counts_by_window("1h")
        overall = norm.totals().iloc[0]
        assert windows["qty_ok"].sum() == overall["qty_ok"]
        assert windows["qty_nok"].sum() == overall["qty_nok"]

    def test_window_boundaries_follow_the_frequency(self, norm):
        windows = norm.counts_by_window("1h")
        assert (windows["end"] - windows["start"] == pd.Timedelta("1h")).all()

    def test_drop_empty_can_be_disabled_for_a_continuous_axis(self, norm):
        assert len(norm.counts_by_window("15min", drop_empty=False)) >= len(
            norm.counts_by_window("15min")
        )

    def test_nok_ppm_and_rates(self, norm):
        row = norm.totals().iloc[0]
        assert row["nok_rate_pct"] == pytest.approx(100 * 2 / 17)
        assert row["nok_ppm"] == pytest.approx(1e6 * 2 / 17)

    def test_diagnostics_have_one_row_per_counter(self, norm):
        diagnostics = norm.diagnostics()
        assert len(diagnostics) == 2
        assert set(diagnostics["source_uuid"]) == {"ok", "nok"}
        assert diagnostics["monotonic"].all()
        assert diagnostics.loc[0, "label"] == "good parts"

    def test_diagnostics_report_readings_and_quantity(self, norm):
        row = norm.diagnostics().set_index("source_uuid").loc["ok"]
        assert row["readings"] == 6
        assert row["quantity"] == 15.0
        assert row["first_value"] == 0.0
        assert row["last_value"] == 15.0

    def test_group_by_rejects_an_absent_column(self, norm):
        with pytest.raises(ValueError, match="are not present on the count events"):
            norm.counts_by_window("1h", group_by=("order_number",))

    def test_counts_by_context_rejects_an_absent_dimension(self, norm):
        with pytest.raises(ValueError, match="not columns on the events"):
            norm.counts_by_context(["order_number"])

    def test_counts_by_context_groups_supplied_columns(self, norm):
        events = norm.count_events(include_zero=True)
        events["order_number"] = ["A"] * 6 + ["B"] * (len(events) - 6)
        grouped = norm.counts_by_context(["order_number"], events=events)
        assert set(grouped["order_number"]) == {"A", "B"}
        assert grouped["qty_ok"].sum() == norm.totals()["qty_ok"].iloc[0]
