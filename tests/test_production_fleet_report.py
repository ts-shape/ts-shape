"""Tests for :class:`FleetProductionReport`.

The fleet report's job is to produce *one* comparable frame from machines that
share nothing but the concept of making parts. So the tests deliberately set up
an unequally instrumented fleet -- a fully mapped press, a partly mapped cell,
and a machine with a single counter and one broken tag -- and assert that the
schema stays stable, that unmeasured figures stay unknown rather than becoming
zero, and that the mis-mapped tag is reported rather than silently absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.production.signals.report import FleetProductionReport
from ts_shape.events.production.signals.spec import (
    ContextSpec,
    CounterKind,
    CounterMode,
    CounterSpec,
    CycleMode,
    CycleSpec,
    FleetSpec,
    MachineSpec,
    MachineState,
    StatusMode,
    StatusSpec,
    TimeUnit,
)

BASE = pd.Timestamp("2025-01-01 06:00:00")
HORIZON = pd.Timestamp("2025-01-01 12:00:00")
STEPS = 25  # 15-minute samples over six hours


def sig(uuid: str, values: list, column: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range(BASE, periods=len(values), freq="15min"),
            "uuid": uuid,
            column: values,
        }
    )


@pytest.fixture()
def fleet() -> FleetSpec:
    press = MachineSpec(
        machine_id="PRESS-01",
        attributes={"plant": "Ingolstadt", "line": "L1"},
        counters=(
            CounterSpec(uuid="p1.total", kind=CounterKind.TOTAL),
            CounterSpec(uuid="p1.nok", kind=CounterKind.NOK),
        ),
        status=StatusSpec(mode=StatusMode.PACKML, uuid="p1.packml"),
        cycle=CycleSpec(
            mode=CycleMode.EXPLICIT,
            uuid="p1.ct",
            unit=TimeUnit.SECONDS,
            cavities=4,
            max_seconds=600,
        ),
        context=(
            ContextSpec(name="order_number", uuid="p1.order"),
            ContextSpec(name="part_number", uuid="p1.part"),
        ),
        ideal_cycle_seconds=55.0,
    )
    cell = MachineSpec(
        machine_id="CELL-07",
        attributes={"plant": "Ingolstadt", "line": "L2"},
        counters=(
            CounterSpec(
                uuid="c7.done",
                kind=CounterKind.OK,
                mode=CounterMode.PULSE,
                parts_per_count=2,
            ),
        ),
        status=StatusSpec(mode=StatusMode.BOOLEAN, uuid="c7.run"),
        context=(ContextSpec(name="order_number", uuid="c7.order"),),
    )
    line = MachineSpec(
        machine_id="LINE-03",
        attributes={"plant": "Regensburg", "line": "L9"},
        counters=(CounterSpec(uuid="l3.total", kind=CounterKind.TOTAL),),
        # A tag that does not exist in the data: the everyday fleet defect.
        context=(ContextSpec(name="order_number", uuid="l3.order_MISSING"),),
    )
    return FleetSpec(machines=(press, cell, line))


@pytest.fixture()
def data() -> pd.DataFrame:
    press = pd.concat(
        [
            sig("p1.total", [i * 4 for i in range(STEPS)], "value_integer"),
            sig("p1.nok", [i // 8 for i in range(STEPS)], "value_integer"),
            sig("p1.packml", [6] * 10 + [9] * 3 + [6] * 12, "value_integer"),
            sig("p1.ct", [220.0] * STEPS, "value_double"),
            pd.DataFrame(
                {
                    "systime": [BASE, BASE + pd.Timedelta("3h")],
                    "uuid": "p1.order",
                    "value_string": ["A-100", "A-101"],
                }
            ),
            pd.DataFrame(
                {
                    "systime": [BASE, BASE + pd.Timedelta("3h")],
                    "uuid": "p1.part",
                    "value_string": ["4711", "4712"],
                }
            ),
        ],
        ignore_index=True,
    )
    cell = pd.concat(
        [
            sig("c7.done", [True, False] * 12 + [True], "value_bool"),
            sig("c7.run", [True] * 20 + [False] * 5, "value_bool"),
            pd.DataFrame(
                {
                    "systime": [BASE],
                    "uuid": "c7.order",
                    "value_string": ["A-100"],
                }
            ),
        ],
        ignore_index=True,
    )
    line = sig("l3.total", [i * 2 for i in range(STEPS)], "value_integer")
    # An unconfigured signal that must be ignored, not counted.
    stray = sig("stranger.signal", [1] * STEPS, "value_integer")
    return pd.concat([press, cell, line, stray], ignore_index=True)


@pytest.fixture()
def report(fleet: FleetSpec, data: pd.DataFrame) -> FleetProductionReport:
    return FleetProductionReport(fleet, data, until=HORIZON)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_requires_a_data_source(self, fleet):
        with pytest.raises(ValueError, match="Provide either dataframe="):
            FleetProductionReport(fleet)

    def test_rejects_two_data_sources(self, fleet, data):
        with pytest.raises(ValueError, match="not both"):
            FleetProductionReport(fleet, data, data_provider=lambda spec: data)

    def test_requires_at_least_one_machine(self, data):
        with pytest.raises(ValueError, match="at least one MachineSpec"):
            FleetProductionReport(FleetSpec(), data)

    def test_accepts_a_bare_sequence_of_machines(self, fleet, data):
        report = FleetProductionReport(list(fleet.machines), data)
        assert len(report.fleet) == 3

    def test_a_signal_on_two_machines_is_rejected_eagerly(self, data):
        shared = CounterSpec(uuid="shared")
        with pytest.raises(ValueError, match="counted twice"):
            FleetProductionReport(
                FleetSpec(
                    machines=(
                        MachineSpec(machine_id="A", counters=(shared,)),
                        MachineSpec(machine_id="B", counters=(shared,)),
                    )
                ),
                data,
            )

    def test_repr_summarises_the_configuration(self, report):
        assert "machines=3" in repr(report)
        assert "source=dataframe" in repr(report)


class TestNormalizers:
    def test_bundles_only_what_is_configured(self, report):
        press = report.normalizers("PRESS-01")
        assert press.counters is not None
        assert press.status is not None
        assert press.cycles is not None
        assert press.context is not None

        line = report.normalizers("LINE-03")
        assert line.counters is not None
        assert line.status is None
        assert line.cycles is None

    def test_are_cached_when_the_report_owns_the_frame(self, report):
        assert report.normalizers("PRESS-01") is report.normalizers("PRESS-01")

    def test_unknown_machine_raises(self, report):
        with pytest.raises(KeyError, match="No machine 'NOPE'"):
            report.normalizers("NOPE")

    def test_repr_lists_the_configured_parts(self, report):
        assert "counters" in repr(report.normalizers("PRESS-01"))

    def test_a_cycle_needing_status_is_skipped_when_status_is_absent(self, data):
        spec = MachineSpec(
            machine_id="X",
            counters=(CounterSpec(uuid="l3.total"),),
            cycle=CycleSpec(mode=CycleMode.STATE_INTERVAL, uuid="l3.total"),
        )
        report = FleetProductionReport(FleetSpec(machines=(spec,)), data)
        # Skipped rather than raising: one mis-configured machine must not take
        # the whole fleet report down.
        assert report.normalizers("X").cycles is None
        assert report.cycle_by_window().empty


# ---------------------------------------------------------------------------
# Production
# ---------------------------------------------------------------------------
class TestProductionByWindow:
    def test_covers_every_machine_with_a_counter(self, report):
        windows = report.production_by_window("2h")
        assert set(windows["machine_id"]) == {"PRESS-01", "CELL-07", "LINE-03"}

    def test_carries_the_asset_attributes(self, report):
        windows = report.production_by_window("2h")
        assert set(windows.columns) >= {"plant", "line"}
        assert (
            windows.loc[windows["machine_id"] == "LINE-03", "plant"] == "Regensburg"
        ).all()

    def test_schema_is_identical_for_unequally_instrumented_machines(self, report):
        windows = report.production_by_window("2h")
        press = windows[windows["machine_id"] == "PRESS-01"]
        line = windows[windows["machine_id"] == "LINE-03"]
        assert list(press.columns) == list(line.columns)

    def test_a_quality_blind_machine_reports_unknown_not_perfect(self, report):
        windows = report.production_by_window("2h")
        line = windows[windows["machine_id"] == "LINE-03"]
        assert line["qty_total"].notna().all()
        assert line["qty_ok"].isna().all()
        assert line["good_rate_pct"].isna().all()

    def test_a_measured_machine_reports_real_rates(self, report):
        windows = report.production_by_window("2h")
        press = windows[windows["machine_id"] == "PRESS-01"]
        assert press["qty_total"].sum() == 96.0
        assert press["good_rate_pct"].notna().all()

    def test_unconfigured_signals_are_ignored(self, report):
        # "stranger.signal" belongs to no machine and must not appear anywhere.
        assert "stranger.signal" not in set(report.signal_coverage()["uuid"])

    def test_selecting_machines(self, report):
        windows = report.production_by_window("2h", machines=["CELL-07"])
        assert set(windows["machine_id"]) == {"CELL-07"}

    def test_windows_are_sized_by_the_frequency(self, report):
        windows = report.production_by_window("1h")
        assert (windows["end"] - windows["start"] == pd.Timedelta("1h")).all()

    def test_empty_data_yields_an_empty_frame(self, fleet, empty_df):
        report = FleetProductionReport(fleet, empty_df)
        assert report.production_by_window("1h").empty


class TestProductionByContext:
    def test_breaks_production_down_per_order(self, report):
        by_order = report.production_by_context(["order_number"])
        press = by_order[by_order["machine_id"] == "PRESS-01"]
        assert set(press["order_number"]) == {"A-100", "A-101"}
        assert press["qty_total"].sum() == 96.0

    def test_a_machine_without_the_dimension_still_appears(self, report):
        by_order = report.production_by_context(["order_number"])
        line = by_order[by_order["machine_id"] == "LINE-03"]
        assert len(line) == 1
        assert line["order_number"].isna().all()
        assert line["qty_total"].iloc[0] == 48.0

    def test_several_dimensions_at_once(self, report):
        by_both = report.production_by_context(["order_number", "part_number"])
        press = by_both[by_both["machine_id"] == "PRESS-01"]
        assert set(zip(press["order_number"], press["part_number"], strict=True)) == {
            ("A-100", "4711"),
            ("A-101", "4712"),
        }

    def test_windows_inside_a_context(self, report):
        by_order = report.production_by_context(["order_number"], freq="2h")
        assert len(by_order) > len(report.production_by_context(["order_number"]))

    def test_requires_a_dimension(self, report):
        with pytest.raises(ValueError, match="at least one dimension"):
            report.production_by_context([])


# ---------------------------------------------------------------------------
# Status and cycle time
# ---------------------------------------------------------------------------
class TestStatusAndCycle:
    def test_status_by_window_covers_machines_with_a_status_signal(self, report):
        status = report.status_by_window("2h")
        assert set(status["machine_id"]) == {"PRESS-01", "CELL-07"}
        assert "availability_pct" in status.columns

    def test_status_intervals_are_concatenated_with_attributes(self, report):
        intervals = report.status_intervals()
        assert set(intervals["machine_id"]) == {"PRESS-01", "CELL-07"}
        assert "plant" in intervals.columns

    def test_status_summary_over_the_whole_range(self, report):
        status = report.status_by_window(None)
        assert len(status) == 2

    def test_cycle_by_window_covers_machines_with_a_cycle_source(self, report):
        cycles = report.cycle_by_window("2h")
        assert set(cycles["machine_id"]) == {"PRESS-01"}
        assert (cycles["mean_seconds"] == 55.0).all()

    def test_cycle_by_context(self, report):
        cycles = report.cycle_by_context(["part_number"])
        press = cycles[cycles["machine_id"] == "PRESS-01"]
        assert set(press["part_number"]) == {"4711", "4712"}


# ---------------------------------------------------------------------------
# OEE
# ---------------------------------------------------------------------------
class TestOEE:
    def test_one_row_per_machine_and_window(self, report):
        oee = report.oee("1D")
        assert set(oee["machine_id"]) == {"PRESS-01", "CELL-07", "LINE-03"}
        assert len(oee) == 3

    def test_availability_comes_from_the_status_intervals(self, report):
        oee = report.oee("1D").set_index("machine_id")
        # PackML: 2.5 h producing, 45 min down, then producing to the horizon.
        assert oee.loc["PRESS-01", "availability_pct"] == pytest.approx(
            100 * 5.25 / 6.0
        )

    def test_performance_uses_the_nameplate_cycle(self, report):
        oee = report.oee("1D").set_index("machine_id")
        row = oee.loc["PRESS-01"]
        expected = 55.0 * row["qty_total"] / row["producing_seconds"] * 100.0
        assert row["performance_pct"] == pytest.approx(expected)

    def test_performance_is_unknown_without_a_nameplate_cycle(self, report):
        oee = report.oee("1D").set_index("machine_id")
        assert pd.isna(oee.loc["CELL-07", "performance_pct"])

    def test_quality_is_unknown_for_a_quality_blind_machine(self, report):
        oee = report.oee("1D").set_index("machine_id")
        assert pd.isna(oee.loc["LINE-03", "quality_pct"])

    def test_oee_is_unknown_when_any_factor_is_unknown(self, report):
        oee = report.oee("1D").set_index("machine_id")
        # An OEE assembled from guessed inputs would be compared against
        # machines whose inputs were real.
        assert pd.isna(oee.loc["LINE-03", "oee_pct"])
        assert pd.isna(oee.loc["CELL-07", "oee_pct"])
        assert not pd.isna(oee.loc["PRESS-01", "oee_pct"])

    def test_oee_is_the_product_of_the_three_factors(self, report):
        row = report.oee("1D").set_index("machine_id").loc["PRESS-01"]
        assert row["oee_pct"] == pytest.approx(
            row["availability_pct"]
            * row["performance_pct"]
            * row["quality_pct"]
            / 10000.0
        )

    def test_windows_are_sized_by_the_frequency(self, report):
        oee = report.oee("1h")
        assert (oee["end"] - oee["start"] == pd.Timedelta("1h")).all()

    def test_attributes_survive_the_join(self, report):
        oee = report.oee("1D")
        assert set(oee.columns) >= {"plant", "line"}
        assert oee["plant"].notna().all()


# ---------------------------------------------------------------------------
# Roll-ups
# ---------------------------------------------------------------------------
class TestSummaries:
    def test_machine_summary_has_one_row_per_machine(self, report):
        summary = report.machine_summary()
        assert len(summary) == 3
        assert set(summary["machine_id"]) == {"PRESS-01", "CELL-07", "LINE-03"}

    def test_machine_summary_carries_performance_and_instrumentation(self, report):
        row = report.machine_summary().set_index("machine_id").loc["PRESS-01"]
        assert row["qty_total"] == 96.0
        assert row["availability_pct"] == pytest.approx(100 * 5.25 / 6.0)
        assert row["cycles"] > 0
        assert row["counter_anomalies"] == 0
        assert row["unmapped_state_values"] == 0
        assert row["context_dimensions"] == "order_number,part_number"

    def test_machine_summary_reports_unknowns_as_unknown(self, report):
        row = report.machine_summary().set_index("machine_id").loc["LINE-03"]
        assert row["qty_total"] == 48.0
        assert pd.isna(row["qty_ok"])
        assert pd.isna(row["availability_pct"])
        assert row["cycles"] == 0

    def test_machine_summary_reports_the_observed_span(self, report):
        row = report.machine_summary().set_index("machine_id").loc["PRESS-01"]
        assert row["first_seen"] == BASE
        assert row["last_seen"] == BASE + pd.Timedelta("6h")

    def test_fleet_summary_groups_by_attributes(self, report):
        rollup = report.fleet_summary(group_by=["plant"]).set_index("plant")
        assert rollup.loc["Ingolstadt", "machines"] == 2
        assert rollup.loc["Regensburg", "machines"] == 1

    def test_fleet_summary_recomputes_rates_from_the_sums(self, fleet, data):
        # Two machines that both measure OK and total: the group rate is the
        # ratio of the sums, not the mean of the per-machine percentages.
        twin = FleetSpec(
            machines=(
                fleet.machine("PRESS-01"),
                MachineSpec(
                    machine_id="PRESS-02",
                    attributes={"plant": "Ingolstadt", "line": "L1"},
                    counters=(
                        CounterSpec(uuid="l3.total", kind=CounterKind.TOTAL),
                        CounterSpec(uuid="c7.done", kind=CounterKind.OK),
                    ),
                ),
            )
        )
        report = FleetProductionReport(twin, data, until=HORIZON)
        machines = report.machine_summary()
        row = report.fleet_summary(group_by=["plant"]).iloc[0]
        assert row["quality_machines"] == 2
        assert row["good_rate_pct"] == pytest.approx(
            machines["qty_ok"].sum() / machines["qty_total"].sum() * 100.0
        )
        assert row["good_rate_pct"] != pytest.approx(machines["good_rate_pct"].mean())

    def test_fleet_summary_rate_only_covers_comparable_machines(self, report):
        # CELL-07 measures good parts but no total, PRESS-01 measures both, and
        # LINE-03 measures only a total. Dividing the group's summed good parts
        # by its summed total would mix unrelated quantities and can exceed 100%.
        rollup = report.fleet_summary(group_by=["plant"]).set_index("plant")
        row = rollup.loc["Ingolstadt"]
        assert row["quality_machines"] == 1
        assert 0.0 <= row["good_rate_pct"] <= 100.0
        press = report.machine_summary().set_index("machine_id").loc["PRESS-01"]
        assert row["good_rate_pct"] == pytest.approx(press["good_rate_pct"])

    def test_fleet_summary_reports_no_comparable_machines_as_zero(self, report):
        rollup = report.fleet_summary(group_by=["plant"]).set_index("plant")
        assert rollup.loc["Regensburg", "quality_machines"] == 0

    def test_fleet_summary_keeps_an_all_unknown_group_unknown(self, report):
        rollup = report.fleet_summary(group_by=["plant"]).set_index("plant")
        # Regensburg's only machine measures no quality; summing to zero would
        # report a 0% quality catastrophe instead of "not measured".
        assert pd.isna(rollup.loc["Regensburg", "qty_ok"])
        assert pd.isna(rollup.loc["Regensburg", "good_rate_pct"])

    def test_fleet_summary_without_grouping_covers_the_whole_fleet(self, report):
        rollup = report.fleet_summary()
        assert len(rollup) == 1
        assert rollup["machines"].iloc[0] == 3

    def test_fleet_summary_by_two_attributes(self, report):
        rollup = report.fleet_summary(group_by=["plant", "line"])
        assert len(rollup) == 3

    def test_fleet_summary_rejects_a_non_attribute(self, report):
        with pytest.raises(ValueError, match="are not machine attributes"):
            report.fleet_summary(group_by=["colour"])


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
class TestDiagnostics:
    def test_signal_coverage_lists_every_configured_signal_with_its_role(self, report):
        coverage = report.signal_coverage()
        press = coverage[coverage["machine_id"] == "PRESS-01"]
        assert set(press["role"]) == {
            "counter_total",
            "counter_nok",
            "status",
            "cycle",
            "context:order_number",
            "context:part_number",
        }

    def test_signal_coverage_flags_the_missing_tag(self, report):
        missing = report.signal_coverage().query("~present")
        assert missing["uuid"].tolist() == ["l3.order_MISSING"]
        assert missing["machine_id"].tolist() == ["LINE-03"]
        assert missing["readings"].tolist() == [0]

    def test_signal_coverage_reports_reading_counts_and_span(self, report):
        row = report.signal_coverage().set_index("uuid").loc["p1.total"]
        assert row["readings"] == STEPS
        assert row["first_seen"] == BASE
        assert row["present"]

    def test_coverage_still_reports_every_signal_when_no_data_arrived(
        self, fleet, empty_df
    ):
        # The commissioning case: nothing reported at all. The aggregates are
        # empty, but coverage must still list what was expected and missing,
        # otherwise a dead gateway looks like a machine that made no parts.
        report = FleetProductionReport(fleet, empty_df)
        coverage = report.signal_coverage()
        assert len(coverage) == sum(len(m.signal_uuids) for m in fleet.machines)
        assert not coverage["present"].any()
        assert report.production_by_window("1h").empty
        assert len(report.machine_summary()) == len(fleet.machines)

    def test_anomalies_are_empty_for_clean_data(self, report):
        assert report.anomalies().empty

    def test_anomalies_aggregate_counter_resets(self, fleet, data):
        broken = data.copy()
        mask = (broken["uuid"] == "l3.total") & (
            broken["systime"] == BASE + pd.Timedelta("1h")
        )
        broken.loc[mask, "value_integer"] = 0
        report = FleetProductionReport(fleet, broken, until=HORIZON)
        anomalies = report.anomalies()
        assert "counter" in set(anomalies["category"])
        row = anomalies[anomalies["source_uuid"] == "l3.total"].iloc[0]
        assert row["anomaly"] == "reset"
        assert row["machine_id"] == "LINE-03"
        assert row["occurrences"] >= 1

    def test_anomalies_aggregate_unmapped_state_codes(self, data):
        spec = MachineSpec(
            machine_id="X",
            counters=(CounterSpec(uuid="p1.total"),),
            status=StatusSpec(
                mode=StatusMode.CODE_MAP,
                uuid="p1.packml",
                state_map={"6": MachineState.PRODUCING},
            ),
        )
        report = FleetProductionReport(FleetSpec(machines=(spec,)), data)
        anomalies = report.anomalies()
        assert anomalies["category"].tolist() == ["status"]
        assert anomalies["anomaly"].iloc[0] == "unmapped_state:9"
        assert anomalies["occurrences"].iloc[0] == 3

    def test_anomalies_are_ordered_worst_first(self, fleet, data):
        broken = data.copy()
        mask = (broken["uuid"] == "l3.total") & (
            broken["systime"] == BASE + pd.Timedelta("1h")
        )
        broken.loc[mask, "value_integer"] = 0
        report = FleetProductionReport(fleet, broken, until=HORIZON)
        severities = report.anomalies()["severity"].tolist()
        rank = {"error": 0, "warning": 1, "info": 2}
        assert severities == sorted(severities, key=lambda s: rank.get(s, 3))


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------
class TestDataProvider:
    def test_each_machine_is_fetched_on_demand(self, fleet, data):
        calls: list[str] = []

        def provider(spec: MachineSpec) -> pd.DataFrame:
            calls.append(spec.machine_id)
            return data[data["uuid"].isin(spec.signal_uuids)]

        report = FleetProductionReport(fleet, data_provider=provider, until=HORIZON)
        windows = report.production_by_window("2h")
        assert set(windows["machine_id"]) == {"PRESS-01", "CELL-07", "LINE-03"}
        assert set(calls) == {"PRESS-01", "CELL-07", "LINE-03"}

    def test_provider_results_are_not_cached(self, fleet, data):
        calls: list[str] = []

        def provider(spec: MachineSpec) -> pd.DataFrame:
            calls.append(spec.machine_id)
            return data[data["uuid"].isin(spec.signal_uuids)]

        report = FleetProductionReport(fleet, data_provider=provider)
        report.normalizers("PRESS-01")
        report.normalizers("PRESS-01")
        assert calls.count("PRESS-01") == 2

    def test_provider_output_matches_the_single_frame_path(self, fleet, data):
        eager = FleetProductionReport(fleet, data, until=HORIZON)
        lazy = FleetProductionReport(
            fleet,
            data_provider=lambda spec: data[data["uuid"].isin(spec.signal_uuids)],
            until=HORIZON,
        )
        pd.testing.assert_frame_equal(
            eager.production_by_window("2h"), lazy.production_by_window("2h")
        )


# ---------------------------------------------------------------------------
# assume_all_ok
# ---------------------------------------------------------------------------
class TestAssumeAllOk:
    def test_off_by_default(self, report):
        line = report.production_by_window("2h").query("machine_id == 'LINE-03'")
        assert line["qty_ok"].isna().all()

    def test_opting_in_treats_total_as_good(self, fleet, data):
        report = FleetProductionReport(fleet, data, until=HORIZON, assume_all_ok=True)
        line = report.production_by_window("2h").query("machine_id == 'LINE-03'")
        assert (line["qty_ok"] == line["qty_total"]).all()
        assert (line["qty_nok"] == 0).all()

    def test_does_not_disturb_a_machine_that_measures_quality(self, fleet, data):
        plain = FleetProductionReport(fleet, data, until=HORIZON)
        assuming = FleetProductionReport(fleet, data, until=HORIZON, assume_all_ok=True)
        for frame in (plain, assuming):
            press = frame.machine_summary().set_index("machine_id").loc["PRESS-01"]
            assert press["qty_nok"] == 3.0


# ---------------------------------------------------------------------------
# Cross-checks
# ---------------------------------------------------------------------------
class TestConsistency:
    def test_windowed_production_sums_to_the_machine_summary(self, report):
        windows = report.production_by_window("1h")
        summary = report.machine_summary().set_index("machine_id")
        for machine_id, group in windows.groupby("machine_id"):
            expected = summary.loc[machine_id, "qty_total"]
            if pd.isna(expected):
                # CELL-07 counts good parts only, so its total is unmeasured
                # and stays unknown in both views.
                assert group["qty_total"].isna().all()
                continue
            assert group["qty_total"].sum() == pytest.approx(expected)

    def test_context_breakdown_sums_to_the_windowed_total(self, report):
        by_order = report.production_by_context(["order_number"])
        windows = report.production_by_window("1h")
        for machine_id, group in by_order.groupby("machine_id"):
            assert group["qty_total"].sum() == pytest.approx(
                windows.query("machine_id == @machine_id")["qty_total"].sum()
            )

    def test_hourly_status_seconds_sum_to_the_overall_summary(self, report):
        hourly = report.status_by_window("1h")
        overall = report.status_by_window(None).set_index("machine_id")
        for machine_id, group in hourly.groupby("machine_id"):
            assert group["producing_seconds"].sum() == pytest.approx(
                overall.loc[machine_id, "producing_seconds"]
            )

    def test_machine_summary_quantities_match_the_normalizer_totals(self, report):
        summary = report.machine_summary().set_index("machine_id")
        counters = report.normalizers("PRESS-01").counters
        assert counters is not None
        assert summary.loc["PRESS-01", "qty_total"] == pytest.approx(
            counters.totals()["qty_total"].iloc[0]
        )

    def test_fleet_summary_totals_match_the_machine_summary(self, report):
        machines = report.machine_summary()
        rollup = report.fleet_summary().iloc[0]
        assert rollup["qty_total"] == pytest.approx(
            np.nansum(machines["qty_total"].to_numpy(dtype="float64"))
        )
