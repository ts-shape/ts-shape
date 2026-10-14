"""Tests for ProductionScheduleAdherenceEvents (plan vs. actual)."""

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pytest  # type: ignore

from ts_shape.events.production import ProductionScheduleAdherenceEvents

COUNTER = "counter:line-1"


def _counter_df(times, values, uuid=COUNTER):
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": [uuid] * len(times),
            "value_double": np.asarray(values, dtype=float),
        }
    )


def _shift_schedule(planned=(400.0, 400.0)):
    return pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-05-02 06:00", "2024-05-02 14:00"]),
            "end": pd.to_datetime(["2024-05-02 14:00", "2024-05-02 22:00"]),
            "planned_qty": list(planned),
            "order_id": ["WO-1001", "WO-1002"],
        }
    )


def _two_shift_counter(first_rate=50.0, second_rate=50.0):
    """One hourly counter across both shifts, each climbing at its own rate."""
    times = pd.date_range("2024-05-02 06:00", "2024-05-02 22:00", freq="1h")
    increments = np.concatenate(
        [[0.0], np.full(8, first_rate), np.full(8, second_rate)]
    )
    return _counter_df(times, np.cumsum(increments))


class TestScheduleValidation:

    def test_missing_column_rejected(self):
        bad = pd.DataFrame({"start": [], "end": []})
        with pytest.raises(ValueError, match="missing required column"):
            ProductionScheduleAdherenceEvents(_two_shift_counter(), bad)

    def test_non_dataframe_schedule_rejected(self):
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            ProductionScheduleAdherenceEvents(_two_shift_counter(), [{"start": 1}])

    def test_inverted_period_rejected(self):
        sched = _shift_schedule()
        sched.loc[0, "end"] = pd.Timestamp("2024-05-02 05:00")
        with pytest.raises(ValueError, match="end before they start"):
            ProductionScheduleAdherenceEvents(_two_shift_counter(), sched)

    def test_schedule_is_sorted_by_start(self):
        sched = _shift_schedule().iloc[::-1].reset_index(drop=True)
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), sched)
        assert det.schedule["order_id"].tolist() == ["WO-1001", "WO-1002"]


class TestAdherenceByPeriod:

    def test_on_plan_line_is_on_track(self):
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), _shift_schedule())
        out = det.adherence_by_period(COUNTER)
        assert len(out) == 2
        assert out["actual_qty"].tolist() == [400.0, 400.0]
        assert out["schedule_adherence_pct"].tolist() == [100.0, 100.0]
        assert out["deviation_type"].tolist() == ["on_track", "on_track"]
        assert out["deviation_units"].tolist() == [0.0, 0.0]
        assert out["order_id"].tolist() == ["WO-1001", "WO-1002"]

    def test_slow_shift_is_behind(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=45.0), _shift_schedule()
        )
        out = det.adherence_by_period(COUNTER)
        assert out["actual_qty"].iloc[0] == 360.0
        assert out["schedule_adherence_pct"].iloc[0] == 90.0
        assert out["deviation_units"].iloc[0] == -40.0
        assert out["deviation_type"].iloc[0] == "behind"

    def test_stopped_shift_is_critical_behind(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=10.0), _shift_schedule()
        )
        out = det.adherence_by_period(COUNTER)
        assert out["deviation_type"].iloc[0] == "critical_behind"

    def test_fast_shift_is_ahead(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=60.0), _shift_schedule()
        )
        out = det.adherence_by_period(COUNTER)
        assert out["deviation_type"].iloc[0] == "ahead"
        assert out["deviation_units"].iloc[0] == 80.0

    def test_counter_reset_inside_a_period_is_handled(self):
        """A part-change reset must not read as a collapsed shift."""
        times = pd.date_range("2024-05-02 06:00", periods=9, freq="1h")
        # 0..200 then the PLC resets and climbs to 200 again → 400 parts.
        values = [0, 50, 100, 150, 200, 0, 50, 150, 200]
        df = _counter_df(times, values)
        det = ProductionScheduleAdherenceEvents(df, _shift_schedule())

        reset_aware = det.adherence_by_period(COUNTER)
        assert reset_aware["actual_qty"].iloc[0] == 400.0
        assert reset_aware["resets"].iloc[0] == 1
        assert reset_aware["deviation_type"].iloc[0] == "on_track"

        naive = det.adherence_by_period(COUNTER, handle_resets=False)
        assert naive["actual_qty"].iloc[0] == 200.0
        assert naive["deviation_type"].iloc[0] == "critical_behind"

    def test_recovery_rate_uses_time_left_in_the_period(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=45.0), _shift_schedule()
        )
        # Four hours before the first shift ends, 40 units short → 10/h.
        out = det.adherence_by_period(COUNTER, as_of="2024-05-02 10:00")
        assert out["recovery_required_rate"].iloc[0] == pytest.approx(10.0)
        assert out["remaining_seconds"].iloc[0] == 4 * 3600

    def test_recovery_rate_is_zero_when_at_or_above_plan(self):
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), _shift_schedule())
        out = det.adherence_by_period(COUNTER, as_of="2024-05-02 10:00")
        assert (out["recovery_required_rate"] == 0.0).all()

    def test_finished_behind_period_has_no_recovery_rate(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=45.0), _shift_schedule()
        )
        out = det.adherence_by_period(COUNTER, as_of="2024-05-03 00:00")
        assert pd.isna(out["recovery_required_rate"].iloc[0])
        assert out["remaining_seconds"].iloc[0] == 0.0

    def test_zero_planned_quantity_does_not_divide_by_zero(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(), _shift_schedule(planned=(0.0, 400.0))
        )
        out = det.adherence_by_period(COUNTER)
        assert pd.isna(out["schedule_adherence_pct"].iloc[0])
        assert out["deviation_type"].iloc[0] == "on_track"

    def test_unordered_thresholds_rejected(self):
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), _shift_schedule())
        with pytest.raises(ValueError, match="critical_pct <= behind_pct"):
            det.adherence_by_period(COUNTER, behind_pct=70.0, critical_pct=90.0)

    def test_unknown_counter_returns_typed_empty_frame(self):
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), _shift_schedule())
        out = det.adherence_by_period("counter:missing")
        assert out.empty
        assert "schedule_adherence_pct" in out.columns

    def test_empty_schedule_returns_typed_empty_frame(self):
        empty = pd.DataFrame(columns=["start", "end", "planned_qty", "order_id"])
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), empty)
        out = det.adherence_by_period(COUNTER)
        assert out.empty
        assert "recovery_required_rate" in out.columns

    def test_custom_schedule_column_names(self):
        sched = _shift_schedule().rename(
            columns={
                "start": "period_from",
                "end": "period_to",
                "planned_qty": "plan",
                "order_id": "wo",
            }
        )
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(),
            sched,
            schedule_start_column="period_from",
            schedule_end_column="period_to",
            planned_qty_column="plan",
            order_id_column="wo",
        )
        out = det.adherence_by_period(COUNTER)
        assert out["order_id"].tolist() == ["WO-1001", "WO-1002"]
        assert out["schedule_adherence_pct"].tolist() == [100.0, 100.0]

    def test_schedule_without_order_id_column(self):
        sched = _shift_schedule().drop(columns=["order_id"])
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), sched)
        out = det.adherence_by_period(COUNTER)
        assert out["order_id"].isna().all()


class TestDeviationEvents:

    def test_only_behind_periods_are_emitted_by_default(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=45.0, second_rate=60.0), _shift_schedule()
        )
        events = det.deviation_events(COUNTER)
        assert len(events) == 1
        assert events["deviation_type"].iloc[0] == "behind"
        assert events["severity"].iloc[0] == "warning"
        assert events["direction"].iloc[0] == "below"
        assert events["systime"].iloc[0] == pd.Timestamp("2024-05-02 14:00")

    def test_include_ahead_adds_the_over_producing_period(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=45.0, second_rate=60.0), _shift_schedule()
        )
        events = det.deviation_events(COUNTER, include_ahead=True)
        assert set(events["deviation_type"]) == {"behind", "ahead"}
        assert set(events["severity"]) == {"warning", "info"}

    def test_critical_period_carries_critical_severity(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=10.0), _shift_schedule()
        )
        events = det.deviation_events(COUNTER)
        assert events["severity"].iloc[0] == "critical"
        assert events["threshold"].iloc[0] == 80.0

    def test_on_plan_line_emits_nothing(self):
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), _shift_schedule())
        out = det.deviation_events(COUNTER)
        assert out.empty
        assert "severity" in out.columns


class TestAdherenceSummary:

    def test_summary_weights_by_planned_quantity(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=45.0), _shift_schedule(planned=(400.0, 100.0))
        )
        row = det.adherence_summary(COUNTER).iloc[0]
        # 360 + 400 actual against 400 + 100 planned.
        assert row["total_actual_qty"] == 760.0
        assert row["total_planned_qty"] == 500.0
        assert row["overall_adherence_pct"] == 152.0
        assert row["periods"] == 2
        assert row["periods_behind"] == 1
        assert row["periods_ahead"] == 1

    def test_summary_names_the_worst_order(self):
        det = ProductionScheduleAdherenceEvents(
            _two_shift_counter(first_rate=10.0), _shift_schedule()
        )
        row = det.adherence_summary(COUNTER).iloc[0]
        assert row["worst_order_id"] == "WO-1001"
        assert row["worst_adherence_pct"] == 20.0
        assert row["on_plan_rate_pct"] == 50.0

    def test_summary_spans_the_scheduled_horizon(self):
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), _shift_schedule())
        row = det.adherence_summary(COUNTER).iloc[0]
        assert row["start"] == pd.Timestamp("2024-05-02 06:00")
        assert row["end"] == pd.Timestamp("2024-05-02 22:00")
        assert row["duration_seconds"] == 16 * 3600

    def test_empty_input_returns_typed_empty_frame(self):
        det = ProductionScheduleAdherenceEvents(_two_shift_counter(), _shift_schedule())
        out = det.adherence_summary("counter:missing")
        assert out.empty
        assert "overall_adherence_pct" in out.columns


class TestNonNumericValuesWarn:
    def test_non_numeric_planned_qty_warns(self):
        from ts_shape.errors import DataQualityWarning

        sched = _shift_schedule()
        sched["planned_qty"] = ["400", "n/a"]
        with pytest.warns(DataQualityWarning, match="1 non-null value"):
            tracker = ProductionScheduleAdherenceEvents(_two_shift_counter(), sched)
        assert np.isnan(tracker.schedule["planned_qty"].iloc[1])

    def test_non_numeric_counter_values_warn(self):
        from ts_shape.errors import DataQualityWarning

        df = _two_shift_counter()
        df["value_double"] = df["value_double"].astype(object)
        df.loc[3, "value_double"] = "err"
        with pytest.warns(DataQualityWarning, match="1 non-null value"):
            tracker = ProductionScheduleAdherenceEvents(df, _shift_schedule())
        tracker.adherence_by_period(COUNTER)
