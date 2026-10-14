"""Per-group counter totals add up to the signal total (no boundary loss).

Computing a counter quantity separately inside each group (shift, day,
operator, part, period, batch ...) loses the increment between the last
reading of one group and the first reading of the next. Every detector that
reports per-group counter totals now computes the reset-aware increments once
over the whole signal and books each to the group of the reading that reports
it, so the groups add up to ``counter_quantity`` of the whole signal.
``handle_resets=False`` keeps the naive per-group ``max(0, last - first)``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.events.production.batch_tracking import BatchTrackingEvents
from ts_shape.events.production.oee_calculator import OEEEvents
from ts_shape.events.production.operator_performance import (
    OperatorPerformanceEvents,
)
from ts_shape.events.production.period_summary import PeriodSummaryEvents
from ts_shape.events.production.quality_tracking import QualityEvents
from ts_shape.events.production.rework_tracking import ReworkEvents
from ts_shape.events.production.schedule_adherence import (
    ProductionScheduleAdherenceEvents,
)
from ts_shape.events.production.scrap_tracking import ScrapEvents
from ts_shape.events.production.shift_handover import ShiftHandoverEvents
from ts_shape.events.production.shift_reporting import ShiftReportEvents
from ts_shape.events.production.target_tracking import TargetEvents
from ts_shape.utils.counters import counter_quantity
from ts_shape.utils.shifts import assign_shift, production_dates

START = "2025-01-01 06:00"
N = 96  # two days at 30-minute resolution: crosses shift and day boundaries
TIMES = pd.date_range(START, periods=N, freq="30min")
SHIFTS = {
    "shift_1": ("06:00", "14:00"),
    "shift_2": ("14:00", "22:00"),
    "shift_3": ("22:00", "06:00"),
}


def _counter(step: int, *, resets: tuple[int, ...] = ()) -> list[int]:
    """A counter rising ``step`` per reading, dropping to 0 at ``resets``."""
    values, value = [], 100
    for i in range(N):
        if i in resets:
            value = 0
        elif i:
            value += step
        values.append(value)
    return values


# The second reset sits exactly on the 22:00 shift boundary (index 32).
OK = _counter(10, resets=(20, 32))
NOK = _counter(1, resets=(50,))


def _signal(uuid: str, values: list, column: str = "value_integer") -> pd.DataFrame:
    return pd.DataFrame({"systime": TIMES, "uuid": uuid, column: values})


def _labels(uuid: str, labels: list[str], every: int) -> pd.DataFrame:
    values = [labels[(i // every) % len(labels)] for i in range(N)]
    return _signal(uuid, values, "value_string")


def _total(values: list) -> float:
    return counter_quantity(pd.Series(values, dtype=float))


def _naive(values: list, keys: pd.DataFrame) -> pd.Series:
    """Per-group ``max(0, last - first)`` -- the handle_resets=False quantity."""
    frame = keys.assign(v=values)
    cols = list(keys.columns)
    g = frame.groupby(cols, sort=True)["v"]
    return (g.last() - g.first()).clip(lower=0).astype(float)


def _shift_keys() -> pd.DataFrame:
    shift = assign_shift(pd.Series(TIMES), SHIFTS)
    date = production_dates(pd.Series(TIMES), shift, SHIFTS)
    return pd.DataFrame({"date": date, "shift": shift})


def test_counter_crosses_shift_and_day_boundaries() -> None:
    keys = _shift_keys()
    # Sanity: the naive per-shift sum really loses production here.
    assert _naive(OK, keys).sum() < _total(OK)
    assert keys.drop_duplicates().shape[0] >= 6


# ----------------------------------------------------------------------
# Shift reporting
# ----------------------------------------------------------------------


def test_shift_production_adds_up() -> None:
    rep = ShiftReportEvents(_signal("c", OK))
    out = rep.shift_production("c")
    assert out["quantity"].sum() == _total(OK)
    # Both resets are seen, including the one on the 22:00 boundary.
    assert out["resets"].sum() == 2


def test_shift_production_issue_example() -> None:
    # 10 parts per 30 min from 06:00 to 21:30: 310 parts, 150 + 160 (not 150 + 150).
    df = pd.DataFrame(
        {
            "systime": pd.date_range(START, periods=32, freq="30min"),
            "uuid": "c",
            "value_integer": range(100, 420, 10),
        }
    )
    out = ShiftReportEvents(df).shift_production("c")
    assert out["quantity"].tolist() == [150, 160]
    naive = ShiftReportEvents(df).shift_production("c", handle_resets=False)
    assert naive["quantity"].tolist() == [150, 150]


def test_shift_production_naive_unchanged() -> None:
    out = ShiftReportEvents(_signal("c", OK)).shift_production("c", handle_resets=False)
    expected = _naive(OK, _shift_keys())
    assert out["quantity"].tolist() == expected.astype(int).tolist()


def test_shift_production_date_filter_keeps_first_increment() -> None:
    rep = ShiftReportEvents(_signal("c", OK))
    day2 = rep.shift_production("c", date="2025-01-02", production_day=False)
    # The readings of Jan 2 report the increments since the reading before
    # each of them, including the one from Jan 1 23:30 to Jan 2 00:00.
    times = pd.Series(TIMES)
    inc = pd.Series(OK, dtype=float).diff()
    inc = inc.where(inc >= 0, pd.Series(OK, dtype=float))
    expected = inc[times.dt.date == pd.Timestamp("2025-01-02").date()].sum()
    assert day2["quantity"].sum() == expected
    full = rep.shift_production("c", production_day=False)
    assert full["quantity"].sum() == _total(OK)


def test_shift_production_by_part_adds_up() -> None:
    df = pd.concat([_signal("c", OK), _labels("part", ["A", "B"], 7)])
    out = ShiftReportEvents(df).shift_production("c", part_id_uuid="part")
    assert out["quantity"].sum() == _total(OK)


def test_shift_targets_and_comparison_use_full_quantities() -> None:
    rep = ShiftReportEvents(_signal("c", OK))
    targets = rep.shift_targets("c", targets={s: 100 for s in SHIFTS})
    assert targets["actual"].sum() == _total(OK)


# ----------------------------------------------------------------------
# Quality
# ----------------------------------------------------------------------


def _quality_frame() -> pd.DataFrame:
    return pd.concat(
        [
            _signal("ok", OK),
            _signal("nok", NOK),
            _labels("part", ["A", "B"], 5),
            _labels("reason", ["Scratch", "Dent"], 3),
        ]
    )


def test_nok_by_shift_adds_up() -> None:
    out = QualityEvents(_quality_frame()).nok_by_shift("ok", "nok")
    assert out["ok_parts"].sum() == _total(OK)
    assert out["nok_parts"].sum() == _total(NOK)
    assert out["resets"].sum() == 3


def test_nok_by_shift_naive_unchanged() -> None:
    out = QualityEvents(_quality_frame()).nok_by_shift("ok", "nok", handle_resets=False)
    keys = _shift_keys()
    assert out["ok_parts"].tolist() == _naive(OK, keys).tolist()
    assert out["nok_parts"].tolist() == _naive(NOK, keys).tolist()


def test_daily_quality_summary_adds_up() -> None:
    out = QualityEvents(_quality_frame()).daily_quality_summary("ok", "nok")
    assert out["ok_parts"].sum() == _total(OK)
    assert out["nok_parts"].sum() == _total(NOK)


def test_quality_by_part_and_reason_add_up() -> None:
    q = QualityEvents(_quality_frame())
    by_part = q.quality_by_part("ok", "nok", "part")
    assert by_part["ok_parts"].sum() == _total(OK)
    assert by_part["nok_parts"].sum() == _total(NOK)
    by_reason = q.nok_by_reason("nok", "reason")
    assert by_reason["nok_parts"].sum() == _total(NOK)


# ----------------------------------------------------------------------
# Rework and scrap (mirror implementations)
# ----------------------------------------------------------------------


@pytest.fixture
def rework() -> ReworkEvents:
    df = pd.concat(
        [_signal("rw", NOK), _signal("prod", OK), _labels("part", ["A", "B"], 5)]
    )
    return ReworkEvents(df)


@pytest.fixture
def scrap() -> ScrapEvents:
    df = pd.concat(
        [_signal("scrap", NOK, "value_double"), _labels("part", ["A", "B"], 5)]
    )
    return ScrapEvents(df)


def test_rework_by_shift_adds_up(rework: ReworkEvents) -> None:
    out = rework.rework_by_shift("rw")
    assert out["rework_count"].sum() == _total(NOK)
    naive = rework.rework_by_shift("rw", handle_resets=False)
    assert naive["rework_count"].tolist() == _naive(NOK, _shift_keys()).tolist()


def test_rework_rate_adds_up(rework: ReworkEvents) -> None:
    out = rework.rework_rate("rw", "prod")
    assert out["rework_count"].sum() == _total(NOK)
    assert out["total_produced"].sum() == _total(OK)
    naive = rework.rework_rate("rw", "prod", handle_resets=False)
    assert naive["total_produced"].tolist() == _naive(OK, _shift_keys()).tolist()


def test_rework_cost_adds_up(rework: ReworkEvents) -> None:
    out = rework.rework_cost("rw", "part", {"A": 2.0, "B": 3.0})
    assert out["rework_count"].sum() == _total(NOK)


def test_rework_trend_adds_up(rework: ReworkEvents) -> None:
    out = rework.rework_trend("rw", window="6h")
    assert out["rework_count"].sum() == _total(NOK)
    naive = rework.rework_trend("rw", window="6h", handle_resets=False)
    periods = pd.DataFrame({"p": pd.Series(TIMES).dt.floor("6h")})
    assert naive["rework_count"].tolist() == _naive(NOK, periods).tolist()


def test_scrap_by_shift_adds_up(scrap: ScrapEvents) -> None:
    out = scrap.scrap_by_shift("scrap")
    assert out["scrap_quantity"].sum() == _total(NOK)
    naive = scrap.scrap_by_shift("scrap", handle_resets=False)
    assert naive["scrap_quantity"].tolist() == _naive(NOK, _shift_keys()).tolist()


def test_scrap_cost_and_trend_add_up(scrap: ScrapEvents) -> None:
    cost = scrap.scrap_cost("scrap", "part", {"A": 1.0, "B": 1.0})
    assert cost["scrap_quantity"].sum() == _total(NOK)
    trend = scrap.scrap_trend("scrap", window="1D")
    assert trend["scrap_quantity"].sum() == _total(NOK)


# ----------------------------------------------------------------------
# Operators, periods, targets, OEE, handover
# ----------------------------------------------------------------------


def test_operator_totals_add_up() -> None:
    df = pd.concat(
        [_signal("c", OK), _signal("nok", NOK), _labels("op", ["Ann", "Bob"], 6)]
    )
    perf = OperatorPerformanceEvents(df)
    prod = perf.production_by_operator("op", "c")
    assert prod["total_produced"].sum() == _total(OK)
    quality = perf.quality_by_operator("op", "c", "nok")
    assert quality["ok_count"].sum() == _total(OK)
    assert quality["nok_count"].sum() == _total(NOK)


def test_period_summary_adds_up() -> None:
    summary = PeriodSummaryEvents(_signal("c", OK))
    weekly = summary.weekly_summary("c")
    assert weekly["total_production"].sum() == _total(OK)
    monthly = summary.monthly_summary("c")
    assert monthly["total_production"].sum() == _total(OK)


def test_target_tracking_adds_up() -> None:
    tt = TargetEvents(_signal("c", OK))
    shifts = tt.compare_to_target("c", {s: 100 for s in SHIFTS})
    assert shifts["actual"].sum() == _total(OK)
    days = tt.target_achievement_summary("c", 500)
    assert days["actual"].sum() == _total(OK)
    naive = tt.compare_to_target("c", {s: 100 for s in SHIFTS}, handle_resets=False)
    expected = _naive(OK, _shift_keys()).astype(int).tolist()
    assert naive["actual"].tolist() == expected


def test_oee_daily_counts_add_up() -> None:
    oee = OEEEvents(pd.concat([_signal("c", OK), _signal("nok", NOK)]))
    perf = oee.performance("c", 10.0)
    assert perf["actual_parts"].sum() == _total(OK)
    quality = oee.quality("c", "nok")
    assert quality["total_parts"].sum() == _total(OK)
    assert quality["reject_parts"].sum() == _total(NOK)


def test_shift_handover_adds_up() -> None:
    df = pd.concat(
        [
            _signal("c", OK),
            _signal("ok", OK),
            _signal("nok", NOK),
            _labels("state", ["Running"], 1),
        ]
    )
    handover = ShiftHandoverEvents(df)
    # One report per production day; the increments are computed on the full
    # signal before the day is selected, so the days add up.
    report = pd.concat(
        [
            handover.generate_report("c", "ok", "nok", "state", report_date=day)
            for day in ("2025-01-01", "2025-01-02")
        ]
    )
    assert report["production"].sum() == _total(OK)
    assert report["ok_parts"].sum() == _total(OK)
    assert report["nok_parts"].sum() == _total(NOK)


# ----------------------------------------------------------------------
# Batches and scheduled periods
# ----------------------------------------------------------------------


def test_batch_yield_adds_up() -> None:
    # Batch IDs change every 8 readings; the counter covers the batch signal.
    df = pd.concat([_signal("c", OK), _labels("batch", ["X", "Y", "Z"], 8)])
    bt = BatchTrackingEvents(df, "batch")
    out = bt.batch_yield("c")
    # Readings after the first batch-ID sample: everything but the baseline.
    assert out["quantity"].sum() == _total(OK)
    assert out["resets"].sum() == 2
    # handle_resets=False keeps the naive [start, end] window per batch.
    naive = bt.batch_yield("c", handle_resets=False)
    keys = pd.DataFrame({"b": [i // 8 for i in range(N)]})
    assert naive["quantity"].tolist() == _naive(OK, keys).astype(int).tolist()


def test_batch_yield_production_between_batches_goes_to_earlier_batch() -> None:
    t = pd.date_range("2025-01-01", periods=4, freq="10s")
    batches = pd.DataFrame(
        {"systime": t, "uuid": "b", "value_string": ["A", "A", "B", "B"]}
    )
    # The counter is read between the batch-ID samples.
    counter = pd.DataFrame(
        {
            "systime": t + pd.Timedelta("5s"),
            "uuid": "c",
            "value_integer": [0, 5, 10, 15],
        }
    )
    out = BatchTrackingEvents(pd.concat([batches, counter]), "b").batch_yield("c")
    # A: readings at 5 s (baseline) and 15 s; the 25 s reading reports the
    # production while B ran (B's ID was seen at 20 s). The reading at 35 s
    # is after the last batch-ID sample and belongs to no batch.
    assert out["quantity"].tolist() == [5, 5]


def test_schedule_adherence_adds_up() -> None:
    df = _signal("c", OK, "value_double")
    edges = pd.date_range(START, periods=7, freq="8h")
    schedule = pd.DataFrame(
        {
            "start": edges[:-1],
            "end": edges[1:],
            "planned_qty": 100.0,
            "order_id": [f"o{i}" for i in range(6)],
        }
    )
    sa = ProductionScheduleAdherenceEvents(df, schedule)
    out = sa.adherence_by_period("c")
    # The periods are contiguous and cover the signal: each increment is
    # booked once, a reading on a shared boundary to the period ending there.
    assert out["actual_qty"].sum() == _total(OK)
    naive = sa.adherence_by_period("c", handle_resets=False)
    # Naive: [start, end] windows, both ends inclusive, per period.
    times = pd.Series(TIMES)
    for row, (s, e) in zip(naive.itertuples(), zip(edges[:-1], edges[1:])):
        window = pd.Series(OK, dtype=float)[(times >= s) & (times <= e)]
        expected = counter_quantity(window, handle_resets=False)
        assert row.actual_qty == expected
