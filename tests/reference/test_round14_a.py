"""Round 14 (a): definitions checked on hand-computed probes.

Each test states the definition it checks in its docstring; the expected
values follow from that definition, not from the implementation.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.engineering.rate_of_change import RateOfChangeEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents
from ts_shape.events.production.operating_state import OperatingStateEvents
from ts_shape.events.quality.calibration_validation import (
    CalibrationValidationEvents,
)
from ts_shape.events.supplychain.demand_pattern import DemandPatternEvents
from ts_shape.events.supplychain.lead_time_analysis import LeadTimeAnalysisEvents

# ---------------------------------------------------------------------------
# DemandPatternEvents: empty periods are zero demand
# ---------------------------------------------------------------------------


def _orders_jan1_jan5():
    t = pd.to_datetime(["2024-01-01", "2024-01-05"])
    return pd.DataFrame({"systime": t, "uuid": "d", "value_double": [5.0, 7.0]})


def test_demand_by_period_zero_fills_days_without_orders():
    """Each row is one order: Jan 2-4 had no orders, so zero demand.

    Daily periods Jan 1 .. Jan 5: totals 5, 0, 0, 0, 7; one order on Jan 1
    and Jan 5; the peak of an empty day is 0 and its mean order size is
    undefined.
    """
    out = DemandPatternEvents(_orders_jan1_jan5(), "d").demand_by_period(window="1D")
    assert out["start"].tolist() == list(pd.date_range("2024-01-01", periods=5))
    assert out["total_demand"].tolist() == [5.0, 0.0, 0.0, 0.0, 7.0]
    assert out["order_count"].tolist() == [1, 0, 0, 0, 1]
    assert out["peak_demand"].tolist() == [5.0, 0.0, 0.0, 0.0, 7.0]
    assert out["avg_demand"].iloc[[0, 4]].tolist() == [5.0, 7.0]
    assert out["avg_demand"].iloc[1:4].isna().all()


def test_demand_by_period_drop_keeps_only_periods_with_orders():
    """``empty_periods="drop"``: the two days with orders, nothing else."""
    out = DemandPatternEvents(_orders_jan1_jan5(), "d").demand_by_period(
        window="1D", empty_periods="drop"
    )
    assert out["total_demand"].tolist() == [5.0, 7.0]
    assert out["order_count"].tolist() == [1, 1]


def test_demand_by_period_counts_orders_and_averages_order_size():
    """Two orders of 2 and 4 on one day: total 6, two orders, mean size 3."""
    t = pd.to_datetime(["2024-01-01 08:00", "2024-01-01 12:00", "2024-01-03 00:00"])
    df = pd.DataFrame({"systime": t, "uuid": "d", "value_double": [2.0, 4.0, 1.0]})
    out = DemandPatternEvents(df, "d").demand_by_period(window="1D")
    assert out["total_demand"].tolist() == [6.0, 0.0, 1.0]
    assert out["order_count"].tolist() == [2, 0, 1]
    assert out["avg_demand"].iloc[0] == 3.0


def test_unknown_empty_periods_raises():
    ev = DemandPatternEvents(_orders_jan1_jan5(), "d")
    with pytest.raises(ValueError, match="empty_periods"):
        ev.demand_by_period(window="1D", empty_periods="nan")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty_periods"):
        ev.detect_demand_spikes(empty_periods="x")  # type: ignore[arg-type]


def test_demand_spike_baseline_includes_zero_days():
    """Spikes against mean + k * std of all daily totals, zeros included.

    Totals 5, 0, 0, 0, 7: mean 2.4, sample std sqrt(12.3) = 3.507, so with
    k = 1 the threshold is 5.907 and only Jan 5 (7) is a spike, by 4.6.
    Without the zero days (5, 7: mean 6, std 1.414) the threshold 7.414 is
    above both, so nothing is flagged.
    """
    ev = DemandPatternEvents(_orders_jan1_jan5(), "d")
    spikes = ev.detect_demand_spikes(threshold_factor=1.0, window="1D")
    assert spikes["start"].tolist() == [pd.Timestamp("2024-01-05")]
    assert spikes["baseline_mean"].iloc[0] == pytest.approx(2.4)
    assert spikes["spike_magnitude"].iloc[0] == pytest.approx(4.6)
    assert ev.detect_demand_spikes(
        threshold_factor=1.0, window="1D", empty_periods="drop"
    ).empty


def test_seasonality_counts_weekdays_without_orders():
    """Mondays: orders on Jan 1 (10) and none on Jan 8 -> average 5.

    Data Jan 1 (Mon) .. Jan 9 (Tue): every day in between is a zero day.
    """
    t = pd.to_datetime(["2024-01-01", "2024-01-09"])
    df = pd.DataFrame({"systime": t, "uuid": "d", "value_double": [10.0, 3.0]})
    ev = DemandPatternEvents(df, "d")
    by_day = ev.seasonality_summary(window="1D").set_index("period_label")
    assert by_day.loc["Monday", "avg_demand"] == 5.0
    assert by_day.loc["Wednesday", "avg_demand"] == 0.0
    dropped = ev.seasonality_summary(window="1D", empty_periods="drop")
    assert dropped.set_index("period_label").loc["Monday", "avg_demand"] == 10.0


# ---------------------------------------------------------------------------
# LeadTimeAnalysisEvents: one-to-one pairing within an order id
# ---------------------------------------------------------------------------


def _po(orders, deliveries):
    rows = [(pd.Timestamp(t), "order", i) for t, i in orders] + [
        (pd.Timestamp(t), "delivery", i) for t, i in deliveries
    ]
    return pd.DataFrame(rows, columns=["systime", "uuid", "value_string"])


def test_lead_times_pair_each_delivery_once_per_order_id():
    """Two PO-1 orders (Jan 1, Jan 2) and two deliveries (Jan 5, Jan 6).

    FIFO within PO-1: Jan 1 -> Jan 5 and Jan 2 -> Jan 6, 96 h each. Every
    order and delivery is paired, so no warning.
    """
    df = _po(
        [("2024-01-01", "PO-1"), ("2024-01-02", "PO-1")],
        [("2024-01-05", "PO-1"), ("2024-01-06", "PO-1")],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        out = LeadTimeAnalysisEvents(df).lead_times("order", "delivery")
    assert out["lead_time_hours"].tolist() == [96.0, 96.0]
    assert out["delivery_time"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-06"),
    ]


def test_lead_times_one_delivery_closes_one_order_and_warns():
    """Two PO-1 orders, one delivery Jan 5: the first order gets it (96 h).

    The second order stays open, which the DataQualityWarning reports.
    """
    df = _po(
        [("2024-01-01", "PO-1"), ("2024-01-02", "PO-1")],
        [("2024-01-05", "PO-1")],
    )
    with pytest.warns(DataQualityWarning, match="1 order") as record:
        out = LeadTimeAnalysisEvents(df).lead_times("order", "delivery")
    assert len([w for w in record if w.category is DataQualityWarning]) == 1
    assert out["lead_time_hours"].tolist() == [96.0]


def test_lead_times_ids_do_not_cross():
    """PO-2 delivered before PO-1: each id pairs with its own delivery.

    PO-1 Jan 1 -> Jan 4 (72 h), PO-2 Jan 2 -> Jan 3 (24 h); rows in
    order-time order.
    """
    df = _po(
        [("2024-01-01", "PO-1"), ("2024-01-02", "PO-2")],
        [("2024-01-03", "PO-2"), ("2024-01-04", "PO-1")],
    )
    out = LeadTimeAnalysisEvents(df).lead_times("order", "delivery")
    assert out["order_id"].tolist() == ["PO-1", "PO-2"]
    assert out["lead_time_hours"].tolist() == [72.0, 24.0]


def test_lead_times_early_delivery_within_an_id_is_skipped():
    """PO-1 orders Jan 2, Jan 3; PO-1 deliveries Jan 1, Jan 4, Jan 5.

    The Jan 1 delivery precedes every PO-1 order and stays unpaired; FIFO
    pairs Jan 2 -> Jan 4 (48 h) and Jan 3 -> Jan 5 (48 h), and warns.
    """
    df = _po(
        [("2024-01-02", "PO-1"), ("2024-01-03", "PO-1")],
        [("2024-01-01", "PO-1"), ("2024-01-04", "PO-1"), ("2024-01-05", "PO-1")],
    )
    with pytest.warns(DataQualityWarning, match="1 delivery"):
        out = LeadTimeAnalysisEvents(df).lead_times("order", "delivery")
    assert out["lead_time_hours"].tolist() == [48.0, 48.0]


def _greedy_by_id(orders, deliveries):
    """Reference: per id, each order (time order) takes the next free delivery >= it."""
    pairs = []
    for key in sorted({k for _, k in orders}):
        free = sorted(t for t, k in deliveries if k == key)
        j = 0
        for t in sorted(t for t, k in orders if k == key):
            while j < len(free) and free[j] < t:
                j += 1
            if j < len(free):
                pairs.append((t, free[j]))
                j += 1
    return sorted(pairs)


def test_lead_times_by_id_match_the_greedy_reference():
    """Random ids with early and missing deliveries: same pairs as the loop."""
    rng = np.random.default_rng(7)
    t0 = pd.Timestamp("2024-01-01")
    for _ in range(20):
        orders = [
            (t0 + pd.Timedelta(hours=int(h)), f"PO-{int(k)}")
            for h, k in zip(rng.integers(0, 200, 30), rng.integers(0, 6, 30))
        ]
        deliveries = [
            (t0 + pd.Timedelta(hours=int(h)), f"PO-{int(k)}")
            for h, k in zip(rng.integers(0, 260, 28), rng.integers(0, 6, 28))
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = LeadTimeAnalysisEvents(_po(orders, deliveries)).lead_times(
                "order", "delivery"
            )
        got = sorted(zip(out["order_time"], out["delivery_time"]))
        assert got == _greedy_by_id(orders, deliveries)
        assert out["order_time"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# RateOfChangeEvents: *_abs_rate twins
# ---------------------------------------------------------------------------


def test_rate_statistics_abs_rate_twins():
    """Ramp 10, 8, 6, 4, 2 at 1 s: rate -2 /s at every step, |rate| = 2.

    The window statistics are of |rate|: mean, max and min 2, std 0. The
    ``*_rate`` columns (absolute today) equal their ``*_abs_rate`` twins.
    """
    t = pd.date_range("2024-01-01", periods=5, freq="1s")
    df = pd.DataFrame(
        {"systime": t, "uuid": "s", "value_double": [10.0, 8.0, 6.0, 4.0, 2.0]}
    )
    out = RateOfChangeEvents(df, "s").rate_statistics(window="1h")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["max_abs_rate"] == 2.0
    assert row["mean_abs_rate"] == 2.0
    assert row["min_abs_rate"] == 2.0
    assert row["std_abs_rate"] == pytest.approx(0.0)
    cols = list(out.columns)
    for stat in ("mean", "std", "max", "min"):
        assert cols.index(f"{stat}_abs_rate") == cols.index(f"{stat}_rate") + 1
        assert row[f"{stat}_rate"] == row[f"{stat}_abs_rate"]


def test_empty_rate_statistics_have_the_twins():
    df = pd.DataFrame({"systime": [pd.Timestamp("2024-01-01")], "uuid": "s"})
    df["value_double"] = 1.0
    out = RateOfChangeEvents(df, "s").rate_statistics()
    assert out.empty
    assert "max_abs_rate" in out.columns


# ---------------------------------------------------------------------------
# CalibrationValidationEvents.calibration_curve: power column
# ---------------------------------------------------------------------------


def test_calibration_curve_power_highest_first():
    """measured = 2 * reference + 1 on {0: 1, 1: 3, 2: 5}.

    np.polyfit order, highest power first: [2, 1] with powers [1, 0].
    """
    out = CalibrationValidationEvents.calibration_curve({0: [1], 1: [3], 2: [5]})
    assert list(out.columns[:3]) == ["coefficient_index", "power", "coefficient_value"]
    assert out["coefficient_index"].tolist() == [0, 1]
    assert out["power"].tolist() == [1, 0]
    assert np.allclose(out["coefficient_value"], [2.0, 1.0])


def test_calibration_curve_power_with_capped_degree():
    """degree=2 with two points: the fit is capped at degree 1, powers [1, 0]."""
    out = CalibrationValidationEvents.calibration_curve({0: [1], 2: [5]}, degree=2)
    assert out["power"].tolist() == [1, 0]
    assert np.allclose(out["coefficient_value"], [2.0, 1.0])


def test_calibration_curve_quadratic_powers():
    """measured = reference**2 on 0, 1, 2, 3: coefficients [1, 0, 0], powers [2, 1, 0]."""
    out = CalibrationValidationEvents.calibration_curve(
        {0: [0], 1: [1], 2: [4], 3: [9]}, degree=2
    )
    assert out["power"].tolist() == [2, 1, 0]
    assert np.allclose(out["coefficient_value"], [1.0, 0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Bare counts: interval_count, micro_stop_count
# ---------------------------------------------------------------------------


def test_state_duration_statistics_interval_count():
    """low at 0/10/20 s, high at 30-60 s, low at 70 s: 2 low and 1 high interval."""
    t = pd.Timestamp("2024-01-01 06:00", tz="UTC") + pd.to_timedelta(
        np.arange(0, 80, 10), unit="s"
    )
    values = [1.0, 1.1, 0.9, 9.0, 9.2, 8.8, 9.1, 1.2]
    df = pd.DataFrame({"systime": t, "uuid": "sig", "value_double": values})
    stats = OperatingStateEvents(df, "sig").state_duration_statistics(n_states=2)
    cols = list(stats.columns)
    assert cols.index("interval_count") == cols.index("count") + 1
    by_label = stats.set_index("state_label")
    assert by_label.loc["low", "interval_count"] == 2
    assert by_label.loc["high", "interval_count"] == 1
    assert (stats["count"] == stats["interval_count"]).all()


def test_micro_stop_frequency_micro_stop_count():
    """Stops of 10 s (07:00) and 20 s (07:30) in one hour: two micro-stops."""
    rows = []
    for start, seconds in (("2024-01-01 07:00", 10), ("2024-01-01 07:30", 20)):
        t0 = pd.Timestamp(start)
        rows += [
            (t0 - pd.Timedelta("1min"), True),
            (t0, False),
            (t0 + pd.Timedelta(seconds=seconds), True),
        ]
    df = pd.DataFrame(rows, columns=["systime", "value_bool"]).assign(uuid="run")
    out = MicroStopEvents(df, "run").micro_stop_frequency(window="1h")
    hit = out[out["micro_stop_count"] > 0]
    assert hit["micro_stop_count"].tolist() == [2]
    assert hit["total_lost_time_seconds"].tolist() == [30.0]
    cols = list(out.columns)
    assert cols.index("micro_stop_count") == cols.index("count") + 1
