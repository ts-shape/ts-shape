"""Regression tests: order/delivery pairing never gives negative lead times."""

import warnings

import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.supplychain.lead_time_analysis import LeadTimeAnalysisEvents


def _signal(times, values, uuid="sig", column="value_double"):
    return pd.DataFrame(
        {"systime": pd.to_datetime(times), "uuid": uuid, column: values}
    ).assign(is_delta=True)


def _minutes(n, start="2024-01-01 00:00"):
    return pd.date_range(start, periods=n, freq="1min")


# --- LeadTimeAnalysisEvents ---------------------------------------------------


def _orders_deliveries(order_times, delivery_times):
    return pd.concat(
        [
            _signal(order_times, None, uuid="order", column="value_string"),
            _signal(delivery_times, None, uuid="delivery", column="value_string"),
        ],
        ignore_index=True,
    )


def test_lead_times_extra_early_delivery_never_negative():
    # A delivery before the first order (e.g. of an order placed before the
    # data starts) shifted every positional pair: negative lead times.
    df = _orders_deliveries(
        ["2024-01-01 10:00", "2024-01-01 12:00"],
        ["2024-01-01 09:00", "2024-01-01 11:00", "2024-01-01 13:00"],
    )
    with pytest.warns(DataQualityWarning, match="dropped"):
        out = LeadTimeAnalysisEvents(df).lead_times("order", "delivery")
    assert (out["duration_seconds"] >= 0).all()
    assert (out["end"] >= out["start"]).all()
    assert out["lead_time_hours"].tolist() == [1.0, 1.0]


def test_lead_times_extra_order_is_dropped():
    df = _orders_deliveries(
        ["2024-01-01 10:00", "2024-01-01 12:00", "2024-01-01 14:00"],
        ["2024-01-01 11:00", "2024-01-01 13:00"],
    )
    with pytest.warns(DataQualityWarning):
        out = LeadTimeAnalysisEvents(df).lead_times("order", "delivery")
    assert out["lead_time_hours"].tolist() == [1.0, 1.0]


def test_lead_times_matching_streams_do_not_warn():
    df = _orders_deliveries(
        ["2024-01-01 10:00", "2024-01-01 12:00"],
        ["2024-01-01 11:00", "2024-01-01 13:30"],
    )
    ev = LeadTimeAnalysisEvents(df)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        out = ev.lead_times("order", "delivery")
    assert out["lead_time_hours"].tolist() == [1.0, 1.5]
