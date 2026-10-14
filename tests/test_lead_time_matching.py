"""Regression tests: lead times match by order id and keep the timezone."""

import pandas as pd
import pytest

from ts_shape.events.supplychain.lead_time_analysis import LeadTimeAnalysisEvents

TZ = "Europe/Berlin"


def _df(delivery_ids=("PO1", "PO2")):
    d = pd.Timestamp("2024-01-01", tz=TZ)
    rows = [
        (d + pd.Timedelta(hours=8), "order", "PO2"),
        (d + pd.Timedelta(hours=9), "order", "PO1"),
        (d + pd.Timedelta(days=1, hours=8), "delivery", delivery_ids[0]),
        (d + pd.Timedelta(days=2, hours=7), "delivery", delivery_ids[1]),
    ]
    return pd.DataFrame(rows, columns=["systime", "uuid", "value_string"])


def test_matches_by_order_id():
    out = LeadTimeAnalysisEvents(_df()).lead_times("order", "delivery")
    hours = dict(zip(out["order_id"], out["lead_time_hours"]))
    assert hours["PO2"] == pytest.approx(47.0)
    assert hours["PO1"] == pytest.approx(23.0)


def test_keeps_timezone():
    out = LeadTimeAnalysisEvents(_df()).lead_times("order", "delivery")
    assert str(out["order_time"].dt.tz) == TZ
    first = out.sort_values("order_time")["order_time"].iloc[0]
    assert first.hour == 8


def test_without_ids_falls_back_to_sequential_pairing():
    out = LeadTimeAnalysisEvents(_df(delivery_ids=(None, None))).lead_times(
        "order", "delivery"
    )
    assert list(out["lead_time_hours"]) == pytest.approx([24.0, 46.0])
