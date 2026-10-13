"""Regression test: PowerIntegrator windows follow the calendar across DST."""

import pandas as pd
import pytest

from ts_shape.transform.calculator.power_integration import PowerIntegrator

BERLIN = "Europe/Berlin"


@pytest.fixture()
def constant_60kw():
    t = pd.date_range(
        "2024-03-30", "2024-04-01", freq="10min", tz=BERLIN, inclusive="left"
    )
    return pd.DataFrame({"systime": t, "uuid": "p", "value_double": 60.0})


def test_daily_windows_on_spring_forward(constant_60kw):
    out = PowerIntegrator.energy_by_window(
        constant_60kw, freq="1D", data_end=pd.Timestamp("2024-04-01", tz=BERLIN)
    )
    assert list(out["duration_seconds"]) == [86400.0, 82800.0]
    assert list(out["end"]) == [
        pd.Timestamp("2024-03-31", tz=BERLIN),
        pd.Timestamp("2024-04-01", tz=BERLIN),
    ]
    assert out["kwh"].tolist() == pytest.approx([24 * 60.0, 23 * 60.0])
    assert (out["coverage"] == 1.0).all()


def test_hourly_windows_are_real_hours(constant_60kw):
    out = PowerIntegrator.energy_by_window(
        constant_60kw, freq="1h", data_end=pd.Timestamp("2024-04-01", tz=BERLIN)
    )
    assert len(out) == 47  # 24 + 23 real hours
    assert (out["duration_seconds"] == 3600.0).all()
    assert (out["kwh"] == 60.0).all()
