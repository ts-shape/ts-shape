"""Shared fixtures for the cross-facility reporting tests."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import Facility, FacilityRegistry


@pytest.fixture()
def registry() -> FacilityRegistry:
    """Three sites, three timezones, three currencies, two regions."""
    return FacilityRegistry(
        [
            Facility(
                "DE-STR",
                name="Stuttgart",
                timezone="Europe/Berlin",
                region="EMEA",
                country="DE",
                currency="EUR",
                capacity_per_day=1000.0,
            ),
            Facility(
                "US-DTW",
                name="Detroit",
                timezone="America/Detroit",
                region="AMER",
                country="US",
                currency="USD",
                capacity_per_day=800.0,
            ),
            Facility(
                "CN-SHA",
                name="Shanghai",
                timezone="Asia/Shanghai",
                region="APAC",
                country="CN",
                currency="CNY",
                capacity_per_day=2000.0,
            ),
        ]
    )


@pytest.fixture()
def site_kpis() -> pd.DataFrame:
    """Two periods of KPIs for three sites, with differing exposure."""
    return pd.DataFrame(
        {
            "facility": [
                "DE-STR",
                "DE-STR",
                "US-DTW",
                "US-DTW",
                "CN-SHA",
                "CN-SHA",
            ],
            "period": pd.to_datetime(["2026-01-01", "2026-02-01"] * 3),
            "parts_produced": [1000.0, 1100.0, 500.0, 450.0, 2000.0, 2400.0],
            "good": [980.0, 1078.0, 475.0, 423.0, 1900.0, 2304.0],
            "total": [1000.0, 1100.0, 500.0, 450.0, 2000.0, 2400.0],
            "oee_pct": [70.0, 72.0, 60.0, 58.0, 85.0, 86.0],
            "scrap_rate_pct": [2.0, 2.0, 5.0, 6.0, 5.0, 4.0],
            "planned_minutes": [10000.0, 10000.0, 5000.0, 5000.0, 20000.0, 30000.0],
            "scrap_cost": [1000.0, 1200.0, 2000.0, 2400.0, 9000.0, 8000.0],
        }
    )
