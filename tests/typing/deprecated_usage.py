"""Uses deprecated names; tests/typing/test_mypy.py expects mypy to flag each.

Only type-checked, never run (the names would warn at runtime).
"""

from __future__ import annotations

import pandas as pd

from ts_shape import OEECalculator, OEEEvents, ScrapTracking
from ts_shape.loader.timeseries.timescale_loader import TimescaleDBDataAccess


def use(df: pd.DataFrame, loader: TimescaleDBDataAccess) -> None:
    OEECalculator(df)  # deprecated class
    ScrapTracking(df)  # deprecated class
    OEEEvents(df).calculate_oee("run", "count", 1.0)  # deprecated method
    print(loader.start_timestamp)  # deprecated attribute
