"""Scale smoke test: production / energy methods that used to time out.

Shift assignment used to run row by row (~2 ms/row), so the ``*_by_shift``
reports took minutes on a few hundred thousand rows. They are vectorized now;
this runs them on ~50k rows with a generous total bound (not a micro-timing).
Also checks the hourly / windowed methods that crashed on a DST fall-back day.
"""

import time
import warnings

import numpy as np
import pandas as pd
import pytest
from invariants import harness as H

import ts_shape


def _plant(n: int, start: str, tz: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    t = pd.date_range(start, periods=n, freq="1min")
    if tz is not None:
        t = t.tz_localize("UTC").tz_convert(tz)
    counter = np.cumsum(rng.integers(0, 3, n))
    cut = int(n * 200 / 360)
    counter[cut:] -= counter[cut - 1]
    state = (np.arange(n) // 30) % 3 != 0
    k = max(n // 60, 1)
    rows = [
        pd.DataFrame(
            {"systime": t, "uuid": H.CONT, "value_double": 50 + rng.normal(0, 2, n)}
        ),
        pd.DataFrame(
            {"systime": t, "uuid": H.CONT2, "value_double": 20 + rng.normal(0, 1, n)}
        ),
        pd.DataFrame({"systime": t, "uuid": H.STATE, "value_bool": state}),
        pd.DataFrame({"systime": t, "uuid": H.COUNTER, "value_integer": counter}),
        pd.DataFrame(
            {
                "systime": t[::60],
                "uuid": H.SETPOINT,
                "value_double": np.tile([50.0, 55.0], k)[: len(t[::60])],
            }
        ),
        pd.DataFrame(
            {
                "systime": t,
                "uuid": H.TEXT,
                "value_string": np.where(state, "Running", "Stopped"),
            }
        ),
    ]
    df = pd.concat(rows, ignore_index=True)
    for col in ("value_double", "value_integer", "value_bool", "value_string"):
        if col not in df.columns:
            df[col] = np.nan
    df["is_delta"] = True
    return df.sort_values(["systime", "uuid"]).reset_index(drop=True)


FORMERLY_SLOW = [
    ("QualityEvents", "nok_by_shift"),
    ("QualityEvents", "daily_quality_summary"),
    ("ReworkEvents", "rework_by_shift"),
    ("ReworkEvents", "rework_rate"),
    ("ScrapEvents", "scrap_by_shift"),
    ("ShiftHandoverEvents", "generate_report"),
    ("ShiftReportEvents", "best_and_worst_shifts"),
    ("ShiftReportEvents", "shift_comparison"),
    ("ShiftReportEvents", "shift_production"),
    ("DowntimeEvents", "downtime_by_shift"),
    ("IdleEnergyDetectionEvents", "idle_energy_by_shift"),
    ("OperatorPerformanceEvents", "operator_comparison"),
    ("OperatorPerformanceEvents", "production_by_operator"),
    ("EnergyEfficiencyEvents", "efficiency_comparison"),
    ("RoutingTraceabilityEvents", "build_routing_timeline"),
    ("OperatingStateEvents", "infer_states"),
    ("OperatingStateEvents", "unusual_state_sequence"),
]


def test_formerly_slow_methods_finish_on_50k_rows():
    df = _plant(n=10000, start="2024-03-29 03:00", tz="Europe/Berlin")  # ~50k rows
    assert len(df) > 49_000
    t0 = time.perf_counter()
    for cls_name, method in FORMERLY_SLOW:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            H.build_call(getattr(ts_shape, cls_name), method, df)()
    assert time.perf_counter() - t0 < 30.0


DST_FLOOR = [
    ("BottleneckDetectionEvents", "detect_bottleneck"),
    ("BottleneckDetectionEvents", "shifting_bottleneck"),
    ("BottleneckDetectionEvents", "station_utilization"),
    ("CycleTimeEvents", "hourly_cycle_time_summary"),
    ("DutyCycleEvents", "duty_cycle_by_window"),
    ("LineBalancingEvents", "balance_metrics"),
    ("LineBalancingEvents", "station_cycle_times"),
    ("RuntimeAccountingEvents", "operating_hours_meter"),
]


@pytest.mark.parametrize("cls_name,method", DST_FLOOR)
def test_windowed_methods_survive_dst_fall_back(cls_name, method):
    # Regression: ``.dt.floor`` raised "Cannot infer dst time" on the repeated
    # 02:00 hour of 2024-10-27 in Europe/Berlin.
    df = _plant(n=1800, start="2024-10-26 20:00", tz="Europe/Berlin")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = H.build_call(getattr(ts_shape, cls_name), method, df)()
    assert isinstance(out, pd.DataFrame)


def test_hourly_windows_distinct_across_fall_back():
    df = _plant(n=1800, start="2024-10-26 20:00", tz="Europe/Berlin")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = H.build_call(ts_shape.DutyCycleEvents, "duty_cycle_by_window", df)()
    starts = out["start"]
    # Both 02:00 hours (CEST and CET) are separate one-hour windows.
    assert starts.is_unique
    assert (out["end"] - out["start"] == pd.Timedelta("1h")).all()
    assert (starts.dt.strftime("%Y-%m-%d %H:%M") == "2024-10-27 02:00").sum() == 2
