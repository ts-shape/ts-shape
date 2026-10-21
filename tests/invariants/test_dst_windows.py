"""Invariant: summary windows on DST days carry their real length.

Every summary-shape detector method of
:data:`ts_shape.eventlog.taxonomy.REGISTRY` is called on Europe/Berlin data
around a DST change -- 2024-03-30 .. 2024-04-01 (2024-03-31 lasts 23 h) and
2024-10-26 .. 2024-10-28 (2024-10-27 lasts 25 h) -- with a day window
(``window`` / ``freq`` / ``period`` = ``"1D"``) where the method takes one.
The harness builds the calls (:func:`harness.build_kwargs`). On every row:

* ``end - start == duration_seconds``;
* every time share in seconds -- each ``*_seconds`` column other than
  ``duration_seconds``, plus the legacy share columns without the suffix in
  :data:`SHARE_SECONDS` -- is at most ``duration_seconds`` (+ 1e-6);
* every percentage share (``*_pct`` / ``pct_*``) is at most 100 (+ 1e-9).

A fixed 24 h stand-in for the window length breaks the bounds on the 23 h
day (an always-true share reads 86400 s / 104 %). Columns that are not
shares of the window are listed with a reason in :data:`NOT_A_SHARE`; a new
seconds or percent column fails here until it is classified.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.eventlog.taxonomy import REGISTRY

from . import harness

DAY_WINDOW_PARAMS = ("window", "freq", "frequency", "period", "resample")

#: First local day of each frame (three local days, every 10 min).
DST_STARTS = {"spring": "2024-03-30", "fall": "2024-10-26"}

#: Share columns in seconds that have no ``_seconds`` suffix (legacy names).
SHARE_SECONDS = (
    "time_above",
    "total_duration",
    "on_time",
    "off_time",
    "total_lost_time",
    "total_run_time",
    "total_micro_stop_time",
)

#: Seconds / percent columns that are not a share of the window, with why.
NOT_A_SHARE: dict[str, str] = {
    # -- durations of single episodes, booked to the window they start in
    "avg_idle_duration_seconds": "mean idle episode; may run past the window",
    "avg_run_duration_seconds": "mean run episode; may run past the window",
    "avg_steady_duration_seconds": "mean steady episode; may run past the window",
    "avg_on_duration_seconds": "mean on episode; may run past the window",
    "avg_off_duration_seconds": "mean off episode; may run past the window",
    "avg_recovery_seconds": "mean recovery time after a disturbance",
    "longest_run_seconds": "one run episode, not clipped to the window",
    "mean_run_seconds": "mean run episode, not clipped to the window",
    "deviation_from_median_duration_seconds": "warm-up length vs. median",
    "time_to_target_seconds": "warm-up time to reach target",
    # -- cycle and flow times
    "bottleneck_cycle_time_seconds": "a cycle time",
    "cycle_time_mean_seconds": "a cycle time",
    "cycle_time_median_seconds": "a cycle time",
    "cycle_time_std_seconds": "spread of cycle times",
    "takt_seconds": "takt time from demand",
    "lead_time_mean_seconds": "FIFO lead time per unit",
    "littles_law_lead_time_seconds": "Little's Law lead time (WIP / rate)",
    # -- sampling statistics
    "max_gap_seconds": "a sampling interval",
    "max_interval_seconds": "a sampling interval",
    "mean_interval_seconds": "a sampling interval",
    "min_interval_seconds": "a sampling interval",
    "std_interval_seconds": "spread of sampling intervals",
    # -- control response times, measured from a setpoint change
    "period_seconds": "oscillation period",
    "rise_time_seconds": "response time after a setpoint step",
    "t_peak_seconds": "response time after a setpoint step",
    "t_settle_seconds": "response time after a setpoint step",
    "t_settle_derivative_seconds": "response time after a setpoint step",
    "t_undershoot_seconds": "response time after a setpoint step",
    # -- forecasts
    "estimated_seconds": "forecast time until a threshold is reached",
    # -- percentages that are ratios, not time shares
    "overshoot_pct": "peak excursion relative to the step size",
    "undershoot_pct": "excursion relative to the step size",
    "deviation_pct": "signed deviation from a baseline",
    "imbalance_pct": "imbalance relative to the input total",
    "jitter_pct": "sampling-interval spread relative to the mean",
    "performance_pct": "speed ratio; above 100 when faster than ideal",
    "oee_pct": "product with performance, which may exceed 100",
    "production_achievement_pct": "actual vs. target output",
}


def _is_pct(col: str) -> bool:
    return col.endswith("_pct") or col.startswith("pct_")


def _is_share_seconds(col: str) -> bool:
    if col == "duration_seconds":
        return False
    return col.endswith("_seconds") or col in SHARE_SECONDS


def dst_frame(first_day: str) -> pd.DataFrame:
    """The harness signals over three local Berlin days, every 10 minutes."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp(first_day, tz=harness.BERLIN)
    t = pd.date_range(
        start.tz_convert("UTC"),
        (start + pd.DateOffset(days=3)).tz_convert("UTC"),
        freq="10min",
        inclusive="left",
    ).tz_convert(harness.BERLIN)
    n = len(t)
    counter = np.cumsum(rng.integers(0, 3, n))
    state = (np.arange(n) // 6) % 3 != 0
    setpoint_t = t[::36]
    rows = [
        pd.DataFrame(
            {
                "systime": t,
                "uuid": harness.CONT,
                "value_double": 50 + rng.normal(0, 2, n),
            }
        ),
        pd.DataFrame(
            {
                "systime": t,
                "uuid": harness.CONT2,
                "value_double": 20 + rng.normal(0, 1, n),
            }
        ),
        pd.DataFrame({"systime": t, "uuid": harness.STATE, "value_bool": state}),
        pd.DataFrame({"systime": t, "uuid": harness.COUNTER, "value_integer": counter}),
        pd.DataFrame(
            {
                "systime": setpoint_t,
                "uuid": harness.SETPOINT,
                "value_double": np.resize([50.0, 55.0], len(setpoint_t)),
            }
        ),
        pd.DataFrame(
            {
                "systime": t,
                "uuid": harness.TEXT,
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


_FRAMES = {name: dst_frame(day) for name, day in DST_STARTS.items()}

#: Extra variants of methods whose default harness call returns no rows or
#: skips the fixed-length branch: ``(id, class, method, ctor kwargs,
#: method kwargs)``.
VARIANTS: list[tuple[str, str, str, dict[str, Any], dict[str, Any]]] = [
    (
        "DutyCycleEvents.duty_cycle_by_window[state,sample_mean]",
        "DutyCycleEvents",
        "duty_cycle_by_window",
        {"signal_uuid": harness.STATE},
        {"time_weighted": False},
    ),
    (
        "DutyCycleEvents.duty_cycle_by_window[state]",
        "DutyCycleEvents",
        "duty_cycle_by_window",
        {"signal_uuid": harness.STATE},
        {},
    ),
    (
        "MicroStopEvents.micro_stop_frequency[2h stops]",
        "MicroStopEvents",
        "micro_stop_frequency",
        {},
        {"max_duration": "2h"},
    ),
    (
        "FlowMetricsEvents.throughput[state]",
        "FlowMetricsEvents",
        "throughput",
        {"entry_uuid": harness.STATE, "exit_uuid": harness.STATE},
        {},
    ),
    (
        "FlowMetricsEvents.flow_summary[state]",
        "FlowMetricsEvents",
        "flow_summary",
        {"entry_uuid": harness.STATE, "exit_uuid": harness.STATE},
        {},
    ),
    (
        "ThresholdMonitoringEvents.time_above_threshold[last_sample]",
        "ThresholdMonitoringEvents",
        "time_above_threshold",
        {},
        {"interval_end": "last_sample"},
    ),
]


def _cases() -> list[Any]:
    """Every callable summary method (plus :data:`VARIANTS`) per DST frame."""
    probe = _FRAMES["spring"]
    base: list[tuple[str, str, str, dict[str, Any], dict[str, Any]]] = []
    for (cls_name, method), rule in sorted(REGISTRY.items()):
        if rule.shape != "summary":
            continue
        try:
            harness.build_kwargs(getattr(ts_shape, cls_name), method, probe)
        except harness.Unsupported:
            continue  # pinned in test_contract_summary.UNCALLABLE
        base.append((f"{cls_name}.{method}", cls_name, method, {}, {}))
    return [
        pytest.param(frame, *case[1:], id=f"{frame}:{case[0]}")
        for frame in _FRAMES
        for case in base + VARIANTS
    ]


def _call(
    cls_name: str,
    method: str,
    df: pd.DataFrame,
    ctor_extra: dict[str, Any],
    method_extra: dict[str, Any],
) -> pd.DataFrame:
    cls = getattr(ts_shape, cls_name)
    c_kwargs, m_kwargs, is_cm = harness.build_kwargs(cls, method, df)
    c_kwargs.update(ctor_extra)
    m_kwargs.update(method_extra)
    params = inspect.signature(getattr(cls, method)).parameters
    day = {p: "1D" for p in DAY_WINDOW_PARAMS if p in params}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if is_cm:
            return getattr(cls, method)(df.copy(), **m_kwargs, **day)
        return getattr(cls(df.copy(), **c_kwargs), method)(**m_kwargs, **day)


def test_not_a_share_entries_are_classified_columns():
    """Every allowlisted column is a seconds or percent column."""
    stray = [c for c in NOT_A_SHARE if not (_is_pct(c) or _is_share_seconds(c))]
    assert not stray


@pytest.mark.parametrize(
    ("frame", "cls_name", "method", "ctor_extra", "method_extra"), _cases()
)
def test_summary_window_shares_fit_the_real_window(
    frame, cls_name, method, ctor_extra, method_extra
):
    try:
        out = _call(cls_name, method, _FRAMES[frame], ctor_extra, method_extra)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing ({exc.name or exc})")
    assert isinstance(out, pd.DataFrame)
    if out.empty:
        return
    duration = pd.to_numeric(out["duration_seconds"], errors="coerce").astype(float)
    span = (
        pd.to_datetime(out["end"]) - pd.to_datetime(out["start"])
    ).dt.total_seconds()
    known = span.notna() & duration.notna()
    assert np.allclose(
        span[known], duration[known], rtol=0, atol=1e-6
    ), "end - start != duration_seconds"

    problems = []
    for col in map(str, out.columns):
        if col in NOT_A_SHARE:
            continue
        s = out[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        values = pd.to_numeric(s, errors="coerce").astype(float)
        if _is_pct(col):
            bad = values > 100 + 1e-9
        elif _is_share_seconds(col):
            bad = values > duration + 1e-6
        else:
            continue
        if bool(bad.any()):
            i = int(np.flatnonzero(bad.to_numpy())[0])
            problems.append(
                f"{col}={values.iloc[i]!r} > bound on window starting "
                f"{out['start'].iloc[i]} ({duration.iloc[i]} s)"
            )
    assert not problems, problems
