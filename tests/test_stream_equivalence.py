"""Chunked runs equal in-memory runs, for every method in ``stream.SUPPORT``.

For each registry method the invariant harness builds a call
(:func:`invariants.harness.build_kwargs`) on a 3-day frame that crosses the
Europe/Berlin spring-forward change (2024-03-31, a 23-hour day), with runs,
stops and excursions across midnight, a counter reset exactly at a slice
border and a change-encoded setpoint. The method runs once in memory on the
whole frame and once through the chunked engine on an iterator of daily
frames; both results must be equal (rows in a canonical order, float
tolerance 1e-9, NaN equal to NaN).

``SUPPORT`` is data-driven: it holds the non-static methods that pass here
and do not depend on whole-signal statistics. Regenerate the report with
(``src`` first on the path)::

    PYTHONPATH=src python tests/test_stream_equivalence.py
"""

from __future__ import annotations

import multiprocessing
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from invariants import harness as H  # noqa: E402

import ts_shape  # noqa: E402
from ts_shape import NotChunkableError, run_chunked  # noqa: E402
from ts_shape.eventlog.taxonomy import REGISTRY  # noqa: E402
from ts_shape.pipeline import _param_names  # noqa: E402
from ts_shape.stream import SUPPORT, _run_chunked  # noqa: E402

BERLIN = "Europe/Berlin"


#: Values for required parameters the invariant harness has no rule for.
EXTRA: dict[str, Any] = {
    "tolerance": 2.0,
    "baseline_rms": 1.0,
    "target_per_shift": 100,
    "daily_target": 1000,
    "part_number": "Running",
    "series_id": H.CONT,
    "production_df": lambda df: df,
}


#: Per-method arguments so that the method finds something in the frame
#: (routed to the constructor or the method by parameter name).
CALLS: dict[str, dict[str, Any]] = {
    "AlarmManagementEvents.chattering_detection": {
        "min_transitions": 4,
        "window": "15min",
    },
    "AnomalyClassificationEvents.detect_drift": {"min_slope": 0.0003},
    "AnomalyClassificationEvents.detect_oscillation": {"window": "30min"},
    "ControlLoopHealthEvents.output_saturation": {"upper_limit": 55.0},
    "DataGapAnalysisEvents.detect_gaps": {"min_gap": "5min"},
    "DataGapAnalysisEvents.interpolation_candidates": {
        "max_gap": "30min",
        "min_gap": "2min",
    },
    "DisturbanceRecoveryEvents.detect_disturbances": {"n_sigma": 2.0},
    "DisturbanceRecoveryEvents.before_after_comparison": {"n_sigma": 2.0},
    "DisturbanceRecoveryEvents.recovery_time": {"n_sigma": 2.0},
    "DutyCycleEvents.on_off_intervals": {"signal_uuid": H.STATE},
    "DutyCycleEvents.cycle_count": {"signal_uuid": H.STATE},
    "DutyCycleEvents.duty_cycle_by_window": {"signal_uuid": H.STATE},
    "DutyCycleEvents.excessive_cycling": {
        "signal_uuid": H.STATE,
        "max_transitions": 3,
    },
    "EnvironmentalCompensationEvents.detect_env_exceedance": {
        "valid_range": (19.0, 23.0)
    },
    "InventoryMonitoringEvents.detect_low_stock": {"min_level": 48.0},
    "InventoryMonitoringEvents.reorder_point_breach": {"reorder_level": 48.0},
    "LongDowntimeEvents.detect_long_downtime": {"min_gap": "1h"},
    "LongDowntimeEvents.count_events_between_gaps": {"min_gap": "1h"},
    "MachineStateEvents.detect_rapid_transitions": {
        "threshold": "3min",
        "min_count": 2,
    },
    "MicroStopEvents.detect_micro_stops": {"max_duration": "5min"},
    "MicroStopEvents.micro_stop_frequency": {"max_duration": "5min"},
    "MicroStopEvents.micro_stop_impact": {"max_duration": "5min"},
    "PerformanceLossEvents.performance_by_shift": {"cycle_uuid": H.COUNTER},
    "PerformanceLossEvents.performance_trend": {"cycle_uuid": H.COUNTER},
    "PerformanceLossEvents.slow_periods": {"cycle_uuid": H.COUNTER},
    "RateOfChangeEvents.detect_rapid_change": {"threshold": 0.05},
    "RateOfChangeEvents.detect_step_changes": {
        "min_delta": 5.0,
        "max_duration": "2min",
    },
    "SignalQualityEvents.detect_out_of_range": {
        "min_value": 46.0,
        "max_value": 56.0,
    },
    "ThresholdMonitoringEvents.threshold_with_hysteresis": {
        "upper_limit": 56.0,
        "lower_limit": 53.0,
    },
    "ThresholdMonitoringEvents.time_above_threshold": {"threshold": 53.0},
    # Multi-hour windows cannot run chunked on DST data (see
    # stream._check_dst_window): checked with hourly windows.
    "CapabilityTrendingEvents.capability_over_time": {"window": "1h"},
    "CapabilityTrendingEvents.yield_estimate": {"window": "1h"},
    "EnvironmentalCompensationEvents.detect_correlation": {"window": "1h"},
    "EnvironmentalCompensationEvents.sensitivity": {"window": "1h"},
    "MeasurementUncertaintyEvents.combined_uncertainty": {"window": "1h"},
    "MeasurementUncertaintyEvents.type_a_uncertainty": {"window": "1h"},
    "SensorDriftEvents.calibration_health": {"window": "1h"},
    "SetupTimeEvents.setup_durations": {
        "run_state_uuid": H.TEXT,
        "setup_value": "Stopped",
    },
}


#: Methods that pass here but use whole-signal context (read in their code),
#: so they are not in ``SUPPORT``. Kept in sync with ``stream._REASONS``.
AUDIT_EXCLUDED = (
    "AnomalyClassificationEvents.detect_flatline",
    "ChangeoverEvents.changeover_window",
    "DataGapAnalysisEvents.coverage_by_period",
    "DisturbanceRecoveryEvents.before_after_comparison",
    "DisturbanceRecoveryEvents.detect_disturbances",
    "DisturbanceRecoveryEvents.recovery_time",
    "LongDowntimeEvents.count_events_between_gaps",
    "PeriodSummaryEvents.weekly_summary",
)


def chunk_frame() -> pd.DataFrame:
    """Seven signals over three local days at one-minute resolution."""
    rng = np.random.default_rng(11)
    t = pd.date_range(
        "2024-03-30", "2024-04-02", freq="1min", inclusive="left", tz=BERLIN
    )
    n = len(t)
    minute = np.arange(n)
    # Irregular run / idle pattern; a long run across the first midnight and a
    # stop across the second one.
    state = (minute // 47) % 4 != 0
    local = pd.Series(t)
    hour = local.dt.hour.to_numpy() + local.dt.minute.to_numpy() / 60
    wall = local.dt.tz_localize(None).dt.normalize()
    day = (wall - wall.iloc[0]).dt.days.to_numpy()
    state[(day == 0) & (hour >= 20)] = True
    state[(day == 1) & (hour < 3)] = True
    state[(day == 1) & (hour >= 23.5)] = False
    state[(day == 2) & (hour < 1.2)] = False
    state[(day == 2) & (hour >= 15) & (hour < 17)] = False
    # Two-minute micro stops and a chattering burst.
    state[(minute % 113 < 2) & (hour > 4) & (hour < 19)] = False
    burst = (day == 2) & (hour >= 9) & (hour < 9.2)
    state[burst] = minute[burst] % 2 == 0
    cont = 50 + 3 * np.sin(minute / 90) + rng.normal(0, 1.5, n)
    # Excursions above 55 / 60 across midnight.
    cont[(day == 0) & (hour >= 23)] += 9
    cont[(day == 1) & (hour < 1.5)] += 9
    cont[(day == 2) & (hour >= 11) & (hour < 12)] += 12
    # A flat stretch across the second midnight.
    flat = ((day == 1) & (hour >= 23 + 2 / 3)) | ((day == 2) & (hour < 1 / 3))
    cont[flat] = 52.0
    counter = np.cumsum(rng.integers(0, 3, n) * state)
    border = int(np.flatnonzero(day == 1)[0])  # first reading of day 2
    counter[border:] -= counter[border] - 1
    mid = int(np.flatnonzero((day == 2) & (hour >= 13))[0])
    counter[mid:] -= counter[mid] - 2
    rows = [
        pd.DataFrame({"systime": t, "uuid": H.CONT, "value_double": cont}),
        pd.DataFrame(
            {
                "systime": t,
                "uuid": H.CONT2,
                "value_double": 20 + 0.02 * cont + rng.normal(0, 1, n),
            }
        ),
        pd.DataFrame({"systime": t, "uuid": H.STATE, "value_bool": state}),
        pd.DataFrame({"systime": t, "uuid": H.COUNTER, "value_integer": counter}),
        pd.DataFrame(
            {
                "systime": t[::60],
                "uuid": H.SETPOINT,
                "value_double": np.resize([50.0, 55.0], len(t[::60])),
            }
        ),
        pd.DataFrame(
            {
                "systime": t,
                "uuid": H.TEXT,
                "value_string": np.where(state, "Running", "Stopped"),
            }
        ),
        pd.DataFrame(
            {"systime": t, "uuid": H.NOISE, "value_double": rng.normal(500, 50, n)}
        ),
    ]
    df = pd.concat(rows, ignore_index=True)
    # Logging gaps of the continuous signal, one across the first midnight.
    wall_time = df["systime"].dt.tz_localize(None)
    gap = (df["uuid"] == H.CONT) & (
        wall_time.between(
            pd.Timestamp("2024-03-30 23:50"), pd.Timestamp("2024-03-31 00:10")
        )
        | wall_time.between(
            pd.Timestamp("2024-04-01 14:00"), pd.Timestamp("2024-04-01 14:30")
        )
    )
    df = df[~gap.to_numpy()]
    for col in ("value_double", "value_integer", "value_bool", "value_string"):
        if col not in df.columns:
            df[col] = np.nan
    df["is_delta"] = True
    return df.sort_values(["systime", "uuid"], kind="stable").reset_index(drop=True)


def daily(df: pd.DataFrame) -> list[pd.DataFrame]:
    """The frame cut into local calendar days."""
    days = df["systime"].dt.normalize()
    return [g.reset_index(drop=True) for _, g in df.groupby(days, sort=True)]


def skewed(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with a first day of very different statistics.

    A method that uses whole-signal statistics (mean, sigma, quantiles of all
    data) gives different results on later days when the first day changes;
    a chunked run does not see the first day there, so the comparison fails.
    """
    out = df.copy()
    first = (out["systime"] < out["systime"].min() + pd.Timedelta(days=1)).to_numpy()
    for uuid, scale, shift in ((H.CONT, 2.0, -48.0), (H.CONT2, 3.0, -35.0)):
        hit = first & (out["uuid"] == uuid).to_numpy()
        out.loc[hit, "value_double"] = out.loc[hit, "value_double"] * scale + shift
    hit = first & (out["uuid"] == H.SETPOINT).to_numpy()
    out.loc[hit, "value_double"] = out.loc[hit, "value_double"] + 20.0
    hit = first & (out["uuid"] == H.COUNTER).to_numpy()
    out.loc[hit, "value_integer"] = out.loc[hit, "value_integer"] * 4
    return out


def shifted(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` 9 h 13 min later: every event meets the slice borders elsewhere."""
    return df.assign(systime=df["systime"] + pd.Timedelta(hours=9, minutes=13))


_FRAMES: dict[str, pd.DataFrame] = {}


def _frame(name: str = "base") -> pd.DataFrame:
    if name not in _FRAMES:
        base = _FRAMES.get("base")
        if base is None:
            base = _FRAMES["base"] = chunk_frame()
        if name != "base":
            _FRAMES[name] = {"skewed": skewed, "shifted": shifted}[name](base)
    return _FRAMES[name]


def canonical(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows in a canonical order (time, uuid, then every other column)."""
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return frame
    return H._canonical(frame)


def compare(memory: pd.DataFrame, chunked: pd.DataFrame) -> str:
    if list(memory.columns) != list(chunked.columns):
        return f"fail: columns {list(memory.columns)} != {list(chunked.columns)}"
    if len(memory) != len(chunked):
        return f"fail: {len(chunked)} rows chunked, {len(memory)} in memory"
    try:
        pd.testing.assert_frame_equal(
            canonical(memory),
            canonical(chunked),
            check_dtype=False,
            check_index_type=False,
            check_column_type=False,
            rtol=1e-9,
            atol=1e-9,
        )
    except AssertionError as exc:
        return f"fail: {str(exc).splitlines()[0][:100]}"
    return "ok"


def equivalence(cls_name: str, method: str) -> str:
    """``"ok"``, ``"fail: ..."`` or ``"n/a: ..."`` for one registry method.

    Checked on the base frame and on its :func:`skewed` and :func:`shifted`
    variants.
    """
    status = "ok"
    for name in ("base", "skewed", "shifted"):
        status = _equivalence(cls_name, method, _frame(name))
        if status != "ok":
            return status if name == "base" else f"{status} ({name})"
    return status


def _equivalence(cls_name: str, method: str, df: pd.DataFrame) -> str:
    cls = getattr(ts_shape, cls_name)
    try:
        c_kwargs, m_kwargs, is_cm = H.build_kwargs(cls, method, df, EXTRA)
    except H.Unsupported as exc:
        return f"n/a: {exc}"
    init_params = _param_names(cls)
    for key, value in CALLS.get(f"{cls_name}.{method}", {}).items():
        if key in init_params and not is_cm:
            c_kwargs[key] = value
        else:
            m_kwargs[key] = value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if is_cm:
                memory = getattr(cls, method)(df, **m_kwargs)
            else:
                memory = getattr(cls(df, **c_kwargs), method)(**m_kwargs)
        except Exception as exc:  # noqa: BLE001 - the harness records every failure
            return f"n/a: in memory {H._short(exc)}"
        if not isinstance(memory, pd.DataFrame):
            return f"n/a: returns {type(memory).__name__}"
        try:
            chunked = _run_chunked(
                iter(daily(df)),
                cls,
                method,
                chunk="1D",
                overlap=None,
                time_column="systime",
                # A method that only passes by carrying all data is not
                # chunked; the frame's longest open event is a few hours.
                max_carry="1D",
                kwargs={**c_kwargs, **m_kwargs},
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            return f"fail: chunked {H._short(exc)}"
    status = compare(memory, chunked)
    if status == "ok" and memory.empty:
        return "n/a: empty result (nothing to compare)"
    return status


def _pair(pair: tuple[str, str]) -> str:
    return equivalence(*pair)


def statuses(pairs: list[tuple[str, str]]) -> dict[str, str]:
    """:func:`equivalence` for every pair (in worker processes when many)."""
    jobs = H._jobs()
    if jobs <= 1 or len(pairs) < 20:
        return {f"{c}.{m}": equivalence(c, m) for c, m in pairs}
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        out = list(pool.map(_pair, pairs, chunksize=2))
    return {f"{c}.{m}": s for (c, m), s in zip(pairs, out, strict=True)}


_SUPPORTED = sorted(tuple(name.split(".", 1)) for name in SUPPORT)


@pytest.fixture(scope="module")
def support_statuses() -> dict[str, str]:
    return statuses([(c, m) for c, m in _SUPPORTED])


def test_support_is_registered_and_not_static():
    for name in SUPPORT:
        cls_name, method = name.split(".", 1)
        rule = REGISTRY.get((cls_name, method))
        assert rule is not None, name
        assert rule.shape != "static", name


def test_audited_exclusions_are_not_supported():
    from ts_shape.stream import _REASONS

    assert not set(AUDIT_EXCLUDED) & SUPPORT
    assert set(AUDIT_EXCLUDED) <= set(_REASONS)
    assert not set(_REASONS) & SUPPORT
    for name in _REASONS:
        assert tuple(name.split(".", 1)) in REGISTRY, name


@pytest.mark.parametrize("name", [f"{c}.{m}" for c, m in _SUPPORTED])
def test_supported_method_equals_in_memory(name, support_statuses):
    assert support_statuses[name] == "ok", support_statuses[name]


@pytest.mark.parametrize(
    "pair", sorted(k for k in REGISTRY if f"{k[0]}.{k[1]}" not in SUPPORT)
)
def test_unsupported_method_raises(pair):
    cls_name, method = pair
    cls = getattr(ts_shape, cls_name)
    with pytest.raises(NotChunkableError, match="cannot run chunked"):
        run_chunked([], cls, method)


def test_frame_crosses_dst_and_borders():
    df = _frame()
    days = daily(df)
    assert len(days) == 3
    span = days[1]["systime"].max() - days[1]["systime"].min()
    assert span == pd.Timedelta(hours=22, minutes=59)  # the 23-hour day


if __name__ == "__main__":  # pragma: no cover - maintenance entry point
    pairs = sorted(REGISTRY)
    result = statuses(pairs)
    passing = []
    for (c, m), status in zip(pairs, result.values(), strict=True):
        shape = REGISTRY[(c, m)].shape
        print(f"{shape:9s} {c}.{m}: {status}", flush=True)
        if status == "ok" and shape != "static":
            passing.append(f"{c}.{m}")
    passing = [name for name in passing if name not in AUDIT_EXCLUDED]
    print(f"\n{len(passing)} non-static methods pass (audit exclusions removed):")
    for name in passing:
        print(f'    "{name}",')
