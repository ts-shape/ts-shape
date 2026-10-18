"""Setpoint KPIs from the shared window gather == the former per-change code.

``tests/data/setpoint_kpis_reference.csv`` holds the output of
``control_quality_metrics`` (and of each KPI method) computed by the
implementation before the shared gather (commit 40210c2), which sliced a
pandas window and re-derived the change table in every KPI method. Floats
are stored with 17 significant digits, so they round-trip exactly:
the comparison is exact.

The ``control_quality_metrics`` calls pin ``deadband=0``: the former code
counted every sign change of the error, and the settle-band default that
replaced it is covered by ``tests/reference/test_kpi_references_misc.py``.

Regenerate (only when a KPI definition changes on purpose) with
``python tests/test_setpoint_shared_gather.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ts_shape.events.engineering.setpoint_events import SetpointChangeEvents

REFERENCE = Path(__file__).parent / "data" / "setpoint_kpis_reference.csv"


def _frame(tz: str | None) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_changes, per = 24, 300
    t = pd.date_range("2024-01-01", periods=n_changes * per, freq="1s", tz=tz)
    levels = np.repeat(rng.integers(0, 8, n_changes) * 5.0, per)
    sp_times = t[::per]
    sp_values = levels[::per].copy()
    # Two changes at one instant, and a change between two samples.
    sp_times = sp_times.append(
        pd.DatetimeIndex([sp_times[5], t[3001] + pd.Timedelta("0.5s")])
    )
    sp_values = np.r_[sp_values, sp_values[5] + 2.5, 17.0]
    sp = pd.DataFrame({"systime": sp_times, "uuid": "sp", "value_double": sp_values})
    # An underdamped second-order response with noise and missing samples.
    y, v = levels[0], 0.0
    resp = np.empty(len(t))
    for i in range(len(t)):
        v += 0.08 * (levels[i] - y) - 0.15 * v
        y += v
        resp[i] = y
    resp += rng.normal(0, 0.05, len(t))
    resp[rng.random(len(t)) < 0.02] = np.nan
    resp[1200:1260] = levels[1200]  # sits exactly on the setpoint
    act = pd.DataFrame({"systime": t, "uuid": "act", "value_double": resp})
    return pd.concat([sp, act], ignore_index=True).assign(is_delta=True)


def _outputs() -> pd.DataFrame:
    parts = []
    for tz in (None, "Europe/Berlin"):
        df = _frame(tz)
        calls = {
            "cqm_pct": lambda e: e.control_quality_metrics(
                "act", settle_pct=0.05, hold="10s", lookahead="5min", deadband=0
            ),
            "cqm_tol": lambda e: e.control_quality_metrics(
                "act",
                tol=0.5,
                hold="20s",
                lookahead="12min",
                rate_threshold=0.05,
                deadband=0,
            ),
            "settle": lambda e: e.time_to_settle("act", tol=1.0, lookahead="2min"),
            "overshoot": lambda e: e.overshoot_metrics("act", window="3min"),
            "derivative": lambda e: e.time_to_settle_derivative(
                "act", rate_threshold=0.2, hold="5s", lookahead="4min"
            ),
            "rise": lambda e: e.rise_time("act", start_pct=0.2, end_pct=0.8),
            "decay": lambda e: e.decay_rate("act", lookahead="2min", min_points=8),
            "frequency": lambda e: e.oscillation_frequency(
                "act", window="4min", min_oscillations=1
            ),
        }
        for name, call in calls.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = call(SetpointChangeEvents(df, "sp"))
            out = out.drop(columns=["uuid", "source_uuid"])
            for col in ("start", "end"):
                out[col] = out[col].astype(str)
            long = out.reset_index(drop=True).melt(
                id_vars=["start"],
                var_name="column",
                value_name="value",
                ignore_index=False,
            )
            long.insert(0, "row", long.index)
            long.insert(0, "call", f"{name}|{tz}")
            long["value"] = [
                repr(float(v)) if isinstance(v, float | np.floating) else str(v)
                for v in long["value"]
            ]
            parts.append(long)
    return pd.concat(parts, ignore_index=True)


def test_kpis_match_the_former_per_change_implementation():
    expected = pd.read_csv(REFERENCE, dtype=str, keep_default_na=False)
    got = _outputs().astype(str).reset_index(drop=True)
    # The reference was written with pandas 3 / numpy 2.5 on one machine.
    # Other numpy versions and CPUs (SIMD paths) round some sums differently
    # in the last bit, and the former implementation differs the same way
    # there. Numbers are compared to 1e-13 relative plus 1e-12 absolute (a
    # value near zero that is a difference of terms around 300 carries an
    # absolute rounding of ~1e-16 * 300), everything else exactly.
    pd.testing.assert_frame_equal(
        got.drop(columns="value"), expected.drop(columns="value")
    )
    got_num = pd.to_numeric(got["value"], errors="coerce")
    exp_num = pd.to_numeric(expected["value"], errors="coerce")
    text = exp_num.isna()
    assert (got_num.isna() == text).all()
    assert (got["value"][text] == expected["value"][text]).all()
    np.testing.assert_allclose(got_num[~text], exp_num[~text], rtol=1e-13, atol=1e-12)


if __name__ == "__main__":  # pragma: no cover - regenerates the reference
    REFERENCE.parent.mkdir(exist_ok=True)
    _outputs().to_csv(REFERENCE, index=False)
