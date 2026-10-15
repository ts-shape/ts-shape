"""profile(): the longest-constant-run span == the former per-run lambda."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ts_shape.profiling import _longest_constant_seconds


def _reference(values: pd.Series, times: pd.Series) -> float:
    """Frozen copy of the former implementation (one Python call per run)."""
    if len(values) < 2:
        return 0.0
    run = (values != values.shift()).cumsum()
    frame = pd.DataFrame({"run": run.to_numpy(), "t": times.to_numpy()})
    spans = frame.groupby("run")["t"].agg(lambda t: t.iloc[-1] - t.iloc[0])
    return float(spans.max().total_seconds()) if len(spans) else 0.0


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("tz", [None, "UTC", "Europe/Berlin"])
@pytest.mark.parametrize("kind", ["float", "object", "bool"])
def test_longest_constant_matches_reference(unit: str, tz: str | None, kind: str):
    rng = np.random.default_rng(len(unit) + len(kind))
    for n, step in ((0, 1), (1, 1), (300, 7), (300, 86400 * 30)):
        seconds = np.cumsum(rng.integers(0, 3, n)) * step
        t = pd.Series(pd.to_datetime(seconds, unit="s")).dt.as_unit(unit)
        if tz:
            t = t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
            t = t.dropna()
        v = pd.Series(np.round(rng.normal(0, 1, len(t))))
        if kind == "object":
            v = v.astype(object).where(rng.random(len(t)) < 0.5, "a")
        elif kind == "bool":
            v = v > 0
        # Non-default, shuffled labels shared by values and times.
        v.index = t.index = rng.permutation(len(t)) + 5
        got = _longest_constant_seconds(v, t)
        assert got == _reference(v, t)
        assert type(got) is float
