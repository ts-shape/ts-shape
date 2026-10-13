"""Property-based tests for the signal primitives (hypothesis).

These pin the invariants that every detector built on the primitives
inherits: no time is lost or double-counted, no production is lost at
counter resets or window boundaries, and results do not depend on the
timezone the timestamps are expressed in.
"""

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from ts_shape.signals import SignalSpec  # noqa: E402
from ts_shape.signals.primitives import increments, intervals, per_window  # noqa: E402

T0 = pd.Timestamp("2024-03-31 00:00", tz="UTC")
_SETTINGS = settings(max_examples=60, deadline=None)

gaps_seconds = st.lists(
    st.integers(min_value=1, max_value=7200), min_size=1, max_size=60
)


def _times(gaps):
    return T0 + pd.to_timedelta(np.concatenate([[0], np.cumsum(gaps)]), unit="s")


@_SETTINGS
@given(gaps=gaps_seconds, data=st.data())
def test_intervals_partition_the_time_span(gaps, data):
    times = _times(gaps)
    states = data.draw(
        st.lists(st.booleans(), min_size=len(times), max_size=len(times))
    )
    df = pd.DataFrame({"systime": times, "uuid": "s", "value_bool": states})
    spec = SignalSpec(
        "s", kind="state", encoding="change_encoded", value_column="value_bool"
    )
    out = intervals(df, spec)
    assert out["duration_seconds"].sum() == pytest.approx(
        (times[-1] - times[0]).total_seconds()
    )
    assert (out["end"].iloc[:-1].to_numpy() == out["start"].iloc[1:].to_numpy()).all()
    assert (
        (out["state"] != out["state"].shift()).iloc[1:].all()
    )  # adjacent intervals differ


@_SETTINGS
@given(gaps=gaps_seconds, data=st.data())
def test_intervals_are_timezone_invariant(gaps, data):
    times = _times(gaps)
    states = data.draw(
        st.lists(st.booleans(), min_size=len(times), max_size=len(times))
    )
    spec = SignalSpec("s", kind="state", value_column="value_bool", max_gap="1h")
    utc = pd.DataFrame({"systime": times, "uuid": "s", "value_bool": states})
    berlin = utc.assign(systime=utc["systime"].dt.tz_convert("Europe/Berlin"))
    a, b = intervals(utc, spec), intervals(berlin, spec)
    assert a["duration_seconds"].tolist() == b["duration_seconds"].tolist()
    assert (a["start"] == b["start"]).all()  # same instants


@st.composite
def resetting_counter(draw):
    n = draw(st.integers(min_value=2, max_value=80))
    incs = draw(
        st.lists(st.integers(min_value=0, max_value=50), min_size=n, max_size=n)
    )
    resets = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    values, produced = [draw(st.integers(min_value=0, max_value=1000))], 0
    for inc, reset in zip(incs[1:], resets[1:], strict=True):
        prev = values[-1]
        if reset and prev > inc:  # PLC restart: counts again from zero
            values.append(inc)
        else:
            values.append(prev + inc)
        produced += inc
    return values, produced


@_SETTINGS
@given(
    counter=resetting_counter(),
    gaps=st.data(),
    window=st.sampled_from(["5min", "1h", "1D"]),
)
def test_counter_totals_survive_resets_and_windowing(counter, gaps, window):
    values, produced = counter
    steps = gaps.draw(
        st.lists(
            st.integers(min_value=1, max_value=3600),
            min_size=len(values) - 1,
            max_size=len(values) - 1,
        )
    )
    df = pd.DataFrame({"systime": _times(steps), "uuid": "c", "value_integer": values})
    spec = SignalSpec("c", kind="counter", value_column="value_integer")
    assert increments(df, spec).sum() == produced
    assert per_window(df, spec, window).sum() == produced


@_SETTINGS
@given(
    gaps=gaps_seconds, level=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)
)
def test_time_weighted_mean_of_constant_is_constant(gaps, level):
    times = _times(gaps)
    df = pd.DataFrame({"systime": times, "uuid": "x", "value_double": level})
    out = per_window(df, SignalSpec("x"), "1h").dropna()
    assert np.allclose(out.to_numpy(), level)
