"""Property-based tests (hypothesis) for the core utilities.

* ``utils.counters``: on a monotonic counter the reset-aware quantity is
  ``last - first``; on any non-negative counter it is at least the naive
  ``max(0, last - first)``; every increment is booked in exactly one window;
  results do not depend on the dtype backend.
* ``utils.intervals.state_intervals`` (``next_change``): the intervals tile
  the time span exactly, without gaps or overlaps.
* ``events._windows.window_bounds``: every window has ``end > start``,
  consecutive windows are contiguous, each sample lies in its window, and a
  calendar day is 23 / 25 hours on DST-change days.
"""

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from ts_shape.events._windows import window_bounds  # noqa: E402
from ts_shape.utils.counters import (  # noqa: E402
    count_resets,
    counter_increments,
    counter_per_window,
    counter_quantity,
)
from ts_shape.utils.intervals import state_intervals  # noqa: E402

_SETTINGS = settings(max_examples=100, deadline=None)
T0 = pd.Timestamp("2024-03-30 00:00")

readings = st.lists(st.integers(min_value=0, max_value=10_000), min_size=1, max_size=80)
steps = st.lists(st.integers(min_value=0, max_value=500), min_size=1, max_size=80)
gaps = st.lists(st.integers(min_value=1, max_value=4 * 3600), min_size=1, max_size=60)


def _backends():
    out = ["int64", "float64", "Int64", "Float64"]
    try:
        import pyarrow  # noqa: F401

        out += ["int64[pyarrow]", "double[pyarrow]"]
    except ImportError:  # pragma: no cover - optional dependency
        pass
    return out


BACKENDS = _backends()


# ---------------------------------------------------------------------------
# utils.counters
# ---------------------------------------------------------------------------


@_SETTINGS
@given(start=st.integers(min_value=0, max_value=10_000), steps=steps)
def test_monotonic_counter_increments_sum_to_last_minus_first(start, steps):
    values = pd.Series(np.cumsum([start, *steps]), dtype="int64")
    inc = counter_increments(values)
    assert inc.iloc[0] == 0.0
    assert (inc >= 0).all()
    assert inc.sum() == values.iloc[-1] - values.iloc[0]
    assert counter_quantity(values) == counter_quantity(values, handle_resets=False)
    assert count_resets(values) == 0


@_SETTINGS
@given(values=readings)
def test_reset_aware_quantity_is_at_least_naive(values):
    s = pd.Series(values, dtype="int64")
    aware = counter_quantity(s)
    naive = counter_quantity(s, handle_resets=False)
    assert naive == max(0, values[-1] - values[0])
    assert aware >= naive
    # Equal exactly when nothing reset.
    if count_resets(s) == 0:
        assert aware == naive


@_SETTINGS
@given(values=readings, gaps=st.data())
def test_every_increment_is_booked_in_exactly_one_window(values, gaps):
    secs = gaps.draw(
        st.lists(
            st.integers(min_value=1, max_value=7200),
            min_size=len(values),
            max_size=len(values),
        )
    )
    idx = T0 + pd.to_timedelta(np.cumsum(secs), unit="s")
    s = pd.Series(values, index=idx, dtype="int64")
    per_window = counter_per_window(s, "1h")
    assert per_window.sum() == pytest.approx(counter_quantity(s))
    assert (per_window >= 0).all()


@settings(max_examples=40, deadline=None)
@given(values=readings)
def test_counters_do_not_depend_on_dtype_backend(values):
    idx = T0 + pd.to_timedelta(np.arange(len(values)) * 600, unit="s")
    ref_inc = counter_increments(pd.Series(values, index=idx, dtype="int64"))
    ref_win = counter_per_window(pd.Series(values, index=idx, dtype="int64"), "1h")
    for dtype in BACKENDS:
        s = pd.Series(values, index=idx).astype(dtype)
        np.testing.assert_array_equal(
            counter_increments(s).to_numpy(dtype=float), ref_inc.to_numpy(dtype=float)
        )
        np.testing.assert_array_equal(
            counter_per_window(s, "1h").to_numpy(dtype=float),
            ref_win.to_numpy(dtype=float),
        )
        assert counter_quantity(s) == counter_quantity(s.astype("int64"))
        assert count_resets(s) == count_resets(s.astype("int64"))


# ---------------------------------------------------------------------------
# utils.intervals
# ---------------------------------------------------------------------------


@_SETTINGS
@given(
    gaps=gaps,
    data=st.data(),
    tz=st.sampled_from([None, "UTC", "Europe/Berlin"]),
    tail=st.integers(min_value=0, max_value=3600),
)
def test_next_change_intervals_tile_the_time_span(gaps, data, tz, tail):
    times = pd.Series(
        T0 + pd.to_timedelta(np.concatenate([[0], np.cumsum(gaps)]), unit="s")
    )
    if tz is not None:
        times = times.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
        times = times.drop_duplicates().sort_values().reset_index(drop=True)
    states = data.draw(
        st.lists(
            st.sampled_from(["run", "idle", "fault"]),
            min_size=len(times),
            max_size=len(times),
        )
    )
    df = pd.DataFrame({"t": times, "s": states})
    data_end = times.iloc[-1] + pd.Timedelta(seconds=tail)
    iv = state_intervals(
        df,
        time_column="t",
        state_column="s",
        interval_end="next_change",
        data_end=data_end,
    )
    # Contiguous: each interval ends where the next starts.
    assert iv["start"].iloc[0] == times.iloc[0]
    assert (iv["end"].iloc[:-1].to_numpy() == iv["start"].iloc[1:].to_numpy()).all()
    assert iv["end"].iloc[-1] == data_end
    # No overlap, no negative durations, and the durations add up to the span.
    assert (iv["duration_seconds"] >= 0).all()
    assert iv["duration_seconds"].sum() == pytest.approx(
        (data_end - times.iloc[0]).total_seconds()
    )
    # Maximal runs: neighbors differ and every sample is counted once.
    assert (iv["state"].iloc[1:].to_numpy() != iv["state"].iloc[:-1].to_numpy()).all()
    assert iv["sample_count"].sum() == len(df)


# ---------------------------------------------------------------------------
# events._windows.window_bounds
# ---------------------------------------------------------------------------

WINDOWS = ["15min", "1h", "6h", "1D", "W", "MS", "ME"]
TZS = [None, "UTC", "Europe/Berlin", "America/New_York"]


@_SETTINGS
@given(
    offsets=st.lists(
        st.integers(min_value=0, max_value=120 * 86400), min_size=1, max_size=40
    ),
    window=st.sampled_from(WINDOWS),
    tz=st.sampled_from(TZS),
)
def test_window_bounds_are_contiguous_and_contain_their_samples(offsets, window, tz):
    times = pd.Timestamp("2024-01-01") + pd.to_timedelta(sorted(offsets), unit="s")
    if tz is not None:
        times = times.tz_localize("UTC").tz_convert(tz)
    s = pd.Series(1, index=times)
    counts = s.resample(window).count()
    start, end = window_bounds(pd.Series(counts.index), window)
    assert (end > start).all()
    # Consecutive resample bins are contiguous windows.
    assert (end.iloc[:-1].to_numpy() == start.iloc[1:].to_numpy()).all()
    # Every sample lies in the window of its bin.
    for (lab, n), st_, en in zip(counts.items(), start, end):
        inside = ((times >= st_) & (times < en)).sum()
        assert inside == n, (window, tz, lab)


@_SETTINGS
@given(
    day=st.dates(
        min_value=pd.Timestamp("2023-01-01").date(),
        max_value=pd.Timestamp("2026-12-31").date(),
    ),
    tz=st.sampled_from(["Europe/Berlin", "America/New_York", "UTC"]),
)
def test_calendar_day_length_follows_dst(day, tz):
    label = pd.Timestamp(day).tz_localize(tz)
    start, end = window_bounds(pd.Series([label]), "1D")
    hours = (end - start).iloc[0] / pd.Timedelta(hours=1)
    # Real length: next local midnight minus this local midnight.
    expected = (
        (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(tz) - label
    ) / pd.Timedelta(hours=1)
    assert hours == expected
    assert hours in (23.0, 24.0, 25.0)


@pytest.mark.parametrize(
    "tz,day,hours",
    [
        ("Europe/Berlin", "2024-03-31", 23),
        ("Europe/Berlin", "2024-10-27", 25),
        ("America/New_York", "2024-03-10", 23),
        ("America/New_York", "2024-11-03", 25),
        ("Europe/Berlin", "2024-06-15", 24),
    ],
)
def test_dst_days_are_23_and_25_hours(tz, day, hours):
    label = pd.Series([pd.Timestamp(day).tz_localize(tz)])
    start, end = window_bounds(label, "1D")
    assert (end - start).iloc[0] == pd.Timedelta(hours=hours)
