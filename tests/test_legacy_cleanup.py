"""Regression tests for the legacy-code cleanup (downcasting, UTC folders, ...)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.events._compat import state_mask
from ts_shape.utils.coerce import bool_or_false
from ts_shape.utils.frames import ffill_values

# ---- no silent downcasting -------------------------------------------------


@pytest.fixture()
def no_future_warnings():
    """Turn pandas' FutureWarning (deprecated silent downcasting) into errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        yield


def _object_bools() -> pd.Series:
    return pd.Series([True, None, False, np.nan, True], dtype=object)


def test_bool_or_false_matches_fillna_astype(no_future_warnings):
    for values in (
        _object_bools(),
        pd.Series([True, pd.NA, False], dtype="boolean"),
        pd.Series([1.0, np.nan, 0.0]),
        pd.Series([True, False]),
    ):
        out = bool_or_false(values)
        assert out.dtype == bool
        assert out.tolist() == [bool(v) if pd.notna(v) else False for v in values]
    idx = pd.date_range("2024-01-01", periods=3, freq="h")
    keep = bool_or_false(pd.Series([True, None, False], index=idx, name="s"))
    assert keep.index.equals(idx) and keep.name == "s"


def test_ffill_values_keeps_object_dtype(no_future_warnings):
    filled = ffill_values(pd.Series([None, True, None, False, None], dtype=object))
    assert filled.dtype == object
    assert filled.tolist()[1:] == [True, True, False, False]
    assert pd.isna(filled.iloc[0])
    idx = pd.Index([10, 11, 12])
    same = ffill_values(pd.Series(["a", None, "b"], index=idx))
    assert same.index.equals(idx) and same.tolist() == ["a", "a", "b"]


def test_state_mask_object_bool_column(no_future_warnings):
    data = pd.DataFrame({"value_bool": _object_bools(), "value_string": None})
    running = state_mask(data, "value_string", True, bool_value=True)
    assert running.tolist() == [True, False, False, False, True]
    stopped = state_mask(data, "value_string", False, bool_value=False)
    assert stopped.tolist() == [False, False, True, False, False]


def test_cycle_extractors_on_object_bools(no_future_warnings):
    from ts_shape.features.cycles.cycles_extractor import CycleExtractor

    times = pd.date_range("2024-01-01", periods=6, freq="min")
    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": "s",
            "value_bool": pd.Series(
                [False, True, None, False, True, False], dtype=object
            ),
        }
    )
    cycles = CycleExtractor(df, start_uuid="s").process_persistent_cycle()
    assert len(cycles) == 2
    changes = CycleExtractor(df, start_uuid="s").process_value_change_cycle()
    assert len(changes) >= 1


def test_future_no_silent_downcasting_path():
    """The helpers give the same answer with pandas' opt-in future behavior."""
    if not hasattr(pd.options, "future") or not hasattr(
        pd.options.future, "no_silent_downcasting"
    ):
        pytest.skip("option removed in this pandas version")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.option_context("future.no_silent_downcasting", True):
            out = bool_or_false(_object_bools())
            data = pd.DataFrame({"value_bool": _object_bools(), "value_string": None})
            mask = state_mask(data, "value_string", True, bool_value=True)
    assert out.tolist() == [True, False, False, False, True]
    assert mask.tolist() == [True, False, False, False, True]


# ---- merged duplicates -----------------------------------------------------


def test_flow_helpers_shared_by_flow_and_line_balancing():
    from ts_shape.events.production import _flow, flow_metrics, line_balancing

    assert flow_metrics.to_seconds is _flow.to_seconds
    assert line_balancing.rising_edge_times is _flow.rising_edge_times
    assert _flow.to_seconds("1min") == 60.0 and _flow.to_seconds(5) == 5.0
    assert _flow.to_seconds(None) is None
    times = pd.date_range("2024-01-01", periods=5, freq="min")
    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": "s",
            "value_bool": pd.Series([True, False, None, True, True], dtype=object),
        }
    )
    edges = _flow.rising_edge_times(
        df, "s", time_column="systime", value_column="value_bool"
    )
    assert list(edges) == [times[0], times[3]]


def test_localize_in_matches_localize_like_on_dst_days():
    from ts_shape.utils.frames import localize_in, localize_like

    ref = pd.Series(
        pd.date_range("2024-10-27", periods=2, freq="h", tz="Europe/Berlin")
    )
    for value in ("2024-10-27 02:30", "2024-03-31 02:30", "2024-01-01 00:00"):
        assert localize_in(value, "Europe/Berlin") == localize_like(value, ref)
    # ambiguous -> standard time (+01:00); nonexistent -> shifted forward
    ambiguous = localize_in("2024-10-27 02:30", "Europe/Berlin")
    assert ambiguous.utcoffset() == pd.Timedelta(hours=1)
    assert localize_in("2024-03-31 02:30", "Europe/Berlin").hour == 3
    assert localize_in("2024-01-01 05:00+01:00", None) == pd.Timestamp(
        "2024-01-01 04:00"
    )


def test_batch_quantity_without_resets_is_last_minus_first():
    from ts_shape.events.production.batch_tracking import BatchTrackingEvents

    t = pd.date_range("2024-01-01", periods=6, freq="10min")
    batch = pd.DataFrame(
        {"systime": t, "uuid": "batch", "value_string": list("AAABBB")}
    )
    counter = pd.DataFrame(
        {"systime": t, "uuid": "cnt", "value_integer": [10, 15, 20, 20, 18, 30]}
    )
    df = pd.concat([batch, counter], ignore_index=True)
    out = BatchTrackingEvents(df, batch_uuid="batch").batch_yield(
        "cnt", handle_resets=False
    )
    assert list(out["quantity"]) == [10, 10]
