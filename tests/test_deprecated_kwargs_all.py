"""Every renamed keyword argument in ts_shape still works under its old name.

The shims come from :func:`ts_shape.utils.deprecation.renamed_kwargs`. This
module finds all of them by walking the package, then checks each
``(callable, old, new)`` triple:

* the reported signature shows the new name and not the old one;
* passing both names raises :class:`TypeError`;
* passing the old name warns and forwards the value under the new name.

The forwarding check swaps the wrapped function for a recorder, so it runs
in microseconds and needs no domain data. A handful of real calls, one per
module family, then checks that the old and the new name give identical
results.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types
import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.utils.deprecation import renamed_kwargs

_SHIM_CODE = renamed_kwargs(old="new")(lambda **kw: kw).__code__


def _modules():
    for info in pkgutil.walk_packages(ts_shape.__path__, "ts_shape."):
        try:
            yield importlib.import_module(info.name)
        except ImportError:  # loader backends with optional dependencies
            continue


def _shim(obj):
    """The ``renamed_kwargs`` wrapper in ``obj``'s ``__wrapped__`` chain."""
    seen = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        if getattr(obj, "__code__", None) is _SHIM_CODE:
            return obj
        obj = getattr(obj, "__wrapped__", None)
    return None


def _collect():
    found = {}
    for mod in _modules():
        for name, obj in vars(mod).items():
            if getattr(obj, "__module__", None) != mod.__name__:
                continue
            if inspect.isfunction(obj):
                members = [(name, obj, obj)]
            elif inspect.isclass(obj):
                members = []
                for attr, raw in vars(obj).items():
                    func = (
                        raw.__func__
                        if isinstance(raw, staticmethod | classmethod)
                        else raw
                    )
                    if callable(func):
                        members.append((f"{name}.{attr}", func, getattr(obj, attr)))
            else:
                continue
            for qualname, func, public in members:
                shim = _shim(func)
                if shim is not None:
                    found[f"{mod.__name__}:{qualname}"] = (public, shim)
    return found


SHIMS = _collect()
CASES = sorted(
    (key, old, new)
    for key, (_, shim) in SHIMS.items()
    for old, new in shim.__deprecated_kwargs__.items()
)


def test_shims_were_found():
    # Guards the walker itself: there are well over a hundred renamed kwargs.
    assert len(CASES) > 100
    assert "ts_shape.utils.base:Base.__init__" in SHIMS


def _recording_shim(shim, calls):
    """A copy of ``shim`` whose wrapped function records its keyword arguments."""

    def record(*args, **kwargs):
        calls.append(kwargs)
        return "sentinel-result"

    cells = []
    for var, cell in zip(shim.__code__.co_freevars, shim.__closure__):
        cells.append(types.CellType(record) if var == "func" else cell)
    return types.FunctionType(
        shim.__code__, shim.__globals__, shim.__name__, shim.__defaults__, tuple(cells)
    )


@pytest.mark.parametrize("key,old,new", CASES, ids=[f"{k}:{o}" for k, o, _ in CASES])
def test_renamed_kwarg(key, old, new):
    public, shim = SHIMS[key]
    assert old != new

    params = inspect.signature(public).parameters
    assert new in params, f"{key}: new name {new!r} missing from signature"
    assert old not in params, f"{key}: old name {old!r} still in signature"

    sentinel = object()
    with pytest.raises(TypeError, match=rf"{old!r}.*{new!r}"):
        shim(**{old: sentinel, new: sentinel})

    calls = []
    recorder = _recording_shim(shim, calls)
    with pytest.warns(DeprecationWarning, match=rf"{old!r} is deprecated, use {new!r}"):
        assert recorder(**{old: sentinel}) == "sentinel-result"
    assert calls == [{new: sentinel}]

    calls.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        recorder(**{new: sentinel})
    assert calls == [{new: sentinel}]


# ---------------------------------------------------------------------------
# Real calls: old and new name give the same result
# ---------------------------------------------------------------------------


def _frame():
    rng = np.random.default_rng(0)
    n = 48
    return pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "uuid": "sig:1",
            "value_double": rng.normal(50.0, 5.0, n),
            "value_integer": rng.integers(0, 10, n),
            "value_bool": rng.integers(0, 2, n).astype(bool),
            "value_string": rng.choice(["A", "B"], n),
            "is_delta": True,
        }
    )


def _numeric_stats(**kw):
    from ts_shape.features.stats.numeric_stats import NumericStatistics

    return NumericStatistics.column_mean(_frame(), **kw)


def _timestamp_stats(**kw):
    from ts_shape.features.stats.timestamp_stats import TimestampStatistics

    return TimestampStatistics.earliest_timestamp(_frame(), **kw)


def _numeric_filter(**kw):
    from ts_shape.transform.filter.numeric_filter import DoubleFilter

    return DoubleFilter.filter_value_double_between(
        _frame(), min_value=45.0, max_value=55.0, **kw
    )


def _time_grouped(**kw):
    from ts_shape.features.time_stats.time_stats_numeric import TimeGroupedStatistics

    return TimeGroupedStatistics.calculate_statistic(
        _frame(), "systime", "value_double", stat_method="mean", **kw
    )


def _harmonizer(**kw):
    from ts_shape.transform.harmonization import DataHarmonizer

    return DataHarmonizer(_frame()).resample_to_uniform(**kw)


def _data_gaps(**kw):
    from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents

    return DataGapAnalysisEvents(_frame(), "sig:1").coverage_by_period(**kw)


def _operating_range(**kw):
    from ts_shape.events.engineering.operating_range import OperatingRangeEvents

    return OperatingRangeEvents(_frame(), "sig:1").time_in_range(window="6h", **kw)


def _feature_matrix(**kw):
    from ts_shape.features.export import FeatureMatrixExporter

    return FeatureMatrixExporter.to_feature_matrix(**kw)


def _base_init(**kw):
    from ts_shape.utils.base import Base

    return Base(_frame(), **kw).dataframe


REAL_CALLS = [
    (
        "numeric_stats",
        _numeric_stats,
        {"column_name": "value_double"},
        {"value_column": "value_double"},
    ),
    (
        "timestamp_stats",
        _timestamp_stats,
        {"column_name": "systime"},
        {"time_column": "systime"},
    ),
    (
        "numeric_filter",
        _numeric_filter,
        {"column_name": "value_double"},
        {"value_column": "value_double"},
    ),
    ("time_grouped", _time_grouped, {"freq": "2h"}, {"window": "2h"}),
    ("harmonizer", _harmonizer, {"freq": "30min"}, {"window": "30min"}),
    ("data_gaps", _data_gaps, {"freq": "6h"}, {"window": "6h"}),
    (
        "operating_range",
        _operating_range,
        {"lower": 45.0, "upper": 55.0},
        {"lower_limit": 45.0, "upper_limit": 55.0},
    ),
    ("feature_matrix", _feature_matrix, {"df": _frame()}, {"dataframe": _frame()}),
    ("base_init", _base_init, {"column_name": "systime"}, {"time_column": "systime"}),
]


def _assert_same(a, b):
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize(
    "call,old_kw,new_kw",
    [c[1:] for c in REAL_CALLS],
    ids=[c[0] for c in REAL_CALLS],
)
def test_old_name_gives_same_result(call, old_kw, new_kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        expected = call(**new_kw)
    with pytest.warns(DeprecationWarning, match="is deprecated, use"):
        got = call(**old_kw)
    _assert_same(got, expected)
