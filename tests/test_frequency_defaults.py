"""Guard: every frequency-like string default parses on the installed pandas.

pandas 3 rejects 'm' (and 'H', 'T') as offsets; a default like ``window="5m"``
works with ``pd.Timedelta`` but crashes the moment it reaches ``resample``.
"""

import inspect

import pandas as pd
import pytest
from pandas.tseries.frequencies import to_offset

import ts_shape

_FREQ_NAMES = (
    "window",
    "freq",
    "frequency",
    "resample",
    "period",
    "baseline_window",
    "comparison_window",
)


def _string_defaults():
    seen = set()
    for name in ts_shape.__all__:
        obj = getattr(ts_shape, name)
        targets = []
        if inspect.isclass(obj):
            targets = [
                getattr(obj, m)
                for m in dir(obj)
                if not m.startswith("_") or m == "__init__"
            ]
        elif callable(obj):
            targets = [obj]
        for fn in targets:
            if not callable(fn):
                continue
            try:
                params = inspect.signature(fn).parameters.values()
            except (TypeError, ValueError):
                continue
            for p in params:
                if isinstance(p.default, str) and p.name in _FREQ_NAMES:
                    key = (getattr(fn, "__qualname__", str(fn)), p.name, p.default)
                    if key not in seen:
                        seen.add(key)
                        yield key


_DEFAULTS = sorted(_string_defaults())


def test_found_frequency_defaults():
    assert len(_DEFAULTS) > 20


@pytest.mark.parametrize(
    ("where", "param", "default"), _DEFAULTS, ids=[f"{w}.{p}" for w, p, _ in _DEFAULTS]
)
def test_frequency_default_is_valid_offset(where, param, default):
    try:
        to_offset(default)
    except ValueError:
        # Durations that are not offsets are fine if they are valid Timedeltas
        # *and* not used as a resample rule; offsets are the strict case.
        pd.Timedelta(default)
        pytest.fail(f"{where}({param}={default!r}) is not a valid pandas offset")
