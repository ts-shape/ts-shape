"""Naming lint: every detector signature follows docs/naming.md.

* no retired parameter names (they live on only as deprecated keyword
  aliases, which are not part of the signature);
* optional arguments are keyword-only;
* ``event_uuid`` defaults look like ``"<pack>:<name>"`` and are unique.
"""

from __future__ import annotations

import inspect
import re

import pytest

import ts_shape
from ts_shape.eventlog.taxonomy import CLASS_ALIASES, REGISTRY

#: retired name -> canonical name (docs/naming.md)
RETIRED = {
    "column_name": "time_column",
    "timestamp_column": "time_column",
    "systime_col": "time_column",
    "time_col": "time_column",
    "value_col": "value_column",
    "uuid_col": "uuid_column",
    "sensor_uuids": "signal_uuids",
    "uuid_list": "signal_uuids",
    "run_uuid": "run_state_uuid",
    "freq": "window",
    "period": "window",
    "resample": "window",
    "z_threshold": "n_sigma",
    "threshold_sigma": "n_sigma",
    "low": "lower_limit",
    "high": "upper_limit",
    "lower": "lower_limit",
    "upper": "upper_limit",
    "lower_bound": "lower_limit",
    "upper_bound": "upper_limit",
    "low_limit": "lower_limit",
    "high_limit": "upper_limit",
    "minor_threshold": "warning_threshold",
    "moderate_threshold": "critical_threshold",
    "include_first": "include_initial",
    "count_initial_active": "include_initial",
    "significant_only": "only_significant",
    "on_crossing_only": "only_crossings",
    "counter_column": "value_column_counter",
    "energy_column": "value_column_energy",
    "state_column": "value_column_state",
}

CLASSES = sorted(
    {c for c, _ in REGISTRY if c in dir(ts_shape) and c not in CLASS_ALIASES}
)


def _callables(cls):
    yield "__init__", cls.__init__
    for name, member in vars(cls).items():
        if name.startswith("_") or not callable(getattr(cls, name, None)):
            continue
        if type(member).__name__ == "_Forward":  # deprecated method alias
            continue
        yield name, getattr(cls, name)


def _params(fn):
    try:
        return list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return []


@pytest.mark.parametrize("cls_name", CLASSES)
def test_no_retired_parameter_names(cls_name):
    cls = getattr(ts_shape, cls_name)
    bad = [
        f"{cls_name}.{name}({p.name}) -> {RETIRED[p.name]}"
        for name, fn in _callables(cls)
        for p in _params(fn)
        if p.name in RETIRED
    ]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("cls_name", CLASSES)
def test_optional_arguments_are_keyword_only(cls_name):
    cls = getattr(ts_shape, cls_name)
    bad = []
    for name, fn in _callables(cls):
        for p in _params(fn):
            if p.kind is p.POSITIONAL_OR_KEYWORD and p.default is not p.empty:
                bad.append(f"{cls_name}.{name}({p.name}=...)")
    assert not bad, "pass these by keyword (add `*`):\n" + "\n".join(bad)


_EVENT_UUID = re.compile(r"^[a-z]+:[a-z0-9_]+$")


def test_event_uuid_defaults_are_namespaced_and_unique():
    seen: dict[str, str] = {}
    problems = []
    classes = {getattr(ts_shape, c): c for c in reversed(CLASSES)}  # dedupe aliases
    for cls, cls_name in sorted(classes.items(), key=lambda kv: kv[1]):
        p = inspect.signature(cls).parameters.get("event_uuid")
        if p is None or p.default is p.empty or p.default is None:
            continue
        if not _EVENT_UUID.match(str(p.default)):
            problems.append(f"{cls_name}: {p.default!r} is not '<pack>:<name>'")
        if p.default in seen:
            problems.append(
                f"{cls_name} and {seen[p.default]} share the default {p.default!r}"
            )
        seen[p.default] = cls_name
    assert not problems, "\n".join(problems)
