"""Renamed classes and methods keep working, warn, and resolve in the event log."""

import inspect
import warnings

import pandas as pd
import pytest

import ts_shape
from ts_shape.eventlog import taxonomy
from ts_shape.eventlog.introspect import describe

# Required constructor arguments beyond the DataFrame, per alias.
DOMAIN_ARGS = {
    "OrderTraceabilityEvents": {"station_uuids": {"st:1": "Station 1"}},
    "ProcessStabilityIndex": {"signal_uuid": "sig:1"},
    "StatisticalProcessControlRuleBased": {
        "value_column": "value_double",
        "tolerance_uuid": "tol:1",
        "actual_uuid": "act:1",
        "event_uuid": "spc:1",
    },
}


@pytest.mark.parametrize("old,new", sorted(taxonomy.CLASS_ALIASES.items()))
def test_old_class_name_is_a_warning_alias(old, new):
    old_cls, new_cls = getattr(ts_shape, old), getattr(ts_shape, new)
    assert issubclass(old_cls, new_cls)
    assert old_cls.__deprecated_alias_of__ is new_cls
    params = list(inspect.signature(new_cls).parameters.values())
    required = [
        p.name
        for p in params[1:]
        if p.default is p.empty and p.kind is p.POSITIONAL_OR_KEYWORD
    ]
    kwargs = DOMAIN_ARGS.get(old, {})
    assert set(required) <= set(kwargs), f"add {required} for {old} to DOMAIN_ARGS"
    frame = pd.DataFrame(
        {"systime": pd.to_datetime([]), "uuid": [], "value_double": []}
    )
    with pytest.warns(DeprecationWarning, match=f"{old} is deprecated, use {new}"):
        obj = old_cls(frame, **kwargs)
    assert isinstance(obj, new_cls)


@pytest.mark.parametrize(
    "cls,old,new",
    sorted((c, o, n) for (c, o), n in taxonomy.METHOD_ALIASES.items()),
)
def test_old_method_name_forwards(cls, old, new):
    klass = getattr(ts_shape, cls)
    assert callable(getattr(klass, new))
    raw = inspect.getattr_static(klass, old)
    assert type(raw).__name__ == "_Forward"
    assert inspect.signature(getattr(klass, old)) == inspect.signature(
        getattr(klass, new)
    )


def test_aliases_in_code_match_taxonomy_maps():
    """Every deprecated alias defined in the code is known to the taxonomy."""
    found_classes, found_methods = {}, {}
    for name in dir(ts_shape):
        obj = getattr(ts_shape, name, None)
        if not inspect.isclass(obj):
            continue
        target = vars(obj).get("__deprecated_alias_of__")
        if target is not None:
            found_classes[name] = target.__name__
            continue
        for attr, raw in vars(obj).items():
            if type(raw).__name__ == "_Forward":
                found_methods[(name, attr)] = getattr(obj, attr).__deprecated_alias_of__
    assert found_classes == taxonomy.CLASS_ALIASES
    assert found_methods == taxonomy.METHOD_ALIASES


def test_event_log_accepts_old_detector_names():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        old = describe("OEECalculator.calculate_oee")
    assert (old["class_name"], old["method_name"]) == ("OEEEvents", "oee")
    assert taxonomy.get("StatisticalProcessControlRuleBased", "process") is (
        taxonomy.REGISTRY[("StatisticalProcessControlEvents", "detect_violations")]
    )
