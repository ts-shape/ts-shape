"""Every categorical severity also comes as severity_level (info/warn/critical)."""

import warnings

import pandas as pd
import pytest
from invariants import harness as H

import ts_shape
from ts_shape.events._output import SEVERITY_LEVELS, add_severity_level


def _severity_results():
    base = H.plant_frame()
    for cls_name, method in sorted(H.REGISTRY):
        for frame in (base, base.iloc[0:0]):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = H.build_call(getattr(ts_shape, cls_name), method, frame)()
            except Exception:  # noqa: BLE001 - other tests cover failures
                continue
            if isinstance(out, pd.DataFrame) and "severity" in out.columns:
                yield f"{cls_name}.{method}", out


def test_categorical_severity_has_a_level_in_the_shared_vocabulary():
    seen = 0
    for name, out in _severity_results():
        if pd.api.types.is_numeric_dtype(out["severity"]) or name.startswith(
            "OutlierDetectionEvents."
        ):
            assert "severity_level" not in out.columns, name
            continue
        seen += 1
        assert "severity_level" in out.columns, name
        cols = list(out.columns)
        assert cols.index("severity_level") == cols.index("severity") + 1, name
        assert set(out["severity_level"].dropna()) <= set(SEVERITY_LEVELS), name
    assert seen >= 10


@pytest.mark.parametrize(
    "overrides,label,level",
    [
        (None, "minor", "info"),  # the shared table (as the event log had it)
        ({"minor": "warn"}, "minor", "warn"),  # ToleranceDeviationEvents
        ({"minor": "info"}, "minor", "info"),  # FlowConstraintEvents
        (None, "high", "critical"),  # SPC rules
        ({"high": "warn"}, "high", "warn"),  # CUSUM
        (None, "warning", "warn"),
        (None, "none", None),
    ],
)
def test_label_mapping(overrides, label, level):
    out = add_severity_level(pd.DataFrame({"severity": [label]}), overrides)
    value = out["severity_level"].iloc[0]
    assert (value is None or pd.isna(value)) if level is None else value == level


def test_class_overrides_are_declared_where_labels_differ():
    from ts_shape.events.production.flow_constraints import FlowConstraintEvents
    from ts_shape.events.quality.statistical_process_control import (
        StatisticalProcessControlEvents,
    )
    from ts_shape.events.quality.tolerance_deviation import ToleranceDeviationEvents

    assert ToleranceDeviationEvents._severity_levels == {"*": {"minor": "warn"}}
    assert FlowConstraintEvents._severity_levels == {"*": {"minor": "info"}}
    assert StatisticalProcessControlEvents._severity_levels == {
        "detect_cusum_shifts": {"high": "warn"}
    }
