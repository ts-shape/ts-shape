"""Tests for align_columns — making per-detector event frames share a schema."""

from __future__ import annotations

import pandas as pd

from ts_shape.eventlog import (
    align_columns,
    concat,
    to_event_log,
)
from ts_shape.eventlog.schema import EVENT_OPTIONAL_COLUMNS, EVENT_REQUIRED_COLUMNS
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents

_CORE = set(EVENT_REQUIRED_COLUMNS) | set(EVENT_OPTIONAL_COLUMNS)


def _interval_log():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2026-05-07", periods=12, freq="30s", tz="UTC"),
            "value_bool": [True] * 3 + [False] * 3 + [True] * 3 + [False] * 3,
            "uuid": ["A"] * 12,
            "source_uuid": ["A"] * 12,
        }
    )
    return to_event_log(
        MachineStateEvents(df, run_state_uuid="A").detect_run_idle(),
        detector="MachineStateEvents.detect_run_idle",
    )


def _point_log():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2026-05-07", periods=12, freq="1min", tz="UTC"),
            "value_double": [1.0] * 5 + [50.0] + [1.0] * 5 + [-30.0],
            "uuid": ["A"] * 12,
            "source_uuid": ["A"] * 12,
        }
    )
    return to_event_log(
        OutlierDetectionEvents(
            df, value_column="value_double"
        ).detect_outliers_zscore(),
        detector="OutlierDetectionEvents.detect_outliers_zscore",
    )


def test_align_columns_makes_frames_identical():
    a, b = _interval_log(), _point_log()
    # Precondition: the raw frames differ in total column count.
    assert list(a.events.columns) != list(b.events.columns)

    aligned = align_columns(a, b)
    cols = [list(log.events.columns) for log in aligned]
    # Every aligned frame has the exact same columns, in the same order.
    assert cols[0] == cols[1]
    # The union preserves every original column.
    assert set(cols[0]) == set(a.events.columns) | set(b.events.columns)


def test_align_columns_core_leads_in_canonical_order():
    aligned = align_columns(_point_log(), _interval_log())
    leading = list(aligned[0].events.columns)[: len(_CORE)]
    assert set(leading) == _CORE
    # Core block comes before any extra column.
    rest = list(aligned[0].events.columns)[len(_CORE) :]
    assert not (set(rest) & _CORE)


def test_align_columns_preserves_rows_and_values():
    a, b = _interval_log(), _point_log()
    aligned = align_columns(a, b)
    assert len(aligned[0].events) == len(a.events)
    assert len(aligned[1].events) == len(b.events)
    # Columns the point log never had are filled with NA (no spurious values).
    extra = "ts_shape:lifecycle_state"
    if extra in aligned[1].events.columns:
        assert aligned[1].events[extra].isna().all()


def test_align_columns_appendable_with_plain_concat():
    aligned = align_columns(_interval_log(), _point_log())
    stacked = pd.concat([log.events for log in aligned], ignore_index=True)
    assert len(stacked) == sum(len(log.events) for log in aligned)
    # Matches what eventlog.concat produces, column-set-wise.
    merged = concat(_interval_log(), _point_log())
    assert set(stacked.columns) == set(merged.events.columns)


def test_align_columns_empty_call():
    assert align_columns() == []
