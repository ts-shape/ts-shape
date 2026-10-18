"""A user script written against the public API, checked with ``mypy --strict``.

tests/typing/test_mypy.py type-checks this file (every name must be typed
through the lazy ``from ts_shape import ...``, not ``object`` or ``Any``) and
also runs it, so the calls stay real.
"""

from __future__ import annotations

import warnings
from typing import assert_type

import pandas as pd

from ts_shape import (
    ColumnNotFoundError,
    DataHarmonizer,
    DataQualityWarning,
    DoubleFilter,
    EventLog,
    IntegerCalc,
    MachineStateEvents,
    NumericStatistics,
    OEEEvents,
    OutlierDetectionEvents,
    Pipeline,
    PipelineResult,
    ProfileReport,
    SignalSpecs,
    TsShapeWarning,
    ValueMapper,
    concat,
    infer_specs,
    list_detectors,
    make_timeseries,
    profile,
    to_event_log,
)
from ts_shape.context import ValueMapper as ContextValueMapper
from ts_shape.events import OEEEvents as EventsOEEEvents
from ts_shape.features import NumericStatistics as FeaturesNumericStatistics
from ts_shape.transform import DoubleFilter as TransformDoubleFilter

# The lazy re-exports are the real classes for the type checker.
assert_type(OutlierDetectionEvents, type[OutlierDetectionEvents])
assert_type(make_timeseries("a:b", n_points=3), pd.DataFrame)
assert_type(EventsOEEEvents, type[OEEEvents])
assert_type(FeaturesNumericStatistics, type[NumericStatistics])
assert_type(TransformDoubleFilter, type[DoubleFilter])
assert_type(ContextValueMapper, type[ValueMapper])


def main() -> None:
    temp: pd.DataFrame = make_timeseries(
        ["oven:temp"], n_points=600, freq="1min", n_outliers=4
    )
    running: pd.DataFrame = make_timeseries(
        ["oven:running"], n_points=600, freq="1min", value_column="value_bool"
    )
    df = pd.concat([temp, running], ignore_index=True)

    report: ProfileReport = profile(df)
    summary: str = report.summary()
    specs: SignalSpecs = infer_specs(df)

    outliers: pd.DataFrame = OutlierDetectionEvents(
        temp, value_column="value_double"
    ).detect_outliers_zscore(n_sigma=4.0)
    intervals: pd.DataFrame = MachineStateEvents(
        df, run_state_uuid="oven:running"
    ).detect_run_idle(min_duration="5min")

    log: EventLog = concat(
        to_event_log(
            outliers, detector="OutlierDetectionEvents.detect_outliers_zscore"
        ),
        to_event_log(intervals, detector="MachineStateEvents.detect_run_idle"),
    )
    events: pd.DataFrame = log.events

    mean: float = NumericStatistics.column_mean(temp, "value_double")
    in_band: pd.DataFrame = DoubleFilter.filter_value_double_between(
        temp, min_value=0.0, max_value=200.0
    )
    wide: pd.DataFrame = DataHarmonizer(df).pivot_to_wide()

    result: PipelineResult = (
        Pipeline(name="typed")
        .transform(IntegerCalc, "scale_column", value_column="value_double", factor=0.1)
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="outliers",
            value_column="value_double",
            n_sigma=3.0,
        )
        .run(temp)
    )
    pipeline_events: dict[str, pd.DataFrame] = result.events

    catalog: pd.DataFrame = list_detectors("events.quality")

    try:
        NumericStatistics.column_mean(temp, "no_such_column")
    except ColumnNotFoundError as exc:
        missing: str = str(exc)
    else:
        raise AssertionError("expected ColumnNotFoundError")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        assert issubclass(DataQualityWarning, TsShapeWarning)

    assert summary and len(specs) == 2 and len(events) == len(log.events)
    assert mean == mean and len(in_band) <= len(temp) and not wide.empty
    assert "outliers" in pipeline_events and not catalog.empty and missing


if __name__ == "__main__":
    main()
