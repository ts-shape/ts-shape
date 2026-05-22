"""Tests for the declarative Pipeline orchestrator."""

from __future__ import annotations

import pandas as pd  # type: ignore
import pytest

import ts_shape
from ts_shape.datasets import make_timeseries
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents
from ts_shape.pipeline import Pipeline, PipelineResult
from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.transform.filter.numeric_filter import DoubleFilter
from ts_shape.utils.base import Base

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _signal(n_outliers: int = 5) -> pd.DataFrame:
    """A value_double signal with injected outliers."""
    return make_timeseries(
        ["sensor:x"], n_points=200, n_outliers=n_outliers, value_column="value_double"
    )


def _run_signal() -> pd.DataFrame:
    """A boolean run-state signal."""
    return make_timeseries(["machine:run"], n_points=200, value_column="value_bool")


class _Doubler(Base):
    """A minimal instance-style transform: doubles a numeric column."""

    def __init__(
        self, dataframe: pd.DataFrame, *, column: str = "value_double"
    ) -> None:
        super().__init__(dataframe)
        self._column = column

    def double(self) -> pd.DataFrame:
        df = self.dataframe.copy()
        df[self._column] = df[self._column] * 2
        return df


# ---------------------------------------------------------------------------
# Transform steps
# ---------------------------------------------------------------------------


def test_transform_only_replaces_channel():
    df = _signal()
    result = (
        Pipeline(name="t")
        .transform(IntegerCalc, "scale_column", column_name="value_double", factor=0.0)
        .run(df)
    )
    assert isinstance(result, PipelineResult)
    assert (result.data["value_double"] == 0.0).all()
    assert result.events == {}


def test_instance_transform_step():
    df = _signal(n_outliers=0)
    original = df["value_double"].to_numpy(copy=True)
    result = Pipeline().transform(_Doubler, "double", column="value_double").run(df)
    assert result.data["value_double"].to_numpy() == pytest.approx(original * 2)


def test_callable_transform_step():
    df = _signal(n_outliers=0)
    result = (
        Pipeline()
        .transform(lambda d: d.assign(value_double=d["value_double"] + 1), name="plus1")
        .run(df)
    )
    assert result.data["value_double"].to_numpy() == pytest.approx(
        df["value_double"].to_numpy() + 1
    )


# ---------------------------------------------------------------------------
# Detector steps
# ---------------------------------------------------------------------------


def test_detect_step_stores_events_and_leaves_channel_unchanged():
    df = _signal()
    result = (
        Pipeline()
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="outliers",
            value_column="value_double",
            threshold=2.0,
        )
        .run(df)
    )
    assert "outliers" in result.events
    for col in ("systime", "uuid", "source_uuid"):
        assert col in result.events["outliers"].columns
    # The detect step must not alter the working channel.
    pd.testing.assert_frame_equal(result.data, df)


def test_full_transform_then_detect():
    df = _signal()
    result = (
        Pipeline(name="quality")
        .transform(DoubleFilter, "filter_nan_value_double", column_name="value_double")
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="outliers",
            value_column="value_double",
            threshold=2.0,
        )
        .run(df)
    )
    assert not result.events["outliers"].empty
    assert "value_double" in result.data.columns


# ---------------------------------------------------------------------------
# kwarg routing & errors
# ---------------------------------------------------------------------------


def test_kwarg_routing_splits_init_and_method():
    # value_column -> __init__, threshold -> method. A successful run proves
    # routing: __init__ rejects 'threshold' and the method rejects
    # 'value_column' if mis-routed.
    df = _signal()
    result = (
        Pipeline()
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="o",
            value_column="value_double",
            threshold=3.0,
        )
        .run(df)
    )
    assert "o" in result.events


def test_unknown_kwarg_raises_value_error():
    with pytest.raises(ValueError, match="unknown argument"):
        Pipeline().detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="o",
            value_column="value_double",
            bogus_argument=1,
        )


def test_missing_method_raises_attribute_error():
    with pytest.raises(AttributeError, match="no method"):
        Pipeline().detect(OutlierDetectionEvents, "no_such_method", name="o")


def test_duplicate_detector_name_raises():
    pipe = Pipeline().detect(
        OutlierDetectionEvents,
        "detect_outliers_zscore",
        name="dup",
        value_column="value_double",
    )
    with pytest.raises(ValueError, match="duplicate detector step"):
        pipe.detect(
            OutlierDetectionEvents,
            "detect_outliers_iqr",
            name="dup",
            value_column="value_double",
        )


def test_step_failure_raises_with_step_context():
    def _boom(_df: pd.DataFrame) -> pd.DataFrame:
        raise KeyError("missing")

    pipe = Pipeline(name="p").transform(_boom, name="exploding")
    with pytest.raises(RuntimeError, match="exploding"):
        pipe.run(_signal())


def test_non_dataframe_result_raises_type_error():
    pipe = Pipeline().transform(lambda _df: "not a frame", name="bad")
    with pytest.raises(TypeError, match="expected a DataFrame"):
        pipe.run(_signal())


def test_run_rejects_non_dataframe_input():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        Pipeline().run([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reusability, eventlog, introspection
# ---------------------------------------------------------------------------


def test_pipeline_is_reusable_across_dataframes():
    pipe = Pipeline().detect(
        OutlierDetectionEvents,
        "detect_outliers_zscore",
        name="o",
        value_column="value_double",
        threshold=2.0,
    )
    few = pipe.run(_signal(n_outliers=2))
    many = pipe.run(_signal(n_outliers=40))
    assert len(many.events["o"]) >= len(few.events["o"])


def test_to_event_log_combined_and_per_step():
    df = _signal()
    result = (
        Pipeline()
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="o",
            value_column="value_double",
            threshold=2.0,
        )
        .run(df)
    )
    combined = result.to_event_log()
    assert isinstance(combined, ts_shape.EventLog)

    per_step = result.to_event_log(concat=False)
    assert set(per_step) == {"o"}
    assert isinstance(per_step["o"], ts_shape.EventLog)


def test_to_event_log_without_detector_identity_raises():
    df = _signal()
    detector = OutlierDetectionEvents(df, value_column="value_double")
    # Callable-form detect step -> no "Class.method" identity.
    result = (
        Pipeline()
        .detect(lambda _d: detector.detect_outliers_zscore(threshold=2.0), name="o")
        .run(df)
    )
    assert "o" in result.events
    with pytest.raises(ValueError, match="no detector step"):
        result.to_event_log()


def test_runtime_detector_in_pipeline():
    result = (
        Pipeline(name="utilization")
        .detect(
            RuntimeAccountingEvents,
            "runtime_summary",
            name="runtime",
            run_uuid="machine:run",
        )
        .run(_run_signal())
    )
    for col in ("start", "end", "duration_seconds"):
        assert col in result.events["runtime"].columns


def test_empty_input_flows_through(empty_df):
    result = (
        Pipeline()
        .transform(IntegerCalc, "scale_column", column_name="value_double", factor=2)
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="o",
            value_column="value_double",
        )
        .run(empty_df)
    )
    assert result.events["o"].empty


def test_steps_describe_and_repr():
    pipe = (
        Pipeline(name="demo")
        .transform(IntegerCalc, "scale_column", column_name="value_double", factor=2)
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="outliers",
            value_column="value_double",
        )
    )
    assert pipe.steps == [("transform", "scale_column"), ("detect", "outliers")]
    text = pipe.describe()
    assert text == repr(pipe)
    assert "demo" in text
    assert "transform" in text and "detect" in text
    # describe() shows step kwargs
    assert "factor=2" in text
    assert Pipeline(name="empty").describe() == "Pipeline 'empty' (no steps)"


def test_callable_form_forwards_kwargs():
    # A bound classmethod passed as a callable still receives its kwargs.
    df = _signal(n_outliers=0)
    result = (
        Pipeline()
        .transform(IntegerCalc.scale_column, column_name="value_double", factor=10)
        .run(df)
    )
    assert result.data["value_double"].to_numpy() == pytest.approx(
        df["value_double"].to_numpy() * 10
    )


def test_input_and_prev_sentinels():
    df = _signal(n_outliers=0)

    def _tag(frame, *, original):
        # `original` is wired to $input; `frame` is the working ($prev) frame.
        return frame.assign(input_rows=len(original), working_rows=len(frame))

    result = (
        Pipeline()
        .transform(lambda d: d.head(10), name="head10")
        .transform(_tag, name="tag", original="$input")
        .run(df)
    )
    assert (result.data["input_rows"] == len(df)).all()
    assert (result.data["working_rows"] == 10).all()


def test_unknown_sentinel_raises():
    with pytest.raises(ValueError, match="unknown sentinel"):
        Pipeline().transform(lambda d, **k: d, name="x", bad="$PREV")


def test_run_steps_returns_every_intermediate():
    df = _signal()
    steps = (
        Pipeline()
        .transform(IntegerCalc, "scale_column", column_name="value_double", factor=2)
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="outliers",
            value_column="value_double",
            threshold=2.0,
        )
        .run_steps(df)
    )
    assert set(steps) == {"input", "scale_column", "outliers"}
    for frame in steps.values():
        assert isinstance(frame, pd.DataFrame)
    # 'input' is the untouched original; 'scale_column' is the transformed signal.
    assert steps["input"]["value_double"].to_numpy() == pytest.approx(
        df["value_double"].to_numpy()
    )
    assert steps["scale_column"]["value_double"].to_numpy() == pytest.approx(
        df["value_double"].to_numpy() * 2
    )


def test_pipeline_exported_at_top_level():
    assert ts_shape.Pipeline is Pipeline
