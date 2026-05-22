#!/usr/bin/env python3
"""
Pipeline Orchestrator Demo for ts-shape
========================================

Demonstrates `Pipeline` -- a declarative, reusable chain of transform and
detector steps.

- A `.transform(...)` step's output REPLACES the working signal.
- A `.detect(...)` step's output is STORED under a name; the signal is
  untouched (detectors branch off).

Two pipelines are shown:

1. Sensor-quality  -- scale a raw signal, then detect outliers on it.
2. Machine-utilization -- runtime accounting on a boolean run signal, then
   normalize the detector output into a single OCEL event log.

All data comes from `ts_shape.make_timeseries` -- no external files needed.
"""

from ts_shape import Pipeline, make_timeseries
from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents
from ts_shape.transform.calculator.numeric_calc import IntegerCalc


def demo_sensor_quality() -> None:
    print("=" * 72)
    print("1. SENSOR-QUALITY PIPELINE  (transform -> detect)")
    print("=" * 72)

    df = make_timeseries(
        ["sensor:x"], n_points=500, n_outliers=8, value_column="value_double"
    )

    pipe = (
        Pipeline(name="sensor-quality")
        # Transform: scale raw counts into engineering units (replaces signal).
        .transform(IntegerCalc, "scale_column", column_name="value_double", factor=0.1)
        # Detect: outliers on the scaled signal (branch-off result).
        .detect(
            OutlierDetectionEvents,
            "detect_outliers_zscore",
            name="outliers",
            value_column="value_double",
            threshold=3.0,
        )
    )
    print(pipe)

    result = pipe.run(df)
    print(f"\nFinal signal: {len(result.data)} rows " f"(value_double scaled by 0.1)")
    print(f"Detector results: {sorted(result.events)}")
    print(f"Outlier events detected: {len(result.events['outliers'])}")
    print()


def demo_machine_utilization() -> None:
    print("=" * 72)
    print("2. MACHINE-UTILIZATION PIPELINE  (detect -> event log)")
    print("=" * 72)

    run_signal = make_timeseries(
        ["machine:running"], n_points=600, value_column="value_bool"
    )

    pipe = Pipeline(name="machine-utilization").detect(
        RuntimeAccountingEvents,
        "runtime_summary",
        name="runtime",
        run_uuid="machine:running",
    )
    print(pipe)

    result = pipe.run(run_signal)
    print("\nRuntime summary:")
    print(result.events["runtime"].to_string(index=False))

    # The pipeline knows each detect step's "Class.method" identity, so it can
    # normalize detector output into a canonical OCEL event log automatically.
    event_log = result.to_event_log()
    print(
        f"\nNormalized event log: {type(event_log).__name__} "
        f"with {len(event_log.events)} event(s)"
    )
    print()


def demo_reuse() -> None:
    print("=" * 72)
    print("3. ONE PIPELINE, MANY DATAFRAMES")
    print("=" * 72)

    pipe = Pipeline(name="reusable").detect(
        OutlierDetectionEvents,
        "detect_outliers_zscore",
        name="outliers",
        value_column="value_double",
        threshold=3.0,
    )

    for label, n_outliers in [("calm day", 2), ("rough day", 30)]:
        df = make_timeseries(
            ["sensor:x"],
            n_points=500,
            n_outliers=n_outliers,
            value_column="value_double",
        )
        result = pipe.run(df)
        print(f"  {label:<10s}: {len(result.events['outliers'])} outlier events")
    print()


if __name__ == "__main__":
    demo_sensor_quality()
    demo_machine_utilization()
    demo_reuse()

    print("=" * 72)
    print("Pipeline demo complete.")
    print("=" * 72)
