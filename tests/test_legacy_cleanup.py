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


# ---- no silent numeric coercion ----------------------------------------------


def _text_in_numbers() -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=60, freq="min")
    values = pd.Series(np.sin(np.arange(60) / 3).astype(object))
    values[5] = "n/a"
    return pd.DataFrame({"systime": t, "uuid": "s", "value_double": values})


def test_feature_functions_report_text_in_numeric_columns():
    from ts_shape.errors import DataQualityWarning
    from ts_shape.features.export import FeatureMatrixExporter
    from ts_shape.features.pattern_recognition import PatternRecognition
    from ts_shape.features.segment_analysis._grouped_stats import summarize_groups

    df = _text_in_numbers()
    with pytest.warns(DataQualityWarning, match="1 non-null value.*'value_double'"):
        PatternRecognition.discover_motifs(df, window_size=5, top_k=1)
    with pytest.warns(DataQualityWarning, match="'value_double' of signal 's'"):
        FeatureMatrixExporter.to_feature_matrix(df, value_columns=["value_double"])
    with pytest.warns(DataQualityWarning, match="'value_double' are not numeric"):
        stats = summarize_groups(df, ["uuid"], "value_double")
    assert int(stats["sample_count"].iloc[0]) == 59


def test_cycle_processor_reports_text_in_metric_column():
    from ts_shape.errors import DataQualityWarning
    from ts_shape.features.cycles.cycle_processor import CycleDataProcessor

    df = _text_in_numbers()
    t = df["systime"]
    cycles = pd.DataFrame(
        {"cycle_start": [t[0]], "cycle_end": [t[30]], "cycle_uuid": ["c1"]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        proc = CycleDataProcessor(cycles, df)
    with pytest.warns(DataQualityWarning, match="'value_double' are not numeric"):
        proc.compute_cycle_statistics()


# ---- column checks -----------------------------------------------------------


def _cycles_and_values():
    t = pd.date_range("2024-01-01", periods=10, freq="min")
    values = pd.DataFrame({"systime": t, "uuid": "s", "value_double": range(10)})
    cycles = pd.DataFrame(
        {"cycle_start": [t[0], t[5]], "cycle_end": [t[4], t[9]], "cycle_uuid": "c"}
    )
    cycles["cycle_uuid"] = ["c1", "c2"]
    return cycles, values


def test_validate_column_keeps_its_message():
    from ts_shape.errors import ColumnNotFoundError
    from ts_shape.utils.base import Base

    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ColumnNotFoundError, match=r"^Column 'b' not found\. "):
        Base._validate_column(df, "b")
    with pytest.raises(ColumnNotFoundError, match=r"^f: column 'b' not found"):
        Base._validate_column(df, "b", where="f")
    Base._validate_column(df, "a")


def test_critical_parameter_ranking_missing_outcome_column():
    from ts_shape.errors import ColumnNotFoundError
    from ts_shape.events.development.critical_parameter_ranking import (
        CriticalParameterRankingEvents,
    )

    runs = pd.DataFrame({"temp": [1.0, 2.0, 3.0], "yield": [1.0, 2.0, 2.5]})
    with pytest.raises(ColumnNotFoundError, match="'nope'"):
        CriticalParameterRankingEvents().rank(runs, ["temp"], "nope")


def test_cycle_processor_column_checks():
    from ts_shape.errors import ColumnNotFoundError
    from ts_shape.features.cycles.cycle_processor import CycleDataProcessor

    cycles, values = _cycles_and_values()
    with pytest.raises(ColumnNotFoundError, match="'cycle_start', 'cycle_end'"):
        CycleDataProcessor(cycles.drop(columns=["cycle_start", "cycle_end"]), values)
    with pytest.raises(ColumnNotFoundError, match="values_df: column 'systime'"):
        CycleDataProcessor(cycles, values.rename(columns={"systime": "t"}))
    proc = CycleDataProcessor(cycles, values)
    groups = proc.group_by_cycle_uuid()
    assert len(proc.split_dataframes_by_group(groups, "uuid")) == 2
    # A frame without the group column raises instead of being skipped.
    with pytest.raises(ColumnNotFoundError, match=r"dfs\[1\]: column 'uuid'"):
        proc.split_dataframes_by_group(
            [groups[0], groups[1].drop(columns="uuid")], "uuid"
        )


# ---- data-quality warnings instead of log lines -----------------------------


def test_degenerate_statistics_warn():
    from ts_shape.errors import DataQualityWarning
    from ts_shape.events.quality.multivariate_spc import MultivariateSPCEvents

    t = pd.date_range("2024-01-01", periods=50, freq="min")
    temp = np.sin(np.arange(50) / 5.0)
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "temp", "value_double": temp}),
            pd.DataFrame({"systime": t, "uuid": "press", "value_double": 2 * temp}),
        ],
        ignore_index=True,
    )
    with pytest.warns(DataQualityWarning, match="covariance is singular"):
        MultivariateSPCEvents(df, ["temp", "press"]).hotelling_t2()


def test_weibull_fit_failure_is_named_in_the_warning(monkeypatch):
    from ts_shape.errors import DataQualityWarning
    from ts_shape.events.maintenance import reliability

    times, states = [], []
    t = pd.Timestamp("2024-01-01")
    for hours in (8, 10, 13, 17):
        times += [t, t + pd.Timedelta(hours=hours)]
        states += [False, True]
        t += pd.Timedelta(hours=hours + 1)
    times.append(t)
    states.append(False)
    df = pd.DataFrame({"systime": times, "uuid": "fault", "value_bool": states})

    def broken_fit(*args, **kwargs):
        raise RuntimeError("did not converge")

    monkeypatch.setattr(reliability.stats.weibull_min, "fit", broken_fit)
    with pytest.warns(DataQualityWarning, match="RuntimeError: did not converge"):
        reliability.ReliabilityEvents(df, "fault").weibull_fit()


def test_segment_processor_warns_for_ignored_and_unmatched_ranges():
    from ts_shape.errors import DataQualityWarning
    from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor

    t = pd.date_range("2024-01-01", periods=5, freq="min")
    df = pd.DataFrame({"systime": t, "uuid": "x", "value_double": range(5)})
    ranges = pd.DataFrame(
        {
            "segment_start": [t[0], pd.NaT],
            "segment_end": [t[2], t[4]],
            "segment_value": ["P", "Q"],
            "segment_index": [0, 1],
        }
    )
    with pytest.warns(DataQualityWarning, match="1 time range.*ignored"):
        out = SegmentProcessor.apply_ranges(df, ranges)
    assert set(out["segment_value"]) == {"P"}
    later = ranges.iloc[:1].assign(
        segment_start=pd.Timestamp("2025-01-01"),
        segment_end=pd.Timestamp("2025-01-02"),
    )
    with pytest.warns(DataQualityWarning, match="no data matched"):
        assert SegmentProcessor.apply_ranges(df, later).empty


def test_feature_tables_warn_when_nothing_to_compute():
    from ts_shape.errors import DataQualityWarning
    from ts_shape.features.export import FeatureMatrixExporter
    from ts_shape.features.segment_analysis.time_windowed_features import (
        TimeWindowedFeatureTable,
    )

    empty = pd.DataFrame(
        {
            "systime": pd.Series(dtype="datetime64[ns]"),
            "uuid": pd.Series(dtype=object),
            "value_double": pd.Series(dtype=float),
        }
    )
    with pytest.warns(DataQualityWarning, match="empty DataFrame"):
        TimeWindowedFeatureTable.compute_long(empty, segment_column=None)
    one = pd.DataFrame(
        {"systime": [pd.Timestamp("2024-01-01")], "uuid": ["s"], "value_double": [1.0]}
    )
    with pytest.warns(DataQualityWarning, match="no windows with sufficient data"):
        TimeWindowedFeatureTable.compute_long(one, segment_column=None)
    with pytest.warns(DataQualityWarning, match="no data to aggregate"):
        FeatureMatrixExporter.to_feature_matrix(empty, value_columns=["value_double"])


def test_cycle_validation_warns():
    from ts_shape.errors import DataQualityWarning
    from ts_shape.features.cycles.cycles_extractor import CycleExtractor

    t = pd.date_range("2024-01-01", periods=6, freq="min")
    df = pd.DataFrame({"systime": t, "uuid": "s", "value_bool": [True] * 6})
    ce = CycleExtractor(df, start_uuid="s")
    cycles = pd.DataFrame(
        {
            "cycle_start": [t[0], t[1]],
            "cycle_end": [t[3], t[5]],
            "cycle_uuid": ["a", "b"],
            "is_complete": [True, True],
        }
    )
    with pytest.warns(
        DataQualityWarning, match="validate_cycles: found 2 cycles longer"
    ):
        ce.validate_cycles(cycles, max_duration="1m")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ce.validate_cycles(cycles, max_duration="1m", warn=False)
    with pytest.warns(DataQualityWarning, match="1 overlapping cycle pairs"):
        ce.detect_overlapping_cycles(cycles)


# ---- deprecated legacy inputs ------------------------------------------------


def test_cycle_processor_old_attribute_names_warn():
    from ts_shape.features.cycles.cycle_processor import CycleDataProcessor

    cycles, values = _cycles_and_values()
    proc = CycleDataProcessor(cycles, values)
    with pytest.warns(DeprecationWarning, match="cycle_uuid_col is deprecated"):
        assert proc.cycle_uuid_col == "cycle_uuid"
    with pytest.warns(DeprecationWarning, match="systime_col is deprecated"):
        assert proc.systime_col == "systime"
    with pytest.warns(DeprecationWarning) as record:
        proc.cycle_uuid_col = "cycle_uuid"
    assert record[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert proc.cycle_uuid_column == "cycle_uuid"


def test_recipe_spec_duration_s_warns():
    from ts_shape.events.development.recipe_phase_adherence import (
        RecipePhaseAdherenceEvents,
    )

    t = pd.date_range("2024-01-01", periods=10, freq="min")
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "phase", "value_string": "heat"}),
            pd.DataFrame({"systime": t, "uuid": "temp", "value_double": range(10)}),
        ],
        ignore_index=True,
    )
    with pytest.warns(
        DeprecationWarning, match="'duration_s'.*'heat'.*'duration_seconds'"
    ):
        RecipePhaseAdherenceEvents(
            df, "phase", "temp", {"heat": {"duration_s": (0, 60)}}
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        RecipePhaseAdherenceEvents(
            df, "phase", "temp", {"heat": {"duration_seconds": (0, 60)}}
        )


def test_handover_downtime_minutes_input_warns():
    from ts_shape.events.production.shift_handover import ShiftHandoverEvents

    prod = pd.DataFrame({"date": ["2024-01-15"], "shift": ["A"], "quantity": [100]})
    minutes = pd.DataFrame(
        {"date": ["2024-01-15"], "shift": ["A"], "downtime_minutes": [12.0]}
    )
    with pytest.warns(
        DeprecationWarning, match="'downtime_minutes'.*'downtime_seconds'"
    ):
        out = ShiftHandoverEvents.from_shift_data(prod, downtime_df=minutes)
    assert out["downtime_seconds"].iloc[0] == 720.0
    seconds = minutes.rename(columns={"downtime_minutes": "downtime_seconds"})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        ShiftHandoverEvents.from_shift_data(prod, downtime_df=seconds)


def test_read_only_aliases_warn():
    from ts_shape.events.production.runtime_accounting import RuntimeAccountingEvents
    from ts_shape.events.quality.multi_sensor_validation import (
        MultiSensorValidationEvents,
    )

    t = pd.date_range("2024-01-01", periods=4, freq="min")
    run = pd.DataFrame({"systime": t, "uuid": "m", "value_bool": [True] * 4})
    with pytest.warns(DeprecationWarning, match="run_uuid is deprecated"):
        assert RuntimeAccountingEvents(run, run_state_uuid="m").run_uuid == "m"
    sensors = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": u, "value_double": 1.0})
            for u in ("a", "b")
        ],
        ignore_index=True,
    )
    msv = MultiSensorValidationEvents(sensors, ["a", "b"])
    with pytest.warns(DeprecationWarning, match="sensor_uuids is deprecated"):
        assert msv.sensor_uuids == ["a", "b"]
