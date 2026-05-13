"""Edge-case and data-quality tests for Base and FeatureMatrixExporter."""

import warnings

import pandas as pd
import numpy as np
import pytest

from ts_shape.utils.base import Base
from ts_shape.errors import DataQualityWarning
from ts_shape.features.export import FeatureMatrixExporter

# ---------------------------------------------------------------------------
# Base — empty / single-row / all-NaN / duplicate-timestamp inputs
# ---------------------------------------------------------------------------


class ConcreteBase(Base):
    """Minimal concrete subclass so we can instantiate Base."""


def test_base_accepts_empty_dataframe(empty_df):
    obj = ConcreteBase(empty_df)
    assert obj.dataframe.empty


def test_base_accepts_single_row(single_row_df):
    obj = ConcreteBase(single_row_df)
    assert len(obj.dataframe) == 1


def test_base_raises_on_non_dataframe():
    with pytest.raises(TypeError):
        ConcreteBase([1, 2, 3])


def test_base_warns_on_all_nan_columns(all_nan_df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ConcreteBase(all_nan_df)
    dq_warnings = [w for w in caught if issubclass(w.category, DataQualityWarning)]
    assert dq_warnings, "Expected a DataQualityWarning for all-NaN columns"
    assert any("NaN" in str(w.message) for w in dq_warnings)


def test_base_warns_on_duplicate_timestamps(duplicate_timestamps_df):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ConcreteBase(duplicate_timestamps_df)
    dq_warnings = [w for w in caught if issubclass(w.category, DataQualityWarning)]
    assert dq_warnings, "Expected a DataQualityWarning for duplicate timestamps"
    assert any("duplicate" in str(w.message).lower() for w in dq_warnings)


def test_base_no_warning_on_clean_data():
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "uuid": ["s1", "s2"],
            "value_double": [1.0, 2.0],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ConcreteBase(df)
    dq_warnings = [w for w in caught if issubclass(w.category, DataQualityWarning)]
    assert not dq_warnings


# ---------------------------------------------------------------------------
# FeatureMatrixExporter
# ---------------------------------------------------------------------------


def _make_signal_df():
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 00:01",
                    "2024-01-01 00:02",
                    "2024-01-01 00:03",
                    "2024-01-01 00:04",
                    "2024-01-01 00:05",
                ]
            ),
            "uuid": ["temp", "temp", "temp", "pressure", "pressure", "pressure"],
            "value_double": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        }
    )


def test_feature_matrix_basic():
    df = _make_signal_df()
    matrix = FeatureMatrixExporter.to_feature_matrix(df)
    assert not matrix.empty
    # Should have columns for both UUIDs
    assert any("temp" in c for c in matrix.columns)
    assert any("pressure" in c for c in matrix.columns)


def test_feature_matrix_column_naming():
    df = _make_signal_df()
    matrix = FeatureMatrixExporter.to_feature_matrix(
        df, value_cols=["value_double"], agg_funcs={"mean": np.mean}
    )
    assert "temp__value_double__mean" in matrix.columns
    assert "pressure__value_double__mean" in matrix.columns


def test_feature_matrix_values_correct():
    df = _make_signal_df()
    matrix = FeatureMatrixExporter.to_feature_matrix(
        df, value_cols=["value_double"], agg_funcs={"mean": np.mean}
    )
    assert matrix.loc[0, "temp__value_double__mean"] == pytest.approx(20.0)
    assert matrix.loc[0, "pressure__value_double__mean"] == pytest.approx(200.0)


def test_feature_matrix_with_group_col():
    df = _make_signal_df()
    df["cycle"] = ["c1", "c1", "c2", "c1", "c1", "c2"]
    matrix = FeatureMatrixExporter.to_feature_matrix(
        df, value_cols=["value_double"], group_col="cycle", agg_funcs={"mean": np.mean}
    )
    assert set(matrix.index) == {"c1", "c2"}


def test_feature_matrix_missing_uuid_col_raises():
    df = pd.DataFrame({"value_double": [1.0, 2.0]})
    with pytest.raises(ValueError, match="uuid_col"):
        FeatureMatrixExporter.to_feature_matrix(df)


def test_feature_matrix_missing_value_col_raises():
    df = _make_signal_df()
    with pytest.raises(ValueError, match="value_cols not found"):
        FeatureMatrixExporter.to_feature_matrix(df, value_cols=["nonexistent"])


def test_feature_matrix_empty_df_returns_empty():
    df = pd.DataFrame(
        {"uuid": pd.Series(dtype=str), "value_double": pd.Series(dtype=float)}
    )
    matrix = FeatureMatrixExporter.to_feature_matrix(df, value_cols=["value_double"])
    assert matrix.empty


# ---------------------------------------------------------------------------
# CycleExtractor — edge cases
# ---------------------------------------------------------------------------

from ts_shape.features.cycles.cycles_extractor import CycleExtractor


def make_cycle_df():
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(
                [
                    "2023-01-01 00:00:00",
                    "2023-01-01 00:05:00",
                    "2023-01-01 00:10:00",
                    "2023-01-01 00:15:00",
                    "2023-01-01 00:20:00",
                ]
            ),
            "value_bool": [True, True, False, True, False],
            "value_integer": [0, 0, 1, 0, 1],
            "value_double": [0.0, 0.0, 0.0, 0.0, 0.0],
            "value_string": ["", "", "", "", ""],
        }
    )


def test_cycle_extractor_empty_df_returns_empty():
    df = pd.DataFrame(
        {
            "systime": pd.Series(dtype="datetime64[ns]"),
            "value_bool": pd.Series(dtype="bool"),
            "value_integer": pd.Series(dtype="int64"),
            "value_double": pd.Series(dtype="float64"),
            "value_string": pd.Series(dtype="str"),
        }
    )
    ce = CycleExtractor(df, start_uuid="start")
    result = ce.process_persistent_cycle()
    assert result.empty


def test_cycle_extractor_no_ends_marks_incomplete():
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "value_bool": [True, True],  # only starts, no False ends
            "value_integer": [0, 0],
            "value_double": [0.0, 0.0],
            "value_string": ["", ""],
        }
    )
    ce = CycleExtractor(df, start_uuid="start")
    result = ce.process_persistent_cycle()
    assert result["is_complete"].all() == False


def test_cycle_extractor_stats_populated():
    df = make_cycle_df()
    ce = CycleExtractor(df, start_uuid="start")
    ce.process_persistent_cycle()
    stats = ce.get_extraction_stats()
    assert stats["total_cycles"] >= 1
    assert "success_rate" in stats
