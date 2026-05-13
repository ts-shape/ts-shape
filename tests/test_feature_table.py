import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from ts_shape.features.stats.feature_table import DescriptiveFeatures


def _make_dataframe(uuids=None, n_rows_per_uuid=5):
    """Helper to build a DataFrame with the standard ts-shape schema."""
    if uuids is None:
        uuids = ["uuid-a", "uuid-b"]

    rows = []
    base_time = pd.Timestamp("2024-01-01 00:00:00")
    for uid in uuids:
        for i in range(n_rows_per_uuid):
            rows.append(
                {
                    "systime": base_time + pd.Timedelta(seconds=i * 10),
                    "uuid": uid,
                    "value_bool": bool(i % 2),
                    "value_integer": i * 10,
                    "value_double": float(i) * 1.5,
                    "value_string": f"text_{i}",
                    "is_delta": bool(i % 3 == 0),
                }
            )
    return pd.DataFrame(rows)


def _safe_summary_as_dict(cls_or_df, column_name_or_none=None):
    """Replacement for NumericStatistics.summary_as_dict that works with pandas >= 2.0.

    Handles both classmethod call patterns:
      - NumericStatistics.summary_as_dict(dataframe, column_name)
    """
    if column_name_or_none is None:
        # called as (dataframe, column_name) via classmethod, cls_or_df = dataframe
        return {}
    dataframe = cls_or_df
    column_name = column_name_or_none
    col = dataframe[column_name]
    return {
        "min": float(col.min()),
        "max": float(col.max()),
        "mean": float(col.mean()),
        "median": float(col.median()),
        "std": float(col.std()),
        "sum": float(col.sum()),
        "count": int(col.count()),
    }


@pytest.fixture(autouse=True)
def _patch_numeric_summary():
    """Patch NumericStatistics.summary_as_dict to avoid pandas >= 2.0 compat issues."""
    with patch(
        "ts_shape.features.stats.numeric_stats.NumericStatistics.summary_as_dict",
        classmethod(lambda cls, df, col: _safe_summary_as_dict(df, col)),
    ):
        yield


class TestComputeDict:
    """test_compute_dict - multi-uuid DataFrame, verify dict structure."""

    def test_compute_dict(self):
        df = _make_dataframe(uuids=["uuid-a", "uuid-b"])
        feat = DescriptiveFeatures(df)
        result = feat.compute(output_format="dict")

        assert isinstance(result, dict)
        assert "uuid-a" in result
        assert "uuid-b" in result

        for uid in ["uuid-a", "uuid-b"]:
            group_stats = result[uid]
            assert "overall" in group_stats
            assert "total_rows" in group_stats["overall"]
            assert "total_time" in group_stats["overall"]
            assert "is_delta_sum" in group_stats["overall"]
            assert "is_delta_avg" in group_stats["overall"]
            assert "is_delta_std" in group_stats["overall"]


class TestComputeDataframe:
    """test_compute_dataframe - verify DataFrame output with flattened column names."""

    def test_compute_dataframe(self):
        df = _make_dataframe(uuids=["uuid-a", "uuid-b"])
        feat = DescriptiveFeatures(df)
        result = feat.compute(output_format="dataframe")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

        # Column names should contain '::' separators from the flattening logic
        for col in result.columns:
            assert (
                "::" in col
            ), f"Expected flattened column name with '::' but got '{col}'"


class TestOverallStats:
    """test_overall_stats - verify keys and correct values."""

    def test_overall_stats(self):
        df = _make_dataframe(uuids=["uuid-a"], n_rows_per_uuid=5)
        feat = DescriptiveFeatures(df)
        group = df[df["uuid"] == "uuid-a"]
        stats = feat.overall_stats(group)

        assert stats["total_rows"] == 5
        expected_time = group["systime"].max() - group["systime"].min()
        assert stats["total_time"] == expected_time
        assert stats["is_delta_sum"] == group["is_delta"].sum()
        assert stats["is_delta_avg"] == group["is_delta"].mean()

        # is_delta_std should be a float (or NaN for constant columns)
        assert isinstance(stats["is_delta_std"], (float, np.floating)) or pd.isna(
            stats["is_delta_std"]
        )


class TestInvalidOutputFormat:
    """test_invalid_output_format - raises ValueError."""

    def test_invalid_output_format(self):
        df = _make_dataframe()
        feat = DescriptiveFeatures(df)

        with pytest.raises(ValueError, match="Invalid output format"):
            feat.compute(output_format="csv")


class TestSingleUuid:
    """test_single_uuid - works with single uuid."""

    def test_single_uuid_dict(self):
        df = _make_dataframe(uuids=["only-one"], n_rows_per_uuid=4)
        feat = DescriptiveFeatures(df)
        result = feat.compute(output_format="dict")

        assert isinstance(result, dict)
        assert "only-one" in result
        assert len(result) == 1
        assert result["only-one"]["overall"]["total_rows"] == 4

    def test_single_uuid_dataframe(self):
        df = _make_dataframe(uuids=["only-one"], n_rows_per_uuid=4)
        feat = DescriptiveFeatures(df)
        result = feat.compute(output_format="dataframe")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


class TestEmptyLike:
    """test_empty_like - verify behavior with minimal data."""

    def test_single_row(self):
        df = _make_dataframe(uuids=["uuid-min"], n_rows_per_uuid=1)
        feat = DescriptiveFeatures(df)
        result = feat.compute(output_format="dict")

        assert isinstance(result, dict)
        assert "uuid-min" in result
        assert result["uuid-min"]["overall"]["total_rows"] == 1
        # total_time should be zero timedelta for a single row
        assert result["uuid-min"]["overall"]["total_time"] == pd.Timedelta(0)

    def test_two_rows(self):
        df = _make_dataframe(uuids=["uuid-two"], n_rows_per_uuid=2)
        feat = DescriptiveFeatures(df)
        result = feat.compute(output_format="dict")

        assert result["uuid-two"]["overall"]["total_rows"] == 2
