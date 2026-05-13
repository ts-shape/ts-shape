import pytest
import pandas as pd
import numpy as np
from ts_shape.transform.harmonization import DataHarmonizer


@pytest.fixture
def long_format_df():
    """Create a multi-UUID long-format DataFrame with irregular timestamps."""
    np.random.seed(42)
    times_a = pd.date_range("2024-01-01", periods=100, freq="1s")
    times_b = pd.date_range("2024-01-01 00:00:00.500", periods=100, freq="1s")
    times_c = pd.date_range("2024-01-01", periods=100, freq="2s")

    rows = []
    for t in times_a:
        rows.append(
            {
                "systime": t,
                "uuid": "sensor_a",
                "value_double": np.sin(t.timestamp() % 60),
            }
        )
    for t in times_b:
        rows.append(
            {
                "systime": t,
                "uuid": "sensor_b",
                "value_double": np.cos(t.timestamp() % 60),
            }
        )
    for t in times_c:
        rows.append(
            {"systime": t, "uuid": "sensor_c", "value_double": np.random.randn()}
        )

    return pd.DataFrame(rows)


@pytest.fixture
def gapped_df():
    """Create a DataFrame with intentional time gaps."""
    rows = []
    # First segment: 0-10s
    for i in range(10):
        t = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=i)
        rows.append({"systime": t, "uuid": "sensor_a", "value_double": float(i)})
    # Gap of 30 seconds
    # Second segment: 40-50s
    for i in range(10):
        t = pd.Timestamp("2024-01-01") + pd.Timedelta(seconds=40 + i)
        rows.append({"systime": t, "uuid": "sensor_a", "value_double": float(40 + i)})
    return pd.DataFrame(rows)


class TestPivotToWide:
    def test_pivot_produces_correct_columns(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        wide = harmonizer.pivot_to_wide()
        assert set(wide.columns) == {"sensor_a", "sensor_b", "sensor_c"}

    def test_pivot_index_is_datetime(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        wide = harmonizer.pivot_to_wide()
        assert isinstance(wide.index, pd.DatetimeIndex)


class TestResampleToUniform:
    def test_uniform_frequency(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        resampled = harmonizer.resample_to_uniform(freq="1s")
        diffs = pd.Series(resampled.index).diff().dropna()
        assert (diffs == pd.Timedelta("1s")).all()

    def test_interpolation_fills_nans(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        resampled = harmonizer.resample_to_uniform(freq="1s")
        # Interior values should mostly be filled
        interior = resampled.iloc[1:-1]
        nan_ratio = interior.isna().sum().sum() / interior.size
        assert nan_ratio < 0.5


class TestDetectGaps:
    def test_detect_gaps_above_threshold(self, gapped_df):
        harmonizer = DataHarmonizer(gapped_df)
        gaps = harmonizer.detect_gaps(threshold="10s")
        assert len(gaps) == 1
        assert gaps.iloc[0]["uuid"] == "sensor_a"
        assert gaps.iloc[0]["gap_duration"] > pd.Timedelta("10s")

    def test_detect_gaps_below_threshold(self, gapped_df):
        harmonizer = DataHarmonizer(gapped_df)
        gaps = harmonizer.detect_gaps(threshold="60s")
        assert len(gaps) == 0

    def test_detect_gaps_empty_result(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        gaps = harmonizer.detect_gaps(threshold="100s")
        assert len(gaps) == 0


class TestFillGaps:
    def test_fill_interpolate(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        filled = harmonizer.fill_gaps(strategy="interpolate")
        assert isinstance(filled, pd.DataFrame)

    def test_fill_constant(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        filled = harmonizer.fill_gaps(strategy="constant", fill_value=-999.0)
        # Check that any previously NaN values are now -999.0
        wide = harmonizer.pivot_to_wide()
        nan_positions = wide.isna()
        if nan_positions.any().any():
            for col in wide.columns:
                mask = nan_positions[col]
                if mask.any():
                    assert (filled.loc[mask, col] == -999.0).all()

    def test_fill_ffill(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        filled = harmonizer.fill_gaps(strategy="ffill")
        assert isinstance(filled, pd.DataFrame)

    def test_fill_invalid_strategy(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        with pytest.raises(ValueError):
            harmonizer.fill_gaps(strategy="invalid")


class TestAlignAsof:
    def test_align_produces_both_values(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        aligned = harmonizer.align_asof("sensor_a", "sensor_b", tolerance="2s")
        assert "value_left" in aligned.columns
        assert "value_right" in aligned.columns
        assert len(aligned) > 0

    def test_align_tight_tolerance(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        aligned = harmonizer.align_asof("sensor_a", "sensor_b", tolerance="100ms")
        # With 500ms offset and 100ms tolerance, most matches should be NaN
        nan_count = aligned["value_right"].isna().sum()
        assert nan_count > 0


class TestMergeMultiSignals:
    def test_merge_all_signals(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        merged = harmonizer.merge_multi_signals()
        assert set(merged.columns) == {"sensor_a", "sensor_b", "sensor_c"}

    def test_merge_selected_uuids(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        merged = harmonizer.merge_multi_signals(uuids=["sensor_a", "sensor_b"])
        assert set(merged.columns) == {"sensor_a", "sensor_b"}

    def test_merge_with_resample(self, long_format_df):
        harmonizer = DataHarmonizer(long_format_df)
        merged = harmonizer.merge_multi_signals(freq="2s")
        diffs = pd.Series(merged.index).diff().dropna()
        assert (diffs == pd.Timedelta("2s")).all()
