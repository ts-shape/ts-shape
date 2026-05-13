import pytest
import pandas as pd
import numpy as np
from ts_shape.events.quality.data_gap_analysis import DataGapAnalysisEvents


@pytest.fixture
def gapped_df():
    """Signal with three segments separated by two gaps (30s and 120s)."""
    rows = []
    base = pd.Timestamp("2024-01-01")
    # Segment 1: 0-99s  (100 samples at 1 Hz)
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.sin(i / 10.0),
            }
        )
    # Gap of 30s (100s to 130s missing)
    # Segment 2: 130-199s  (70 samples)
    for i in range(70):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=130 + i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.sin((130 + i) / 10.0),
            }
        )
    # Gap of 120s (200s to 320s missing)
    # Segment 3: 320-419s  (100 samples)
    for i in range(100):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=320 + i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.sin((320 + i) / 10.0),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def continuous_df():
    """Perfectly continuous 1 Hz signal with no gaps."""
    base = pd.Timestamp("2024-01-01")
    rows = [
        {
            "systime": base + pd.Timedelta(seconds=i),
            "uuid": "sensor_1",
            "value_double": float(i),
        }
        for i in range(200)
    ]
    return pd.DataFrame(rows)


class TestFindGaps:
    def test_finds_two_gaps(self, gapped_df):
        det = DataGapAnalysisEvents(gapped_df, "sensor_1")
        gaps = det.find_gaps(min_gap="5s")
        assert len(gaps) == 2
        durations = gaps["gap_duration_seconds"].tolist()
        assert any(d >= 29 for d in durations)
        assert any(d >= 119 for d in durations)

    def test_min_gap_filters(self, gapped_df):
        det = DataGapAnalysisEvents(gapped_df, "sensor_1")
        gaps = det.find_gaps(min_gap="60s")
        assert len(gaps) == 1
        assert gaps.iloc[0]["gap_duration_seconds"] >= 119

    def test_no_gaps(self, continuous_df):
        det = DataGapAnalysisEvents(continuous_df, "sensor_1")
        gaps = det.find_gaps(min_gap="5s")
        assert len(gaps) == 0

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = DataGapAnalysisEvents(df, "sensor_1")
        assert len(det.find_gaps()) == 0


class TestGapSummary:
    def test_summary_values(self, gapped_df):
        det = DataGapAnalysisEvents(gapped_df, "sensor_1")
        summary = det.gap_summary(min_gap="5s")
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["total_gaps"] == 2
        assert row["longest_gap_seconds"] >= 119
        assert row["shortest_gap_seconds"] >= 29
        assert 0 < row["gap_fraction"] < 1

    def test_no_gaps_summary(self, continuous_df):
        det = DataGapAnalysisEvents(continuous_df, "sensor_1")
        summary = det.gap_summary(min_gap="5s")
        assert summary.iloc[0]["total_gaps"] == 0
        assert summary.iloc[0]["gap_fraction"] == 0.0


class TestCoverageByPeriod:
    def test_coverage_below_100(self, gapped_df):
        det = DataGapAnalysisEvents(gapped_df, "sensor_1")
        cov = det.coverage_by_period(freq="5min")
        assert len(cov) > 0
        # At least one window should have gaps
        assert any(cov["gap_count"] > 0)

    def test_full_coverage(self, continuous_df):
        det = DataGapAnalysisEvents(continuous_df, "sensor_1")
        cov = det.coverage_by_period(freq="5min")
        assert len(cov) > 0
        # Continuous 1 Hz signal should have ~100% coverage
        assert all(cov["coverage_pct"] > 99.0)


class TestInterpolationCandidates:
    def test_short_gap_is_candidate(self, gapped_df):
        det = DataGapAnalysisEvents(gapped_df, "sensor_1")
        # 30s gap should be a candidate at max_gap=60s
        cands = det.interpolation_candidates(max_gap="60s", min_gap="5s")
        assert len(cands) == 1
        assert cands.iloc[0]["gap_duration_seconds"] >= 29
        # Smooth sine signal — should be safe to interpolate
        assert cands.iloc[0]["safe_to_interpolate"] == True

    def test_large_max_gap_returns_both(self, gapped_df):
        det = DataGapAnalysisEvents(gapped_df, "sensor_1")
        cands = det.interpolation_candidates(max_gap="300s", min_gap="5s")
        assert len(cands) == 2

    def test_empty_signal(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        det = DataGapAnalysisEvents(df, "sensor_1")
        assert len(det.interpolation_candidates()) == 0
