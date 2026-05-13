import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import pytest

from ts_shape.events.maintenance import (
    DegradationDetectionEvents,
    FailurePredictionEvents,
    VibrationAnalysisEvents,
)

# ---------------------------------------------------------------------------
# Helpers for synthetic manufacturing data
# ---------------------------------------------------------------------------


def _make_df(times, values, uuid="sig1"):
    """Build a standard ts-shape DataFrame from times and values."""
    return pd.DataFrame(
        {
            "systime": times,
            "uuid": [uuid] * len(times),
            "value_double": values,
            "is_delta": [True] * len(times),
        }
    )


def _empty_df():
    """Return an empty DataFrame with standard columns."""
    return pd.DataFrame(
        {
            "systime": pd.Series(dtype="datetime64[ns]"),
            "uuid": pd.Series(dtype="str"),
            "value_double": pd.Series(dtype="float64"),
            "is_delta": pd.Series(dtype="bool"),
        }
    )


# ===================================================================
# DegradationDetectionEvents
# ===================================================================


class TestDegradationDetectionEvents:

    def test_detect_trend_degradation_decreasing(self):
        """Linear downward signal should produce at least one degradation event."""
        t = pd.date_range("2024-01-01", periods=120, freq="1min")
        # Steady decline: 100 -> ~40 over 2 hours
        values = 100.0 - np.arange(120) * 0.5
        df = _make_df(t, values)

        det = DegradationDetectionEvents(df, signal_uuid="sig1")
        result = det.detect_trend_degradation(
            window="30min", min_slope=0.0001, direction="decreasing"
        )
        assert not result.empty
        assert "avg_slope" in result.columns
        assert (result["avg_slope"] < 0).all()
        assert result["uuid"].iloc[0] == "maint:degradation"
        assert result["is_delta"].all()

    def test_detect_trend_degradation_increasing(self):
        """Linear upward signal flagged when direction='increasing'."""
        t = pd.date_range("2024-01-01", periods=60, freq="1min")
        values = np.arange(60) * 0.3
        df = _make_df(t, values)

        det = DegradationDetectionEvents(df, signal_uuid="sig1")
        result = det.detect_trend_degradation(
            window="20min", min_slope=0.0001, direction="increasing"
        )
        assert not result.empty
        assert (result["avg_slope"] > 0).all()

    def test_detect_variance_increase(self):
        """Signal with increasing noise should be flagged."""
        t = pd.date_range("2024-01-01", periods=120, freq="1min")
        np.random.seed(42)
        # First 60 points: low noise, last 60: high noise
        low_noise = 50.0 + np.random.normal(0, 0.5, 60)
        high_noise = 50.0 + np.random.normal(0, 5.0, 60)
        values = np.concatenate([low_noise, high_noise])
        df = _make_df(t, values)

        det = DegradationDetectionEvents(df, signal_uuid="sig1")
        result = det.detect_variance_increase(window="30min", threshold_factor=2.0)
        assert not result.empty
        assert "ratio" in result.columns
        assert (result["ratio"] >= 2.0).all()

    def test_detect_level_shift(self):
        """Signal with a step change should produce a level-shift event."""
        t = pd.date_range("2024-01-01", periods=120, freq="1min")
        np.random.seed(0)
        # Stable at 50 for 60 points, then jumps to 70
        values = np.concatenate(
            [
                50.0 + np.random.normal(0, 0.5, 60),
                70.0 + np.random.normal(0, 0.5, 60),
            ]
        )
        df = _make_df(t, values)

        det = DegradationDetectionEvents(df, signal_uuid="sig1")
        result = det.detect_level_shift(min_shift=10.0, hold="5min")
        assert not result.empty
        assert "shift_magnitude" in result.columns
        assert abs(result["shift_magnitude"].iloc[0]) >= 10.0

    def test_health_score_bounds(self):
        """Health score must be between 0 and 100."""
        t = pd.date_range("2024-01-01", periods=200, freq="1min")
        np.random.seed(7)
        # Degrading signal: increasing noise + downward drift
        values = (
            100.0
            - np.arange(200) * 0.2
            + np.random.normal(0, 1, 200) * np.linspace(1, 5, 200)
        )
        df = _make_df(t, values)

        det = DegradationDetectionEvents(df, signal_uuid="sig1")
        result = det.health_score(window="30min", baseline_window="1h")
        assert not result.empty
        assert (result["health_score"] >= 0).all()
        assert (result["health_score"] <= 100).all()
        assert "mean_drift_pct" in result.columns
        assert "variance_ratio" in result.columns
        assert "trend_slope" in result.columns

    def test_empty_dataframe_degradation(self):
        """All methods should handle empty DataFrames gracefully."""
        df = _empty_df()
        det = DegradationDetectionEvents(df, signal_uuid="sig1")

        assert det.detect_trend_degradation().empty
        assert det.detect_variance_increase().empty
        assert det.detect_level_shift(min_shift=1.0).empty
        assert det.health_score().empty


# ===================================================================
# FailurePredictionEvents
# ===================================================================


class TestFailurePredictionEvents:

    def test_remaining_useful_life_linear_signal(self):
        """Linearly increasing signal should give plausible RUL estimates."""
        t = pd.date_range("2024-01-01", periods=60, freq="1min")
        # Signal rising from 0 toward threshold 100 at ~1 unit/min
        values = np.arange(60, dtype=float)
        df = _make_df(t, values)

        fpe = FailurePredictionEvents(df, signal_uuid="sig1")
        result = fpe.remaining_useful_life(
            degradation_rate=1.0 / 60, failure_threshold=100.0
        )
        assert not result.empty
        assert "rul_seconds" in result.columns
        assert "rul_hours" in result.columns
        assert "confidence" in result.columns
        # At the last point (value ~59), RUL should be roughly (100-59)/rate seconds
        last = result.iloc[-1]
        assert last["rul_seconds"] is not None
        assert last["rul_seconds"] > 0

    def test_detect_exceedance_pattern(self):
        """Signal with increasing exceedances should show escalation."""
        t = pd.date_range("2024-01-01", periods=120, freq="1min")
        np.random.seed(3)
        # First hour: mostly below 80, second hour: many above 80/90
        values = np.concatenate(
            [
                70 + np.random.normal(0, 3, 60),
                85 + np.random.normal(0, 5, 60),
            ]
        )
        df = _make_df(t, values)

        fpe = FailurePredictionEvents(df, signal_uuid="sig1")
        result = fpe.detect_exceedance_pattern(
            warning_threshold=80.0, critical_threshold=90.0, window="1h"
        )
        assert not result.empty
        assert "warning_count" in result.columns
        assert "critical_count" in result.columns
        assert "escalation_detected" in result.columns

    def test_time_to_threshold_increasing(self):
        """Increasing signal should give positive time-to-threshold estimates."""
        t = pd.date_range("2024-01-01", periods=30, freq="1min")
        values = 10.0 + np.arange(30) * 2.0  # 10, 12, 14, ..., 68
        df = _make_df(t, values)

        fpe = FailurePredictionEvents(df, signal_uuid="sig1")
        result = fpe.time_to_threshold(threshold=100.0, direction="increasing")
        assert not result.empty
        assert "rate_of_change" in result.columns
        assert "estimated_time_seconds" in result.columns
        # Later points should have shorter time to threshold
        last = result.iloc[-1]
        first_with_est = result.dropna(subset=["estimated_time_seconds"])
        if not first_with_est.empty:
            assert first_with_est.iloc[-1]["estimated_time_seconds"] >= 0

    def test_time_to_threshold_decreasing(self):
        """Decreasing signal with direction='decreasing'."""
        t = pd.date_range("2024-01-01", periods=30, freq="1min")
        values = 100.0 - np.arange(30) * 2.0
        df = _make_df(t, values)

        fpe = FailurePredictionEvents(df, signal_uuid="sig1")
        result = fpe.time_to_threshold(threshold=20.0, direction="decreasing")
        assert not result.empty

    def test_empty_dataframe_failure(self):
        """All methods should handle empty DataFrames gracefully."""
        df = _empty_df()
        fpe = FailurePredictionEvents(df, signal_uuid="sig1")

        assert fpe.remaining_useful_life(0.01, 100).empty
        assert fpe.detect_exceedance_pattern(80, 90).empty
        assert fpe.time_to_threshold(100).empty


# ===================================================================
# VibrationAnalysisEvents
# ===================================================================


class TestVibrationAnalysisEvents:

    def test_detect_rms_exceedance_sinusoidal(self):
        """Sinusoidal signal with amplitude growth should trigger RMS exceedance."""
        t = pd.date_range("2024-01-01", periods=600, freq="1s")
        # First 300s: amplitude 1, next 300s: amplitude 5
        phase = np.arange(600) * 2 * np.pi / 20  # 20-second period
        amp = np.concatenate([np.ones(300), np.ones(300) * 5.0])
        values = amp * np.sin(phase)
        df = _make_df(t, values)

        vae = VibrationAnalysisEvents(df, signal_uuid="sig1")
        # Baseline RMS for amplitude 1 sine: ~0.707
        result = vae.detect_rms_exceedance(
            baseline_rms=0.707, threshold_factor=2.0, window="30s"
        )
        assert not result.empty
        assert "rms_value" in result.columns
        assert "ratio" in result.columns
        assert result["is_delta"].all()

    def test_detect_amplitude_growth(self):
        """Signal with growing amplitude should be flagged."""
        t = pd.date_range("2024-01-01", periods=600, freq="1s")
        phase = np.arange(600) * 2 * np.pi / 10
        # Amplitude grows linearly
        amp = 1.0 + np.arange(600) * 0.01
        values = amp * np.sin(phase)
        df = _make_df(t, values)

        vae = VibrationAnalysisEvents(df, signal_uuid="sig1")
        result = vae.detect_amplitude_growth(window="1min", growth_threshold=0.1)
        assert not result.empty
        assert "amplitude" in result.columns
        assert "growth_pct" in result.columns
        # Later windows should show positive growth
        assert result["growth_pct"].iloc[-1] > 0

    def test_bearing_health_indicators(self):
        """Sinusoidal vibration should produce valid bearing indicators."""
        t = pd.date_range("2024-01-01", periods=600, freq="1s")
        phase = np.arange(600) * 2 * np.pi / 10
        values = 2.0 * np.sin(phase)
        df = _make_df(t, values)

        vae = VibrationAnalysisEvents(df, signal_uuid="sig1")
        result = vae.bearing_health_indicators(window="1min")
        assert not result.empty
        assert "rms" in result.columns
        assert "peak" in result.columns
        assert "crest_factor" in result.columns
        assert "kurtosis" in result.columns
        # Crest factor for pure sine = sqrt(2) ~ 1.414
        for _, row in result.iterrows():
            assert row["rms"] > 0
            assert row["peak"] > 0
            assert row["crest_factor"] > 0

    def test_bearing_health_impulsive_signal(self):
        """Impulsive signal should show elevated kurtosis."""
        t = pd.date_range("2024-01-01", periods=300, freq="1s")
        np.random.seed(99)
        # Background noise + occasional spikes (bearing defect simulation)
        values = np.random.normal(0, 0.5, 300)
        # Add spikes every ~30 samples
        values[::30] += 10.0
        df = _make_df(t, values)

        vae = VibrationAnalysisEvents(df, signal_uuid="sig1")
        result = vae.bearing_health_indicators(window="2min")
        assert not result.empty
        # Kurtosis of impulsive signal should be significantly positive
        assert (result["kurtosis"] > 0).all()

    def test_empty_dataframe_vibration(self):
        """All methods should handle empty DataFrames gracefully."""
        df = _empty_df()
        vae = VibrationAnalysisEvents(df, signal_uuid="sig1")

        assert vae.detect_rms_exceedance(1.0).empty
        assert vae.detect_amplitude_growth().empty
        assert vae.bearing_health_indicators().empty


# ===================================================================
# Cross-class: custom event_uuid propagation
# ===================================================================


class TestCustomEventUuid:

    def test_custom_event_uuid_propagates(self):
        """Custom event_uuid should appear in output rows."""
        t = pd.date_range("2024-01-01", periods=60, freq="1min")
        values = np.arange(60, dtype=float)
        df = _make_df(t, values)

        det = DegradationDetectionEvents(
            df, signal_uuid="sig1", event_uuid="custom:deg"
        )
        result = det.detect_trend_degradation(
            window="20min", min_slope=0.0, direction="increasing"
        )
        if not result.empty:
            assert (result["uuid"] == "custom:deg").all()

        fpe = FailurePredictionEvents(df, signal_uuid="sig1", event_uuid="custom:fail")
        result = fpe.remaining_useful_life(0.01, 100)
        if not result.empty:
            assert (result["uuid"] == "custom:fail").all()
