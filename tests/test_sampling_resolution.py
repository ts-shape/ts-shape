import numpy as np
import pandas as pd
import pytest

from ts_shape.events.quality.sampling_resolution import SamplingResolutionEvents


@pytest.fixture
def quantized_signal_df():
    """Signal quantized to 0.1 steps (simulating low-resolution A/D)."""
    np.random.seed(42)
    n = 600
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        # Continuous value rounded to nearest 0.1
        raw = 50.0 + np.random.randn() * 2.0
        quantized = round(raw * 10) / 10  # 0.1 step quantization
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": quantized,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def continuous_signal_df():
    """High-resolution continuous signal (no quantization)."""
    np.random.seed(42)
    n = 600
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.random.randn() * 0.5,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def regular_sampling_df():
    """Signal with perfectly regular 1-second sampling."""
    np.random.seed(42)
    n = 300
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.random.randn() * 0.1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def jittery_sampling_df():
    """Signal with irregular sampling (high jitter)."""
    np.random.seed(42)
    n = 300
    base = pd.Timestamp("2024-01-01")
    rows = []
    t = 0
    for i in range(n):
        interval = 1000 + np.random.randint(-400, 400)  # 600-1400ms intervals
        t += max(100, interval)
        rows.append(
            {
                "systime": base + pd.Timedelta(milliseconds=t),
                "uuid": "sensor_1",
                "value_double": 50.0 + np.random.randn() * 0.1,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def high_freq_signal_df():
    """Signal with fast oscillation sampled at 1 Hz (aliasing scenario)."""
    np.random.seed(42)
    n = 300
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        # 0.4 Hz oscillation sampled at 1 Hz (close to Nyquist)
        val = 50.0 + 2.0 * np.sin(2 * np.pi * 0.4 * i)
        rows.append(
            {
                "systime": base + pd.Timedelta(seconds=i),
                "uuid": "sensor_1",
                "value_double": val,
            }
        )
    return pd.DataFrame(rows)


class TestDetectQuantization:
    def test_quantized_signal(self, quantized_signal_df):
        sr = SamplingResolutionEvents(quantized_signal_df, "sensor_1")
        result = sr.detect_quantization(window="5min")
        assert len(result) > 0
        # Quantization step should be close to 0.1
        first = result.iloc[0]
        assert 0.05 <= first["quantization_step"] <= 0.15

    def test_continuous_not_quantized(self, continuous_signal_df):
        sr = SamplingResolutionEvents(continuous_signal_df, "sensor_1")
        result = sr.detect_quantization(window="5min")
        assert len(result) > 0
        # Continuous signal should not be flagged as quantized
        assert not result.iloc[0]["quantized"]

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        sr = SamplingResolutionEvents(df, "sensor_1")
        assert len(sr.detect_quantization()) == 0


class TestEffectiveResolution:
    def test_quantized_signal(self, quantized_signal_df):
        sr = SamplingResolutionEvents(quantized_signal_df, "sensor_1")
        result = sr.effective_resolution(window="5min")
        assert len(result) > 0
        first = result.iloc[0]
        assert first["theoretical_resolution"] > 0
        assert first["assessment"] in ("adequate", "degraded", "insufficient")

    def test_continuous_signal(self, continuous_signal_df):
        sr = SamplingResolutionEvents(continuous_signal_df, "sensor_1")
        result = sr.effective_resolution(window="5min")
        assert len(result) > 0

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        sr = SamplingResolutionEvents(df, "sensor_1")
        assert len(sr.effective_resolution()) == 0


class TestDetectAliasing:
    def test_high_frequency(self, high_freq_signal_df):
        sr = SamplingResolutionEvents(high_freq_signal_df, "sensor_1")
        result = sr.detect_aliasing(window="5min")
        assert len(result) > 0
        first = result.iloc[0]
        assert first["sampling_rate_hz"] > 0
        assert first["nyquist_hz"] > 0
        # 0.4 Hz signal at 1 Hz sampling → 0.4/0.5 = 80% of Nyquist → high risk
        assert first["aliasing_risk"] in ("medium", "high")

    def test_with_expected_frequency(self, regular_sampling_df):
        sr = SamplingResolutionEvents(regular_sampling_df, "sensor_1")
        # 0.1 Hz expected at 1 Hz sampling → safe
        result = sr.detect_aliasing(window="5min", expected_frequency=0.1)
        assert len(result) > 0
        assert result.iloc[0]["aliasing_risk"] == "none"

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        sr = SamplingResolutionEvents(df, "sensor_1")
        assert len(sr.detect_aliasing()) == 0


class TestSamplingJitter:
    def test_regular_sampling(self, regular_sampling_df):
        sr = SamplingResolutionEvents(regular_sampling_df, "sensor_1")
        result = sr.sampling_jitter(window="5min")
        assert len(result) > 0
        first = result.iloc[0]
        # Regular 1-second sampling should have low jitter
        assert first["jitter_pct"] < 5.0
        assert first["regularity_score"] > 80

    def test_jittery_sampling(self, jittery_sampling_df):
        sr = SamplingResolutionEvents(jittery_sampling_df, "sensor_1")
        result = sr.sampling_jitter(window="5min")
        assert len(result) > 0
        first = result.iloc[0]
        # High jitter should be detected
        assert first["jitter_pct"] > 10.0
        assert first["regularity_score"] < 90

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        sr = SamplingResolutionEvents(df, "sensor_1")
        assert len(sr.sampling_jitter()) == 0
