import pytest
import pandas as pd
import numpy as np
from ts_shape.events.quality.multi_sensor_validation import MultiSensorValidationEvents


@pytest.fixture
def three_sensor_df():
    """Three sensors: sensor_1 and sensor_2 agree, sensor_3 has a +5 bias."""
    np.random.seed(42)
    n = 300
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        t = base + pd.Timedelta(minutes=i)
        base_val = 100.0 + np.random.randn() * 0.5
        rows.append({"systime": t, "uuid": "sensor_1", "value_double": base_val})
        rows.append(
            {
                "systime": t,
                "uuid": "sensor_2",
                "value_double": base_val + np.random.randn() * 0.1,
            }
        )
        rows.append({"systime": t, "uuid": "sensor_3", "value_double": base_val + 5.0})
    return pd.DataFrame(rows)


@pytest.fixture
def agreeing_sensors_df():
    """Two sensors that agree closely."""
    np.random.seed(42)
    n = 200
    base = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(n):
        t = base + pd.Timedelta(minutes=i)
        v = 50.0 + np.random.randn() * 0.1
        rows.append({"systime": t, "uuid": "s1", "value_double": v})
        rows.append(
            {"systime": t, "uuid": "s2", "value_double": v + np.random.randn() * 0.01}
        )
    return pd.DataFrame(rows)


class TestDetectDisagreement:
    def test_finds_biased_sensor(self, three_sensor_df):
        msv = MultiSensorValidationEvents(
            three_sensor_df,
            ["sensor_1", "sensor_2", "sensor_3"],
        )
        result = msv.detect_disagreement(threshold=2.0, window="1h")
        assert len(result) > 0
        # sensor_3 should always be the high sensor
        assert all(r == "sensor_3" for r in result["sensor_high"].tolist())

    def test_no_disagreement(self, agreeing_sensors_df):
        msv = MultiSensorValidationEvents(
            agreeing_sensors_df,
            ["s1", "s2"],
        )
        result = msv.detect_disagreement(threshold=1.0, window="1h")
        assert len(result) == 0

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        msv = MultiSensorValidationEvents(df, ["s1", "s2"])
        assert len(msv.detect_disagreement(threshold=1.0)) == 0


class TestPairwiseBias:
    def test_bias_between_pairs(self, three_sensor_df):
        msv = MultiSensorValidationEvents(
            three_sensor_df,
            ["sensor_1", "sensor_2", "sensor_3"],
        )
        result = msv.pairwise_bias(window="2h")
        assert len(result) > 0
        # There should be 3 pairs per window
        first_window = result[result["window_start"] == result["window_start"].iloc[0]]
        assert len(first_window) == 3
        # sensor_3 pairs should have higher abs_bias
        s3_rows = first_window[
            (first_window["sensor_a"] == "sensor_3")
            | (first_window["sensor_b"] == "sensor_3")
        ]
        assert all(b > 2.0 for b in s3_rows["abs_bias"].tolist())


class TestConsensusScore:
    def test_high_consensus(self, agreeing_sensors_df):
        msv = MultiSensorValidationEvents(
            agreeing_sensors_df,
            ["s1", "s2"],
        )
        result = msv.consensus_score(window="2h")
        assert len(result) > 0
        assert result.iloc[0]["consensus_score"] > 0.9

    def test_low_consensus_with_bias(self, three_sensor_df):
        msv = MultiSensorValidationEvents(
            three_sensor_df,
            ["sensor_1", "sensor_2", "sensor_3"],
        )
        result = msv.consensus_score(window="2h")
        assert len(result) > 0
        # With a 5-unit bias, consensus should be lower
        assert result.iloc[0]["consensus_score"] < 0.99


class TestIdentifyOutlierSensor:
    def test_identifies_biased_sensor(self, three_sensor_df):
        msv = MultiSensorValidationEvents(
            three_sensor_df,
            ["sensor_1", "sensor_2", "sensor_3"],
        )
        result = msv.identify_outlier_sensor(window="2h", method="median")
        assert len(result) > 0
        assert all(s == "sensor_3" for s in result["outlier_sensor"].tolist())

    def test_mean_method(self, three_sensor_df):
        msv = MultiSensorValidationEvents(
            three_sensor_df,
            ["sensor_1", "sensor_2", "sensor_3"],
        )
        result = msv.identify_outlier_sensor(window="2h", method="mean")
        assert len(result) > 0
        assert all(s == "sensor_3" for s in result["outlier_sensor"].tolist())

    def test_empty(self):
        df = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        msv = MultiSensorValidationEvents(df, ["s1", "s2"])
        assert len(msv.identify_outlier_sensor()) == 0


class TestConstructorValidation:
    def test_single_sensor_raises(self):
        df = pd.DataFrame(
            {
                "systime": [pd.Timestamp("2024-01-01")],
                "uuid": ["s1"],
                "value_double": [1.0],
            }
        )
        with pytest.raises(ValueError, match="At least 2"):
            MultiSensorValidationEvents(df, ["s1"])
