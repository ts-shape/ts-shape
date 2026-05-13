import pandas as pd  # type: ignore
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents


def test_outlier_detection_zscore_and_iqr():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=10, freq="min"),
            "value": [1, 1, 1, 1, 10, 1, 1, 1, 1, 1],
        }
    )

    det = OutlierDetectionEvents(
        df, value_column="value", event_uuid="outlier_evt", time_threshold="2min"
    )
    z = det.detect_outliers_zscore(threshold=2.0)
    # Should create events around the outlier timestamp
    assert "uuid" in z.columns

    i = det.detect_outliers_iqr(threshold=(1.5, 1.5))
    assert "uuid" in i.columns
