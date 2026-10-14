import pandas as pd

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


def test_outlier_output_is_canonical_point_shape():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=10, freq="min"),
            "uuid": ["sensor:x"] * 10,
            "value_double": [1, 1, 1, 1, 50, 1, 1, 1, 1, 1],
        }
    )
    det = OutlierDetectionEvents(
        df, value_column="value_double", event_uuid="outlier_evt"
    )
    z = det.detect_outliers_zscore(threshold=2.0)
    # Canonical point schema: systime / uuid / source_uuid.
    for col in ("systime", "uuid", "source_uuid"):
        assert col in z.columns
    assert not z.empty
    assert (z["uuid"] == "outlier_evt").all()
    assert (z["source_uuid"] == "sensor:x").all()


def test_signal_uuids_restricts_the_analysis():
    import numpy as np

    t = pd.date_range("2024-01-01", periods=50, freq="1min")
    a = np.full(50, 10.0)
    a[25] = 100.0
    b = np.full(50, 5.0)
    b[10] = -80.0
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "a", "value_double": a}),
            pd.DataFrame({"systime": t, "uuid": "b", "value_double": b}),
        ],
        ignore_index=True,
    )
    both = OutlierDetectionEvents(df, "value_double").detect_outliers_iqr()
    only_a = OutlierDetectionEvents(
        df, "value_double", signal_uuids=["a"]
    ).detect_outliers_iqr()
    assert set(both["source_uuid"]) == {"a", "b"}
    assert set(only_a["source_uuid"]) == {"a"}
