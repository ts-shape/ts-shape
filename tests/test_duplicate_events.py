"""Regression tests: duplicated input samples do not duplicate events."""

import numpy as np
import pandas as pd

from ts_shape.events.engineering.warmup_analysis import WarmUpCoolDownEvents
from ts_shape.events.quality.outlier_detection import OutlierDetectionEvents


def _signal(times, values, uuid="sig", column="value_double"):
    return pd.DataFrame(
        {"systime": pd.to_datetime(times), "uuid": uuid, column: values}
    ).assign(is_delta=True)


def _minutes(n, start="2024-01-01 00:00"):
    return pd.date_range(start, periods=n, freq="1min")


# --- Duplicate exact timestamps ------------------------------------------------


def test_outliers_from_duplicated_samples_are_reported_once():
    values = np.r_[np.zeros(20), 100.0, np.zeros(20)]
    df = _signal(_minutes(41), values)
    dup = pd.concat([df, df.iloc[[20]]]).sort_values("systime")
    ev = OutlierDetectionEvents(dup, "value_double", signal_uuids=["sig"])
    out = ev.detect_outliers_iqr()
    assert len(out) == len(out.drop_duplicates())
    assert len(out) == 1


def test_warmup_not_split_by_duplicated_sample():
    df = _signal(_minutes(10), np.arange(10.0) * 10)
    dup = pd.concat([df, df.iloc[[5]]]).sort_values("systime", kind="stable")
    plain = WarmUpCoolDownEvents(df, "sig").detect_warmup(min_rise=50.0)
    doubled = WarmUpCoolDownEvents(dup, "sig").detect_warmup(min_rise=50.0)
    assert len(plain) == len(doubled) == 1
    pd.testing.assert_frame_equal(plain, doubled)
