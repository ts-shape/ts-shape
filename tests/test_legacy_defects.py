"""Regressions for defects found in the legacy-code audit."""

import subprocess
import sys
import textwrap
import warnings

import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.features.segment_analysis.profile_comparison import ProfileComparison
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor


def test_apply_ranges_mixed_time_units():
    t = pd.date_range("2024-01-01", periods=4, freq="1h").as_unit("ns")
    df = pd.DataFrame({"systime": t, "uuid": "a", "value_double": range(4)})
    ranges = pd.DataFrame(
        {
            "segment_start": pd.Series([t[0]]).astype("datetime64[s]"),
            "segment_end": pd.Series([t[-1] + pd.Timedelta("1h")]).astype(
                "datetime64[s]"
            ),
            "segment_value": ["A"],
            "segment_index": [0],
        }
    )
    out = SegmentProcessor.apply_ranges(df, ranges, target_uuids=["a"])
    assert len(out) == 4


def test_base_fallback_keeps_numeric_time_named_columns():
    profiles = pd.DataFrame(
        {
            "segment_value": ["A", "B"],
            "uuid": ["s", "s"],
            "cycle_time_mean": [1.5, 2.5],
            "update_rate": [10.0, 20.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        pc = ProfileComparison(profiles)
    assert pc.dataframe["cycle_time_mean"].dtype == float
    assert pc.dataframe["update_rate"].dtype == float


def test_deprecation_warning_visible_under_default_filters(tmp_path):
    script = tmp_path / "user_script.py"
    script.write_text(textwrap.dedent("""
            import pandas as pd
            from ts_shape.events.engineering.threshold_monitoring import (
                ThresholdMonitoringEvents,
            )

            df = pd.DataFrame({"systime": pd.date_range("2024", periods=4),
                               "uuid": "a", "value_double": [1.0, 6.0, 2.0, 0.5]})
            ThresholdMonitoringEvents(df, "a").threshold_with_hysteresis(
                high=5, low=1
            )
            """))
    res = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=True
    )
    assert "DeprecationWarning" in res.stderr
    assert "user_script.py" in res.stderr
