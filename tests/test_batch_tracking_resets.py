"""Reset-aware behaviour of BatchTrackingEvents.batch_yield."""

import pandas as pd  # type: ignore
import pytest

from ts_shape.events.production import BatchTrackingEvents


@pytest.fixture
def reset_batch_data():
    """A single batch whose counter resets mid-run.

    Counter values: 0,100,200,[reset]0,150 during BATCH_A.
    naive last-first = 150 ; reset-aware = 350 ; 1 reset.
    """
    times = pd.date_range("2024-01-15 06:00:00", periods=5, freq="1h")
    values = [0, 100, 200, 0, 150]
    rows = []
    for t, v in zip(times, values):
        rows.append(
            {
                "systime": t,
                "uuid": "batch_signal",
                "value_integer": None,
                "value_string": "BATCH_A",
                "is_delta": False,
            }
        )
        rows.append(
            {
                "systime": t,
                "uuid": "batch_counter",
                "value_integer": v,
                "value_string": None,
                "is_delta": True,
            }
        )
    return pd.DataFrame(rows)


def test_batch_yield_default_under_reports(reset_batch_data):
    tracker = BatchTrackingEvents(reset_batch_data, batch_uuid="batch_signal")
    result = tracker.batch_yield("batch_counter")
    assert "resets" not in result.columns
    assert len(result) == 1
    assert result["quantity"].iloc[0] == 150


def test_batch_yield_handles_resets(reset_batch_data):
    tracker = BatchTrackingEvents(reset_batch_data, batch_uuid="batch_signal")
    result = tracker.batch_yield("batch_counter", handle_resets=True)
    assert "resets" in result.columns
    assert result["quantity"].iloc[0] == 350
    assert result["resets"].iloc[0] == 1
