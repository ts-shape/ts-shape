"""Counter-reset handling for ContinuousProcessAlignmentEvents.segment_by_cut."""

import pandas as pd
import pytest

from ts_shape.events.production import ContinuousProcessAlignmentEvents

LINE_CONFIG = [{"name": "zone_a", "offset": 30.0, "uuids": ["station:a"]}]


def _make_reset_line_df():
    """Numeric part counter that resets mid-run.

    Counter (every 60 s): 1..10 then [reset] 1, 2, 3.

    A reset produces a negative delta. Without reset handling that step is
    skipped, so the piece cut immediately after the reset is lost: 11 pieces
    are detected instead of 12.
    """
    base = pd.Timestamp("2024-01-01 00:00:00")
    rows = []

    # station:a every 15 s, long enough to cover every cut boundary
    for i, t in enumerate(pd.date_range(base, periods=80, freq="15s")):
        rows.append(
            dict(
                systime=t,
                uuid="station:a",
                value_double=float(i),
                value_bool=None,
                value_integer=None,
            )
        )

    # constant 10 m/min line speed
    for t in pd.date_range(base, periods=20, freq="60s"):
        rows.append(
            dict(
                systime=t,
                uuid="speed:line",
                value_double=10.0,
                value_bool=None,
                value_integer=None,
            )
        )

    # integer part counter with a reset
    counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3]
    for t, v in zip(
        pd.date_range(base + pd.Timedelta(seconds=60), periods=len(counts), freq="60s"),
        counts,
    ):
        rows.append(
            dict(
                systime=t,
                uuid="counter:int",
                value_double=None,
                value_bool=None,
                value_integer=v,
            )
        )

    return pd.DataFrame(rows)


@pytest.fixture
def aligner():
    df = _make_reset_line_df()
    return ContinuousProcessAlignmentEvents(
        df, "speed:line", LINE_CONFIG, speed_unit="m/min"
    )


def test_segment_by_cut_loses_post_reset_piece_when_naive(aligner):
    aligned = aligner.align_to_reference(station_uuids=["station:a"])
    result = aligner.segment_by_cut(
        aligned, part_counter_uuid="counter:int", handle_resets=False
    )

    # Original columns preserved, no new reset column.
    assert "piece_id" in result.columns
    assert "resets" not in result.columns
    # Reset step skipped -> the post-reset piece is lost (11 instead of 12).
    assert int(result["piece_id"].max()) == 11


def test_segment_by_cut_handles_resets(aligner):
    aligned = aligner.align_to_reference(station_uuids=["station:a"])
    default = aligner.segment_by_cut(
        aligned, part_counter_uuid="counter:int", handle_resets=False
    )
    aware = aligner.segment_by_cut(
        aligned, part_counter_uuid="counter:int", handle_resets=True
    )

    # The reset's post-reset value is now counted as a cut.
    assert int(aware["piece_id"].max()) == 12
    assert int(aware["piece_id"].max()) == int(default["piece_id"].max()) + 1


def test_segment_by_cut_default_is_reset_aware(aligner):
    """The default ``handle_resets=True`` counts post-reset production."""
    aligned = aligner.align_to_reference(station_uuids=["station:a"])

    def call(**kw):
        return aligner.segment_by_cut(aligned, part_counter_uuid="counter:int", **kw)

    default = call()
    pd.testing.assert_frame_equal(default, call(handle_resets=True))
    assert int(default["piece_id"].max()) == 12
