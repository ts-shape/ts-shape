"""Reset-aware behavior tests for ScrapEvents.

A scrap counter can reset back to a lower value at a shift change or a
controller restart. With ``handle_resets=False`` a window containing a reset
under-reports; by default (``handle_resets=True``), the post-reset scrap is counted and a ``resets``
column is added.
"""

import pandas as pd

from ts_shape.events.production.scrap_tracking import ScrapEvents


def _make_reset_scrap_df():
    """Scrap counter that resets mid-shift, plus reason and part signals.

    Single shift (shift_1, 06:00-14:00). Counter:
        10, 30, 55, [reset] 5, 20, 40
    Naive last-first: 40 - 10 = 30.
    Reset-aware: (20 + 25) + (5 + 15 + 20) = 45 + 40 = 85.
    """
    times = pd.date_range("2024-01-01 06:00", periods=6, freq="1h")
    counts = [10.0, 30.0, 55.0, 5.0, 20.0, 40.0]
    rows = []
    for t, c in zip(times, counts):
        rows.append({"systime": t, "uuid": "scrap_weight", "value_double": c})
    # One reason for the whole (contiguous) window.
    for t in times:
        rows.append({"systime": t, "uuid": "scrap_reason", "value_string": "Material"})
    # One part for the whole window.
    for t in times:
        rows.append({"systime": t, "uuid": "part_number", "value_string": "PART_A"})
    return pd.DataFrame(rows)


def test_scrap_by_shift_naive_under_reports():
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)
    result = tracker.scrap_by_shift(scrap_uuid="scrap_weight", handle_resets=False)

    # Default clamps to last - first = 30, losing the pre-reset production.
    assert list(result.columns) == [
        "start",
        "end",
        "duration_seconds",
        "date",
        "shift",
        "scrap_quantity",
    ]
    assert "resets" not in result.columns
    assert result["scrap_quantity"].sum() == 30.0


def test_scrap_by_shift_handles_resets():
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)
    result = tracker.scrap_by_shift(scrap_uuid="scrap_weight", handle_resets=True)

    assert "resets" in result.columns
    assert result["scrap_quantity"].sum() == 85.0
    assert result["resets"].sum() == 1


def test_scrap_by_reason_handles_resets():
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)

    default = tracker.scrap_by_reason(
        scrap_uuid="scrap_weight", reason_uuid="scrap_reason", handle_resets=False
    )
    assert "resets" not in default.columns
    assert default["scrap_quantity"].sum() == 30.0

    aware = tracker.scrap_by_reason(
        scrap_uuid="scrap_weight", reason_uuid="scrap_reason", handle_resets=True
    )
    assert "resets" in aware.columns
    assert aware["scrap_quantity"].sum() == 85.0
    assert aware["resets"].sum() == 1


def test_scrap_cost_handles_resets():
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)

    default = tracker.scrap_cost(
        scrap_uuid="scrap_weight",
        part_id_uuid="part_number",
        material_costs={"PART_A": 2.0},
        handle_resets=False,
    )
    assert "resets" not in default.columns
    assert default["scrap_quantity"].iloc[0] == 30.0
    assert default["total_cost"].iloc[0] == 60.0

    aware = tracker.scrap_cost(
        scrap_uuid="scrap_weight",
        part_id_uuid="part_number",
        material_costs={"PART_A": 2.0},
        handle_resets=True,
    )
    assert "resets" in aware.columns
    assert aware["scrap_quantity"].iloc[0] == 85.0
    assert aware["total_cost"].iloc[0] == 170.0
    assert aware["resets"].iloc[0] == 1


def test_scrap_trend_handles_resets():
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)

    default = tracker.scrap_trend(
        scrap_uuid="scrap_weight", window="1D", handle_resets=False
    )
    assert "resets" not in default.columns
    assert default["scrap_quantity"].sum() == 30.0

    aware = tracker.scrap_trend(
        scrap_uuid="scrap_weight", window="1D", handle_resets=True
    )
    assert "resets" in aware.columns
    assert aware["scrap_quantity"].sum() == 85.0
    assert aware["resets"].sum() == 1


def test_scrap_by_shift_default_is_reset_aware():
    """The default ``handle_resets=True`` counts post-reset production."""
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)

    def call(**kw):
        return tracker.scrap_by_shift(scrap_uuid="scrap_weight", **kw)

    default = call()
    pd.testing.assert_frame_equal(default, call(handle_resets=True))
    assert default["resets"].sum() > 0


def test_scrap_by_reason_default_is_reset_aware():
    """The default ``handle_resets=True`` counts post-reset production."""
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)
    default = tracker.scrap_by_reason(
        scrap_uuid="scrap_weight", reason_uuid="scrap_reason", handle_resets=False
    )
    assert "resets" not in default.columns
    assert default["scrap_quantity"].sum() == 30.0

    def call(**kw):
        return tracker.scrap_by_reason(
            scrap_uuid="scrap_weight", reason_uuid="scrap_reason", **kw
        )

    default = call()
    pd.testing.assert_frame_equal(default, call(handle_resets=True))
    assert default["resets"].sum() > 0


def test_scrap_cost_default_is_reset_aware():
    """The default ``handle_resets=True`` counts post-reset production."""
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)
    default = tracker.scrap_cost(
        scrap_uuid="scrap_weight",
        part_id_uuid="part_number",
        material_costs={"PART_A": 2.0},
        handle_resets=False,
    )
    assert "resets" not in default.columns
    assert default["scrap_quantity"].iloc[0] == 30.0
    assert default["total_cost"].iloc[0] == 60.0

    def call(**kw):
        return tracker.scrap_cost(
            scrap_uuid="scrap_weight",
            part_id_uuid="part_number",
            material_costs={"PART_A": 2.0},
            **kw,
        )

    default = call()
    pd.testing.assert_frame_equal(default, call(handle_resets=True))
    assert default["resets"].sum() > 0


def test_scrap_trend_default_is_reset_aware():
    """The default ``handle_resets=True`` counts post-reset production."""
    df = _make_reset_scrap_df()
    tracker = ScrapEvents(df)
    default = tracker.scrap_trend(
        scrap_uuid="scrap_weight", window="1D", handle_resets=False
    )
    assert "resets" not in default.columns
    assert default["scrap_quantity"].sum() == 30.0

    def call(**kw):
        return tracker.scrap_trend(scrap_uuid="scrap_weight", window="1D", **kw)

    default = call()
    pd.testing.assert_frame_equal(default, call(handle_resets=True))
    assert default["resets"].sum() > 0
