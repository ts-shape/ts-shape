import pytest
import pandas as pd

from ts_shape.events.production import LongDowntimeEvents

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_df(times, values, uuid="machine", col="value_bool"):
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(times),
            "uuid": uuid,
            col: values,
            "is_delta": True,
        }
    )


def _make_prod_df(times, values, uuid="prod", col="value_integer"):
    return pd.DataFrame(
        {
            "systime": pd.to_datetime(times),
            "uuid": uuid,
            col: values,
            "is_delta": True,
        }
    )


# ---------------------------------------------------------------------------
# detect_long_downtime
# ---------------------------------------------------------------------------


class TestDetectLongDowntime:
    def test_basic_two_long_downtimes(self):
        """Two idle gaps of ~4 h each, both should be returned."""
        times = [
            "2024-01-01 00:00",  # down
            "2024-01-01 04:00",  # down → end of gap 1 (4 h)
            "2024-01-01 04:01",  # running
            "2024-01-01 08:00",  # running
            "2024-01-01 08:01",  # down
            "2024-01-01 12:30",  # down → end of gap 2 (~4.5 h)
        ]
        values = [False, False, True, True, False, False]
        df = _make_state_df(times, values)

        lde = LongDowntimeEvents(df, "machine")
        result = lde.detect_long_downtime(min_gap="3h")

        assert len(result) == 2
        assert list(result["downtime_index"]) == [0, 1]
        assert result.iloc[0]["duration_seconds"] == pytest.approx(4 * 3600)
        assert result.iloc[1]["duration_seconds"] == pytest.approx(4.5 * 3600 - 60)
        assert set(result.columns) >= {
            "start",
            "end",
            "duration_seconds",
            "downtime_index",
            "uuid",
            "source_uuid",
            "is_delta",
        }

    def test_integer_signal_with_value_range(self):
        """Integer signal: value_range=(1,1) means running when value==1."""
        times = [
            "2024-01-02 00:00",  # 0 = down
            "2024-01-02 05:00",  # 0 = down (5 h gap)
            "2024-01-02 05:01",  # 1 = running
            "2024-01-02 10:00",  # 1 = running
        ]
        values = [0, 0, 1, 1]
        df = _make_state_df(times, values, col="value_integer")

        lde = LongDowntimeEvents(
            df, "machine", value_column="value_integer", value_range=(1, 1)
        )
        result = lde.detect_long_downtime(min_gap="3h")

        assert len(result) == 1
        assert result.iloc[0]["duration_seconds"] == pytest.approx(5 * 3600)

    def test_short_gaps_filtered_out(self):
        """1 h gap must NOT be returned; 4 h gap must be returned."""
        times = [
            "2024-01-03 00:00",  # down
            "2024-01-03 01:00",  # down → 1 h gap (too short)
            "2024-01-03 01:01",  # running
            "2024-01-03 02:00",  # running
            "2024-01-03 02:01",  # down
            "2024-01-03 06:01",  # down → 4 h gap
        ]
        values = [False, False, True, True, False, False]
        df = _make_state_df(times, values)

        lde = LongDowntimeEvents(df, "machine")
        result = lde.detect_long_downtime(min_gap="3h")

        assert len(result) == 1
        assert result.iloc[0]["duration_seconds"] == pytest.approx(4 * 3600)

    def test_no_qualifying_downtimes(self):
        """All idle periods shorter than min_gap → empty result."""
        times = [
            "2024-01-04 00:00",
            "2024-01-04 01:00",
            "2024-01-04 01:01",
            "2024-01-04 02:00",
        ]
        values = [False, False, True, True]
        df = _make_state_df(times, values)

        lde = LongDowntimeEvents(df, "machine")
        result = lde.detect_long_downtime(min_gap="3h")

        assert result.empty
        assert "start" in result.columns

    def test_empty_dataframe(self):
        """Empty input returns empty result without raising."""
        df = pd.DataFrame(
            {
                "systime": pd.Series(dtype="datetime64[ns]"),
                "uuid": pd.Series(dtype="str"),
                "value_bool": pd.Series(dtype="bool"),
                "is_delta": pd.Series(dtype="bool"),
            }
        )
        lde = LongDowntimeEvents(df, "machine")
        result = lde.detect_long_downtime()

        assert result.empty
        assert "start" in result.columns


# ---------------------------------------------------------------------------
# count_events_between_gaps
# ---------------------------------------------------------------------------


def _build_bounded_scenario():
    """
    State signal:
      gap 0: 00:00 → 05:00 (5 h down)
      running: 05:00 → 10:00
      gap 1: 10:00 → 15:00 (5 h down)
      running: 15:00 → 20:00
      gap 2: 20:00 → 25:00 (5 h down)
    """
    state_times = [
        "2024-01-05 00:00",
        "2024-01-05 05:00",  # down
        "2024-01-05 05:01",
        "2024-01-05 10:00",  # running
        "2024-01-05 10:01",
        "2024-01-05 15:00",  # down
        "2024-01-05 15:01",
        "2024-01-05 20:00",  # running
        "2024-01-05 20:01",
        "2024-01-06 01:00",  # down (~4.5 h)
    ]
    state_values = [False, False, True, True, False, False, True, True, False, False]
    state_df = _make_state_df(state_times, state_values)
    return state_df


class TestCountEventsBetweenGaps:
    def test_count_aggregation(self):
        """5 production rows in window 0→1 → event_count == 5."""
        state_df = _build_bounded_scenario()

        prod_times = [
            "2024-01-05 06:00",
            "2024-01-05 07:00",
            "2024-01-05 08:00",
            "2024-01-05 09:00",
            "2024-01-05 09:30",
        ]
        prod_df = _make_prod_df(prod_times, [1, 1, 1, 1, 1])

        lde = LongDowntimeEvents(state_df, "machine")
        result = lde.count_events_between_gaps(prod_df, "prod", aggregation="count")

        assert len(result) == 2
        assert int(result.iloc[0]["event_count"]) == 5
        assert int(result.iloc[1]["event_count"]) == 0  # nothing in second window
        assert result.iloc[0]["downtime_index"] == 1
        assert result.iloc[1]["downtime_index"] == 2

    def test_sum_aggregation(self):
        """Sum value_integer for events in window."""
        state_df = _build_bounded_scenario()

        prod_times = ["2024-01-05 06:00", "2024-01-05 08:00", "2024-01-05 09:00"]
        prod_df = _make_prod_df(prod_times, [10, 20, 30])

        lde = LongDowntimeEvents(state_df, "machine")
        result = lde.count_events_between_gaps(
            prod_df, "prod", aggregation="sum", value_column="value_integer"
        )

        assert result.iloc[0]["event_count"] == pytest.approx(60)

    def test_transitions_aggregation(self):
        """Count value changes (state transitions) in window."""
        state_df = _build_bounded_scenario()

        # Alternating 0/1 in window 0 → 4 transitions (0→1, 1→0, 0→1, 1→0)
        prod_times = [
            "2024-01-05 06:00",
            "2024-01-05 07:00",
            "2024-01-05 08:00",
            "2024-01-05 09:00",
            "2024-01-05 09:30",
        ]
        prod_df = _make_prod_df(prod_times, [0, 1, 0, 1, 0])

        lde = LongDowntimeEvents(state_df, "machine")
        result = lde.count_events_between_gaps(
            prod_df, "prod", aggregation="transitions", value_column="value_integer"
        )

        # First row has a NaN shift so counts include the first row as a "change"
        # (0 != NaN → True), plus 4 actual transitions → total 5
        assert int(result.iloc[0]["event_count"]) == 5

    def test_window_duration_seconds(self):
        """window_duration_seconds reflects the actual window span."""
        state_df = _build_bounded_scenario()
        prod_df = _make_prod_df([], [])

        lde = LongDowntimeEvents(state_df, "machine")
        result = lde.count_events_between_gaps(prod_df, "prod")

        # window 0: ends at 05:00, window 1 starts at 10:01 → ~5h1min
        assert result.iloc[0]["window_duration_seconds"] > 0
        assert "start" in result.columns
        assert "end" in result.columns

    def test_fewer_than_two_gaps_returns_empty(self):
        """Only one long downtime → no windows → empty result."""
        times = [
            "2024-01-06 00:00",
            "2024-01-06 05:00",  # down (5 h)
            "2024-01-06 05:01",
            "2024-01-06 10:00",  # running
        ]
        values = [False, False, True, True]
        state_df = _make_state_df(times, values)
        prod_df = _make_prod_df(["2024-01-06 07:00"], [1])

        lde = LongDowntimeEvents(state_df, "machine")
        result = lde.count_events_between_gaps(prod_df, "prod")

        assert result.empty
        assert "event_count" in result.columns

    def test_empty_state_returns_empty(self):
        """Empty state dataframe → count_events_between_gaps returns empty."""
        state_df = pd.DataFrame(
            {
                "systime": pd.Series(dtype="datetime64[ns]"),
                "uuid": pd.Series(dtype="str"),
                "value_bool": pd.Series(dtype="bool"),
                "is_delta": pd.Series(dtype="bool"),
            }
        )
        prod_df = _make_prod_df(["2024-01-07 10:00"], [1])

        lde = LongDowntimeEvents(state_df, "machine")
        result = lde.count_events_between_gaps(prod_df, "prod")

        assert result.empty

    def test_invalid_aggregation_raises(self):
        """Unknown aggregation value → ValueError."""
        state_df = _build_bounded_scenario()
        prod_df = _make_prod_df(["2024-01-05 07:00"], [1])

        lde = LongDowntimeEvents(state_df, "machine")
        with pytest.raises(ValueError, match="Unknown aggregation"):
            lde.count_events_between_gaps(prod_df, "prod", aggregation="median")


# ---------------------------------------------------------------------------
# Init guards
# ---------------------------------------------------------------------------


class TestInitGuards:
    def test_invalid_uuid_raises(self):
        """UUID not present in the dataframe → ValueError."""
        times = ["2024-01-08 00:00", "2024-01-08 01:00"]
        df = _make_state_df(times, [True, False])

        with pytest.raises(ValueError, match="not found in dataframe"):
            LongDowntimeEvents(df, "nonexistent_uuid")
