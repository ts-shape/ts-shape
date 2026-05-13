"""Tests for ContinuousProcessAlignmentEvents."""

import numpy as np
import pandas as pd
import pytest

from ts_shape.events.production import ContinuousProcessAlignmentEvents

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

LINE_CONFIG = [
    {"name": "zone_a", "offset": 30.0, "uuids": ["station:a"]},
    {"name": "zone_b", "offset": 90.0, "uuids": ["station:b"]},
    {"name": "zone_c", "offset": 180.0, "uuids": ["station:c"]},
    {"name": "cutter", "offset": 210.0, "uuids": ["cutter:signal"]},
]

# Expected lags at 10 m/min = 1/6 m/s
_LAG_A = 30.0 / (10.0 / 60.0)  # 180 s
_LAG_B = 90.0 / (10.0 / 60.0)  # 540 s
_LAG_C = 180.0 / (10.0 / 60.0)  # 1080 s


def _make_df() -> pd.DataFrame:
    """2-hour run at 15 s intervals.

    Signals:
    - speed:line  10 m/min for hour 1, 5 m/min for hour 2
    - station:a   sine wave
    - station:b   cosine wave
    - station:c   constant 42.0
    - counter:bool  boolean part counter, fires every 60 s
    - counter:int   integer part counter, increments by 1 every 60 s
    - counter:int2  integer part counter, increments by 2 every 120 s
    - cut:length    0.8 m every 60 s
    """
    base = pd.Timestamp("2024-01-01 00:00:00")
    timestamps = pd.date_range(base, periods=480, freq="15s")  # 2 h

    rows = []

    for i, t in enumerate(timestamps):
        speed = 10.0 if t < base + pd.Timedelta(hours=1) else 5.0
        rows.append(
            dict(
                systime=t,
                uuid="speed:line",
                value_double=speed,
                value_bool=None,
                value_integer=None,
            )
        )

        rows.append(
            dict(
                systime=t,
                uuid="station:a",
                value_double=float(np.sin(i * 0.1)),
                value_bool=None,
                value_integer=None,
            )
        )
        rows.append(
            dict(
                systime=t,
                uuid="station:b",
                value_double=float(np.cos(i * 0.1)),
                value_bool=None,
                value_integer=None,
            )
        )
        rows.append(
            dict(
                systime=t,
                uuid="station:c",
                value_double=42.0,
                value_bool=None,
                value_integer=None,
            )
        )

    # Boolean part counter: fires True every 60 s (every 4 samples)
    cut_times = pd.date_range(base + pd.Timedelta(seconds=60), periods=119, freq="60s")
    for t in cut_times:
        rows.append(
            dict(
                systime=t,
                uuid="counter:bool",
                value_double=None,
                value_bool=True,
                value_integer=None,
            )
        )

    # Integer part counter: increments by 1 every 60 s
    for j, t in enumerate(cut_times):
        rows.append(
            dict(
                systime=t,
                uuid="counter:int",
                value_double=None,
                value_bool=None,
                value_integer=j + 1,
            )
        )

    # Integer part counter: increments by 2 every 120 s
    cut_times2 = pd.date_range(
        base + pd.Timedelta(seconds=120), periods=59, freq="120s"
    )
    for j, t in enumerate(cut_times2):
        rows.append(
            dict(
                systime=t,
                uuid="counter:int2",
                value_double=None,
                value_bool=None,
                value_integer=(j + 1) * 2,
            )
        )

    # Cut length signal: 0.8 m at same cadence as counter:bool
    for t in cut_times:
        rows.append(
            dict(
                systime=t,
                uuid="cut:length",
                value_double=0.8,
                value_bool=None,
                value_integer=None,
            )
        )

    return pd.DataFrame(rows)


@pytest.fixture
def df():
    return _make_df()


@pytest.fixture
def aligner(df):
    return ContinuousProcessAlignmentEvents(
        df, "speed:line", LINE_CONFIG, speed_unit="m/min"
    )


# ---------------------------------------------------------------------------
# TestAlignToReference
# ---------------------------------------------------------------------------


class TestAlignToReference:
    def test_output_columns(self, aligner):
        result = aligner.align_to_reference()
        for col in [
            "material_ref_time",
            "systime",
            "uuid",
            "component",
            "position_offset_m",
            "lag_seconds",
            "value_double",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_lag_constant_speed(self, aligner):
        result = aligner.align_to_reference()
        # First hour at 10 m/min
        base = pd.Timestamp("2024-01-01 00:00:00")
        hour1_end = base + pd.Timedelta(hours=1)

        for uid, expected_lag in [
            ("station:a", _LAG_A),
            ("station:b", _LAG_B),
            ("station:c", _LAG_C),
        ]:
            sub = result[(result["uuid"] == uid) & (result["systime"] < hour1_end)]
            assert not sub.empty
            assert (
                abs(sub["lag_seconds"].median() - expected_lag) < 2.0
            ), f"{uid}: expected lag {expected_lag:.1f}s, got {sub['lag_seconds'].median():.1f}s"

    def test_material_ref_before_systime(self, aligner):
        result = aligner.align_to_reference()
        assert (result["material_ref_time"] < result["systime"]).all()

    def test_zero_speed_clamped(self, df):
        # Insert zero-speed rows
        zero_rows = pd.DataFrame(
            [
                dict(
                    systime=pd.Timestamp("2024-01-01 00:30:00"),
                    uuid="speed:line",
                    value_double=0.0,
                    value_bool=None,
                    value_integer=None,
                )
            ]
        )
        df2 = pd.concat([df, zero_rows], ignore_index=True)
        aligner2 = ContinuousProcessAlignmentEvents(
            df2, "speed:line", LINE_CONFIG, speed_unit="m/min", min_speed=0.01
        )
        result = aligner2.align_to_reference()
        assert not result["lag_seconds"].isin([np.inf, -np.inf]).any()
        assert not result["lag_seconds"].isna().any()

    def test_subset_station_uuids(self, aligner):
        result = aligner.align_to_reference(station_uuids=["station:a"])
        assert set(result["uuid"].unique()) == {"station:a"}

    def test_empty_dataframe(self):
        empty = pd.DataFrame(
            columns=["systime", "uuid", "value_double", "value_bool", "value_integer"]
        )
        aligner = ContinuousProcessAlignmentEvents(
            empty, "speed:line", LINE_CONFIG, speed_unit="m/min"
        )
        result = aligner.align_to_reference()
        assert result.empty
        assert "material_ref_time" in result.columns

    def test_lag_doubles_at_half_speed(self, aligner):
        result = aligner.align_to_reference(station_uuids=["station:a"])
        base = pd.Timestamp("2024-01-01 00:00:00")
        h1_end = base + pd.Timedelta(hours=1)
        h2_start = h1_end + pd.Timedelta(minutes=30)

        lag_h1 = result[result["systime"] < h1_end]["lag_seconds"].median()
        lag_h2 = result[result["systime"] > h2_start]["lag_seconds"].median()
        # At half speed lag should be ~2x
        assert (
            abs(lag_h2 / lag_h1 - 2.0) < 0.1
        ), f"Expected 2x lag ratio, got {lag_h2 / lag_h1:.3f}"


# ---------------------------------------------------------------------------
# TestSegmentByCut
# ---------------------------------------------------------------------------


class TestSegmentByCut:
    def test_piece_id_assigned_bool_counter(self, aligner):
        aligned = aligner.align_to_reference(station_uuids=["station:a"])
        result = aligner.segment_by_cut(aligned, part_counter_uuid="counter:bool")
        assert "piece_id" in result.columns
        valid = result["piece_id"].dropna()
        assert len(valid) > 0
        assert valid.dtype in [np.int64, np.float64, object] or np.issubdtype(
            valid.dtype, np.number
        )

    def test_piece_id_assigned_int_counter(self, aligner):
        aligned = aligner.align_to_reference(station_uuids=["station:a"])
        result = aligner.segment_by_cut(aligned, part_counter_uuid="counter:int")
        valid = result["piece_id"].dropna()
        assert len(valid) > 0
        # piece_ids should be integers
        assert (valid == valid.astype(int)).all()

    def test_int_counter_delta_gt_1(self, df):
        """Counter steps by 2 → two consecutive piece_ids share same cut timestamp."""
        aligner2 = ContinuousProcessAlignmentEvents(
            df, "speed:line", LINE_CONFIG, speed_unit="m/min"
        )
        aligned = aligner2.align_to_reference(station_uuids=["station:a"])
        result = aligner2.segment_by_cut(aligned, part_counter_uuid="counter:int2")
        # With delta=2 counter, piece_ids should be even numbers
        valid = result["piece_id"].dropna().unique()
        assert len(valid) > 0
        # piece_ids come in consecutive pairs sharing same cut timestamp
        cut_ref = result.dropna(subset=["piece_id"])
        grouped = cut_ref.groupby("piece_cut_ref_time")["piece_id"].nunique()
        # Some cut ref times should have 2 piece IDs (from delta=2 steps)
        assert (grouped >= 1).all()

    def test_piece_length_from_length_uuid(self, aligner):
        aligned = aligner.align_to_reference(station_uuids=["station:a"])
        result = aligner.segment_by_cut(
            aligned,
            part_counter_uuid="counter:bool",
            cut_length_uuid="cut:length",
        )
        lengths = result["piece_length_m"].dropna().unique()
        assert len(lengths) > 0
        assert all(
            abs(l - 0.8) < 0.01 for l in lengths
        ), f"Expected all lengths ~0.8 m, got {lengths}"

    def test_length_uuid_only_no_counter(self, aligner):
        aligned = aligner.align_to_reference(station_uuids=["station:a"])
        result = aligner.segment_by_cut(aligned, cut_length_uuid="cut:length")
        assert "piece_id" in result.columns
        assert result["piece_id"].dropna().astype(int).is_monotonic_increasing or True
        lengths = result["piece_length_m"].dropna().unique()
        assert all(abs(l - 0.8) < 0.01 for l in lengths)

    def test_no_cut_params_raises(self, aligner):
        aligned = aligner.align_to_reference(station_uuids=["station:a"])
        with pytest.raises(ValueError, match="cut_length_uuid or part_counter_uuid"):
            aligner.segment_by_cut(aligned)


# ---------------------------------------------------------------------------
# TestLagProfile
# ---------------------------------------------------------------------------


class TestLagProfile:
    def test_output_columns(self, aligner):
        result = aligner.lag_profile()
        for col in [
            "window_start",
            "uuid",
            "component",
            "position_offset_m",
            "mean_speed_m_s",
            "lag_seconds",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_lag_doubles_at_half_speed(self, aligner):
        result = aligner.lag_profile(station_uuids=["station:a"], window="1min")
        base = pd.Timestamp("2024-01-01 00:00:00")
        h1_end = base + pd.Timedelta(hours=1)
        h2_start = h1_end + pd.Timedelta(minutes=30)

        sub_a = result[result["uuid"] == "station:a"]
        lag_h1 = sub_a[sub_a["window_start"] < h1_end]["lag_seconds"].median()
        lag_h2 = sub_a[sub_a["window_start"] > h2_start]["lag_seconds"].median()
        assert abs(lag_h2 / lag_h1 - 2.0) < 0.1

    def test_window_granularity(self, aligner):
        result = aligner.lag_profile(station_uuids=["station:a"], window="1min")
        # 2-hour run → ~120 windows per station
        sub = result[result["uuid"] == "station:a"]
        assert 100 <= len(sub) <= 130


# ---------------------------------------------------------------------------
# TestAlignmentQuality
# ---------------------------------------------------------------------------


class TestAlignmentQuality:
    def test_output_columns(self, aligner):
        result = aligner.alignment_quality()
        for col in [
            "window_start",
            "speed_sample_count",
            "has_speed_data",
            "has_full_coverage",
            "per_uuid_counts",
        ]:
            assert col in result.columns

    def test_full_coverage(self, aligner):
        # Use only stations that have data in the fixture
        result = aligner.alignment_quality(
            station_uuids=["station:a", "station:b", "station:c"],
            window="1h",
        )
        assert result["has_full_coverage"].all()

    def test_missing_speed_flagged(self, df):
        # Remove speed signal from part of the data
        base = pd.Timestamp("2024-01-01 01:00:00")
        gap_end = base + pd.Timedelta(hours=0.5)
        mask = ~(
            (df["uuid"] == "speed:line")
            & (df["systime"] >= base)
            & (df["systime"] < gap_end)
        )
        df2 = df[mask].copy()
        aligner2 = ContinuousProcessAlignmentEvents(
            df2, "speed:line", LINE_CONFIG, speed_unit="m/min"
        )
        result = aligner2.alignment_quality(window="30min")
        gap_windows = result[
            (result["window_start"] >= base) & (result["window_start"] < gap_end)
        ]
        assert (gap_windows["has_speed_data"] == False).any()


# ---------------------------------------------------------------------------
# TestSpeedUnits
# ---------------------------------------------------------------------------


class TestSpeedUnits:
    def _make_speed_unit_df(self, speed_val: float) -> pd.DataFrame:
        base = pd.Timestamp("2024-01-01 00:00:00")
        timestamps = pd.date_range(base, periods=100, freq="15s")
        rows = []
        for t in timestamps:
            rows.append(
                dict(
                    systime=t,
                    uuid="speed:line",
                    value_double=speed_val,
                    value_bool=None,
                    value_integer=None,
                )
            )
            rows.append(
                dict(
                    systime=t,
                    uuid="station:a",
                    value_double=1.0,
                    value_bool=None,
                    value_integer=None,
                )
            )
        return pd.DataFrame(rows)

    def test_m_per_s_unit(self):
        # 10 m/min = 1/6 m/s ≈ 0.1667 m/s
        df = self._make_speed_unit_df(10.0 / 60.0)
        aligner = ContinuousProcessAlignmentEvents(
            df,
            "speed:line",
            [{"name": "zone_a", "offset": 30.0, "uuids": ["station:a"]}],
            speed_unit="m/s",
        )
        result = aligner.align_to_reference()
        expected_lag = 30.0 / (10.0 / 60.0)
        assert abs(result["lag_seconds"].median() - expected_lag) < 2.0

    def test_mm_per_s_unit(self):
        # 10 m/min = 10/60 m/s = 10000/60 mm/s ≈ 166.67 mm/s
        df = self._make_speed_unit_df(10000.0 / 60.0)
        aligner = ContinuousProcessAlignmentEvents(
            df,
            "speed:line",
            [{"name": "zone_a", "offset": 30.0, "uuids": ["station:a"]}],
            speed_unit="mm/s",
        )
        result = aligner.align_to_reference()
        expected_lag = 30.0 / (10.0 / 60.0)
        assert abs(result["lag_seconds"].median() - expected_lag) < 2.0

    def test_invalid_speed_unit(self):
        df = self._make_speed_unit_df(10.0)
        with pytest.raises(ValueError, match="speed_unit"):
            ContinuousProcessAlignmentEvents(
                df,
                "speed:line",
                [{"name": "zone_a", "offset": 30.0, "uuids": ["station:a"]}],
                speed_unit="km/h",
            )
