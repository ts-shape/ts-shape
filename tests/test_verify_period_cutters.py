"""Verification of the period cutters with hand-derived expectations.

An order / recipe / batch signal defines periods; every other signal's rows
must be cut into those periods exactly once, with the period's identity.
Covers SegmentExtractor, SegmentProcessor, TimeWindowedFeatureTable,
ContinuousProcessAlignmentEvents.segment_by_cut and the production-day
helper used by shift windows.
"""

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.events.production import ContinuousProcessAlignmentEvents
from ts_shape.features.segment_analysis import _grouped_stats
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor
from ts_shape.features.segment_analysis.time_windowed_features import (
    TimeWindowedFeatureTable,
)
from ts_shape.features.stats.numeric_stats import NumericStatistics
from ts_shape.utils.shifts import DEFAULT_SHIFTS, assign_shift, production_dates

TZ = "Europe/Berlin"


def _t(text: str, tz: str | None = TZ) -> pd.Timestamp:
    return pd.Timestamp(text, tz=tz)


def _dst_line() -> pd.DataFrame:
    """Order signal (change-encoded) + two process signals over the spring DST night.

    order: A @00:00, B @01:00, B @03:30 (re-sent), C @05:00 (local, 2024-03-31;
    02:00-03:00 does not exist). temp every 10 min 23:50 .. 06:00, pressure
    every 25 min 00:05 .. 05:55.
    """
    order = pd.DataFrame(
        {
            "systime": [
                _t("2024-03-31 00:00"),
                _t("2024-03-31 01:00"),
                _t("2024-03-31 03:30"),
                _t("2024-03-31 05:00"),
            ],
            "uuid": "order",
            "value_string": ["A", "B", "B", "C"],
        }
    )
    temp_t = pd.date_range(_t("2024-03-30 23:50"), _t("2024-03-31 06:00"), freq="10min")
    temp = pd.DataFrame(
        {"systime": temp_t, "uuid": "temp", "value_double": np.arange(len(temp_t))}
    )
    press_t = pd.date_range(
        _t("2024-03-31 00:05"), _t("2024-03-31 05:55"), freq="25min"
    )
    press = pd.DataFrame({"systime": press_t, "uuid": "press", "value_double": 100.0})
    df = pd.concat([order, temp, press], ignore_index=True)
    # Interleave the signals the way a historian export does.
    df = df.sample(frac=1.0, random_state=3).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def _expected_order(ts: pd.Timestamp) -> str | None:
    if ts < _t("2024-03-31 00:00"):
        return None
    if ts < _t("2024-03-31 01:00"):
        return "A"
    if ts < _t("2024-03-31 05:00"):
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# SegmentExtractor
# ---------------------------------------------------------------------------


class TestExtractTimeRanges:
    def test_dst_change_encoded_ranges(self):
        r = SegmentExtractor.extract_time_ranges(_dst_line(), "order")
        assert list(r["segment_value"]) == ["A", "B", "C"]
        assert list(r["segment_index"]) == [0, 1, 2]
        assert list(r["segment_start"]) == [
            _t("2024-03-31 00:00"),
            _t("2024-03-31 01:00"),
            _t("2024-03-31 05:00"),
        ]
        # The last (open) segment runs to the end of the data (06:00), not to
        # its own single sample.
        assert list(r["segment_end"]) == [
            _t("2024-03-31 01:00"),
            _t("2024-03-31 05:00"),
            _t("2024-03-31 06:00"),
        ]
        # B spans the DST jump: 01:00+01 -> 05:00+02 is 3 real hours.
        assert list(r["duration_seconds"]) == [3600.0, 10800.0, 3600.0]
        assert (r["segment_duration"].dt.total_seconds() == r["duration_seconds"]).all()
        assert (r["start"] == r["segment_start"]).all()
        assert (r["end"] == r["segment_end"]).all()
        assert (r["source_uuid"] == "order").all()
        assert list(r["end_inclusive"]) == [False, False, True]
        assert str(r["segment_start"].dt.tz) == TZ
        assert str(r["end"].dt.tz) == TZ

    def test_data_end_explicit_and_only_extends(self):
        df = _dst_line()
        r = SegmentExtractor.extract_time_ranges(
            df, "order", data_end=pd.Timestamp("2024-03-31 07:00")
        )
        assert r["segment_end"].iloc[-1] == _t("2024-03-31 07:00")
        r = SegmentExtractor.extract_time_ranges(
            df, "order", data_end=_t("2024-03-31 04:00")
        )
        # Never before the segment's own last sample.
        assert r["segment_end"].iloc[-1] == _t("2024-03-31 05:00")

    def test_signal_only_frame_last_segment_is_zero_length(self):
        df = _dst_line()
        r = SegmentExtractor.extract_time_ranges(df[df.uuid == "order"], "order")
        assert r["duration_seconds"].iloc[-1] == 0.0

    def test_last_sample_mode(self):
        r = SegmentExtractor.extract_time_ranges(
            _dst_line(), "order", interval_end="last_sample"
        )
        assert list(r["segment_end"]) == [
            _t("2024-03-31 00:00"),
            _t("2024-03-31 03:30"),
            _t("2024-03-31 05:00"),
        ]
        assert r["end_inclusive"].all()

    def test_nan_joins_empty_string_is_gap_leading_nan_skipped(self):
        times = pd.date_range("2024-01-01 08:00", periods=8, freq="10min")
        df = pd.DataFrame(
            {
                "systime": times,
                "uuid": "batch",
                "value_string": [None, "A", None, "A", "", "  ", "B", np.nan],
            }
        )
        r = SegmentExtractor.extract_time_ranges(df, "batch")
        assert list(r["segment_value"]) == ["A", "B"]
        assert list(r["segment_start"]) == [times[1], times[6]]
        # A ends where the empty-id run starts (08:40), B at the data end.
        assert list(r["segment_end"]) == [times[4], times[7]]
        assert list(r["end_inclusive"]) == [False, True]

    def test_value_returns_later_is_new_segment(self):
        times = pd.date_range("2024-01-01", periods=6, freq="1h")
        df = pd.DataFrame(
            {
                "systime": times,
                "uuid": "o",
                "value_integer": [7, 7, 8, 8, 7, 7],
            }
        )
        r = SegmentExtractor.extract_time_ranges(df, "o", value_column="value_integer")
        assert list(r["segment_value"]) == [7, 8, 7]
        assert list(r["segment_index"]) == [0, 1, 2]
        assert list(r["duration_seconds"]) == [7200.0, 7200.0, 3600.0]

    @pytest.mark.parametrize(
        "dtype,values",
        [
            ("string[pyarrow]", ["A", None, "B", "B", "C"]),
            ("string", ["A", None, "B", "B", "C"]),
            ("Int64", [1, None, 2, 2, 3]),
            ("int64[pyarrow]", [1, None, 2, 2, 3]),
        ],
    )
    def test_nullable_and_pyarrow_ids(self, dtype, values):
        times = pd.date_range("2024-01-01", periods=5, freq="1min", tz="UTC")
        df = pd.DataFrame(
            {
                "systime": times,
                "uuid": "o",
                "v": pd.array(values, dtype=dtype),
            }
        )
        r = SegmentExtractor.extract_time_ranges(df, "o", value_column="v")
        assert list(r["segment_start"]) == [times[0], times[2], times[4]]
        assert list(r["segment_end"]) == [times[2], times[4], times[4]]
        assert [str(v) for v in r["segment_value"]] == [
            str(values[0]),
            str(values[2]),
            str(values[4]),
        ]

    def test_unsorted_and_same_timestamp_keeps_input_order(self):
        t0 = pd.Timestamp("2024-01-01 10:00")
        df = pd.DataFrame(
            {
                "systime": [t0 + pd.Timedelta("20min"), t0, t0],
                "uuid": "o",
                "value_string": ["C", "A", "B"],
            }
        )
        r = SegmentExtractor.extract_time_ranges(df, "o")
        assert list(r["segment_value"]) == ["A", "B", "C"]
        assert r["duration_seconds"].tolist() == [0.0, 1200.0, 0.0]

    def test_min_duration_renumbers(self):
        times = pd.to_datetime(
            ["2024-01-01 00:00:00", "2024-01-01 00:00:05", "2024-01-01 01:00:00"]
        )
        df = pd.DataFrame(
            {"systime": times, "uuid": "o", "value_string": ["A", "X", "B"]}
        )
        r = SegmentExtractor.extract_time_ranges(df, "o", min_duration="1min")
        assert list(r["segment_value"]) == ["X"]
        assert list(r["segment_index"]) == [0]

    def test_empty_result_has_full_schema_and_tz(self):
        df = _dst_line()
        r = SegmentExtractor.extract_time_ranges(df, "nope")
        assert r.empty
        assert "segment_start" in r and "duration_seconds" in r and "end_inclusive" in r
        assert str(r["segment_start"].dtype) == str(df["systime"].dtype)

    def test_nat_rows_warn_and_are_skipped(self):
        df = pd.DataFrame(
            {
                "systime": [
                    pd.Timestamp("2024-01-01"),
                    pd.NaT,
                    pd.Timestamp("2024-01-02"),
                ],
                "uuid": "o",
                "value_string": ["A", "Z", "B"],
            }
        )
        with pytest.warns(DataQualityWarning, match="no timestamp"):
            r = SegmentExtractor.extract_time_ranges(df, "o")
        assert list(r["segment_value"]) == ["A", "B"]


# ---------------------------------------------------------------------------
# SegmentProcessor.apply_ranges
# ---------------------------------------------------------------------------


class TestApplyRanges:
    def test_each_row_exactly_once_with_identity(self):
        df = _dst_line()
        r = SegmentExtractor.extract_time_ranges(df, "order")
        out = SegmentProcessor.apply_ranges(df, r, target_uuids=["temp", "press"])
        assert not out["row_id"].duplicated().any()
        process = df[df.uuid.isin(["temp", "press"])]
        expected = process[process.systime >= _t("2024-03-31 00:00")]
        assert set(out["row_id"]) == set(expected["row_id"])
        # Only the 23:50 temp sample (before the first order) is dropped.
        assert len(process) - len(out) == 1
        for ts, val in zip(out["systime"], out["segment_value"], strict=True):
            assert val == _expected_order(ts)
        # Exact boundary samples go to the period that starts there.
        at = out.set_index("systime")
        assert at.loc[_t("2024-03-31 01:00"), "segment_value"] == "B"
        assert at.loc[_t("2024-03-31 05:00"), "segment_value"] == "C"
        assert at.loc[_t("2024-03-31 06:00"), "segment_value"] == "C"
        assert str(out["systime"].dt.tz) == TZ

    def test_round_trip_all_signals(self):
        df = _dst_line()
        r = SegmentExtractor.extract_time_ranges(df, "order")
        out = SegmentProcessor.apply_ranges(df, r)
        kept = df[df.systime >= r["segment_start"].min()]
        assert sorted(out["row_id"]) == sorted(kept["row_id"])
        # Per-period counts add up and match a hand count of temp rows:
        # A 00:00..00:50 (6), B 01:00..04:50 local (= 3 real hours, 18),
        # C 05:00..06:00 (7).
        counts = out[out.uuid == "temp"].groupby("segment_index").size()
        assert counts.tolist() == [6, 18, 7]

    def test_gap_rows_dropped_boundary_not_attributed(self):
        times = pd.date_range("2024-01-01 08:00", periods=5, freq="10min")
        order = pd.DataFrame(
            {"systime": times[[0, 2, 4]], "uuid": "o", "value_string": ["A", "", "B"]}
        )
        data = pd.DataFrame({"systime": times, "uuid": "x", "value_double": range(5)})
        df = pd.concat([order, data], ignore_index=True)
        r = SegmentExtractor.extract_time_ranges(df, "o")
        out = SegmentProcessor.apply_ranges(df, r, target_uuids=["x"])
        # 08:00, 08:10 -> A; 08:20, 08:30 are in the empty-id gap; 08:40 -> B.
        assert out["value_double"].tolist() == [0, 1, 4]
        assert out["segment_value"].tolist() == ["A", "A", "B"]

    def test_last_sample_ranges_keep_end_sample(self):
        times = pd.date_range("2024-01-01", periods=6, freq="1min")
        order = pd.DataFrame(
            {"systime": times, "uuid": "o", "value_string": list("AAABBB")}
        )
        data = pd.DataFrame(
            {
                "systime": times + pd.Timedelta("30s"),
                "uuid": "x",
                "value_double": range(6),
            }
        )
        df = pd.concat([order, data], ignore_index=True)
        r = SegmentExtractor.extract_time_ranges(df, "o", interval_end="last_sample")
        out = SegmentProcessor.apply_ranges(df, r)
        # last_sample: A = [00:00, 00:02], B = [00:03, 00:05]; x@00:02:30
        # falls between them and is dropped, as is x@00:05:30 after B.
        x = out[out.uuid == "x"]
        assert x["value_double"].tolist() == [0, 1, 3, 4]
        o = out[out.uuid == "o"]
        assert o["segment_value"].tolist() == list("AAABBB")

    def test_duplicate_timestamp_switch_goes_to_later_value(self):
        t0 = pd.Timestamp("2024-01-01 10:00")
        df = pd.DataFrame(
            {
                "systime": [t0, t0, t0, t0 + pd.Timedelta("1min")],
                "uuid": ["o", "o", "x", "x"],
                "value_string": ["A", "B", None, None],
                "value_double": [np.nan, np.nan, 1.0, 2.0],
            }
        )
        r = SegmentExtractor.extract_time_ranges(df, "o")
        out = SegmentProcessor.apply_ranges(df, r, target_uuids=["x"])
        assert out["segment_value"].tolist() == ["B", "B"]

    def test_naive_ranges_on_aware_data_and_aware_on_naive_raises(self):
        df = _dst_line()
        ranges = pd.DataFrame(
            {
                "segment_start": [pd.Timestamp("2024-03-31 00:00")],
                "segment_end": [pd.Timestamp("2024-03-31 01:00")],
                "segment_value": ["A"],
                "segment_index": [0],
            }
        )
        out = SegmentProcessor.apply_ranges(df, ranges, target_uuids=["temp"])
        # Read as Berlin wall clock; end inclusive (nothing starts at 01:00).
        assert out["systime"].min() == _t("2024-03-31 00:00")
        assert out["systime"].max() == _t("2024-03-31 01:00")
        naive = df.assign(systime=df.systime.dt.tz_localize(None))
        aware = ranges.assign(
            segment_start=ranges.segment_start.dt.tz_localize(TZ),
            segment_end=ranges.segment_end.dt.tz_localize(TZ),
        )
        with pytest.raises(ValueError, match="tz-aware"):
            SegmentProcessor.apply_ranges(naive, aware)

    def test_ranges_in_other_zone_are_converted(self):
        df = _dst_line()
        r = SegmentExtractor.extract_time_ranges(df, "order")
        r_utc = r.assign(
            segment_start=r.segment_start.dt.tz_convert("UTC"),
            segment_end=r.segment_end.dt.tz_convert("UTC"),
        )
        a = SegmentProcessor.apply_ranges(df, r)
        b = SegmentProcessor.apply_ranges(df, r_utc)
        pd.testing.assert_frame_equal(a, b)

    def test_overlapping_hand_built_ranges_duplicate_with_warning(self, caplog):
        times = pd.date_range("2024-01-01", periods=5, freq="1min")
        df = pd.DataFrame({"systime": times, "uuid": "x", "value_double": range(5)})
        ranges = pd.DataFrame(
            {
                "segment_start": [times[0], times[2]],
                "segment_end": [times[3], times[4]],
                "segment_value": ["P", "Q"],
                "segment_index": [0, 1],
            }
        )
        out = SegmentProcessor.apply_ranges(df, ranges)
        assert "overlapping" in caplog.text.lower()
        # No end_inclusive column: an end is inclusive when no range starts
        # there, so P = [0, 3] and Q = [2, 4]; rows 2 and 3 appear twice.
        assert out["segment_value"].tolist() == ["P"] * 4 + ["Q"] * 3
        assert out["value_double"].tolist() == [0, 1, 2, 3, 2, 3, 4]

    def test_empty_ranges_and_no_match_keep_columns(self):
        df = _dst_line()
        empty = SegmentExtractor.extract_time_ranges(df, "nope")
        out = SegmentProcessor.apply_ranges(df, empty)
        assert out.empty
        assert {"segment_value", "segment_index", "systime", "uuid"} <= set(out.columns)


# ---------------------------------------------------------------------------
# Profiles and windowed features
# ---------------------------------------------------------------------------


def test_metric_profiles_hand_values_and_matches_numeric_statistics():
    times = pd.date_range("2024-01-01", periods=10, freq="1min")
    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": "x",
            "segment_index": [0] * 5 + [1] * 5,
            "value_double": [1.0, 2.0, 2.0, 3.0, np.nan, 5.0, 5.0, 5.0, 5.0, 9.0],
        }
    )
    prof = SegmentProcessor.compute_metric_profiles(df, group_column="segment_index")
    first = prof.iloc[0]
    assert first["sample_count"] == 4
    assert first["mean"] == 2.0
    assert first["median"] == 2.0
    assert first["mode"] == 2.0
    assert first["q1"] == 1.75 and first["q3"] == 2.25
    # IQR ignores the NaN (it used to be NaN for any group with a gap).
    assert first["iqr"] == pytest.approx(0.5)
    assert first["mad"] == pytest.approx(0.5)
    assert first["var"] == pytest.approx(2.0 / 3.0)
    for _, row in prof.iterrows():
        seg = df[df.segment_index == row["segment_index"]]
        ref = NumericStatistics.summary_as_dict(seg, "value_double")
        for k, v in ref.items():
            if k == "iqr":
                continue
            assert row[k] == pytest.approx(v, nan_ok=True, rel=1e-12), k


def test_grouped_stats_constant_and_tiny_groups():
    codes = np.array([0, 0, 0, 0, 1, 1, 2])
    vals = np.array([3.0, 3.0, 3.0, 3.0, 1.0, 2.0, 4.0])
    out = _grouped_stats.grouped_summary(codes, vals, 3)
    assert out.loc[0, "skewness"] == 0.0 and out.loc[0, "kurtosis"] == 0.0
    assert np.isnan(out.loc[1, "skewness"]) and np.isnan(out.loc[1, "kurtosis"])
    assert out.loc[1, "mode"] == 1.0  # tie -> smallest
    assert np.isnan(out.loc[2, "std"])


def test_windowed_features_split_at_segment_change_and_window_columns():
    df = _dst_line()
    r = SegmentExtractor.extract_time_ranges(df, "order")
    cut = SegmentProcessor.apply_ranges(df, r, target_uuids=["temp"])
    long_df = TimeWindowedFeatureTable.compute_long(cut, freq="1h", metrics=["mean"])
    # 00:00-01:00 window: A 00:00..00:50 (values 1..6).
    w0 = long_df[(long_df.time_window == _t("2024-03-31 00:00"))]
    assert w0["segment_value"].tolist() == ["A"]
    assert w0["sample_count"].tolist() == [6]
    assert w0["mean"].tolist() == [3.5]
    # 05:00 window holds 05:00..05:50 of C (06:00 is in the next window,
    # alone, so it has < 2 values and is left out).
    w5 = long_df[long_df.time_window == _t("2024-03-31 05:00")]
    assert w5["segment_value"].tolist() == ["C"] and w5["sample_count"].tolist() == [6]
    assert (long_df["start"] == long_df["time_window"]).all()
    assert (long_df["duration_seconds"] == 3600.0).all()
    assert list(long_df.columns[-3:]) == ["start", "end", "duration_seconds"]
    wide = TimeWindowedFeatureTable.compute(cut, freq="1h", metrics=["mean"])
    assert "temp__duration_seconds" not in wide.columns
    assert "temp__mean" in wide.columns


def test_windowed_features_daily_window_on_dst_day():
    t = pd.date_range(_t("2024-03-30 12:00"), _t("2024-04-01 12:00"), freq="1h")
    df = pd.DataFrame({"systime": t, "uuid": "x", "value_double": 1.0})
    out = TimeWindowedFeatureTable.compute_long(
        df, freq="1D", segment_column=None, metrics=["sum"]
    )
    day = out[out.time_window == _t("2024-03-31 00:00")].iloc[0]
    assert day["sample_count"] == 23
    assert day["end"] == _t("2024-04-01 00:00")


def test_windowed_features_empty_keeps_columns():
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime([]),
            "uuid": [],
            "value_double": [],
            "segment_value": [],
        }
    )
    out = TimeWindowedFeatureTable.compute_long(df, metrics=["mean"])
    assert list(out.columns) == [
        "time_window",
        "uuid",
        "segment_value",
        "sample_count",
        "mean",
        "start",
        "end",
        "duration_seconds",
    ]


def test_cutting_100k_rows_is_fast():
    rng = np.random.default_rng(0)
    n = 100_000
    t0 = _t("2024-03-30")
    times = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 86400, n)), unit="s")
    df = pd.DataFrame(
        {
            "systime": times,
            "uuid": rng.choice(["a", "b", "c", "d"], n),
            "value_double": rng.normal(size=n),
        }
    )
    ot = t0 + pd.to_timedelta(np.sort(rng.uniform(0, 3 * 86400, 1000)), unit="s")
    orders = pd.DataFrame(
        {
            "systime": ot,
            "uuid": "order",
            "value_string": [f"O{i // 2}" for i in range(1000)],
        }
    )
    df = pd.concat([df, orders], ignore_index=True)
    start = time.perf_counter()
    r = SegmentExtractor.extract_time_ranges(df, "order")
    cut = SegmentProcessor.apply_ranges(df, r, target_uuids=["a", "b", "c", "d"])
    SegmentProcessor.compute_metric_profiles(cut, group_column="segment_index")
    TimeWindowedFeatureTable.compute(cut, freq="10min")
    assert time.perf_counter() - start < 5.0
    assert len(r) == 500
    in_range = (df.uuid != "order") & (df.systime >= r.segment_start.min())
    assert len(cut) == int(in_range.sum())


# ---------------------------------------------------------------------------
# ContinuousProcessAlignmentEvents.segment_by_cut
# ---------------------------------------------------------------------------


def _cut_line(tz: str | None, speed_rows=None, cut_rows=None) -> pd.DataFrame:
    base = pd.Timestamp("2024-03-31 01:59:30", tz=tz)
    rows = [
        dict(systime=base + pd.Timedelta(seconds=s), uuid="st", value_double=float(s))
        for s in range(36)
    ]
    for s, v in speed_rows or [(0, 60.0)]:
        rows.append(
            dict(systime=base + pd.Timedelta(seconds=s), uuid="speed", value_double=v)
        )
    for s, length in cut_rows or [(10, 10.0), (20, 11.0), (30, 12.0)]:
        rows.append(
            dict(
                systime=base + pd.Timedelta(seconds=s),
                uuid="cut:len",
                value_double=length,
            )
        )
    return pd.DataFrame(rows)


_CFG = [
    {"name": "st", "offset": 0.0, "uuids": ["st"]},
    {"name": "cutter", "offset": 5.0, "uuids": ["cut:len"]},
]


@pytest.mark.parametrize("tz", [None, TZ])
def test_segment_by_cut_pieces_are_cut_material(tz):
    df = _cut_line(tz)
    c = ContinuousProcessAlignmentEvents(df, "speed", _CFG)
    aligned = c.align_to_reference(station_uuids=["st"])
    out = c.segment_by_cut(aligned, cut_length_uuid="cut:len")
    # 1 m/s, cutter 5 m downstream: cuts at +10/+20/+30 s reach the reference
    # at +5/+15/+25 s. Material (−inf, 5] -> 1, (5, 15] -> 2, (15, 25] -> 3,
    # after 25 s not cut yet.
    secs = out["value_double"].astype(int)
    exp = np.select([secs <= 5, secs <= 15, secs <= 25], [1, 2, 3], 0)
    got = out["piece_id"].fillna(0).astype(int).to_numpy()
    assert (got == exp).all()
    assert out["piece_id"].isna().sum() == 10
    lengths = out.groupby("piece_id")["piece_length_m"].first().tolist()
    assert lengths == [10.0, 11.0, 12.0]
    base = pd.Timestamp("2024-03-31 01:59:30", tz=tz)
    refs = (
        out.dropna(subset=["piece_id"])
        .groupby("piece_id")["piece_cut_ref_time"]
        .first()
    )
    assert refs.tolist() == [base + pd.Timedelta(seconds=s) for s in (5, 15, 25)]
    assert out["piece_cut_ref_time"].dtype == aligned["material_ref_time"].dtype


def test_segment_by_cut_counter_baseline_gaps_and_steps():
    base = pd.Timestamp("2024-01-01 00:00:00")
    rows = [
        dict(systime=base + pd.Timedelta(seconds=s), uuid="st", value_double=float(s))
        for s in range(0, 60)
    ]
    rows.append(dict(systime=base, uuid="speed", value_double=60.0))
    # Counter: NaN first (gap), baseline 500, +1, NaN, +2 (two pieces in one step).
    for s, v in [(5, None), (10, 500), (20, 501), (25, None), (40, 503)]:
        rows.append(
            dict(systime=base + pd.Timedelta(seconds=s), uuid="cnt", value_integer=v)
        )
    df = pd.DataFrame(rows)
    df["value_integer"] = df["value_integer"].astype("Int64")
    cfg = [{"name": "st", "offset": 0.0, "uuids": ["st"]}]
    c = ContinuousProcessAlignmentEvents(df, "speed", cfg)
    out = c.segment_by_cut(c.align_to_reference(), part_counter_uuid="cnt")
    # Pieces: 1 cut @20 s, pieces 2+3 cut @40 s. The leading NaN is a gap, not
    # a counter at 0 (which used to emit 500 pieces).
    assert int(out["piece_id"].max()) == 2
    secs = out["value_double"].astype(int)
    assert (out.loc[secs <= 20, "piece_id"] == 1).all()
    assert (out.loc[(secs > 20) & (secs <= 40), "piece_id"] == 2).all()
    assert out.loc[secs > 40, "piece_id"].isna().all()


def test_segment_by_cut_speed_gaps_hold_last_speed():
    # Speed: NaN reading at +12 s must not turn into min_speed (a 500 s lag).
    df = _cut_line(None, speed_rows=[(3, 60.0), (12, np.nan)])
    c = ContinuousProcessAlignmentEvents(df, "speed", _CFG)
    aligned = c.align_to_reference(station_uuids=["st", "cut:len"])
    cut_rows = aligned[aligned.uuid == "cut:len"]
    assert (cut_rows["lag_seconds"] == 5.0).all()
    # Rows before the first speed reading use the first reading.
    assert (aligned["lag_seconds"].isin([0.0, 5.0])).all()
    out = c.segment_by_cut(aligned, cut_length_uuid="cut:len")
    assert out["piece_cut_ref_time"].dropna().nunique() == 3


def test_segment_by_cut_100k_rows_is_fast():
    n = 100_000
    b = _t("2024-03-30 22:00")
    t = b + pd.to_timedelta(np.arange(n) * 2.0, unit="s")
    df = pd.concat(
        [
            pd.DataFrame({"systime": t, "uuid": "st", "value_double": 1.0}),
            pd.DataFrame({"systime": t[::30], "uuid": "speed", "value_double": 30.0}),
            pd.DataFrame(
                {
                    "systime": t[::20],
                    "uuid": "cnt",
                    "value_integer": np.arange(len(t[::20])),
                }
            ),
        ],
        ignore_index=True,
    )
    cfg = [
        {"name": "st", "offset": 10.0, "uuids": ["st"]},
        {"name": "cut", "offset": 20.0, "uuids": ["cnt"]},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = ContinuousProcessAlignmentEvents(df, "speed", cfg)
    start = time.perf_counter()
    out = c.segment_by_cut(
        c.align_to_reference(station_uuids=["st"]), part_counter_uuid="cnt"
    )
    assert time.perf_counter() - start < 3.0
    assert int(out["piece_id"].max()) == len(t[::20]) - 1
    assert str(out["piece_cut_ref_time"].dt.tz) == TZ


# ---------------------------------------------------------------------------
# Production day for overnight shifts across DST
# ---------------------------------------------------------------------------


def test_production_date_day_after_dst_change():
    t = pd.Series(
        pd.to_datetime(
            [
                "2024-03-31 23:00",  # night shift starting on the DST day
                "2024-04-01 00:30",  # its post-midnight part (23 h day before)
                "2024-04-01 05:59",
                "2024-10-27 23:30",  # night shift starting on the fall-back day
                "2024-10-28 00:30",
            ]
        ).tz_localize(TZ)
    )
    shifts = assign_shift(t, DEFAULT_SHIFTS)
    assert shifts.tolist() == ["shift_3"] * 5
    days = production_dates(t, shifts, DEFAULT_SHIFTS)
    assert [str(d) for d in days] == [
        "2024-03-31",
        "2024-03-31",
        "2024-03-31",
        "2024-10-27",
        "2024-10-27",
    ]
