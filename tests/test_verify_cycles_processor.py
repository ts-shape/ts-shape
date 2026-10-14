"""Verification of CycleDataProcessor: sample-to-cycle assignment and per-cycle results.

Rule under test: a sample belongs to a cycle when
``cycle_start <= systime < cycle_end``; open cycles get no samples.
"""

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.features.cycles.cycle_processor import CycleDataProcessor
from ts_shape.features.cycles.cycles_extractor import CycleExtractor

BERLIN = "Europe/Berlin"


def T(minutes: float, tz: str | None = None) -> pd.Timestamp:
    return pd.Timestamp("2024-03-05 06:00", tz=tz) + pd.Timedelta(minutes=minutes)


def cycles(spans: list[tuple], tz: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cycle_start": [T(s, tz) for s, _ in spans],
            "cycle_end": [pd.NaT if e is None else T(e, tz) for _, e in spans],
            "cycle_uuid": [f"c{i + 1}" for i in range(len(spans))],
            "is_complete": [e is not None for _, e in spans],
        }
    )


def values(points: list[tuple], tz: str | None = None) -> pd.DataFrame:
    """points: (minute, uuid, value_double)."""
    return pd.DataFrame(
        {
            "systime": [T(m, tz) for m, _, _ in points],
            "uuid": [u for _, u, _ in points],
            "value_double": [v for _, _, v in points],
        }
    )


def minutes(frame: pd.DataFrame, tz: str | None = None) -> list[float]:
    return [(t - T(0, tz)) / pd.Timedelta(minutes=1) for t in frame["systime"]]


BACK_TO_BACK = [(0, 5), (5, 9), (12, None)]
POINTS = [
    (0, "temp", 10.0),
    (1, "temp", 20.0),
    (1, "pressure", 1.0),
    (5, "temp", 30.0),
    (7, "temp", 50.0),
    (9, "temp", 99.0),  # exactly at the end of c2: in no cycle
    (12, "temp", 5.0),  # c3 has no end: in no cycle
    (13, "temp", 6.0),
]


def test_half_open_assignment_back_to_back_and_open_cycle():
    proc = CycleDataProcessor(cycles(BACK_TO_BACK), values(POINTS))
    merged = proc.merge_dataframes_by_cycle()
    assert minutes(merged) == [0, 1, 1, 5, 7]
    assert merged["cycle_uuid"].tolist() == ["c1", "c1", "c1", "c2", "c2"]
    assert merged.index.tolist() == [0, 1, 2, 3, 4]

    stats = proc.compute_cycle_statistics()
    assert stats["cycle_uuid"].tolist() == ["c1", "c2", "c3"]
    assert stats["value_count"].tolist() == [3, 2, 0]
    assert stats["sample_count"].tolist() == [3, 2, 0]
    assert stats["value_count"].sum() == len(merged)  # every sample counted once
    assert stats["unique_uuids"].tolist() == [2, 1, 0]
    assert stats["duration_seconds"].iloc[:2].tolist() == [300.0, 240.0]
    assert np.isnan(stats["duration_seconds"].iloc[2])
    assert stats["mean_value_double"].iloc[:2].tolist() == pytest.approx([31 / 3, 40.0])
    assert stats["std_value_double"].iloc[1] == pytest.approx(np.std([30, 50], ddof=1))
    assert np.isnan(stats["mean_value_double"].iloc[2])
    assert (stats["start"] == stats["cycle_start"]).all()

    split = proc.split_by_cycle()
    assert list(split) == ["c1", "c2"]
    assert minutes(split["c2"]) == [5, 7]
    assert "cycle_uuid" not in split["c1"].columns

    groups = proc.group_by_cycle_uuid()
    assert [len(g) for g in groups] == [3, 2]
    assert proc.split_dataframes_by_group(groups, "uuid")[0][
        "uuid"
    ].unique().tolist() == ["pressure"]


def test_unsorted_values_keep_index_and_come_back_time_sorted():
    shuffled = values(POINTS).sample(frac=1, random_state=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = CycleDataProcessor(cycles(BACK_TO_BACK), shuffled)
    merged = proc.merge_dataframes_by_cycle()
    assert minutes(merged) == [0, 1, 1, 5, 7]
    assert sorted(merged.index.tolist()) == [0, 1, 2, 3, 4]
    assert list(proc.split_by_cycle()) == ["c1", "c2"]


def test_overlapping_cycles_share_samples():
    proc = CycleDataProcessor(cycles([(0, 6), (4, 10)]), values(POINTS))
    merged = proc.merge_dataframes_by_cycle()
    assert list(zip(minutes(merged), merged["cycle_uuid"])) == [
        (0, "c1"),
        (1, "c1"),
        (1, "c1"),
        (5, "c1"),
        (5, "c2"),
        (7, "c2"),
        (9, "c2"),
    ]
    assert proc.compute_cycle_statistics()["value_count"].tolist() == [4, 3]


def test_tz_aware_cycles_match_values_in_another_zone():
    cyc = cycles(BACK_TO_BACK, tz=BERLIN)
    vals = values(POINTS, tz=BERLIN)
    vals["systime"] = vals["systime"].dt.tz_convert("UTC")
    proc = CycleDataProcessor(cyc, vals)
    merged = proc.merge_dataframes_by_cycle()
    assert merged["cycle_uuid"].tolist() == ["c1", "c1", "c1", "c2", "c2"]
    assert str(merged["systime"].dt.tz) == "UTC"
    stats = proc.compute_cycle_statistics()
    assert str(stats["cycle_start"].dt.tz) == BERLIN
    assert str(stats["end"].dt.tz) == BERLIN


def test_tz_aware_against_naive_raises():
    with pytest.raises(TypeError, match="tz-aware"):
        CycleDataProcessor(cycles(BACK_TO_BACK, tz=BERLIN), values(POINTS))


def test_missing_columns_raise():
    with pytest.raises(ColumnNotFoundError, match="cycle_end"):
        CycleDataProcessor(
            cycles(BACK_TO_BACK).drop(columns="cycle_end"), values(POINTS)
        )
    proc = CycleDataProcessor(cycles(BACK_TO_BACK), values(POINTS))
    with pytest.raises(ColumnNotFoundError, match="nope"):
        proc.compare_cycles("c1", metric="nope")


def test_no_cycles_gives_empty_results_with_columns():
    empty = cycles([]).astype(
        {"cycle_start": "datetime64[ns]", "cycle_end": "datetime64[ns]"}
    )
    proc = CycleDataProcessor(empty, values(POINTS))
    merged = proc.merge_dataframes_by_cycle()
    assert merged.empty
    assert list(merged.columns) == ["systime", "uuid", "value_double", "cycle_uuid"]
    assert proc.split_by_cycle() == {}
    assert proc.group_by_cycle_uuid() == []
    stats = proc.compute_cycle_statistics()
    assert stats.empty
    for column in ("cycle_uuid", "duration_seconds", "value_count", "start", "end"):
        assert column in stats.columns


def test_nat_value_timestamps_belong_to_no_cycle():
    vals = values(POINTS)
    vals.loc[0, "systime"] = pd.NaT
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataQualityWarning)
        proc = CycleDataProcessor(cycles(BACK_TO_BACK), vals)
    assert proc.merge_dataframes_by_cycle()["cycle_uuid"].tolist() == [
        "c1",
        "c1",
        "c2",
        "c2",
    ]


@pytest.mark.parametrize("dtype", ["Float64", "double[pyarrow]"])
def test_nullable_value_dtypes(dtype):
    vals = values(POINTS)
    vals["value_double"] = pd.array(
        [10.0, None, 1.0, 30.0, 50.0, 99.0, 5.0, 6.0], dtype=dtype
    )
    proc = CycleDataProcessor(cycles(BACK_TO_BACK), vals)
    stats = proc.compute_cycle_statistics()
    assert stats["value_count"].tolist() == [3, 2, 0]  # samples, missing or not
    assert stats["mean_value_double"].iloc[0] == pytest.approx(5.5)
    comp = proc.compare_cycles("c1")
    assert comp["mean_value"].tolist() == pytest.approx([5.5, 40.0])


def test_compare_cycles_exact_values():
    proc = CycleDataProcessor(cycles(BACK_TO_BACK + [(20, 30)]), values(POINTS))
    comp = proc.compare_cycles("c2")
    # c3 is open and c4 has no samples: both left out
    assert comp["cycle_uuid"].tolist() == ["c1", "c2"]
    assert comp["is_reference"].tolist() == [False, True]
    c1_mean = np.mean([10.0, 20.0, 1.0])
    c1_std = np.std([10.0, 20.0, 1.0], ddof=1)
    ref_std = np.std([30.0, 50.0], ddof=1)
    assert comp["mean_value"].tolist() == pytest.approx([c1_mean, 40.0])
    assert comp["deviation_from_ref"].tolist() == pytest.approx([c1_mean - 40.0, 0.0])
    assert comp["deviation_pct"].tolist() == pytest.approx(
        [(c1_mean - 40.0) / 40.0 * 100, 0.0]
    )
    assert comp["variability_ratio"].tolist() == pytest.approx([c1_std / ref_std, 1.0])


def test_compare_cycles_unknown_or_empty_reference_warns_with_schema():
    proc = CycleDataProcessor(cycles(BACK_TO_BACK), values(POINTS))
    for ref in ("nope", "c3"):
        with pytest.warns(DataQualityWarning, match="reference cycle"):
            comp = proc.compare_cycles(ref)
        assert comp.empty
        assert "deviation_pct" in comp.columns


def test_golden_cycles():
    spans = [(0, 10), (10, 20), (20, 30), (30, 40)]
    pts = [
        (1, "x", -10.0),
        (2, "x", -11.0),  # mean -10.5, std 0.707: cv 0.067
        (11, "x", 100.0),
        (12, "x", 112.0),  # mean 106, std 8.49: cv 0.080
        (21, "x", 1.0),
        (22, "x", 3.0),  # mean 2, std 1.41: cv 0.707
        (31, "x", 50.0),  # one sample: no std
    ]
    proc = CycleDataProcessor(cycles(spans), values(pts))
    assert proc.identify_golden_cycles(top_n=10) == ["c1", "c2", "c3"]
    assert proc.identify_golden_cycles(method="high_mean", top_n=2) == ["c2", "c4"]
    assert proc.identify_golden_cycles(method="target_value", target=2.5, top_n=2) == [
        "c3",  # |2 - 2.5| = 0.5
        "c1",  # |-10.5 - 2.5| = 13
    ]
    with pytest.raises(ValueError, match="target"):
        proc.identify_golden_cycles(method="target_value")
    with pytest.raises(ValueError, match="Unknown method"):
        proc.identify_golden_cycles(method="best")


def test_end_to_end_two_machines_dst_night():
    # one row every 5 min in UTC across the 2024-10-27 fall-back night
    t = pd.date_range("2024-10-26 23:30", periods=24, freq="5min", tz="UTC")
    t = t.tz_convert(BERLIN)
    run_a = ([False] + [True] * 5 + [False] * 2) * 3  # cycles of 25 min every 40 min
    run_b = [True] * 24
    frames = [
        pd.DataFrame({"systime": t, "uuid": "A.run", "value_bool": run_a}),
        pd.DataFrame({"systime": t, "uuid": "B.run", "value_bool": run_b}),
        pd.DataFrame({"systime": t, "uuid": "A.temp", "value_double": np.arange(24.0)}),
        pd.DataFrame(
            {"systime": t, "uuid": "B.temp", "value_double": np.arange(24.0) + 100}
        ),
    ]
    df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=7)
    cyc = CycleExtractor(df, "A.run").process_persistent_cycle()
    assert cyc["cycle_start"].tolist() == [t[1], t[9], t[17]]
    assert cyc["cycle_end"].tolist() == [t[6], t[14], t[22]]
    assert cyc["duration_seconds"].tolist() == [1500.0] * 3

    proc = CycleDataProcessor(cyc, df[df["uuid"] == "A.temp"])
    stats = proc.compute_cycle_statistics()
    assert stats["value_count"].tolist() == [5, 5, 5]
    assert stats["mean_value_double"].tolist() == [3.0, 11.0, 19.0]
    split = proc.split_by_cycle()
    assert list(split) == cyc["cycle_uuid"].tolist()
    assert split[cyc["cycle_uuid"].iloc[1]]["value_double"].tolist() == [
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
    ]


def test_speed_200k_values_10k_cycles():
    n = 200_000
    t = pd.date_range("2024-01-01", periods=n, freq="1s", tz=BERLIN)
    vals = pd.DataFrame({"systime": t, "uuid": "v", "value_double": np.arange(n) % 20})
    starts = t[::20]
    cyc = pd.DataFrame(
        {
            "cycle_start": starts,
            "cycle_end": starts + pd.Timedelta(seconds=20),
            "cycle_uuid": [f"c{i}" for i in range(len(starts))],
        }
    )
    started = time.perf_counter()
    proc = CycleDataProcessor(cyc, vals)
    merged = proc.merge_dataframes_by_cycle()
    stats = proc.compute_cycle_statistics()
    comp = proc.compare_cycles("c0")
    golden = proc.identify_golden_cycles(top_n=3)
    elapsed = time.perf_counter() - started
    assert len(merged) == n
    assert stats["value_count"].eq(20).all()
    assert stats["mean_value_double"].eq(9.5).all()
    assert comp["deviation_from_ref"].abs().max() == 0
    assert golden == ["c0", "c1", "c2"]
    assert elapsed < 3.0, elapsed
