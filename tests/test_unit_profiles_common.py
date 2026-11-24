"""What every mapping-free profiler shares: helpers, degenerate inputs, the
orchestrator (registry, save / load, failures)."""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.events.production._unit_profile import (
    SignalIndex,
    assign_to_cycles,
    cycles_since,
    enrichment,
    free_windows,
    restart_points,
    robust_center_scale,
)
from ts_shape.unit_profiles import PROFILERS, UnitProfilesResult, run_profiler

pytestmark = pytest.mark.filterwarnings("ignore::ts_shape.errors.DataQualityWarning")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_assign_to_cycles_closed_sides() -> None:
    starts, ends = np.array([0, 10, 20]), np.array([10, 20, 25])
    t = np.array([-1, 0, 10, 24, 25, 30])
    pos, inside = assign_to_cycles(t, starts, ends)
    assert pos.tolist() == [-1, 0, 1, 2, 2, 2]
    assert inside.tolist() == [False, True, True, True, False, False]
    pos, inside = assign_to_cycles(t, starts, ends, closed="right")
    # A boundary reading belongs to the cycle that just ended.
    assert pos[2] == 0 and inside[2]
    assert pos[4] == 2 and inside[4]
    with pytest.raises(ValueError, match="closed"):
        assign_to_cycles(t, starts, ends, closed="both")


def test_restart_points_and_cycles_since() -> None:
    cycles = pd.DataFrame(
        {
            "cycle_class": ["normal", "pause", "normal", "normal", "normal"],
            "variant": ["A", "A", "A", "B", "B"],
        }
    )
    rp = restart_points(cycles)
    assert rp["position"].tolist() == [1, 3]
    assert rp["trigger"].tolist() == ["pause", "changeover"]
    since = cycles_since(rp["position"], len(cycles))
    assert since.tolist()[1:] == [0, 1, 0, 1]
    assert since[0] > 10**6  # before the first restart
    assert restart_points(cycles.iloc[:0]).empty


def test_robust_center_scale() -> None:
    assert robust_center_scale([1.0, 2.0, 3.0, 100.0])[0] == pytest.approx(2.5)
    median, scale = robust_center_scale([5.0, 5.0, 5.0, 5.0, 6.0])
    assert median == 5.0 and scale > 0  # MAD 0: the std
    assert np.isnan(robust_center_scale([3.0, 3.0])[1])
    assert np.isnan(robust_center_scale([])[0])


def test_signal_index_sorts_and_drops_nat() -> None:
    frame = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                [
                    "2026-01-01 00:00:02",
                    None,
                    "2026-01-01 00:00:01",
                    "2026-01-01 00:00:00",
                ]
            ),
            "uuid": ["a", "a", "a", "b"],
            "value_double": [2.0, 9.0, 1.0, 5.0],
        }
    )
    index = SignalIndex(frame, "uuid", "systime")
    assert sorted(index.uuids) == ["a", "b"]
    assert index.numeric("a", "value_double").tolist() == [1.0, 2.0]
    assert np.all(np.diff(index.ns("a")) > 0)
    assert len(index.ns("missing")) == 0 and "missing" not in index


def test_free_windows_avoid_busy_and_repeat() -> None:
    busy = [(100, 200), (150, 400)]
    starts, ends = free_windows(50, 40, span=(0, 1000), busy=busy, seed=3)
    assert len(starts) == 40
    assert not np.any((starts < 400) & (ends > 100))
    again, _ = free_windows(50, 40, span=(0, 1000), busy=busy, seed=3)
    assert np.array_equal(starts, again)
    assert len(free_windows(50, 5, span=(0, 40), busy=[])[0]) == 0


def test_enrichment_ties_are_ordered_by_label() -> None:
    units = [{"b", "a", "c"}] * 6
    base = [set()] * 30
    labels = enrichment(units, base)["label"].tolist()
    assert labels == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Every registered profiler
# ---------------------------------------------------------------------------


def test_registry_lists_every_profiler() -> None:
    assert list(PROFILERS) == [
        "stops",
        "changeovers",
        "startups",
        "orders",
        "scrap",
        "handshakes",
        "curves",
        "settings",
        "loops",
    ]
    for cls in PROFILERS.values():
        assert cls.profile_tables()[-1] == "evaluation_window"
        for name in cls.profile_tables():
            assert callable(getattr(cls, name)), (cls.__name__, name)


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.to_datetime([]),
            "uuid": pd.Series([], dtype=object),
            "value_double": pd.Series([], dtype=float),
            "value_integer": pd.Series([], dtype="Int64"),
            "value_bool": pd.Series([], dtype="boolean"),
            "value_string": pd.Series([], dtype=object),
            "is_delta": pd.Series([], dtype=bool),
        }
    )


@pytest.mark.parametrize("kind", list(PROFILERS))
def test_empty_frame(kind: str) -> None:
    cls = PROFILERS[kind]
    result = _quiet(run_profiler, cls, _empty_frame())
    assert result.summary["evaluable"] is False
    assert isinstance(result.summary["reason"], str) and result.summary["reason"]
    window = result.tables["evaluation_window"]
    assert len(window) == 1 and not bool(window["evaluable"].iloc[0])
    for name, table in result.tables.items():
        assert isinstance(table, pd.DataFrame), name
        if name == "evaluation_window":
            continue
        assert table.empty, name
        if "systime" in table.columns:
            assert pd.api.types.is_datetime64_any_dtype(table["systime"]), name


@pytest.fixture(scope="module")
def small_plc() -> pd.DataFrame:
    frame, _ = ts_shape.make_plc_cycles(
        n_cycles=200,
        seed=3,
        stop_causes=2,
        stop_probability=0.05,
        handshake_partner=True,
        setting_changes=[(100, 120.0)],
    )
    return frame


@pytest.mark.parametrize("kind", list(PROFILERS))
def test_timezone_aware_and_naive_agree(small_plc: pd.DataFrame, kind: str) -> None:
    cls = PROFILERS[kind]
    naive = _quiet(run_profiler, cls, small_plc)
    aware_frame = small_plc.assign(
        systime=small_plc["systime"].dt.tz_localize("Europe/Berlin")
    )
    aware = _quiet(run_profiler, cls, aware_frame)
    assert naive.summary == aware.summary
    for name, table in naive.tables.items():
        other = aware.tables[name]
        assert list(table.columns) == list(other.columns), name
        if "systime" in table.columns and len(other):
            assert str(other["systime"].dt.tz) == "Europe/Berlin"
        # Times, and ids derived from them, follow the zone; nothing else may.
        keep = [
            c
            for c in table.columns
            if c not in ("systime", "start", "end", "unit_uuid", "cycle_uuid")
            and not c.endswith("_ns")
        ]
        pd.testing.assert_frame_equal(
            table[keep].reset_index(drop=True),
            other[keep].reset_index(drop=True),
            check_dtype=False,
        )


@pytest.mark.parametrize("kind", list(PROFILERS))
def test_coarse_polling_does_not_break(kind: str) -> None:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=150, seed=2, poll_period="13s")
    result = _quiet(run_profiler, PROFILERS[kind], frame)
    assert set(result.summary) >= {"evaluable", "reason"}
    if kind in ("changeovers", "curves", "settings", "loops", "handshakes"):
        assert result.summary["evaluable"] is False


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_profile_units_round_trip_keeps_contents(tmp_path) -> None:
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=240, seed=4, order_size=40, scrap_probability=0.03
    )
    plan = pd.DataFrame(
        {"order_id": sorted(truth["order_id"].unique()), "planned_quantity": 40}
    )
    result = _quiet(
        ts_shape.profile_units, frame, profilers=("orders", "scrap", "loops"), plan=plan
    )
    assert result["orders"].tables["order_records"]["attainment_ratio"].notna().any()
    assert "loop_register" in result["loops"].tables
    result.save(tmp_path)
    loaded = UnitProfilesResult.load(tmp_path)
    assert loaded.summary == result.summary
    for name in ("orders", "scrap"):
        for table, before in result[name].tables.items():
            after = loaded[name].tables[table]
            assert list(after.columns) == list(before.columns), (name, table)
            assert len(after) == len(before), (name, table)


def test_a_failing_profiler_is_reported(monkeypatch, small_plc) -> None:
    def boom(self):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(PROFILERS["handshakes"], "handshake_pairs", boom)
    result = _quiet(
        ts_shape.profile_units, small_plc, profilers=("stops", "handshakes")
    )
    assert result.summary["handshakes"]["evaluable"] is False
    assert "synthetic failure" in result.summary["handshakes"]["reason"]
    assert "stop_records" in result["stops"].tables


# ---------------------------------------------------------------------------
# Scale (run with -m scale)
# ---------------------------------------------------------------------------


@pytest.mark.scale
def test_profile_units_on_a_large_export() -> None:
    """About 2 M rows and 250 signals: every profiler within a time budget."""
    frame, _ = ts_shape.make_plc_cycles(
        n_cycles=5000,
        seed=1,
        stop_causes=3,
        stop_probability=0.03,
        order_size=200,
        scrap_probability=0.01,
    )
    rng = np.random.default_rng(0)
    t0, t1 = frame["systime"].min(), frame["systime"].max()
    times = pd.date_range(t0, t1, freq="30s")
    extra = [
        pd.DataFrame(
            {
                "systime": times,
                "uuid": f"plc:noise_{k:03d}",
                "value_double": np.cumsum(rng.normal(0, 1, len(times))),
            }
        )
        for k in range(220)
    ]
    frame = pd.concat([frame, *extra], ignore_index=True)
    assert len(frame) > 1_500_000
    start = time.perf_counter()
    result = _quiet(ts_shape.profile_units, frame)
    seconds = time.perf_counter() - start
    assert all("reason" in s for s in result.summary.values())
    assert seconds < 600, f"profile_units took {seconds:.0f} s"
