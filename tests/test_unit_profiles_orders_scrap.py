"""Mapping-free unit profilers, phase 2: counter roles, orders, scrap."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.events.profiling.cycles import CycleProfileEvents
from ts_shape.events.profiling.orders import (
    OrderProfileEvents,
    ScrapProfileEvents,
    counter_roles,
)

pytestmark = pytest.mark.filterwarnings("ignore::ts_shape.errors.DataQualityWarning")


@pytest.fixture(scope="module")
def plant() -> tuple[pd.DataFrame, pd.DataFrame, CycleProfileEvents]:
    frame, truth = ts_shape.make_plc_cycles(
        n_cycles=800,
        seed=4,
        order_size=40,
        scrap_probability=0.02,
        rampup_cycles=4,
        stop_causes=2,
        stop_probability=0.05,
        curve_faults={"spike_in_hold": 0.05},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        profile = CycleProfileEvents(frame)
        profile.units()
    return frame, truth, profile


def test_counter_roles_from_increments() -> None:
    rng = np.random.default_rng(0)
    bad = (rng.random(200) < 0.05).astype(float)
    total = np.ones(200)
    table = pd.DataFrame(
        {
            "total": total,
            "good": total - bad,
            "bad": bad,
            "copy": total,
            "kwh": rng.random(200) * 0.7,
        }
    )
    roles = counter_roles(table).set_index("uuid")["role"]
    assert roles["total"] == "total" and roles["good"] == "good"
    assert roles["bad"] == "bad" and roles["copy"] == "alias"
    assert roles["kwh"] == "energy_like"
    assert counter_roles(pd.DataFrame()).empty


def test_counter_roles_found_in_signals(plant) -> None:
    frame, _, profile = plant
    roles = ScrapProfileEvents(frame, cycle_profile=profile).counter_roles()
    by_role = dict(zip(roles["role"], roles["uuid"], strict=True))
    assert by_role["bad"] == "plc:bad_counter"
    assert by_role["good"] == "plc:good_counter"
    assert by_role["total"] == "plc:part_counter"


def test_orders(plant) -> None:
    frame, truth, profile = plant
    plan = pd.DataFrame(
        {"order_id": sorted(truth["order_id"].unique()), "planned_quantity": 40}
    )
    orders = OrderProfileEvents(frame, cycle_profile=profile, plan=plan)
    window = orders.evaluation_window().iloc[0]
    assert bool(window["evaluable"]) and window["order_uuid"] == "plc:order_no"
    records = orders.order_records()
    assert abs(len(records) - truth["order_id"].nunique()) <= 1
    assert records["cycle_count"].between(30, 45).mean() > 0.9
    assert (records["good_count"] + records["scrap_count"]).sum() == pytest.approx(
        len(truth), abs=45
    )
    assert records["attainment_ratio"].notna().all()
    assert (records["speed_ratio"] > 0.5).all()
    register = orders.order_register()
    assert set(register["variant"]) == set(records["variant"])
    with pytest.raises(ValueError, match="plan needs"):
        OrderProfileEvents(frame, plan=pd.DataFrame({"order": [1]}))


def test_scrap_records_match_truth(plant) -> None:
    frame, truth, profile = plant
    scrap = ScrapProfileEvents(frame, cycle_profile=profile)
    assert bool(scrap.evaluation_window()["evaluable"].iloc[0])
    records = scrap.nok_records()
    left = records.sort_values("systime")
    right = truth.assign(systime=truth["end"]).sort_values("systime")
    matched = pd.merge_asof(
        left,
        right[["systime", "nok"]],
        on="systime",
        direction="nearest",
        tolerance=pd.Timedelta("3s"),
    )
    assert matched["nok"].mean() == 1.0  # every NOK record is a bad cycle
    assert len(records) == int(truth["nok"].sum())


def test_scrap_pareto_finds_the_drivers(plant) -> None:
    frame, _, profile = plant
    pareto = ScrapProfileEvents(frame, cycle_profile=profile).scrap_pareto()
    significant = set(pareto.loc[pareto["is_significant"].astype(bool), "context"])
    assert "after_stop" in significant  # ramp-up after stops (5x)
    assert any(
        c.startswith("curve plc:pressure") or "pressure" in c for c in significant
    )
    assert not any("counter" in c for c in pareto["context"])


def test_no_bad_counter() -> None:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=150, seed=1)
    scrap = ScrapProfileEvents(frame)
    window = scrap.evaluation_window().iloc[0]
    assert not bool(window["evaluable"])
    assert window["reason"].startswith("no_candidate_signal")
    assert scrap.nok_records().empty and scrap.scrap_pareto().empty
    orders = OrderProfileEvents(frame)
    assert orders.evaluation_window()["order_source"].iloc[0] in {"variant", "none"}


def test_counter_roles_with_little_scrap_and_a_late_bad_count() -> None:
    rng = np.random.default_rng(0)
    n = 400
    bad = (rng.random(n) < 0.01).astype(float)
    total = np.ones(n)
    table = pd.DataFrame(
        {
            "total": total,
            "good": total - bad,
            "bad": np.r_[0.0, bad[:-1]],  # reported one cycle late
            "rework": (rng.random(n) < 0.01).astype(float),  # rare, unrelated
            "copy": total,
        }
    )
    roles = counter_roles(table).set_index("uuid")
    # With 1 % scrap good == total in 99 % of the cycles: still not an alias.
    assert roles.at["good", "role"] == "good" and roles.at["total", "role"] == "total"
    assert roles.at["bad", "role"] == "bad"
    # Two rarely moving counters are not equal just because both are mostly 0.
    assert roles.at["rework", "role"] == "counter"
    assert roles.at["copy", "role"] == "alias"
    lag = int(roles.at["bad", "lag_cycles"])
    aligned = table["bad"].shift(lag).fillna(0.0).to_numpy()
    assert np.array_equal(aligned, bad)


def test_plan_with_duplicate_order_ids() -> None:
    frame, _ = ts_shape.make_plc_cycles(n_cycles=60, seed=1, order_size=20)
    plan = pd.DataFrame({"order_id": ["A", "A"], "planned_quantity": [1, 2]})
    with pytest.raises(ValueError, match="more than once"):
        OrderProfileEvents(frame, plan=plan)
