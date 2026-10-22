"""Self-checks of the ``make_object_signals`` identifier benchmark generator.

The labels must be consistent with the frame, the uuids must not give the
answer away, real formats must carry valid check digits, labelled nesting must
hold in the data, the output must be deterministic per seed, and the held-out
split must use formats the dev split never sees.
"""

from __future__ import annotations

import re
from functools import cache

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape._object_signals import (
    DEV_SEEDS,
    FORMAT_VARIANTS,
    HELDOUT_SEEDS,
    NESTING,
    OBJECT_TYPES,
    ROLES,
    gs1_valid,
)

VALUE_COLUMNS = ("value_bool", "value_integer", "value_double", "value_string")


@cache
def _generated(seed: int, noise: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    return ts_shape.make_object_signals(seed=seed, noise=noise)


def _values(frame: pd.DataFrame, uuid: str, column: str) -> pd.Series:
    rows = frame.loc[frame["uuid"] == uuid]
    return rows.set_index("systime")[column]


def _is_empty(value: object) -> bool:
    return value is None or value is pd.NA or value == "" or value == 0


@pytest.mark.parametrize("seed", [0, 3, 100, 107])
def test_labels_are_consistent_with_the_frame(seed):
    frame, labels = _generated(seed)
    assert list(labels.columns[:5]) == [
        "uuid",
        "role",
        "object_type",
        "value_column",
        "parent_uuid",
    ]
    assert labels["uuid"].is_unique
    assert set(labels["uuid"]) == set(frame["uuid"])
    assert set(labels["role"]) <= set(ROLES)
    ids = labels["role"] == "identifier"
    assert labels.loc[ids, "object_type"].isin(OBJECT_TYPES).all()
    assert labels.loc[~ids, "object_type"].isna().all()
    for row in labels.itertuples():
        rows = frame.loc[frame["uuid"] == row.uuid]
        filled = [c for c in VALUE_COLUMNS if rows[c].notna().any()]
        assert filled == [row.value_column], (row.uuid, row.variant, filled)
        if row.role == "identifier":
            assert row.variant in FORMAT_VARIANTS[row.object_type]
            split, column = FORMAT_VARIANTS[row.object_type][row.variant]
            assert column == row.value_column


@pytest.mark.parametrize("seed", [0, 4, 100, 119])
def test_every_role_and_the_hard_negatives_are_present(seed):
    _, labels = _generated(seed, noise=False)
    assert set(labels["role"]) == set(ROLES)
    kinds = set(labels["variant"].str.split(":").str[0])
    assert {"mode_code", "shot_weight_int", "cycle_timestamp"} <= kinds
    assert (labels["role"] == "identifier").sum() >= 15


def test_uuids_are_opaque():
    words = {*OBJECT_TYPES, *ROLES, "order", "id", "count", "text", "temp"}
    assignments = []
    for seed in (*DEV_SEEDS, 100, 101):
        frame, labels = _generated(seed, noise=False)
        assert labels["uuid"].str.fullmatch(r"sig:\d{4}").all()
        lowered = " ".join(labels["uuid"]).lower()
        assert not any(w in lowered for w in words)
        assert set(frame.columns) == {
            "systime",
            "uuid",
            *VALUE_COLUMNS,
            "is_delta",
        }
        ids = labels.loc[labels["role"] == "identifier"]
        assignments.append(dict(zip(ids["object_type"], ids["uuid"], strict=False)))
    # the uuid of a type changes from seed to seed
    for otype in ("material", "serial", "batch"):
        assert len({a.get(otype) for a in assignments}) > 3


def _id_values(seed: int, otype: str) -> list[tuple[str, list[object]]]:
    frame, labels = _generated(seed, noise=False)
    out = []
    for row in labels.loc[labels["object_type"] == otype].itertuples():
        vals = frame.loc[frame["uuid"] == row.uuid, row.value_column].dropna()
        out.append((row.variant, [v for v in vals.unique() if not _is_empty(v)]))
    return out


def test_check_digits_and_epc_structure_are_valid():
    seen = set()
    for seed in (*DEV_SEEDS, *HELDOUT_SEEDS):
        for variant, vals in _id_values(seed, "material"):
            if variant in ("gtin13", "gtin14"):
                seen.add(variant)
                width = 13 if variant == "gtin13" else 14
                assert all(len(v) == width and gs1_valid(v) for v in vals), vals[:3]
        for variant, vals in _id_values(seed, "handling_unit"):
            if variant.startswith("sscc"):
                seen.add(variant)
                codes = [
                    (
                        f"{v:018d}"
                        if variant == "sscc_int"
                        else str(v).removeprefix("(00)")
                    )
                    for v in vals
                ]
                assert all(len(c) == 18 and gs1_valid(c) for c in codes), codes[:3]
        for variant, vals in _id_values(seed, "carrier"):
            if variant == "epc_grai96":
                seen.add(variant)
                assert all(re.fullmatch(r"33[0-9A-F]{22}", v) for v in vals)
    assert seen == {"gtin13", "gtin14", "sscc", "sscc_ai", "sscc_int", "epc_grai96"}


def test_gs1_check_digit_known_values():
    assert gs1_valid("4006381333931")  # a published GTIN-13
    assert gs1_valid("106141411234567897")  # GS1 SSCC example
    assert not gs1_valid("4006381333932")


def _median_step(index: pd.Index) -> float:
    t = index.to_series().diff().dt.total_seconds().dropna()
    return float(t.median()) if len(t) else 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 100, 105, 112])
def test_labelled_nesting_holds(seed):
    """Away from the parent's changes, each child value sits in one parent."""
    frame, labels = _generated(seed, noise=False)
    by_uuid = labels.set_index("uuid")
    pairs = labels.loc[labels["parent_uuid"].notna()]
    assert len(pairs) >= 2
    parent_changes = 0
    for row in pairs.itertuples():
        child_type = row.object_type
        parent = by_uuid.loc[row.parent_uuid]
        assert NESTING[child_type] == parent["object_type"]
        child = _values(frame, row.uuid, row.value_column)
        par = _values(frame, row.parent_uuid, parent["value_column"]).dropna()
        tol = 1.0 + _median_step(child.index) + max(_median_step(par.index), 0)
        changes = par.index[1:][par.to_numpy()[1:] != par.to_numpy()[:-1]]
        joined = pd.merge_asof(
            child.rename("child").reset_index(),
            par.rename("parent").reset_index(),
            on="systime",
        ).dropna()
        joined = joined.loc[~joined["child"].map(_is_empty)]
        t = joined["systime"].to_numpy()
        near = np.zeros(len(t), dtype=bool)
        for c in changes.to_numpy():
            near |= np.abs((t - c) / np.timedelta64(1, "s")) <= tol
        clean = joined.loc[~near]
        parents_per_child = clean.groupby("child")["parent"].nunique()
        assert (parents_per_child == 1).all(), (
            row.variant,
            parents_per_child[parents_per_child > 1].head(),
        )
        parent_changes += clean["parent"].nunique() - 1
    # and parents do change while their children are observed
    assert parent_changes > 0


@pytest.mark.parametrize("seed", [0, 101])
def test_deterministic_per_seed(seed):
    a = ts_shape.make_object_signals(seed=seed, days=1)
    b = ts_shape.make_object_signals(seed=seed, days=1)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])
    c = ts_shape.make_object_signals(seed=seed + 1, days=1)
    assert not a[0].equals(c[0])
    # noise changes the values, never the labels
    clean = ts_shape.make_object_signals(seed=seed, days=1, noise=False)
    pd.testing.assert_frame_equal(a[1], clean[1])
    assert not a[0].equals(clean[0])


def test_heldout_split_uses_formats_the_dev_split_never_sees():
    def variants(seeds):
        used = set()
        for seed in seeds:  # noise draws come after the formats are picked
            _, labels = _generated(seed, noise=False)
            ids = labels.loc[labels["role"] == "identifier"]
            used |= set(zip(ids["object_type"], ids["variant"], strict=True))
        return used

    heldout_only = {
        (otype, v)
        for otype, vs in FORMAT_VARIANTS.items()
        for v, (split, _) in vs.items()
        if split == "heldout"
    }
    dev = variants(DEV_SEEDS)
    held = variants(HELDOUT_SEEDS)
    assert not dev & heldout_only
    assert held & heldout_only == heldout_only
    # every object type is covered by both splits
    assert {o for o, _ in dev} == set(OBJECT_TYPES)
    assert {o for o, _ in held} == set(OBJECT_TYPES)


@pytest.mark.parametrize("seed", [*DEV_SEEDS, 100, 110, 118])
def test_frame_size(seed):
    frame, _ = _generated(seed, noise=False)
    assert 10_000 <= len(frame) <= 260_000


def test_shop_floor_behaviour():
    frame, labels = _generated(2, noise=False)

    def runs(otype):
        row = labels.loc[labels["object_type"] == otype].iloc[0]
        s = _values(frame, row.uuid, row.value_column).dropna()
        s = s[~s.map(_is_empty)]
        return s[s.ne(s.shift())]

    serial = runs("serial")
    assert serial.is_unique  # held for one cycle, never revisited
    assert _median_step(serial.index) < 300
    carriers = runs("carrier")
    assert 20 <= carriers.nunique() <= 60
    assert carriers.value_counts().min() >= 10  # the pool recirculates
    assert runs("tool").nunique() <= 6
    batch = runs("batch")
    assert batch.nunique() >= 0.8 * len(batch)  # rarely revisited
    operator = runs("operator")
    hours = operator.index.hour + operator.index.minute / 60
    to_shift = np.min([np.abs(hours - h) for h in (6, 14, 22, 30, -2)], axis=0)
    assert (to_shift <= 0.5).mean() >= 0.6  # mostly at the shift changes
    # one part counter per serial signal
    assert (labels["variant"].str.startswith("part_counter")).sum() == (
        labels["object_type"] == "serial"
    ).sum()


def test_include_filters_and_drops_missing_parents():
    frame, labels = ts_shape.make_object_signals(
        seed=0, days=1, include=["serial", "counter"]
    )
    assert set(labels["object_type"].dropna()) == {"serial"}
    assert set(labels["role"]) == {"identifier", "counter"}
    assert labels["parent_uuid"].isna().all()
    assert set(frame["uuid"]) == set(labels["uuid"])
    with pytest.raises(ValueError, match="unknown"):
        ts_shape.make_object_signals(include=["widget"])
    with pytest.raises(ValueError, match="days"):
        ts_shape.make_object_signals(days=0)


def test_noise_adds_dirty_reads():
    clean, labels = _generated(100, noise=False)
    noisy, _ = _generated(100, noise=True)
    strings = noisy["value_string"].dropna().astype(str)
    assert strings.isin(["NOREAD", "????????", "ERR", "#N/A", "NULL"]).any()
    assert (strings == "").any()
    assert (
        noisy[list(VALUE_COLUMNS)].isna().all(axis=1).sum()
        > clean[list(VALUE_COLUMNS)].isna().all(axis=1).sum()
    )
