"""identify_objects: hand-built reference cases, negative controls, contract.

Every fixture is built here from values alone; the uuids are single letters
so no test can pass by reading a name.
"""

from __future__ import annotations

import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.eventlog import detect_objects, schema
from ts_shape.signals import (
    IDENTIFIER_TYPES,
    KINDS,
    ObjectReport,
    identify,
    identify_objects,
    infer_specs,
    register_identifier_pattern,
    register_identifier_type,
)

START = pd.Timestamp("2025-03-03 06:00")


def _sig(uuid, times, values, column="value_string"):
    return pd.DataFrame(
        {"systime": pd.DatetimeIndex(times), "uuid": uuid, column: values}
    )


def _held(uuid, values, hold, *, freq="1min", start=START, column="value_string"):
    seq = [v for v in values for _ in range(hold)]
    return _sig(uuid, pd.date_range(start, periods=len(seq), freq=freq), seq, column)


def _stamped(uuid, values, *, gap="10min", start=START):
    """Each id for one sample, cleared by an empty reading a minute later."""
    times, out = [], []
    for i, value in enumerate(values):
        t0 = start + i * pd.Timedelta(gap)
        times += [t0, t0 + pd.Timedelta("1min")]
        out += [value, ""]
    return _sig(uuid, times, out)


def _check_digit(body: str) -> str:
    total = sum(int(c) * (3 if i % 2 == 0 else 1) for i, c in enumerate(reversed(body)))
    return body + str((10 - total % 10) % 10)


def _plant() -> dict[str, pd.DataFrame]:
    """One signal per object type plus negative controls (uuid -> rows)."""
    rng = np.random.default_rng(7)
    f: dict[str, pd.DataFrame] = {}
    f["a"] = _held("a", [f"SN{100000 + i:06d}" for i in range(400)], 2, freq="30s")
    f["cnt"] = _held("cnt", list(range(400)), 2, freq="30s", column="value_integer")
    f["b"] = _held(
        "b",
        [
            f"B{(START + pd.Timedelta(hours=i)).strftime('%y%m%d')}-{i % 24 + 1:03d}"
            for i in range(30)
        ],
        60,
    )
    recipes = ["R-100", "R-200", "R-100", "R-300", "R-200", "R-400"]
    recipes += ["R-100", "R-500", "R-300", "R-200", "R-100", "R-400"]
    holds = rng.integers(60, 180, len(recipes))
    seq = [r for r, h in zip(recipes, holds, strict=True) for _ in range(h)]
    f["c"] = _sig("c", pd.date_range(START, periods=len(seq), freq="1min"), seq)
    f["d"] = _held("d", ["T01", "T02", "T01", "T03"], 72, freq="10min")
    f["e"] = _stamped("e", [f"WT{i % 12 + 1:02d}" for i in range(240)], gap="2min")
    f["f"] = _held(
        "f",
        ["E1042", "E2077", "E3101", "E1042", "E2077", "E3315"] * 2,
        96,
        freq="5min",
    )
    f["g"] = _stamped("g", [str(4500000 + i) for i in range(80)], gap="30min")
    partners = ["C-17", "C-22", "C-35", "C-41"]
    f["h"] = _stamped(
        "h", [partners[int(i)] for i in rng.integers(0, 4, 80)], gap="30min"
    )
    f["i"] = _stamped(
        "i",
        [_check_digit(f"03401234{500000000 + i:09d}") for i in range(80)],
        gap="7min",
    )
    articles = [101, 202, 303, 101, 404, 202, 101, 505, 303]
    f["j"] = _held("j", [_check_digit(f"400123456{a:03d}") for a in articles], 90)
    hexd = list("0123456789ABCDEF")
    tags = ["3034" + "".join(rng.choice(hexd, 20)) for _ in range(40)]
    f["k"] = _stamped("k", [tags[i % 40] for i in range(200)], gap="3min")
    bins = [f"A-{a:02d}-{b:02d}" for a in range(1, 4) for b in range(1, 6)]
    f["l"] = _stamped(
        "l", [bins[int(i)] for i in rng.integers(0, len(bins), 200)], gap="4min"
    )
    f["m"] = _held("m", [f"OP{10 * (i % 5 + 1)}" for i in range(200)], 3)
    materials = [100234, 100567, 100234, 100890, 100567, 100234]
    f["n"] = _held("n", [f"{x:018d}" for x in materials], 120)
    orders = ["FA4711001", "FA4711002", "FA4711003", "FA4711002"]
    orders += ["FA4711004", "FA4711005", "FA4711006"]
    f["o"] = _held("o", orders, 150)
    f["w"] = _held("w", ["L1-M01", "L1-M02", "L1-M01", "L1-M03"], 70, freq="10min")
    lots = ["LOT-8812", "LOT-8813", "LOT-8812", "LOT-8814", "LOT-8815"]
    lots += ["LOT-8814", "LOT-8816", "LOT-8817", "LOT-8818", "LOT-8819"]
    f["x"] = _held("x", lots, 50)
    gs1 = [f"(01){_check_digit('0401234500001')}(21){7000 + i}" for i in range(120)]
    f["y"] = _held("y", gs1, 3)
    # negative controls
    t500 = pd.date_range(START, periods=500, freq="1min")
    f["p"] = _sig("p", t500, 20 + rng.normal(0, 1, 500), "value_double")
    f["q"] = _held("q", ["RUN", "IDLE", "RUN", "STOP", "RUN", "IDLE"] * 10, 8)
    t60 = pd.date_range(START, periods=60, freq="7min")
    notes = [f"Note {i}: {'jam at conveyor ' * (i % 4)}checked" for i in range(60)]
    f["r"] = _sig("r", t60, notes)
    f["s"] = _sig("s", t60, [str(x) for x in t60])
    f["t"] = _held("t", ["X1"], 50)
    f["u"] = _held("u", [1, 2, 3, 1, 2, 3, 2, 1] * 5, 5, column="value_integer")
    walk = (50 + np.cumsum(rng.integers(-1, 2, 500))).astype(int)
    f["v"] = _sig("v", t500, walk, "value_integer")
    return f


EXPECTED_TYPES = {
    "a": "serial",
    "b": "batch",
    "c": "recipe",
    "d": "tool",
    "e": "carrier",
    "f": "operator",
    "g": "order",
    "h": "business_partner",
    "i": "handling_unit",
    "j": "article",
    "k": "rfid_tag",
    "l": "location",
    "m": "station",
    "n": "material",
    "o": "work_order",
    "w": "asset",
    "x": "lot",
    "y": "serial",
}
EXPECTED_ROLES = {
    "cnt": "counter",
    "p": "measurement",
    "q": "state",
    "r": "free_text",
    "s": "timestamp",
    "t": "constant",
    "u": "state",
    "v": "measurement",
}


@pytest.fixture(scope="module")
def plant() -> pd.DataFrame:
    return pd.concat(_plant().values(), ignore_index=True)


@pytest.fixture(scope="module")
def report(plant) -> ObjectReport:
    return identify_objects(plant)


def _row(report: ObjectReport, uuid: str) -> pd.Series:
    return report.table.set_index("uuid").loc[uuid]


# ---------- reference cases --------------------------------------------------


@pytest.mark.parametrize("uuid,expected", sorted(EXPECTED_TYPES.items()))
def test_reference_type(report, uuid, expected):
    row = _row(report, uuid)
    assert row["role"] == "identifier", report.explain(uuid)
    # recipe, tool and fixture fixtures without a format cue or a work order
    # are reported as their group, ``setup``.
    assert row["object_type"] in (
        expected,
        IDENTIFIER_TYPES.get(expected),
    ), report.explain(uuid)
    assert row["type_confidence"] > row["runner_up_confidence"] or np.isnan(
        row["runner_up_confidence"]
    )


@pytest.mark.parametrize("uuid,expected", sorted(EXPECTED_ROLES.items()))
def test_negative_controls(report, uuid, expected):
    row = _row(report, uuid)
    assert row["role"] == expected, report.explain(uuid)
    assert row["object_type"] is None


def test_format_cases_are_confident(report):
    for uuid in ("i", "j", "k", "l", "n", "y"):
        assert _row(report, uuid)["type_confidence"] >= 0.6, report.explain(uuid)


def test_counter_alignment_is_evidence_for_the_serial(report):
    assert "counter" in " ".join(_row(report, "a")["evidence"])


def test_nesting_records_the_parent(report):
    assert _row(report, "a")["parent_uuid"] == "b"
    assert _row(report, "b")["parent_uuid"] is None
    # resources are never nested
    for uuid in ("d", "e", "m", "w"):
        assert _row(report, uuid)["parent_uuid"] is None


def test_nesting_needs_aligned_boundaries():
    # The child changes every 7 minutes, the parent at whole hours: each
    # child presence sits inside one parent presence only by chance.
    child = _held("a", [f"SN{100000 + i}" for i in range(60)], 7)
    parent = _held("b", [f"B{i:04d}" for i in range(7)], 61)
    out = identify_objects(pd.concat([child, parent]))
    assert _row(out, "a")["parent_uuid"] is None


def test_three_level_nesting_gives_depth():
    serial = _held("a", [f"SN{100000 + i}" for i in range(240)], 5)
    batch = _held("b", [f"B2503030{i:02d}" for i in range(24)], 50)
    material = _held("c", [f"{m:018d}" for m in (4711, 4712, 4711, 4713)], 300)
    out = identify_objects(pd.concat([serial, batch, material]))
    assert _row(out, "a")["parent_uuid"] == "b"
    assert _row(out, "b")["parent_uuid"] == "c"
    assert any("depth 2" in e for e in _row(out, "a")["evidence"])


def test_group_is_reported_without_a_format_cue(report):
    g, h = _row(report, "g"), _row(report, "h")
    assert g["object_type"] == "order"
    assert h["object_type"] == "business_partner"
    assert any("not separable" in e for e in g["evidence"])


def test_prefix_family_separates_a_group():
    orders = _stamped("g", [f"SO-{70000 + i}" for i in range(80)], gap="30min")
    out = identify_objects(orders)
    assert _row(out, "g")["object_type"] == "sales_order"
    purchase = _stamped("g", [f"PO-{70000 + i}" for i in range(80)], gap="30min")
    assert _row(identify_objects(purchase), "g")["object_type"] == "purchase_order"


def test_prefix_alone_is_never_decisive():
    # Random tool-prefixed codes with no tool behaviour: the prefix weight
    # (0.25) stays below the type threshold.
    rng = np.random.default_rng(1)
    codes = [f"WZ{int(x)}" for x in rng.integers(1000, 9999, 200)]
    out = identify_objects(_held("z", codes, 2, freq="5s"))
    row = _row(out, "z")
    assert row["object_type"] != "tool"


def test_fallback_type_says_what_was_missing():
    rng = np.random.default_rng(3)
    letters = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
    pool = [
        "".join(rng.choice(letters, 3)) + str(int(x)) for x in rng.integers(10, 99, 30)
    ]
    # held about 12 minutes, a fifth of the runs revisit an earlier code:
    # between a serial (one cycle) and a batch (long holds), no format cue
    codes = [pool[i] if i % 5 else pool[max(i - 7, 0)] for i in range(30)]
    out = identify_objects(_held("z", codes, 12, freq="1min"))
    row = _row(out, "z")
    assert row["role"] == "identifier"
    assert row["object_type"] == "identifier", out.explain("z")
    assert any("no type reached" in e for e in row["evidence"])


def test_gs1_and_check_digits_are_validated():
    good = [_check_digit(f"400123456{i:03d}") for i in range(1, 30)]
    bad = [g[:-1] + str((int(g[-1]) + 1) % 10) for g in good]
    ok = identify_objects(_held("z", good, 30))
    wrong = identify_objects(_held("z", bad, 30))
    assert _row(ok, "z")["object_type"] == "article"
    assert any("gtin" in e for e in _row(ok, "z")["evidence"])
    assert _row(wrong, "z")["object_type"] != "article"
    assert not any("gtin" in e for e in _row(wrong, "z")["evidence"])


def test_sequential_serial_is_not_date_coded():
    # SN250023 reads as YYDDD 25-023 + "" -- but day 23 is weeks before the
    # value first appears, so it is not a date code.
    serials = _held("z", [f"SN{250000 + i}" for i in range(240)], 3)
    row = _row(identify_objects(serials), "z")
    assert not any("date" in e for e in row["evidence"]), row["evidence"]
    dated = _held("z", [f"B250303-{i:03d}" for i in range(1, 30)], 30)
    assert any(
        "date_coded" in e for e in _row(identify_objects(dated), "z")["evidence"]
    )


def test_sequential_integer_serial_is_not_a_counter():
    serial = _held("z", list(range(5_000_000, 5_000_300)), 2, column="value_integer")
    row = _row(identify_objects(serial), "z")
    assert row["role"] == "identifier"
    assert row["object_type"] == "serial"


def test_shift_definitions_drive_operator_alignment():
    ops = _held("f", ["E1", "E2", "E3", "E1", "E2", "E4"] * 2, 96, freq="5min")
    ops["value_string"] = ops["value_string"].str.replace("E", "E10")
    assert _row(identify_objects(ops), "f")["object_type"] == "operator"
    other = {"early": ("07:30", "15:30"), "late": ("15:30", "23:30")}
    out = identify_objects(ops, shift_definitions=other)
    assert _row(out, "f")["object_type"] != "operator"


# ---------- clean-up, codes and states ----------------------------------------


def test_misreads_do_not_hide_a_small_pool():
    # three fixtures, each held for hours, read with no-reads, truncations,
    # lower case and dropped samples: still three code values, an identifier
    seq = [v for v in ["FX-13", "FX-35", "FX-13", "FX-28"] * 3 for _ in range(60)]
    for i, noise in zip(range(7, len(seq), 37), ["NOREAD", "FX-", "fx-13", None] * 30):
        seq[i] = noise
    out = identify_objects(
        _sig("z", pd.date_range(START, periods=len(seq), freq="2min"), seq)
    )
    row = _row(out, "z")
    assert row["role"] == "identifier", out.explain("z")
    assert row["distinct_count"] == 3
    assert any("misread" in e for e in row["evidence"])


def test_small_enumerations_are_states_and_sparse_integers_codes():
    enum = _held("u", [1, 2, 3, 1, 3, 2] * 10, 20, column="value_integer")
    assert _row(identify_objects(enum), "u")["role"] == "state"
    programs = _held("p", [17, 23, 5, 17, 5, 23] * 4, 90, column="value_integer")
    assert _row(identify_objects(programs), "p")["role"] == "identifier"


def test_decimal_text_and_compact_dates_are_not_ids():
    t = pd.date_range(START, periods=600, freq="10min")  # about 4 days
    rng = np.random.default_rng(5)
    temps = _sig("m", t, [f"{60 + x:.1f}" for x in rng.normal(0, 0.4, 600)])
    days = _sig("d", t, [x.strftime("%Y%m%d") for x in t])
    stamps = _sig("s", t, [x.strftime("%d.%m.%y %H:%M") for x in t])
    epoch = _sig("e", t, [str(int(x.timestamp())) for x in t])
    out = identify_objects(pd.concat([temps, days, stamps, epoch]))
    roles = out.table.set_index("uuid")["role"]
    assert roles["m"] == "measurement"
    assert roles[["d", "s", "e"]].tolist() == ["timestamp"] * 3


def test_messages_that_embed_ids_are_free_text():
    t = pd.date_range(START, periods=60, freq="20min")
    msgs = [
        f"Batch B25W10-{i:03d} released" if i % 3 else f"Order WO-{1000 + i} started"
        for i in range(60)
    ]
    assert _row(identify_objects(_sig("z", t, msgs)), "z")["role"] == "free_text"


def test_a_brief_code_from_a_pool_is_a_fault_code():
    # E-codes set at a stop for a few minutes, then cleared for hours
    rng = np.random.default_rng(2)
    t = pd.date_range(START, periods=3000, freq="1min")
    values = [""] * 3000
    for start in rng.choice(np.arange(0, 2990, 50), 40, replace=False):
        code = f"E{rng.choice([1043, 2077, 3101, 4410, 5120, 6033])}"
        for k in range(int(start), int(start) + int(rng.integers(2, 6))):
            values[k] = code
    row = _row(identify_objects(_sig("z", t, values)), "z")
    assert row["role"] == "state", row["evidence"]
    assert any("fault or alarm code" in e for e in row["evidence"])


# ---------- cross-signal relations --------------------------------------------


def test_attribute_of_a_work_order_is_a_recipe_of_an_order_a_partner():
    # production: a work order over batches over serials, with a recipe that
    # changes only when the work order does
    wo_codes = [f"WO-{5000 + i}" for i in range(12)]
    recipe_of = ["R-7", "R-9", "R-7", "R-4", "R-9", "R-7"] * 2
    wo = _held("wo", wo_codes, 240)
    recipe = _held("rcp", recipe_of, 240)
    batch = _held("b", [f"B{100 + i}" for i in range(48)], 60)
    serial = _held("s", [f"SN{100000 + i}" for i in range(960)], 3)
    # business: orders arriving at random, a customer changing with them
    rng = np.random.default_rng(4)
    times = START + pd.to_timedelta(np.cumsum(rng.exponential(1800, 60)), unit="s")
    custs = [f"C{x}" for x in rng.choice([17, 22, 35, 41, 58, 63, 77], 60)]
    order = _sig("so", times, [f"{8800000 + i}" for i in range(60)])
    cust = _sig("cu", times, custs)
    frames = [wo, recipe, batch, serial, order, cust]
    out = identify_objects(pd.concat(frames))
    types = out.table.set_index("uuid")["object_type"]
    assert types["rcp"] == "recipe", out.explain("rcp")
    assert types["wo"] == "work_order", out.explain("wo")
    assert types["b"] == "batch", out.explain("b")
    assert types["cu"] == "business_partner", out.explain("cu")
    assert types["so"] == "order", out.explain("so")


def test_a_fixed_count_of_parts_is_a_handling_unit():
    serial = _held("s", [f"SN{100000 + i}" for i in range(1200)], 2, freq="30s")
    packs = _held("h", [f"PK{7000 + i}" for i in range(50)], 48, freq="30s")
    out = identify_objects(pd.concat([serial, packs]))
    row = _row(out, "h")
    assert row["object_type"] == "handling_unit", out.explain("h")
    assert any("fixed count" in e for e in row["evidence"])


# ---------- report API --------------------------------------------------------


def test_table_columns_and_types(report):
    assert list(report.table.columns) == identify.REPORT_COLUMNS
    assert report.table["uuid"].is_monotonic_increasing
    assert set(report.table["role"]) <= set(identify.ROLES)
    for evidence in report.table["evidence"]:
        assert isinstance(evidence, tuple)
        assert all(isinstance(e, str) for e in evidence)
    assert report.table["sample_count"].dtype == "int64"


def test_identifiers_explain_summary(report):
    ids = report.identifiers()
    assert set(ids["uuid"]) == set(EXPECTED_TYPES)
    text = report.explain("a")
    assert text.startswith("a: identifier") and "serial" in text
    assert "ts-shape object identification" in report.summary()
    assert repr(report) == report.summary()
    with pytest.raises(KeyError):
        report.explain("missing")
    frame = report.to_frame()
    frame.loc[0, "role"] = "changed"
    assert report.table.loc[0, "role"] != "changed"


def test_report_is_a_mapping_of_types(report):
    types = dict(report)
    assert len(report) == len(report.table) == len(types)
    expected = {
        u: report.table.set_index("uuid").at[u, "object_type"] for u in EXPECTED_TYPES
    }
    assert {u: t for u, t in types.items() if t is not None} == expected
    assert all(
        expected[u] in (t, IDENTIFIER_TYPES.get(t)) for u, t in EXPECTED_TYPES.items()
    )
    assert report["p"] is None and report["a"] == "serial"
    with pytest.raises(KeyError):
        report["missing"]


def test_to_object_specs_round_trip(plant, report):
    specs = report.to_object_specs()
    assert specs, report.summary()
    kept = {s.uuid: s.object_type for s in specs}
    for uuid, otype in kept.items():
        assert _row(report, uuid)["type_confidence"] >= 0.6
        assert otype == _row(report, uuid)["object_type"]
    log = detect_objects(plant, specs)
    found = set(log.objects["ocel:type"])
    assert found == set(kept.values())
    assert schema.is_known_object_type("carrier")
    assert schema.is_known_object_type("business_partner")
    loose = report.to_object_specs(0.0, include_generic=True)
    assert len(loose) == len(report.identifiers())


def test_spec_kinds_are_untouched(plant):
    before = infer_specs(plant).to_frame()
    identify_objects(plant)
    pd.testing.assert_frame_equal(before, infer_specs(plant).to_frame())
    assert "identifier" not in KINDS


# ---------- contract ----------------------------------------------------------


def test_empty_frame_keeps_the_columns():
    empty = pd.DataFrame(
        {"systime": pd.Series(dtype="datetime64[ns]"), "uuid": [], "value_string": []}
    )
    out = identify_objects(empty)
    assert out.table.empty
    assert list(out.table.columns) == identify.REPORT_COLUMNS
    assert out.to_object_specs() == []
    assert "0 signal(s)" in out.summary()


def test_missing_columns_raise():
    with pytest.raises(ColumnNotFoundError):
        identify_objects(pd.DataFrame())
    with pytest.raises(ColumnNotFoundError):
        identify_objects(pd.DataFrame({"systime": [START], "value_string": ["A"]}))


def test_tz_aware_and_naive_agree(plant):
    naive = identify_objects(plant).table
    aware = plant.assign(systime=plant["systime"].dt.tz_localize("UTC"))
    pd.testing.assert_frame_equal(naive, identify_objects(aware).table)


def test_operator_alignment_uses_local_wall_clock():
    ops = _held(
        "f", ["E1042", "E2077", "E3101", "E1042", "E2077", "E3315"] * 2, 96, freq="5min"
    )
    berlin = ops.assign(systime=ops["systime"].dt.tz_localize("Europe/Berlin"))
    assert _row(identify_objects(berlin), "f")["object_type"] == "operator"
    utc = ops.assign(systime=ops["systime"].dt.tz_localize("Asia/Kolkata"))
    shifted = utc.assign(systime=utc["systime"].dt.tz_convert("Europe/Berlin"))
    # the same instants read in Berlin are 4.5 h off the shift starts
    assert _row(identify_objects(shifted), "f")["object_type"] != "operator"


@pytest.mark.parametrize("dtype", ["string", "string[pyarrow]", "object"])
def test_nullable_and_pyarrow_strings(dtype):
    pytest.importorskip("pyarrow")
    base = pd.concat([_plant()[k] for k in ("a", "b", "q")], ignore_index=True)
    expected = identify_objects(base).table
    typed = base.assign(value_string=base["value_string"].astype(dtype))
    pd.testing.assert_frame_equal(expected, identify_objects(typed).table)


def test_nullable_and_pyarrow_integers():
    pytest.importorskip("pyarrow")
    serial = _held("z", list(range(5_000_000, 5_000_300)), 2, column="value_integer")
    expected = identify_objects(serial).table
    for dtype in ("Int64", "int64[pyarrow]"):
        typed = serial.assign(value_integer=serial["value_integer"].astype(dtype))
        pd.testing.assert_frame_equal(expected, identify_objects(typed).table)


def test_integral_doubles_are_ids():
    serial = _held(
        "z", [float(x) for x in range(5_000_000, 5_000_300)], 2, column="value_double"
    )
    row = _row(identify_objects(serial), "z")
    assert row["object_type"] == "serial"
    assert row["pattern"] == "9{7}"


def test_nan_and_empty_readings_mean_no_item():
    values = []
    for i in range(150):
        values += [f"SN{100000 + i}", f"SN{100000 + i}", "", None]
    t = pd.date_range(START, periods=len(values), freq="15s")
    out = identify_objects(_sig("z", t, values))
    row = _row(out, "z")
    assert row["empty_ratio"] == pytest.approx(0.5)
    assert row["distinct_count"] == 150
    assert row["run_count"] == 150
    assert row["median_hold_seconds"] == pytest.approx(30.0)
    assert row["object_type"] == "serial"
    integers = _sig(
        "z", t[:8], [1.0, 1.0, np.nan, 2.0, 2.0, np.nan, 3.0, 3.0], "value_integer"
    )
    assert _row(identify_objects(integers, min_samples=1), "z")["distinct_count"] == 3


def test_all_empty_and_short_signals_are_insufficient():
    # The default min_samples is 3 (it was 20): a recipe or tool logged on
    # change has only a handful of rows over days, and 20 rows hid it.
    t = pd.date_range(START, periods=30, freq="1min")
    blank = _sig("z", t, [""] * 30)
    tiny = _held("x", ["SN1", "SN2"], 1)
    short = _held("y", [f"SN{i}" for i in range(10)], 1)
    out = identify_objects(pd.concat([blank, tiny, short]))
    assert _row(out, "z")["role"] == "insufficient_data"
    assert _row(out, "x")["role"] == "insufficient_data"
    assert _row(out, "y")["role"] == "identifier"
    assert _row(out, "y")["sample_count"] == 10
    strict = identify_objects(short, min_samples=20)
    assert _row(strict, "y")["role"] == "insufficient_data"


def test_hints_pin_the_type(plant):
    out = identify_objects(plant, hints={"c": "material", "q": "station"})
    c, q = _row(out, "c"), _row(out, "q")
    assert c["object_type"] == "material" and c["type_confidence"] == 1.0
    assert any("pinned by hints" in e for e in c["evidence"])
    assert q["role"] == "identifier" and q["object_type"] == "station"
    with pytest.raises(ValueError, match="unknown object type"):
        identify_objects(plant, hints={"c": "spaceship"})


def test_unknown_signal_uuid_warns(plant):
    with pytest.warns(DataQualityWarning, match="not found.*Available uuids"):
        out = identify_objects(plant, signal_uuids=["a", "nope"])
    assert out.table["uuid"].tolist() == ["a"]
    with pytest.warns(DataQualityWarning, match="hints"):
        identify_objects(plant, signal_uuids=["a"], hints={"zz": "serial"})


def test_signal_uuids_matches_uuid_as_text():
    serial = _held(7, [f"SN{100000 + i}" for i in range(100)], 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        out = identify_objects(serial, signal_uuids=["7"])
    assert out.table["uuid"].tolist() == ["7"]


def test_deterministic_and_order_independent(plant):
    first = identify_objects(plant)
    pd.testing.assert_frame_equal(first.table, identify_objects(plant).table)
    shuffled = plant.sample(frac=1.0, random_state=3)
    pd.testing.assert_frame_equal(first.table, identify_objects(shuffled).table)
    assert first.type_scores == identify_objects(shuffled).type_scores


def test_input_is_not_changed(plant):
    before = plant.copy()
    identify_objects(plant)
    pd.testing.assert_frame_equal(before, plant)


def test_top_level_exports():
    assert ts_shape.identify_objects is identify_objects
    assert ts_shape.ObjectReport is ObjectReport
    assert ts_shape.register_identifier_type is register_identifier_type


# ---------- extensibility -----------------------------------------------------


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(identify, "IDENTIFIER_TYPES", dict(IDENTIFIER_TYPES))
    monkeypatch.setattr(identify, "_PATTERNS", dict(identify._PATTERNS))


def test_register_type_and_pattern(clean_registry):
    register_identifier_type("coil")
    assert "coil" in identify.IDENTIFIER_TYPES
    assert schema.is_known_object_type("coil")
    register_identifier_pattern("coil-no", r"C\d{2}-\d{4}", "coil", weight=1.2)
    coils = _held("z", [f"C{25}-{1000 + i}" for i in range(40)], 30)
    row = _row(identify_objects(coils), "z")
    assert row["object_type"] == "coil", row["evidence"]
    assert any("coil-no" in e for e in row["evidence"])


def test_register_callable_pattern_and_group(clean_registry):
    register_identifier_type("drum", group="container")
    register_identifier_type("ibc", group="container")
    assert identify._groups()["container"] == ["drum", "ibc"]
    register_identifier_pattern(
        "container", lambda v: v.startswith("CT"), "container", weight=1.0
    )
    rows = _held("z", [f"CT{i:05d}" for i in range(60)], 4)
    assert _row(identify_objects(rows), "z")["object_type"] == "container"


def test_register_rejects_bad_input(clean_registry):
    with pytest.raises(ValueError):
        register_identifier_type("a:b")
    with pytest.raises(ValueError):
        register_identifier_type("drum", group="serial")
    with pytest.raises(ValueError):
        register_identifier_pattern("x", r"X", "spaceship", weight=1.0)
    with pytest.raises(ValueError):
        register_identifier_pattern("x", r"X", "serial", weight=0)


def test_a_failing_pattern_does_not_break_screening(clean_registry):
    def boom(value: str) -> bool:
        raise RuntimeError(value)

    register_identifier_pattern("boom", boom, "serial", weight=1.0)
    out = identify_objects(_held("z", [f"SN{100000 + i}" for i in range(100)], 2))
    assert _row(out, "z")["object_type"] == "serial"


# ---------- scale -------------------------------------------------------------


def _scale_frame(n_rows: int, n_signals: int = 20) -> pd.DataFrame:
    per = n_rows // n_signals
    t = pd.date_range("2025-01-06", periods=per, freq="1s")
    rng = np.random.default_rng(0)
    parts = []
    for k in range(n_signals):
        frame = pd.DataFrame({"systime": t, "uuid": f"s{k:02d}"})
        frame["value_double"] = np.nan
        frame["value_integer"] = np.nan
        frame["value_string"] = None
        kind = k % 5
        if kind == 0:
            codes = np.arange(per) // 30
            pool = np.array([f"SN{1000000 + i}" for i in range(codes[-1] + 1)], object)
            frame["value_string"] = pool[codes]
        elif kind == 1:
            codes = np.arange(per) // 3600
            pool = np.array([f"B{i:06d}" for i in range(codes[-1] + 1)], object)
            frame["value_string"] = pool[codes]
        elif kind == 2:
            frame["value_double"] = 20 + rng.normal(0, 1, per)
        elif kind == 3:
            frame["value_integer"] = (np.arange(per) // 7).astype("int64")
        else:
            pool = np.array(["RUN", "IDLE", "STOP"], dtype=object)
            frame["value_string"] = pool[(np.arange(per) // 600) % 3]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


#: Measured: 1M rows / 20 signals take about 2.3 s with a traced peak of
#: about 60 bytes per row; 10M rows take about 6-8 s (52 bytes per row).
SCALE_SECONDS = 30.0
SCALE_BYTES_PER_ROW = 150


@pytest.mark.scale
def test_scale_one_million_rows():
    df = _scale_frame(1_000_000)
    identify_objects(df.head(20_000))  # warm-up
    tracemalloc.start()
    t0 = time.perf_counter()
    out = identify_objects(df)
    seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert seconds < SCALE_SECONDS, f"{seconds:.1f} s"
    assert peak / len(df) < SCALE_BYTES_PER_ROW, f"{peak / len(df):.0f} bytes/row"
    roles = out.table.set_index("uuid")["role"]
    assert roles["s00"] == "identifier" and roles["s02"] == "measurement"
    assert roles["s03"] == "counter" and roles["s04"] == "state"
    assert _row(out, "s00")["object_type"] == "serial"
