"""The landing-page charts are real ts-shape output (scripts/landing_examples.py).

The script runs each example's code and the page shows that code; this test
runs it too, so an API change that breaks a landing example fails here, and
checks that the committed data file has what the charts read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_PATH = Path(__file__).parents[1] / "scripts" / "landing_examples.py"
_spec = importlib.util.spec_from_file_location("landing_examples", _PATH)
le = importlib.util.module_from_spec(_spec)
sys.modules["landing_examples"] = le
_spec.loader.exec_module(le)


def test_examples_run_and_give_what_the_charts_need():
    data = le.build()
    assert list(data) == ["line", "loops", "spc", "energy", "drivers"]
    line, loops, spc = data["line"], data["loops"], data["spc"]
    energy, drivers = data["energy"], data["drivers"]
    assert line["stop_count"] == len(line["stops"]) > 0
    assert 0 < line["oee_pct"] <= 100
    # the stops are explained by the alarms the simulator raises before them
    assert {c["label"] for c in line["causes"][:2]} == {
        "Die pressure alarm",
        "Web break alarm",
    }
    # the simulator's good / oscillating / sluggish loops are told apart
    assert [lp["diagnosis"] for lp in loops["loops"]] == [
        "ok",
        "oscillating",
        "sluggish",
    ]
    assert all(len(lp["response"]) > 50 for lp in loops["loops"])
    assert len(spc["points"]) == 480 and spc["lcl"] < spc["mean"] < spc["ucl"]
    assert spc["out_of_limits"] and spc["run_spans"]
    assert spc["first_run_rule"] <= spc["drift_from"]
    assert energy["best_kwh_per_part"] < energy["median_kwh_per_part"]
    assert energy["median_kwh_per_part"] < energy["worst_kwh_per_part"]
    # only setpoint and valve output drive the process value, not the look-alikes
    causal = {sg["uuid"] for sg in drivers["signals"] if sg["causal"]}
    assert causal == {"plc:tic101_sp", "plc:tic101_op"}
    for example in data.values():
        assert "import ts_shape" in example["code"]


def test_committed_data_file_has_every_field():
    committed = json.loads(le.OUT.read_text(encoding="utf-8"))
    fresh = le.build()
    for name, example in fresh.items():
        assert set(committed[name]) == set(example), name
