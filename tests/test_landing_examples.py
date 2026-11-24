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
    spc, line, cycles = data["spc"], data["line"], data["cycles"]
    assert len(spc["points"]) == 480 and spc["lcl"] < spc["mean"] < spc["ucl"]
    assert spc["out_of_limits"] and spc["run_spans"]
    assert {iv[2] for iv in line["intervals"]} == {"run", "idle"}
    assert line["stop_count"] > 0 and 0 < line["oee_pct"] <= 100
    assert len(cycles["segments"]) == 6
    potentials = [s["potential_seconds"] for s in cycles["segments"]]
    assert potentials == sorted(potentials, reverse=True)
    for example in data.values():
        assert "import ts_shape" in example["code"]


def test_committed_data_file_has_every_field():
    committed = json.loads(le.OUT.read_text(encoding="utf-8"))
    fresh = le.build()
    for name, example in fresh.items():
        assert set(committed[name]) == set(example), name
