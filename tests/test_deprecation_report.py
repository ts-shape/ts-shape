"""The deprecation report (scripts/deprecation_report.py) finds every kind.

The report is generated from markers the deprecation helpers set, so a helper
that stops setting its marker (or a kind the report stops reading) shows up
here as a missing entry.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

import ts_shape

_PATH = Path(__file__).parents[1] / "scripts" / "deprecation_report.py"
_spec = importlib.util.spec_from_file_location("deprecation_report", _PATH)
dr = importlib.util.module_from_spec(_spec)
sys.modules["deprecation_report"] = dr  # dataclasses look the module up
_spec.loader.exec_module(dr)


@pytest.fixture(scope="module")
def report():
    return dr.collect()


def _plain(rows):
    return {tuple(cell.strip("`") for cell in row) for row in rows}


def test_every_kind_is_found(report):
    empty = [kind for kind, rows in report.rows.items() if not rows]
    assert not empty, f"no entries for {empty}"


@pytest.mark.parametrize(
    "kind, row",
    [
        ("classes", ("OEECalculator", "OEEEvents")),
        ("classes", ("ProfileReport", "SignalReport")),
        ("functions", ("profile", "check_signals")),
        ("modules", ("ts_shape.cycle_profile", "ts_shape.events.profiling.runner")),
        (
            "moved_names",
            (
                "ts_shape.events.production.CycleProfileEvents",
                "ts_shape.events.profiling.cycles.CycleProfileEvents",
            ),
        ),
        ("columns", ("ToleranceDeviationEvents.capability_indices", "Cpk", "cpk")),
    ],
)
def test_known_entries(report, kind, row):
    assert row in _plain(report.rows[kind])


def test_marker_based_kinds_have_entries_from_the_helpers(report):
    # Spot-check one entry of each helper-marked kind by its shape.
    assert any(len(r) == 3 for r in _plain(report.rows["keywords"]))
    assert any(r[1] for r in _plain(report.rows["keyword_only"]))
    assert any(len(r) == 3 for r in _plain(report.rows["methods"]))
    assert any(len(r) == 3 for r in _plain(report.rows["attributes"]))


def test_counts_do_not_shrink_by_accident(report):
    # Minimums as of November 2026; removing deprecations (1.0) lowers them
    # on purpose, together with this test.
    counts = report.counts()
    assert counts["classes"] >= 18
    assert counts["keywords"] >= 290
    assert counts["keyword_only"] >= 160
    assert counts["columns"] >= 90
    assert counts["modules"] >= 11


def test_page_renders_with_counts_and_tables(report, tmp_path):
    out = tmp_path / "deprecations.md"
    out.write_text(dr.render(report), encoding="utf-8")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("---\ntitle: Deprecated names")
    assert f"| **Total** | **{report.total()}** |" in text
    assert "| `OEECalculator` | `OEEEvents` |" in text


def test_dir_lists_only_exports():
    leaked = {"importlib", "logging", "version", "NullHandler"}
    leaked |= {"PackageNotFoundError", "TYPE_CHECKING"}
    assert not leaked & set(dir(ts_shape))
    assert {n for n in dir(ts_shape) if not n.startswith("_")} >= set(ts_shape.__all__)
