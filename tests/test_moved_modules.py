"""Modules and names that moved keep working from their old place, with a warning.

November 2026: the machine profilers moved to ``ts_shape.events.profiling`` and
the data check ``ts_shape.profile`` became ``ts_shape.check_signals`` in
``ts_shape.signals.report``. Every old import path resolves to the same object
and emits a DeprecationWarning naming the new place.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

import ts_shape

MOVED_MODULES = {
    "ts_shape.cycle_profile": ("ts_shape.events.profiling.runner", "profile_cycles"),
    "ts_shape.unit_profiles": ("ts_shape.events.profiling.units", "profile_units"),
    "ts_shape.profiling": ("ts_shape.signals.report", "SignalReport"),
    "ts_shape.events.production.cycle_profiling": (
        "ts_shape.events.profiling.cycles",
        "CycleProfileEvents",
    ),
    "ts_shape.events.production.stop_profiling": (
        "ts_shape.events.profiling.stops",
        "StopProfileEvents",
    ),
    "ts_shape.events.production.changeover_profiling": (
        "ts_shape.events.profiling.changeovers",
        "StartupProfileEvents",
    ),
    "ts_shape.events.production.handshake_profiling": (
        "ts_shape.events.profiling.handshakes",
        "HandshakeProfileEvents",
    ),
    "ts_shape.events.production.order_profiling": (
        "ts_shape.events.profiling.orders",
        "ScrapProfileEvents",
    ),
    "ts_shape.events.production.curve_signature": (
        "ts_shape.events.profiling.curves",
        "CurveSignatureEvents",
    ),
    "ts_shape.events.production.setting_impact": (
        "ts_shape.events.profiling.settings",
        "SettingImpactEvents",
    ),
    "ts_shape.events.engineering.control_loop_profiling": (
        "ts_shape.events.profiling.control_loops",
        "ControlLoopProfileEvents",
    ),
}


@pytest.mark.parametrize("old", sorted(MOVED_MODULES))
def test_old_module_forwards_with_a_warning(old):
    new, name = MOVED_MODULES[old]
    module = importlib.import_module(old)  # importing alone does not warn
    with pytest.warns(DeprecationWarning, match=f"{old} moved to {new}"):
        obj = getattr(module, name)
    assert obj is getattr(importlib.import_module(new), name)


@pytest.mark.parametrize("old", sorted(MOVED_MODULES))
def test_old_module_lists_nothing_and_rejects_unknown_names(old):
    module = importlib.import_module(old)
    assert dir(module) == []
    with pytest.raises(AttributeError):
        module.no_such_name  # noqa: B018


@pytest.mark.parametrize(
    "package, name",
    [
        ("ts_shape.events.production", "CycleProfileEvents"),
        ("ts_shape.events.production", "OrderProfileEvents"),
        ("ts_shape.events.engineering", "ControlLoopProfileEvents"),
    ],
)
def test_moved_class_still_resolves_from_its_old_pack(package, name):
    pack = importlib.import_module(package)
    assert name not in pack.__all__
    with pytest.warns(DeprecationWarning, match="ts_shape.events.profiling"):
        cls = getattr(pack, name)
    assert cls is getattr(ts_shape.events.profiling, name)


def test_profile_is_a_deprecated_alias_of_check_signals():
    df = ts_shape.make_timeseries(["temp"], n_points=20)
    with pytest.warns(DeprecationWarning, match=r"use check_signals\(\)"):
        report = ts_shape.profile(df)
    assert isinstance(report, ts_shape.SignalReport)
    assert report.table.equals(ts_shape.check_signals(df).table)
    with pytest.warns(DeprecationWarning, match="use SignalReport"):
        ts_shape.ProfileReport(
            specs=report.specs, table=report.table, issue_table=report.issue_table
        )


def test_new_names_are_exported_and_old_ones_hidden():
    for name in ("check_signals", "SignalReport", "profile_cycles", "profile_units"):
        assert name in ts_shape.__all__
    assert "check_signals" in ts_shape.signals.__all__
    assert "profile" not in ts_shape.signals.report.__all__
    assert "profile" not in ts_shape.list_detectors()["name"].tolist()


@pytest.mark.parametrize(
    "module", ["ts_shape.events.profiling", "ts_shape.cycle_profile"]
)
def test_cli_runs_from_new_and_old_module(module):
    out = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--container" in out.stdout
