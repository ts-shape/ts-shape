"""Tests for facility master data and its validation."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import Facility, FacilityRegistry


def test_facility_requires_id():
    with pytest.raises(ValueError, match="non-empty"):
        Facility("  ")


def test_label_falls_back_to_id():
    assert Facility("A").label == "A"
    assert Facility("A", name="Alpha").label == "Alpha"


def test_utc_offset_is_dst_aware():
    berlin = Facility("DE", timezone="Europe/Berlin")
    winter = berlin.utc_offset_hours(pd.Timestamp("2026-01-15", tz="UTC"))
    summer = berlin.utc_offset_hours(pd.Timestamp("2026-07-15", tz="UTC"))
    assert winter == 1.0
    assert summer == 2.0


def test_unknown_timezone_raises_naming_the_facility():
    with pytest.raises(ValueError, match="DE-STR"):
        Facility("DE-STR", timezone="Mars/Olympus").tzinfo()


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate facility id"):
        FacilityRegistry([Facility("A"), Facility("A")])


def test_registry_access(registry):
    assert len(registry) == 3
    assert registry.ids == ["CN-SHA", "DE-STR", "US-DTW"]
    assert "DE-STR" in registry
    assert registry["DE-STR"].name == "Stuttgart"
    assert registry.currencies["CN-SHA"] == "CNY"
    assert registry.regions()["EMEA"] == ["DE-STR"]


def test_unknown_facility_lists_known_ids(registry):
    with pytest.raises(KeyError, match="DE-STR"):
        registry.get("NOPE")


def test_filter_narrows_registry(registry):
    emea = registry.filter(region="EMEA")
    assert emea.ids == ["DE-STR"]
    assert registry.ids == ["CN-SHA", "DE-STR", "US-DTW"]  # unchanged


def test_to_frame_shape(registry):
    frame = registry.to_frame()
    assert list(frame["facility"]) == ["CN-SHA", "DE-STR", "US-DTW"]
    assert "timezone" in frame.columns


def test_empty_registry_to_frame_keeps_columns():
    frame = FacilityRegistry([]).to_frame()
    assert frame.empty
    assert "facility" in frame.columns


def test_attach_keeps_unmapped_sites_visible(registry):
    kpis = pd.DataFrame({"facility": ["DE-STR", "GHOST"], "parts": [1, 2]})
    attached = registry.attach(kpis, columns=["region"])
    assert list(attached["facility"]) == ["DE-STR", "GHOST"]
    assert attached.loc[0, "region"] == "EMEA"
    assert pd.isna(attached.loc[1, "region"])


def test_attach_rejects_unknown_column(registry):
    kpis = pd.DataFrame({"facility": ["DE-STR"]})
    with pytest.raises(ValueError, match="Unknown master-data column"):
        registry.attach(kpis, columns=["nonsense"])


def test_validate_is_empty_for_clean_registry(registry):
    issues = registry.validate()
    assert issues.empty
    assert list(issues.columns) == ["facility", "issue", "detail"]


def test_validate_flags_master_data_problems():
    registry = FacilityRegistry(
        [
            Facility(
                "BAD",
                timezone="Mars/Olympus",
                currency="EURO",
                region="",
                shifts={"only": ("06:00", "14:00")},
                working_days=(),
            )
        ]
    )
    issues = set(registry.validate()["issue"])
    assert {
        "unknown_timezone",
        "suspicious_currency",
        "missing_region",
        "shift_gap",
        "no_working_days",
    } <= issues


def test_validate_flags_overlapping_shifts():
    registry = FacilityRegistry(
        [
            Facility(
                "OVERLAP",
                region="EMEA",
                shifts={
                    "a": ("06:00", "15:00"),
                    "b": ("14:00", "22:00"),
                    "c": ("22:00", "06:00"),
                },
            )
        ]
    )
    assert "overlapping_shifts" in set(registry.validate()["issue"])


def test_validate_accepts_a_full_overnight_pattern():
    registry = FacilityRegistry(
        [
            Facility(
                "OK",
                region="EMEA",
                shifts={
                    "day": ("06:00", "14:00"),
                    "late": ("14:00", "22:00"),
                    "night": ("22:00", "06:00"),
                },
            )
        ]
    )
    assert registry.validate().empty


def test_from_frame_builds_registry():
    master = pd.DataFrame(
        {
            "facility": ["A", "B"],
            "name": ["Alpha", "Beta"],
            "timezone": ["Europe/Berlin", "Asia/Shanghai"],
            "region": ["EMEA", "APAC"],
            "currency": ["EUR", "CNY"],
            "working_days": ["0,1,2,3,4,5", [0, 1, 2]],
            "capacity_per_day": [100.0, None],
        }
    )
    registry = FacilityRegistry.from_frame(master)
    assert registry["A"].working_days == (0, 1, 2, 3, 4, 5)
    assert registry["B"].working_days == (0, 1, 2)
    assert registry["A"].capacity_per_day == 100.0
    assert registry["B"].capacity_per_day is None


def test_from_frame_requires_id_column():
    with pytest.raises(ValueError, match="not found"):
        FacilityRegistry.from_frame(pd.DataFrame({"site": ["A"]}))


def test_from_frame_applies_per_site_shifts():
    master = pd.DataFrame({"facility": ["A", "B"]})
    registry = FacilityRegistry.from_frame(
        master,
        shifts={"A": {"single": ("00:00", "23:59")}},
        default_shifts={"day": ("06:00", "18:00"), "night": ("18:00", "06:00")},
    )
    assert set(registry["A"].shifts) == {"single"}
    assert set(registry["B"].shifts) == {"day", "night"}
