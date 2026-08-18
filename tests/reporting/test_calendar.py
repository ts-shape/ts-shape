"""Tests for per-facility local time, shifts and DST-correct periods."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.reporting import Facility, FacilityCalendar, FacilityRegistry


@pytest.fixture()
def calendar(registry) -> FacilityCalendar:
    return FacilityCalendar(registry)


@pytest.fixture()
def readings() -> pd.DataFrame:
    """One UTC instant, seen from three sites, plus an unmapped one."""
    return pd.DataFrame(
        {
            "facility": ["DE-STR", "US-DTW", "CN-SHA", "GHOST"],
            "systime": pd.to_datetime(["2026-03-02 23:30:00"] * 4),
            "parts": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_localize_converts_per_facility(calendar, readings):
    local = calendar.localize(readings)
    by_site = dict(zip(local["facility"], local["local_time"]))
    assert by_site["DE-STR"] == pd.Timestamp("2026-03-03 00:30:00")
    assert by_site["US-DTW"] == pd.Timestamp("2026-03-02 18:30:00")
    assert by_site["CN-SHA"] == pd.Timestamp("2026-03-03 07:30:00")
    assert dict(zip(local["facility"], local["utc_offset_hours"]))["CN-SHA"] == 8.0


def test_localize_leaves_source_column_tz_aware(calendar, readings):
    local = calendar.localize(readings)
    assert str(local["systime"].dt.tz) == "UTC"


def test_unknown_facility_falls_back_but_can_be_strict(calendar, readings):
    local = calendar.localize(readings)
    ghost = local[local["facility"] == "GHOST"].iloc[0]
    assert ghost["timezone"] == "UTC"
    with pytest.raises(ValueError, match="GHOST"):
        calendar.localize(readings, strict=True)


def test_localize_requires_columns(calendar, readings):
    with pytest.raises(ValueError, match="not found"):
        calendar.localize(readings, time_column="nope")


def test_localize_empty_frame_keeps_columns(calendar):
    empty = pd.DataFrame({"facility": [], "systime": []})
    local = calendar.localize(empty)
    assert local.empty
    assert {"local_time", "timezone", "utc_offset_hours"} <= set(local.columns)


def test_assign_shifts_uses_local_time_and_production_day(calendar, readings):
    shifted = calendar.assign_shifts(readings)
    by_site = shifted.set_index("facility")
    # 00:30 local in Stuttgart is the night shift that started the day before.
    assert by_site.loc["DE-STR", "shift"] == "shift_3"
    assert by_site.loc["DE-STR", "production_date"] == pd.Timestamp("2026-03-02").date()
    # 18:30 local in Detroit is the afternoon shift, same calendar day.
    assert by_site.loc["US-DTW", "shift"] == "shift_2"
    assert by_site.loc["US-DTW", "production_date"] == pd.Timestamp("2026-03-02").date()


def test_production_day_can_be_disabled(calendar, readings):
    raw = calendar.assign_shifts(readings, production_day=False)
    stuttgart = raw[raw["facility"] == "DE-STR"].iloc[0]
    assert stuttgart["production_date"] == pd.Timestamp("2026-03-03").date()


def test_site_specific_shift_patterns_are_honoured():
    registry = FacilityRegistry(
        [
            Facility(
                "TWO",
                timezone="UTC",
                shifts={"a": ("00:00", "12:00"), "b": ("12:00", "00:00")},
            ),
            Facility("THREE", timezone="UTC"),
        ]
    )
    frame = pd.DataFrame(
        {
            "facility": ["TWO", "THREE"],
            "systime": pd.to_datetime(["2026-01-01 15:00:00"] * 2),
        }
    )
    shifted = FacilityCalendar(registry).assign_shifts(frame).set_index("facility")
    assert shifted.loc["TWO", "shift"] == "b"
    assert shifted.loc["THREE", "shift"] == "shift_2"


def test_local_periods_carry_auditable_utc_bounds(calendar, readings):
    daily = calendar.local_periods(readings, freq="D")
    shanghai = daily[daily["facility"] == "CN-SHA"].iloc[0]
    assert shanghai["period"] == pd.Timestamp("2026-03-03")
    assert shanghai["period_start_utc"] == pd.Timestamp("2026-03-02 16:00", tz="UTC")
    assert shanghai["period_end_utc"] == pd.Timestamp("2026-03-03 16:00", tz="UTC")


def test_local_periods_supports_month_aliases(calendar, readings):
    monthly = calendar.local_periods(readings, freq="MS")
    assert set(monthly["period"]) == {pd.Timestamp("2026-03-01")}


def test_period_grid_is_dst_correct():
    registry = FacilityRegistry([Facility("DE", timezone="Europe/Berlin")])
    grid = FacilityCalendar(registry).period_grid("2026-03-28", "2026-03-30")
    hours = (
        grid["period_end_utc"] - grid["period_start_utc"]
    ).dt.total_seconds() / 3600.0
    # Spring forward: the 29th is a 23-hour local day.
    assert list(hours) == [24.0, 23.0, 24.0]


def test_period_grid_covers_every_facility(calendar, registry):
    grid = calendar.period_grid("2026-01-01", "2026-01-03")
    assert len(grid) == 3 * len(registry)
    assert set(grid["facility"]) == set(registry.ids)


def test_period_grid_can_drop_non_working_days(calendar):
    # 2026-01-03 and 04 are a Saturday and Sunday.
    grid = calendar.period_grid(
        "2026-01-01", "2026-01-05", facilities=["DE-STR"], working_days_only=True
    )
    assert list(grid["period"].dt.day) == [1, 2, 5]


def test_period_grid_unknown_facility_raises(calendar):
    with pytest.raises(KeyError, match="GHOST"):
        calendar.period_grid("2026-01-01", "2026-01-02", facilities=["GHOST"])


def test_period_grid_empty_selection(calendar):
    assert calendar.period_grid("2026-01-01", "2026-01-02", facilities=[]).empty


def test_utc_window_round_trips(calendar):
    start, end = calendar.utc_window("CN-SHA", "2026-03-01", "2026-03-02")
    assert start == pd.Timestamp("2026-02-28 16:00", tz="UTC")
    assert end == pd.Timestamp("2026-03-01 16:00", tz="UTC")


def test_naive_input_is_assumed_utc_by_default(registry):
    frame = pd.DataFrame(
        {"facility": ["CN-SHA"], "systime": pd.to_datetime(["2026-01-01 00:00"])}
    )
    local = FacilityCalendar(registry).localize(frame)
    assert local.loc[0, "local_time"] == pd.Timestamp("2026-01-01 08:00")


def test_assume_timezone_is_configurable(registry):
    frame = pd.DataFrame(
        {"facility": ["CN-SHA"], "systime": pd.to_datetime(["2026-01-01 00:00"])}
    )
    calendar = FacilityCalendar(registry, assume_timezone="Europe/Berlin")
    local = calendar.localize(frame)
    assert local.loc[0, "local_time"] == pd.Timestamp("2026-01-01 07:00")


def test_registry_type_is_checked():
    with pytest.raises(TypeError, match="FacilityRegistry"):
        FacilityCalendar({"DE": "Europe/Berlin"})  # type: ignore[arg-type]
