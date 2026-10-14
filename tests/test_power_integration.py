"""Power (kW) -> energy (kWh) integration.

Every case here is hand-computable: the expected kWh is stated in the
docstring or comment so a failure points at the physics, not at a fixture.
"""

import pandas as pd  # type: ignore
import pytest  # type: ignore

from ts_shape.events.energy.consumption_analysis import EnergyConsumptionEvents
from ts_shape.transform.calculator.power_integration import PowerIntegrator


def _df(rows, uuid="m1"):
    """Build a ts-shape power frame from (timestamp, kW) pairs."""
    return pd.DataFrame(
        [
            {
                "systime": pd.Timestamp(ts),
                "uuid": uuid,
                "value_double": float(kw),
            }
            for ts, kw in rows
        ]
    )


# ---------------------------------------------------------------------------
# Core integration
# ---------------------------------------------------------------------------


def test_constant_power_hour_equals_power():
    """60 kW held for a full hour is 60 kWh, whatever the sample count."""
    rows = [(f"2024-01-01 00:{m:02d}:00", 60.0) for m in range(0, 60, 10)]
    rows.append(("2024-01-01 01:00:00", 60.0))
    out = PowerIntegrator.energy_by_window(_df(rows))

    assert len(out) == 1
    assert out["kwh"].iloc[0] == pytest.approx(60.0)
    assert out["avg_kw"].iloc[0] == pytest.approx(60.0)
    assert out["coverage"].iloc[0] == pytest.approx(1.0)
    assert out["start"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert out["end"].iloc[0] == pd.Timestamp("2024-01-01 01:00:00")


def test_step_change_is_time_weighted_not_sample_weighted():
    """100 kW for 30 min then 200 kW for 30 min = 150 kWh."""
    out = PowerIntegrator.energy_by_window(
        _df(
            [
                ("2024-01-01 00:00:00", 100.0),
                ("2024-01-01 00:30:00", 200.0),
                ("2024-01-01 01:00:00", 200.0),
            ]
        )
    )
    assert out["kwh"].iloc[0] == pytest.approx(150.0)


def test_sample_count_mean_would_be_wrong():
    """The failure mode this class exists for.

    200 kW for 59 minutes, then a 900 kW spike logged 10 times in the last
    minute. Time-weighted energy is 211.7 kWh; a plain mean of the samples
    reads ~836 kW and would claim ~836 kWh.
    """
    rows = [("2024-01-01 00:00:00", 200.0)]
    rows += [
        (f"2024-01-01 00:59:{s:02d}", 900.0) for s in range(0, 60, 6)
    ]  # 10 samples
    rows.append(("2024-01-01 01:00:00", 900.0))
    df = _df(rows)

    out = PowerIntegrator.energy_by_window(df)
    expected = 200.0 * (59 / 60) + 900.0 * (1 / 60)
    assert out["kwh"].iloc[0] == pytest.approx(expected)
    assert df["value_double"].mean() > 800  # the naive answer, for contrast


def test_segment_straddling_window_boundary_is_split():
    """120 kW from 00:55 to 01:05 books 10 kWh into each hour."""
    out = PowerIntegrator.energy_by_window(
        _df([("2024-01-01 00:55:00", 120.0), ("2024-01-01 01:05:00", 0.0)])
    )
    assert len(out) == 2
    assert out["kwh"].tolist() == pytest.approx([10.0, 10.0])
    # Only 5 of 60 minutes of each hour are actually covered.
    assert out["coverage"].tolist() == pytest.approx([5 / 60, 5 / 60])
    # ...but the mean power while covered is the real 120 kW.
    assert out["avg_kw"].tolist() == pytest.approx([120.0, 120.0])


def test_multi_day_window():
    """1D windows: 24 h at 10 kW = 240 kWh per day."""
    rows = [(f"2024-01-0{d} 00:00:00", 10.0) for d in (1, 2, 3)]
    out = PowerIntegrator.energy_by_window(_df(rows), freq="1D")
    assert out["kwh"].tolist() == pytest.approx([240.0, 240.0])


# ---------------------------------------------------------------------------
# Interpolation method
# ---------------------------------------------------------------------------


def test_linear_ramp_is_exact_trapezoidal():
    """A 0 -> 100 kW ramp over one hour is exactly 50 kWh trapezoidal.

    LOCF lags by half a step (49.17 kWh on 60 samples) -- interpolation bias,
    not an error.
    """
    rows = [
        (f"2024-01-01 00:{m:02d}:00" if m < 60 else "2024-01-01 01:00:00", m * 100 / 60)
        for m in range(61)
    ]
    df = _df(rows)

    linear = PowerIntegrator.energy_by_window(df, method="linear")
    locf = PowerIntegrator.energy_by_window(df, method="locf")

    assert linear["kwh"].iloc[0] == pytest.approx(50.0)
    assert locf["kwh"].iloc[0] == pytest.approx(100 / 60 * 29.5, rel=1e-9)
    assert locf["kwh"].iloc[0] < linear["kwh"].iloc[0]


def test_linear_interpolation_survives_a_boundary_split():
    """0 -> 120 kW over two hours, split at 01:00, still totals 120 kWh.

    Each half is 60 kWh: the ramp is interpolated at the boundary (60 kW),
    not carried across it.
    """
    out = PowerIntegrator.energy_by_window(
        _df([("2024-01-01 00:00:00", 0.0), ("2024-01-01 02:00:00", 120.0)]),
        method="linear",
    )
    assert out["kwh"].tolist() == pytest.approx([30.0, 90.0])
    assert out["kwh"].sum() == pytest.approx(120.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method"):
        PowerIntegrator.energy_by_window(
            _df([("2024-01-01 00:00:00", 1.0)]), method="cubic"
        )


# ---------------------------------------------------------------------------
# Gaps and coverage
# ---------------------------------------------------------------------------


def test_max_gap_suppresses_phantom_energy_across_an_outage():
    """One 500 kW sample before a 3 h logging outage must not bill 1500 kWh."""
    rows = [
        ("2024-01-01 00:00:00", 500.0),
        ("2024-01-01 03:00:00", 500.0),  # logger came back
        ("2024-01-01 03:30:00", 500.0),
    ]
    uncapped = PowerIntegrator.energy_by_window(_df(rows))
    # The cap must sit above the expected logging cadence: every segment
    # longer than it is treated as an outage, not as a value that persisted.
    capped = PowerIntegrator.energy_by_window(_df(rows), max_gap="1h")

    assert uncapped["kwh"].sum() == pytest.approx(1500.0 + 250.0)
    # Only the 30 min segment after the outage survives the cap.
    assert capped["kwh"].sum() == pytest.approx(250.0)
    assert capped["start"].tolist() == [pd.Timestamp("2024-01-01 03:00:00")]
    assert capped["coverage"].iloc[0] == pytest.approx(0.5)


def test_unmeasured_windows_are_omitted_not_zero():
    """A silent hour is absent, so 'not measured' never reads as 'used nothing'."""
    out = PowerIntegrator.energy_by_window(
        _df(
            [
                ("2024-01-01 00:00:00", 100.0),
                ("2024-01-01 01:00:00", 100.0),
                ("2024-01-01 05:00:00", 100.0),
                ("2024-01-01 06:00:00", 100.0),
            ]
        ),
        max_gap="1h",
    )
    starts = out["start"].tolist()
    assert pd.Timestamp("2024-01-01 03:00:00") not in starts
    assert starts == [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2024-01-01 05:00:00"),
    ]


def test_data_end_closes_the_trailing_segment():
    """Without data_end the last sample has no duration and contributes nothing."""
    rows = [("2024-01-01 00:00:00", 60.0)]
    assert PowerIntegrator.energy_by_window(_df(rows)).empty

    out = PowerIntegrator.energy_by_window(_df(rows), data_end="2024-01-01 01:00:00")
    assert out["kwh"].iloc[0] == pytest.approx(60.0)


def test_duplicate_timestamps_do_not_double_count():
    """Two readings at the same instant span zero time; the later one wins."""
    out = PowerIntegrator.energy_by_window(
        _df(
            [
                ("2024-01-01 00:00:00", 10.0),
                ("2024-01-01 00:00:00", 100.0),
                ("2024-01-01 01:00:00", 100.0),
            ]
        )
    )
    assert out["kwh"].iloc[0] == pytest.approx(100.0)


def test_non_numeric_and_missing_values_are_dropped():
    df = _df([("2024-01-01 00:00:00", 60.0), ("2024-01-01 01:00:00", 60.0)])
    df.loc[len(df)] = {
        "systime": pd.Timestamp("2024-01-01 00:30:00"),
        "uuid": "m1",
        "value_double": float("nan"),
    }
    out = PowerIntegrator.energy_by_window(df)
    assert out["kwh"].iloc[0] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Units, signs, multi-signal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unit", "value", "expected"),
    [("kW", 60.0, 60.0), ("W", 60_000.0, 60.0), ("MW", 0.06, 60.0)],
)
def test_power_unit_scaling(unit, value, expected):
    out = PowerIntegrator.energy_by_window(
        _df([("2024-01-01 00:00:00", value), ("2024-01-01 01:00:00", value)]),
        power_unit=unit,
    )
    assert out["kwh"].iloc[0] == pytest.approx(expected)


def test_unknown_power_unit_raises():
    with pytest.raises(ValueError, match="power_unit"):
        PowerIntegrator.energy_by_window(
            _df([("2024-01-01 00:00:00", 1.0)]), power_unit="horsepower"
        )


def test_negative_power_is_net_energy():
    """Export/regeneration nets off against import."""
    out = PowerIntegrator.energy_by_window(
        _df(
            [
                ("2024-01-01 00:00:00", 100.0),
                ("2024-01-01 00:30:00", -100.0),
                ("2024-01-01 01:00:00", 0.0),
            ]
        )
    )
    assert out["kwh"].iloc[0] == pytest.approx(0.0)


def test_energy_adds_across_signals():
    """Two meters, 40 kW and 60 kW: the plant total is 100 kWh, not 50."""
    df = pd.concat(
        [
            _df(
                [("2024-01-01 00:00:00", 40.0), ("2024-01-01 01:00:00", 40.0)],
                uuid="m1",
            ),
            _df(
                [("2024-01-01 00:00:00", 60.0), ("2024-01-01 01:00:00", 60.0)],
                uuid="m2",
            ),
        ]
    )

    total = PowerIntegrator.energy_by_window(df, total=True)
    assert len(total) == 1
    assert total["kwh"].iloc[0] == pytest.approx(100.0)
    assert total["avg_kw"].iloc[0] == pytest.approx(100.0)
    assert total["sample_count"].iloc[0] == 2

    per_meter = PowerIntegrator.energy_by_window(df, total=False)
    assert per_meter["uuid"].tolist() == ["m1", "m2"]
    assert per_meter["kwh"].tolist() == pytest.approx([40.0, 60.0])


def test_total_coverage_reports_the_least_covered_signal():
    """A total built on a partly-logged meter is flagged as under-reported."""
    df = pd.concat(
        [
            _df(
                [("2024-01-01 00:00:00", 40.0), ("2024-01-01 01:00:00", 40.0)],
                uuid="m1",
            ),
            _df(
                [("2024-01-01 00:00:00", 60.0), ("2024-01-01 00:15:00", 60.0)],
                uuid="m2",
            ),
        ]
    )
    total = PowerIntegrator.energy_by_window(df, total=True)
    assert total["coverage"].iloc[0] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Audit trail, schema handoff, edge cases
# ---------------------------------------------------------------------------


def test_segment_energy_sums_to_the_windowed_result():
    df = _df(
        [
            ("2024-01-01 00:40:00", 90.0),
            ("2024-01-01 01:20:00", 30.0),
            ("2024-01-01 02:00:00", 30.0),
        ]
    )
    segments = PowerIntegrator.segment_energy(df)
    windows = PowerIntegrator.energy_by_window(df)

    assert list(segments.columns) == [
        "uuid",
        "window_start",
        "start",
        "end",
        "duration_seconds",
        "kw_start",
        "kw_end",
        "kwh",
    ]
    # The 00:40 sample is split at 01:00, so three segments over three hours.
    assert len(segments) == 3
    assert segments.groupby("window_start")["kwh"].sum().tolist() == pytest.approx(
        windows["kwh"].tolist()
    )


def test_to_energy_deltas_feeds_the_energy_pack():
    df = pd.concat(
        [
            _df(
                [
                    ("2024-01-01 00:00:00", 40.0),
                    ("2024-01-01 01:00:00", 80.0),
                    ("2024-01-01 02:00:00", 80.0),
                ],
                uuid="m1",
            ),
            _df(
                [("2024-01-01 00:00:00", 10.0), ("2024-01-01 02:00:00", 10.0)],
                uuid="m2",
            ),
        ]
    )

    deltas = PowerIntegrator.to_energy_deltas(df)
    assert list(deltas.columns) == ["systime", "uuid", "value_double", "is_delta"]
    assert deltas["is_delta"].all()

    consumption = EnergyConsumptionEvents(deltas).consumption_by_window(
        "m1", window="1h", agg="sum"
    )
    assert consumption["consumption"].tolist() == pytest.approx([40.0, 80.0])


def test_to_energy_deltas_total_uses_the_event_uuid():
    df = pd.concat(
        [
            _df(
                [("2024-01-01 00:00:00", 40.0), ("2024-01-01 01:00:00", 40.0)],
                uuid="m1",
            ),
            _df(
                [("2024-01-01 00:00:00", 60.0), ("2024-01-01 01:00:00", 60.0)],
                uuid="m2",
            ),
        ]
    )
    deltas = PowerIntegrator.to_energy_deltas(df, total=True, event_uuid="plant:kwh")
    assert deltas["uuid"].tolist() == ["plant:kwh"]
    assert deltas["value_double"].tolist() == pytest.approx([100.0])


def test_empty_input_returns_empty_frames_with_schema():
    empty = pd.DataFrame(columns=["systime", "uuid", "value_double"])
    assert list(PowerIntegrator.energy_by_window(empty).columns) == [
        "start",
        "end",
        "duration_seconds",
        "kwh",
        "avg_kw",
        "coverage",
        "sample_count",
    ]
    assert PowerIntegrator.energy_by_window(empty, total=False).empty
    assert list(PowerIntegrator.to_energy_deltas(empty).columns) == [
        "systime",
        "uuid",
        "value_double",
        "is_delta",
    ]
    assert PowerIntegrator.segment_energy(empty).empty


def test_single_sample_without_data_end_returns_empty():
    assert PowerIntegrator.energy_by_window(_df([("2024-01-01 00:00:00", 5.0)])).empty


def test_missing_column_raises_column_not_found():
    from ts_shape.errors import ColumnNotFoundError

    df = _df([("2024-01-01 00:00:00", 5.0)]).drop(columns=["uuid"])
    with pytest.raises(ColumnNotFoundError):
        PowerIntegrator.energy_by_window(df)


def test_alternate_column_names():
    """The raw CSV model (time | id | value) works via the column kwargs."""
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
            "id": "meter-7",
            "value": [75.0, 75.0],
        }
    )
    out = PowerIntegrator.energy_by_window(
        df, time_column="time", uuid_column="id", value_column="value", total=False
    )
    assert out["uuid"].tolist() == ["meter-7"]
    assert out["kwh"].iloc[0] == pytest.approx(75.0)


def test_exported_at_top_level():
    import ts_shape

    assert ts_shape.PowerIntegrator is PowerIntegrator
    assert "PowerIntegrator" in ts_shape.__all__


def test_runs_as_a_pipeline_transform():
    """The documented handoff: integrate, then detect on the kWh deltas."""
    from ts_shape.pipeline import Pipeline

    df = _df(
        [(ts, 60.0) for ts in pd.date_range("2024-01-01", periods=25, freq="5min")]
    )
    result = (
        Pipeline(name="hourly-kwh")
        .transform(PowerIntegrator, "to_energy_deltas", freq="1h", max_gap="15min")
        .detect(
            EnergyConsumptionEvents,
            "consumption_by_window",
            name="kwh",
            meter_uuid="m1",
            window="1h",
            agg="sum",
        )
        .run(df)
    )
    assert result.events["kwh"]["consumption"].tolist() == pytest.approx([60.0, 60.0])


def test_timezone_aware_input_is_preserved():
    """Elapsed time is measured on the tz-aware values, DST included.

    01:00 to 04:00 local on the European spring-forward day is only two real
    hours, because 02:00 never happens.
    """
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2024-03-31 01:00:00", "2024-03-31 04:00:00"]
            ).tz_localize("Europe/Berlin"),
            "uuid": "m1",
            "value_double": [10.0, 10.0],
        }
    )
    out = PowerIntegrator.energy_by_window(df, freq="1h")

    assert str(out["start"].dt.tz) == "Europe/Berlin"
    assert out["kwh"].sum() == pytest.approx(20.0)
    assert len(out) == 2


def test_non_default_index_does_not_mispair_columns():
    df = _df([("2024-01-01 00:00:00", 60.0), ("2024-01-01 01:00:00", 60.0)])
    df.index = [7, 3]
    out = PowerIntegrator.energy_by_window(df)
    assert out["kwh"].iloc[0] == pytest.approx(60.0)


def test_daily_windows_conserve_energy_across_a_dst_changeover():
    """47 real hours at 10 kW is 470 kWh, however the local days are labelled."""
    df = pd.DataFrame(
        {
            "systime": pd.to_datetime(
                ["2024-03-30 12:00:00", "2024-04-01 12:00:00"]
            ).tz_localize("Europe/Berlin"),
            "uuid": "m1",
            "value_double": [10.0, 10.0],
        }
    )
    out = PowerIntegrator.energy_by_window(df, freq="1D")
    assert out["kwh"].sum() == pytest.approx(470.0)


def test_non_numeric_power_and_bad_timestamps_warn():
    from ts_shape.errors import DataQualityWarning

    df = _df([("2024-01-01 00:00:00", 60.0), ("2024-01-01 01:00:00", 60.0)])
    df = df.astype({"value_double": object, "systime": object})
    extra = pd.DataFrame(
        {
            "systime": [pd.Timestamp("2024-01-01 00:30:00"), "not a time"],
            "uuid": "m1",
            "value_double": ["n/a", 10.0],
        }
    )
    df = pd.concat([df, extra], ignore_index=True)
    with pytest.warns(DataQualityWarning) as caught:
        out = PowerIntegrator.energy_by_window(df)
    messages = " ".join(str(w.message) for w in caught)
    assert "1 non-null value(s) in power column 'value_double'" in messages
    assert "1 value(s) in time column 'systime' are not timestamps" in messages
    assert out["kwh"].iloc[0] == pytest.approx(60.0)
