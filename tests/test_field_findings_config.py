"""Field test 0.0.88: wrong configurations fail loudly or warn.

Each case is one of the wrong configurations that 0.0.88 accepted silently
(or rejected with a raw third-party error), rebuilt on synthetic data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import ts_shape
from ts_shape.errors import DataQualityWarning
from ts_shape.utils.validate import (
    check_choice,
    check_keys,
    check_known_names,
    check_levels_ordered,
    check_order,
)

T = pd.date_range("2026-10-05", periods=600, freq="1min")


def _signal(uuid: str, values, column: str = "value_double", times=T) -> pd.DataFrame:
    df = pd.DataFrame({"systime": times, "uuid": uuid})
    for c in ("value_double", "value_integer", "value_bool", "value_string"):
        df[c] = np.nan if c != "value_string" else None
    df[column] = values
    return df


def _wave(uuid: str = "temp") -> pd.DataFrame:
    return _signal(uuid, 50 + 10 * np.sin(np.arange(len(T)) / 20.0))


def _counter(uuid: str = "parts") -> pd.DataFrame:
    return _signal(uuid, np.arange(len(T)), "value_integer")


# ---------------------------------------------------------------------------
# The validation helpers
# ---------------------------------------------------------------------------


def test_check_keys_suggests_the_closest_key() -> None:
    with pytest.raises(ValueError, match="did you mean 'hold_value'"):
        check_keys({"hold_valu": (1, 2)}, ("hold_value", "peak_value"), where="X")
    check_keys(None, ("a",))


def test_check_order_and_levels() -> None:
    with pytest.raises(ValueError, match="swapped"):
        check_order(5, 1, names=("min", "max"))
    check_order(None, 1)
    check_order(1, 1)
    with pytest.raises(ValueError):
        check_order(1, 1, strict=True)
    with pytest.raises(ValueError, match="increasing"):
        check_levels_ordered([90, 80], direction="above")
    with pytest.raises(ValueError, match="empty"):
        check_levels_ordered([])


def test_check_choice_and_known_names() -> None:
    with pytest.raises(ValueError, match="did you mean 'above'"):
        check_choice("abvoe", ("above", "below"), name="direction")
    with pytest.warns(DataQualityWarning, match="not found"):
        assert check_known_names(["x"], ["a"], kind="shift") == ["x"]
    with pytest.raises(ValueError):
        check_known_names(["x"], ["a"], kind="shift", error=True)


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


def test_recipe_spec_typo_and_swapped_bounds_raise() -> None:
    phases = _signal("phase", ["heat"] * 300 + ["hold"] * 300, "value_string")
    df = pd.concat([phases, _wave()], ignore_index=True)
    with pytest.raises(ValueError, match="did you mean 'hold_value'"):
        ts_shape.RecipePhaseAdherenceEvents(
            df, "phase", "temp", {"hold": {"hold_valu": (40, 60)}}
        )
    with pytest.raises(ValueError, match="swapped"):
        ts_shape.RecipePhaseAdherenceEvents(
            df, "phase", "temp", {"hold": {"hold_value": (60, 40)}}
        )
    det = ts_shape.RecipePhaseAdherenceEvents(
        df, "phase", "temp", {"cure": {"hold_value": (40, 60)}}
    )
    with pytest.warns(DataQualityWarning, match="'cure'"):
        det.evaluate()


@pytest.mark.parametrize(
    ("name", "divisor"),
    [("normal", 2.0), ("gaussian", 2.0), ("uniform", np.sqrt(3)), ("U-shaped", None)],
)
def test_type_b_distribution_names(name: str, divisor: float | None) -> None:
    out = ts_shape.MeasurementUncertaintyEvents.type_b_uncertainty(
        [{"source": "cal", "half_width": 1.0, "distribution": name}]
    )
    if divisor is not None:
        assert out["divisor"].iloc[0] == pytest.approx(divisor)


def test_type_b_unknown_distribution_raises() -> None:
    with pytest.raises(ValueError, match="distribution"):
        ts_shape.MeasurementUncertaintyEvents.type_b_uncertainty(
            [{"source": "cal", "half_width": 1.0, "distribution": "lognormal"}]
        )
    with pytest.raises(ValueError, match="half_width"):
        ts_shape.MeasurementUncertaintyEvents.type_b_uncertainty(
            [{"source": "cal", "half_widht": 1.0}]
        )


def test_unknown_shift_names_raise() -> None:
    with pytest.raises(ValueError, match="shift"):
        ts_shape.TargetEvents(_counter()).compare_to_target(
            "parts", {"Frühschicht": 100}
        )
    with pytest.raises(ValueError, match="shift"):
        ts_shape.ShiftReportEvents(_counter()).shift_targets("parts", {"morning": 100})


def test_scrap_cost_table_checked() -> None:
    df = pd.concat(
        [
            _signal("scrap", np.arange(len(T)) // 10.0),
            _signal("part", ["A"] * 300 + ["B"] * 300, "value_string"),
        ],
        ignore_index=True,
    )
    scrap = ts_shape.ScrapEvents(df)
    with pytest.raises(ValueError, match="empty"):
        scrap.scrap_cost("scrap", "part", {})
    with pytest.warns(DataQualityWarning, match="without a cost"):
        out = scrap.scrap_cost("scrap", "part", {"A": 2.0})
    assert out.set_index("part_number").loc["A", "cost_per_unit"] == 2.0


def test_swapped_limits_raise() -> None:
    with pytest.raises(ValueError, match="swapped"):
        ts_shape.SignalQualityEvents(_wave(), "temp").detect_out_of_range(
            min_value=60, max_value=40
        )
    with pytest.raises(ValueError, match="swapped"):
        ts_shape.CapabilityTrendingEvents(_wave(), "temp", upper_spec=40, lower_spec=60)
    limits = pd.concat(
        [
            _wave("actual"),
            _signal("upper", 40.0, times=T[:1]),
            _signal("lower", 60.0, times=T[:1]),
        ],
        ignore_index=True,
    )
    det = ts_shape.ToleranceDeviationEvents(
        limits,
        "value_double",
        "value_double",
        actual_uuid="actual",
        upper_tolerance_uuid="upper",
        lower_tolerance_uuid="lower",
        event_uuid="dev",
    )
    with pytest.raises(ValueError, match="swapped"):
        det.detect_deviations()


def test_threshold_and_failure_prediction_inputs() -> None:
    thr = ts_shape.ThresholdMonitoringEvents(_wave(), "temp")
    with pytest.raises(ValueError, match="direction"):
        thr.multi_level_threshold({"warn": 55}, direction="upwards")
    with pytest.raises(ValueError, match="empty"):
        thr.multi_level_threshold({})
    fp = ts_shape.FailurePredictionEvents(_wave(), "temp")
    with pytest.raises(ValueError, match="degradation_rate"):
        fp.remaining_useful_life(degradation_rate=0, failure_threshold=80)
    with pytest.raises(ValueError, match="swapped"):
        fp.detect_exceedance_pattern(
            warning_threshold=60, critical_threshold=55, direction="rising"
        )


def test_negative_amounts_raise() -> None:
    with pytest.raises(ValueError, match="min_fall"):
        ts_shape.WarmUpCoolDownEvents(_wave(), "temp").detect_cooldown(min_fall=-1)
    run = _signal(
        "run", pd.array(np.arange(len(T)) % 50 < 40, dtype="boolean"), "value_bool"
    )
    rel = ts_shape.ReliabilityEvents(run, "run", failure_states=(False,))
    with pytest.raises(ValueError, match="hours"):
        rel.reliability_at([-5])


def test_compare_periods_checks_order() -> None:
    with pytest.raises(ValueError, match="swapped"):
        ts_shape.PeriodSummaryEvents(_counter()).compare_periods(
            "parts", ("2026-10-06", "2026-10-05"), ("2026-10-05", "2026-10-06")
        )


def test_carbon_factor_checks() -> None:
    df = _signal("kwh", np.ones(len(T)))
    with pytest.raises(ValueError, match=">= 0"):
        ts_shape.CarbonIntensityEvents(df, {"kwh": -0.2})
    with pytest.raises(ValueError, match="scope_map"):
        ts_shape.CarbonIntensityEvents(df, {"kwh": 0.2}, scope_map={"kwh": 3})
    with pytest.warns(DataQualityWarning, match="'missing'"):
        ts_shape.CarbonIntensityEvents(df, {"kwh": 0.2, "missing": 0.1})


def test_carbon_power_reading_is_integrated() -> None:
    t = pd.date_range("2026-10-01", periods=2 * 3600 // 3, freq="3s")
    noise = np.random.default_rng(0).normal(0.0, 1.0, len(t))
    df = _signal("kva", 100.0 + noise - noise.mean(), times=t)
    with pytest.warns(DataQualityWarning, match="power reading"):
        out = ts_shape.CarbonIntensityEvents(df, {"kva": 0.5}).emissions_by_window(
            window="1h"
        )
    # 100 kW for 2 h (one sample interval short) = ~200 kWh -> ~100 kg.
    assert out["kgco2e"].sum() == pytest.approx(100.0, rel=0.01)
    summed = ts_shape.CarbonIntensityEvents(df, {"kva": 0.5}).emissions_by_window(
        window="1h", signal_kind="interval"
    )
    assert summed["kgco2e"].sum() == pytest.approx(len(t) * 50.0, rel=0.01)


def test_carbon_meter_reading_uses_increments() -> None:
    t = pd.date_range("2026-10-01", periods=48, freq="30min")
    df = _signal("meter", 1000.0 + np.arange(48) * 5.0, times=t)
    with pytest.warns(DataQualityWarning, match="cumulative meter"):
        out = ts_shape.CarbonIntensityEvents(df, {"meter": 1.0}).emissions_by_window()
    assert out["kgco2e"].sum() == pytest.approx(47 * 5.0)


def test_spc_collapsed_sigma_warns() -> None:
    rng = np.random.default_rng(1)
    held = np.repeat(rng.normal(50, 2.35, 60), 20)  # a sensor that holds values
    df = pd.concat(
        [
            _signal(
                "ref", held, times=pd.date_range("2026-10-01", periods=1200, freq="1s")
            ),
            _signal(
                "act", held, times=pd.date_range("2026-10-02", periods=1200, freq="1s")
            ),
        ],
        ignore_index=True,
    )
    spc = ts_shape.StatisticalProcessControlEvents(
        df, "value_double", "ref", "act", "spc"
    )
    with pytest.warns(DataQualityWarning, match="moving-range sigma"):
        spc.control_limits()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        spc.detect_rule_violations()
    assert any("beyond 3 sigma" in str(w.message) for w in rec)


def test_design_space_flat_data_is_a_clear_error() -> None:
    df = pd.concat([_wave("a"), _signal("b", np.full(len(T), 3.0))], ignore_index=True)
    ds = ts_shape.DesignSpaceEvents(df, ["a", "b"])
    with pytest.raises(ValueError, match=r"\['b'\] are constant"):
        ds.fit_hull()
    with pytest.warns(DataQualityWarning, match="default box"):
        ds.detect_excursions(df)


def test_environmental_compensation_constant_environment() -> None:
    df = pd.concat(
        [_wave("m"), _signal("env", np.full(len(T), 21.0))], ignore_index=True
    )
    env = ts_shape.EnvironmentalCompensationEvents(df, "m", env_uuid="env")
    with pytest.warns(DataQualityWarning, match="constant"):
        out = env.sensitivity(window="2h")
    assert out.empty
    with pytest.raises(ValueError, match="constant"):
        env.compensated_signal()
    with pytest.raises(TypeError):
        ts_shape.EnvironmentalCompensationEvents(df, "m")  # env_uuid is required


def test_calibration_curve_needs_two_points() -> None:
    cal = ts_shape.CalibrationValidationEvents(_wave(), "temp", reference_value=50.0)
    with pytest.raises(ValueError, match="at least 2"):
        cal.calibration_curve({50.0: [50.1]})


def test_line_config_is_validated() -> None:
    df = pd.concat(
        [_signal("speed", np.full(len(T), 10.0)), _wave()], ignore_index=True
    )
    make = ts_shape.ContinuousProcessAlignmentEvents
    with pytest.raises(ValueError, match="did you mean 'uuids'"):
        make(df, "speed", [{"name": "x", "offset": 0, "uuid": ["temp"]}])
    with pytest.raises(ValueError, match="needs"):
        make(df, "speed", [{"name": "x", "uuids": ["temp"]}])
    with pytest.raises(ValueError, match="list"):
        make(df, "speed", [{"name": "x", "offset": 0, "uuids": "temp"}])
    with pytest.raises(ValueError, match="both"):
        make(
            df,
            "speed",
            [
                {"name": "x", "offset": 0, "uuids": ["temp"]},
                {"name": "y", "offset": 5, "uuids": ["temp"]},
            ],
        )


def test_blank_order_signal_warns() -> None:
    df = pd.concat(
        [_counter(), _signal("order", [""] * len(T), "value_string")], ignore_index=True
    )
    with pytest.warns(DataQualityWarning, match="only empty text"):
        ts_shape.PartProductionEvents(df).production_by_part("order", "parts")


def test_signal_specs_override_unknown_uuid() -> None:
    specs = ts_shape.infer_specs(_wave())
    with pytest.warns(DataQualityWarning, match="did you mean 'temp'"):
        specs.override({"tmep": {"unit": "degC"}})
    with warnings.catch_warnings():
        warnings.simplefilter("error", DataQualityWarning)
        assert "new" in specs.override({"new": {"unit": "degC"}}, add=True)


def test_metadata_duplicate_label() -> None:
    meta = ts_shape.MetadataJsonLoader(
        [
            {"uuid": "a", "label": "Temp", "config": {}},
            {"uuid": "b", "label": "Temp", "config": {}},
            {"uuid": "c", "label": "Speed", "config": {}},
        ]
    )
    with pytest.raises(ValueError, match="2 datapoints"):
        meta.get_by_label("Temp")
    assert [r["uuid"] for r in meta.get_all_by_label("Temp")] == ["a", "b"]
    assert meta.duplicate_labels()["label"].tolist() == ["Temp"]
    assert meta.get_by_label("Speed") is not None


def test_run_chunked_swapped_arguments() -> None:
    with pytest.raises(TypeError, match="swapped"):
        ts_shape.run_chunked(ts_shape.MachineStateEvents, _wave(), "detect_run_idle")


def test_cross_signal_accepts_long_format() -> None:
    df = pd.concat([_wave("a"), _wave("b")], ignore_index=True)
    assert ts_shape.CrossSignalAnalytics(df).signals == ["a", "b"]
