"""Tests for UnitConverter (pint-backed engineering unit conversion)."""

import pandas as pd  # type: ignore
import pytest

import ts_shape
from ts_shape.transform.calculator.unit_conversion import (
    PINT_AVAILABLE,
    UnitConverter,
)

pytestmark = pytest.mark.skipif(
    not PINT_AVAILABLE, reason="pint not installed (optional [units] extra)"
)


def test_temperature_roundtrip():
    assert UnitConverter.convert_value(100, "C", "F") == pytest.approx(212.0)
    assert UnitConverter.convert_value(212, "F", "C") == pytest.approx(100.0)
    assert UnitConverter.convert_value(0, "C", "K") == pytest.approx(273.15)


def test_pressure_and_flow_conversions():
    assert UnitConverter.convert_value(1, "bar", "psi") == pytest.approx(
        14.5037, rel=1e-4
    )
    assert UnitConverter.convert_value(1, "m^3/hour", "L/hour") == pytest.approx(1000.0)


def test_conversion_factor_ratio_and_affine():
    scale, offset = UnitConverter.conversion_factor("bar", "psi")
    assert scale == pytest.approx(14.5037, rel=1e-4)
    assert offset == pytest.approx(0.0)
    # Temperature is affine: a non-zero offset.
    scale, offset = UnitConverter.conversion_factor("C", "F")
    assert scale == pytest.approx(1.8)
    assert offset == pytest.approx(32.0)


def test_convert_column_in_place_on_copy():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2025-01-01", periods=3, freq="h"),
            "uuid": "p",
            "value_double": [1.0, 2.0, 3.0],
        }
    )
    out = UnitConverter.convert_column(df, "bar", "psi", column_name="value_double")
    assert out["value_double"].iloc[0] == pytest.approx(14.5037, rel=1e-4)
    # The input DataFrame is not mutated.
    assert df["value_double"].iloc[0] == 1.0


def test_convert_column_to_target_column():
    df = pd.DataFrame(
        {
            "systime": pd.date_range("2025-01-01", periods=2, freq="h"),
            "uuid": "p",
            "value_double": [10.0, 20.0],
        }
    )
    out = UnitConverter.convert_column(
        df, "C", "F", column_name="value_double", target_column="value_fahrenheit"
    )
    assert "value_fahrenheit" in out.columns
    assert out["value_fahrenheit"].iloc[0] == pytest.approx(50.0)


def test_incompatible_units_raise_value_error():
    with pytest.raises(ValueError):
        UnitConverter.convert_value(1, "bar", "meter")


def test_unknown_unit_raises_value_error():
    with pytest.raises(ValueError):
        UnitConverter.convert_value(1, "not_a_unit", "psi")


def test_missing_column_raises():
    df = pd.DataFrame({"value_double": [1.0]})
    with pytest.raises(ValueError):
        UnitConverter.convert_column(df, "bar", "psi", column_name="missing")


def test_exported_at_top_level():
    assert ts_shape.UnitConverter is UnitConverter
