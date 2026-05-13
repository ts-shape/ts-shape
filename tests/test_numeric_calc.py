import pandas as pd  # type: ignore
from ts_shape.transform.calculator.numeric_calc import IntegerCalc


def _make_df():
    return pd.DataFrame(
        {"value_integer": [1, 2, 3], "systime": pd.date_range("2023-01-01", periods=3)}
    )


def test_scale_and_offset_and_divide_and_subtract():
    df = _make_df()
    scaled = IntegerCalc.scale_column(df.copy(), factor=2)
    assert scaled["value_integer"].tolist() == [2, 4, 6]

    offset = IntegerCalc.offset_column(df.copy(), offset_value=5)
    assert offset["value_integer"].tolist() == [6, 7, 8]

    divided = IntegerCalc.divide_column(df.copy(), divisor=2)
    assert divided["value_integer"].tolist() == [0.5, 1.0, 1.5]

    sub = IntegerCalc.subtract_column(df.copy(), subtract_value=1)
    assert sub["value_integer"].tolist() == [0, 1, 2]


def test_fixed_factors_mod_power():
    df = _make_df()
    calc = IntegerCalc.calculate_with_fixed_factors(
        df.copy(), multiply_factor=3, add_factor=1
    )
    assert calc["value_integer"].tolist() == [4, 7, 10]

    modded = IntegerCalc.mod_column(df.copy(), mod_value=2)
    assert modded["value_integer"].tolist() == [1, 0, 1]

    powered = IntegerCalc.power_column(df.copy(), power_value=2)
    assert powered["value_integer"].tolist() == [1, 4, 9]
