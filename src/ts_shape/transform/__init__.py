"""Transform

Row-level filtering, column transformations, and timestamp utilities.

Classes:
- IntegerFilter: Equality/inequality/range filters for integer columns.
  - filter_value_integer_match: Select rows equal to a value.
  - filter_value_integer_not_match: Select rows not equal to a value.
  - filter_value_integer_between: Select rows within [min, max].

- DoubleFilter: NaN removal and numeric ranges for floating-point columns.
  - filter_nan_value_double: Drop rows with NaN in the column.
  - filter_value_double_between: Select rows within [min, max].

- StringFilter: Equality, contains, regex cleaning, and change detection.
  - filter_na_value_string: Drop rows with NA values.
  - filter_value_string_match: Select rows equal to a string.
  - filter_value_string_not_match: Select rows not equal to a string.
  - filter_string_contains: Select rows containing a substring.
  - regex_clean_value_string: Regex-based cleaning or replacement.
  - detect_changes_in_string: Detect row-to-row changes in a string column.

- BooleanFilter: Detect raising/falling edges of boolean states.
  - filter_falling_value_bool: True→False transitions.
  - filter_raising_value_bool: False→True transitions.

- IsDeltaFilter: Select rows by the is_delta flag.
  - filter_is_delta_true: Only True.
  - filter_is_delta_false: Only False.

- DateTimeFilter: Before/after/between filters for timestamps.
  - filter_after_date: After a given date.
  - filter_before_date: Before a given date.
  - filter_between_dates: Between start and end dates.
  - filter_after_datetime: After a given datetime.
  - filter_before_datetime: Before a given datetime.
  - filter_between_datetimes: Between start and end datetimes.

- CustomFilter: Free-form DataFrame.query string conditions.
  - filter_custom_conditions: Apply a query string to filter rows.

- IntegerCalc: Numeric column calculations.
  - scale_column: Multiply by a factor.
  - offset_column: Add a constant.
  - divide_column: Divide by a constant.
  - subtract_column: Subtract a constant.
  - calculate_with_fixed_factors: Multiply then add.
  - mod_column: Modulo operation.
  - power_column: Raise to a power.

- PowerIntegrator: Integrate an instantaneous power signal (kW) into energy (kWh).
  - energy_by_window: Time-weighted kWh and mean kW per window.
  - to_energy_deltas: Emit kWh deltas in the standard ts-shape schema.
  - segment_energy: Per-sample energy contributions (audit trail).

- UnitConverter: Convert engineering units in scalars and DataFrame columns via
  ``pint`` (the ``units`` extra).

- LambdaProcessor: Apply vectorized callables to columns.
  - apply_function: Apply a Python callable over a column's values.

- TimestampConverter: Convert integer timestamps to tz-aware datetimes.
  - convert_to_datetime: Convert s/ms/us/ns to datetime in a timezone.

- TimezoneShift: Timezone localization/conversion helpers.
  - shift_timezone: Convert timezones in-place.
  - add_timezone_column: Add a converted timestamp column.
  - detect_timezone_awareness: Check tz-awareness of a column.
  - revert_to_original_timezone: Convert back to original tz.
  - calculate_time_difference: Difference between two timestamp columns.

- DataHarmonizer: Multi-signal data harmonization.
  - pivot_to_wide: Pivot long-format to wide-format (one column per UUID).
  - resample_to_uniform: Resample to uniform time grid with interpolation.
  - detect_gaps: Identify time gaps per UUID exceeding a threshold.
  - fill_gaps: Fill gaps using interpolation, forward/back fill, or constant.
  - align_asof: Align two UUID signals using merge_asof.
  - merge_multi_signals: End-to-end harmonization pipeline.
"""

import importlib as _importlib
from typing import TYPE_CHECKING

# Explicit imports for type checkers and IDEs only (so ``from ts_shape import X``
# is typed as X, not object); at runtime __getattr__ below imports lazily.
# tests/typing/test_lazy_exports.py keeps this block in sync with the map.
if TYPE_CHECKING:
    from ts_shape.transform.calculator.numeric_calc import IntegerCalc as IntegerCalc
    from ts_shape.transform.calculator.power_integration import (
        PowerIntegrator as PowerIntegrator,
    )
    from ts_shape.transform.calculator.unit_conversion import (
        UnitConverter as UnitConverter,
    )
    from ts_shape.transform.filter.boolean_filter import BooleanFilter as BooleanFilter
    from ts_shape.transform.filter.boolean_filter import IsDeltaFilter as IsDeltaFilter
    from ts_shape.transform.filter.custom_filter import CustomFilter as CustomFilter
    from ts_shape.transform.filter.datetime_filter import (
        DateTimeFilter as DateTimeFilter,
    )
    from ts_shape.transform.filter.numeric_filter import DoubleFilter as DoubleFilter
    from ts_shape.transform.filter.numeric_filter import IntegerFilter as IntegerFilter
    from ts_shape.transform.filter.string_filter import StringFilter as StringFilter
    from ts_shape.transform.functions.lambda_func import (
        LambdaProcessor as LambdaProcessor,
    )
    from ts_shape.transform.harmonization import DataHarmonizer as DataHarmonizer
    from ts_shape.transform.time_functions.timestamp_converter import (
        TimestampConverter as TimestampConverter,
    )
    from ts_shape.transform.time_functions.timezone_shift import (
        TimezoneShift as TimezoneShift,
    )

# The classes listed above, importable from the pack. They are imported on
# first access, so a class whose optional dependency is missing only fails
# when it is used.
_CLASSES = {
    "BooleanFilter": "ts_shape.transform.filter.boolean_filter",
    "CustomFilter": "ts_shape.transform.filter.custom_filter",
    "DataHarmonizer": "ts_shape.transform.harmonization",
    "DateTimeFilter": "ts_shape.transform.filter.datetime_filter",
    "DoubleFilter": "ts_shape.transform.filter.numeric_filter",
    "IntegerCalc": "ts_shape.transform.calculator.numeric_calc",
    "IntegerFilter": "ts_shape.transform.filter.numeric_filter",
    "IsDeltaFilter": "ts_shape.transform.filter.boolean_filter",
    "LambdaProcessor": "ts_shape.transform.functions.lambda_func",
    "PowerIntegrator": "ts_shape.transform.calculator.power_integration",
    "StringFilter": "ts_shape.transform.filter.string_filter",
    "TimestampConverter": "ts_shape.transform.time_functions.timestamp_converter",
    "TimezoneShift": "ts_shape.transform.time_functions.timezone_shift",
    "UnitConverter": "ts_shape.transform.calculator.unit_conversion",
}

__all__ = sorted(_CLASSES)


def __getattr__(name: str) -> object:
    if name in _CLASSES:
        return getattr(_importlib.import_module(_CLASSES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CLASSES))
