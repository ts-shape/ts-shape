"""
Transform Operations Demo for ts-shape
========================================

Demonstrates all transform classes with practical examples:

Filters:
  - IntegerFilter / DoubleFilter   (numeric_filter)
  - StringFilter                   (string_filter)
  - BooleanFilter / IsDeltaFilter  (boolean_filter)
  - DateTimeFilter                 (datetime_filter)

Calculators:
  - IntegerCalc                    (numeric_calc)

Functions:
  - LambdaProcessor                (lambda_func)

Time functions:
  - TimezoneShift                  (timezone_shift)
  - TimestampConverter             (timestamp_converter)

Scenario: An industrial IoT gateway collects heterogeneous sensor data from a
packaging line.  Each row carries one of several uuid-tagged signals.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.transform.filter.boolean_filter import BooleanFilter, IsDeltaFilter
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.filter.numeric_filter import DoubleFilter, IntegerFilter
from ts_shape.transform.filter.string_filter import StringFilter
from ts_shape.transform.functions.lambda_func import LambdaProcessor
from ts_shape.transform.time_functions.timestamp_converter import TimestampConverter
from ts_shape.transform.time_functions.timezone_shift import TimezoneShift

# ---------------------------------------------------------------------------
# Helper: build a mixed-signal packaging-line DataFrame
# ---------------------------------------------------------------------------


def create_packaging_line_data(n_rows: int = 200) -> pd.DataFrame:
    """
    Simulate heterogeneous IoT data from a packaging line.

    Signals:
      - 'conveyor_speed'    : integer RPM values (value_integer)
      - 'fill_weight'       : double weight in grams (value_double)
      - 'product_label'     : string label identifier (value_string)
      - 'lid_sealed'        : boolean seal status (value_bool)
    """
    np.random.seed(12)
    start = datetime(2025, 7, 1, 8, 0, 0)

    rows = []
    for i in range(n_rows):
        t = start + timedelta(seconds=15 * i)

        # Conveyor speed (integer RPM, mostly 60, occasional 0 for stops)
        rpm = 60 if np.random.random() > 0.05 else 0
        rows.append(
            {
                "systime": t,
                "uuid": "conveyor_speed",
                "value_bool": None,
                "value_integer": rpm,
                "value_double": None,
                "value_string": None,
                "is_delta": True,
            }
        )

        # Fill weight (double, target 500 g)
        weight = 500.0 + np.random.normal(0, 2.5)
        if i in (50, 120, 180):
            weight = np.nan  # simulate missing readings
        rows.append(
            {
                "systime": t + timedelta(seconds=1),
                "uuid": "fill_weight",
                "value_bool": None,
                "value_integer": None,
                "value_double": round(weight, 3),
                "value_string": None,
                "is_delta": True,
            }
        )

        # Product label string
        labels = ["SKU-A100", "SKU-B200", "SKU-C300", "SKU-A100", "SKU-A100"]
        label = np.random.choice(labels)
        rows.append(
            {
                "systime": t + timedelta(seconds=2),
                "uuid": "product_label",
                "value_bool": None,
                "value_integer": None,
                "value_double": None,
                "value_string": label,
                "is_delta": True,
            }
        )

        # Lid sealed boolean (True most of the time, occasional False)
        sealed = True if np.random.random() > 0.08 else False
        rows.append(
            {
                "systime": t + timedelta(seconds=3),
                "uuid": "lid_sealed",
                "value_bool": sealed,
                "value_integer": None,
                "value_double": None,
                "value_string": None,
                "is_delta": bool(i % 2 == 0),  # alternate is_delta for demo
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------


def demo_numeric_filters(df: pd.DataFrame):
    """Filter numeric columns (integer and double)."""
    print("=" * 72)
    print("1. NUMERIC FILTERS")
    print("=" * 72)

    # --- Integer filters ---
    speed_df = df[df["uuid"] == "conveyor_speed"].copy()
    print(f"\nConveyor speed rows: {len(speed_df)}")

    # Match a specific RPM value
    stopped = IntegerFilter.filter_value_integer_match(
        speed_df, value_column="value_integer", integer_value=0
    )
    print(f"  Conveyor stopped (RPM=0): {len(stopped)} rows")

    running = IntegerFilter.filter_value_integer_not_match(
        speed_df, value_column="value_integer", integer_value=0
    )
    print(f"  Conveyor running (RPM!=0): {len(running)} rows")

    # Range filter
    in_range = IntegerFilter.filter_value_integer_between(
        speed_df, value_column="value_integer", min_value=50, max_value=70
    )
    print(f"  RPM between 50 and 70: {len(in_range)} rows")

    # --- Double filters ---
    weight_df = df[df["uuid"] == "fill_weight"].copy()
    print(f"\nFill weight rows: {len(weight_df)}")

    # Remove NaN weights
    valid_weights = DoubleFilter.filter_nan_value_double(
        weight_df, value_column="value_double"
    )
    print(f"  Valid (non-NaN) weights: {len(valid_weights)} rows")

    # Filter weights within an acceptable range
    in_spec = DoubleFilter.filter_value_double_between(
        valid_weights, value_column="value_double", min_value=495.0, max_value=505.0
    )
    print(f"  Weights in spec (495-505 g): {len(in_spec)} rows")
    print()


def demo_string_filters(df: pd.DataFrame):
    """Filter and inspect string columns."""
    print("=" * 72)
    print("2. STRING FILTERS")
    print("=" * 72)

    label_df = df[df["uuid"] == "product_label"].copy()
    print(f"\nProduct label rows: {len(label_df)}")

    # Remove NAs
    clean = StringFilter.filter_na_value_string(label_df, value_column="value_string")
    print(f"  Non-null labels: {len(clean)}")

    # Exact match
    sku_a = StringFilter.filter_value_string_match(
        clean, string_value="SKU-A100", value_column="value_string"
    )
    print(f"  SKU-A100 rows: {len(sku_a)}")

    # Not match
    not_a = StringFilter.filter_value_string_not_match(
        clean, string_value="SKU-A100", value_column="value_string"
    )
    print(f"  Not SKU-A100: {len(not_a)}")

    # Substring contains
    b_products = StringFilter.filter_string_contains(
        clean, substring="B200", value_column="value_string"
    )
    print(f"  Labels containing 'B200': {len(b_products)}")

    # Detect label changes (e.g., product changeover)
    changes = StringFilter.detect_changes_in_string(clean, value_column="value_string")
    print(f"  Label change events: {len(changes)}")

    # Regex clean: strip the alphabetic prefix to keep only the numeric part
    cleaned = StringFilter.regex_clean_value_string(
        clean.copy(),
        value_column="value_string",
        regex_pattern=r"SKU-[A-Z]",
        replacement="ITEM-",
    )
    print(
        f"  After regex replace (first 5): {cleaned['value_string'].head(5).tolist()}"
    )
    print()


def demo_boolean_filters(df: pd.DataFrame):
    """Filter boolean and is_delta columns."""
    print("=" * 72)
    print("3. BOOLEAN / IS_DELTA FILTERS")
    print("=" * 72)

    seal_df = df[df["uuid"] == "lid_sealed"].copy().reset_index(drop=True)
    print(f"\nLid sealed rows: {len(seal_df)}")

    # is_delta filters
    delta_true = IsDeltaFilter.filter_is_delta_true(seal_df)
    delta_false = IsDeltaFilter.filter_is_delta_false(seal_df)
    print(f"  is_delta=True:  {len(delta_true)} rows")
    print(f"  is_delta=False: {len(delta_false)} rows")

    # Boolean edge detection: rising (False -> True) and falling (True -> False)
    falling = BooleanFilter.filter_falling_value_bool(
        seal_df.copy(), value_column="value_bool"
    )
    print(f"  Falling edges (seal lost): {len(falling)} rows")

    raising = BooleanFilter.filter_raising_value_bool(
        seal_df.copy(), value_column="value_bool"
    )
    print(f"  Rising edges (seal restored): {len(raising)} rows")
    print()


def demo_datetime_filters(df: pd.DataFrame):
    """Filter rows by datetime ranges."""
    print("=" * 72)
    print("4. DATETIME FILTERS")
    print("=" * 72)

    # Ensure systime is datetime
    df = df.copy()
    df["systime"] = pd.to_datetime(df["systime"])
    print(f"\nTotal rows: {len(df)}")
    print(f"Time range: {df['systime'].min()} to {df['systime'].max()}")

    # Filter rows after a specific date/time
    after = DateTimeFilter.filter_after_datetime(
        df, time_column="systime", datetime="2025-07-01 08:30:00"
    )
    print(f"  After 08:30:00: {len(after)} rows")

    # Filter rows before a specific date/time
    before = DateTimeFilter.filter_before_datetime(
        df, time_column="systime", datetime="2025-07-01 08:15:00"
    )
    print(f"  Before 08:15:00: {len(before)} rows")

    # Filter between two datetimes
    between = DateTimeFilter.filter_between_datetimes(
        df,
        time_column="systime",
        start_datetime="2025-07-01 08:10:00",
        end_datetime="2025-07-01 08:20:00",
    )
    print(f"  Between 08:10 and 08:20: {len(between)} rows")

    # Date-level filters
    after_date = DateTimeFilter.filter_after_date(
        df, time_column="systime", date="2025-06-30"
    )
    print(f"  After 2025-06-30: {len(after_date)} rows (should be all)")

    before_date = DateTimeFilter.filter_before_date(
        df, time_column="systime", date="2025-07-02"
    )
    print(f"  Before 2025-07-02: {len(before_date)} rows (should be all)")

    between_dates = DateTimeFilter.filter_between_dates(
        df, time_column="systime", start_date="2025-06-30", end_date="2025-07-02"
    )
    print(f"  Between 2025-06-30 and 2025-07-02: {len(between_dates)} rows")
    print()


def demo_numeric_calc(df: pd.DataFrame):
    """Demonstrate numeric calculations on integer columns."""
    print("=" * 72)
    print("5. NUMERIC CALCULATIONS (IntegerCalc)")
    print("=" * 72)

    speed_df = df[df["uuid"] == "conveyor_speed"][
        ["systime", "uuid", "value_integer"]
    ].copy()
    original_sum = speed_df["value_integer"].sum()
    print(f"\nOriginal RPM sum: {original_sum}")

    # Scale: convert RPM to radians/sec (multiply by 2*pi/60)
    scaled = IntegerCalc.scale_column(
        speed_df.copy(), value_column="value_integer", factor=2 * np.pi / 60
    )
    print(
        f"  After scale (RPM -> rad/s), first 3 values: {[round(v, 4) for v in scaled['value_integer'].head(3).tolist()]}"
    )

    # Offset: add a calibration offset of +2 RPM
    offset = IntegerCalc.offset_column(
        df[df["uuid"] == "conveyor_speed"][["systime", "uuid", "value_integer"]].copy(),
        value_column="value_integer",
        offset_value=2,
    )
    print(
        f"  After offset +2, first 3 values: {offset['value_integer'].head(3).tolist()}"
    )

    # Divide: convert RPM to RPS
    divided = IntegerCalc.divide_column(
        df[df["uuid"] == "conveyor_speed"][["systime", "uuid", "value_integer"]].copy(),
        value_column="value_integer",
        divisor=60,
    )
    print(
        f"  After divide by 60 (RPM -> RPS), first 3 values: {divided['value_integer'].head(3).tolist()}"
    )

    # Combined multiply + add (linear transform: y = mx + b)
    combined = IntegerCalc.calculate_with_fixed_factors(
        df[df["uuid"] == "conveyor_speed"][["systime", "uuid", "value_integer"]].copy(),
        value_column="value_integer",
        multiply_factor=1.05,
        add_factor=-3,
    )
    print(
        f"  After y=1.05x-3, first 3 values: {combined['value_integer'].head(3).tolist()}"
    )

    # Power: square the values
    powered = IntegerCalc.power_column(
        df[df["uuid"] == "conveyor_speed"][["systime", "uuid", "value_integer"]].copy(),
        value_column="value_integer",
        power_value=2,
    )
    print(
        f"  After power(2), first 3 values: {powered['value_integer'].head(3).tolist()}"
    )

    # Modulus
    modded = IntegerCalc.mod_column(
        df[df["uuid"] == "conveyor_speed"][["systime", "uuid", "value_integer"]].copy(),
        value_column="value_integer",
        mod_value=7,
    )
    print(f"  After mod 7, first 3 values: {modded['value_integer'].head(3).tolist()}")
    print()


def demo_lambda_processor(df: pd.DataFrame):
    """Demonstrate custom lambda/function application."""
    print("=" * 72)
    print("6. LAMBDA PROCESSOR")
    print("=" * 72)

    weight_df = df[df["uuid"] == "fill_weight"][
        ["systime", "uuid", "value_double"]
    ].copy()
    weight_df = weight_df.dropna(subset=["value_double"])
    print(f"\nFill weight rows (non-null): {len(weight_df)}")

    # Apply a lambda to convert grams to kilograms
    kg_df = LambdaProcessor.apply_function(
        weight_df.copy(),
        value_column="value_double",
        func=lambda x: x / 1000.0,
    )
    print(
        f"  After g -> kg, first 3: {[round(v, 5) for v in kg_df['value_double'].head(3).tolist()]}"
    )

    # Apply numpy log transform
    log_df = LambdaProcessor.apply_function(
        weight_df.copy(),
        value_column="value_double",
        func=lambda x: np.log(x),
    )
    print(
        f"  After log transform, first 3: {[round(v, 4) for v in log_df['value_double'].head(3).tolist()]}"
    )

    # Apply a clipping function
    clipped = LambdaProcessor.apply_function(
        weight_df.copy(),
        value_column="value_double",
        func=lambda x: np.clip(x, 497.0, 503.0),
    )
    print(
        f"  After clip [497, 503], min={clipped['value_double'].min():.1f}, max={clipped['value_double'].max():.1f}"
    )
    print()


def demo_timezone_shift(df: pd.DataFrame):
    """Demonstrate timezone conversion and related utilities."""
    print("=" * 72)
    print("7. TIMEZONE SHIFT")
    print("=" * 72)

    # Work with a small slice
    small = df.head(12).copy()
    small["systime"] = pd.to_datetime(small["systime"])
    print(
        f"\nOriginal timezone-aware? {TimezoneShift.detect_timezone_awareness(small, 'systime')}"
    )

    # Shift from UTC to US Eastern
    shifted = TimezoneShift.shift_timezone(
        small.copy(),
        time_column="systime",
        input_timezone="UTC",
        target_timezone="America/New_York",
    )
    print("  After shift to America/New_York:")
    print(f"    First timestamp: {shifted['systime'].iloc[0]}")
    print(
        f"    Timezone-aware?  {TimezoneShift.detect_timezone_awareness(shifted, 'systime')}"
    )

    # Add a second timezone column without modifying the original
    dual = TimezoneShift.add_timezone_column(
        small.copy(),
        time_column="systime",
        input_timezone="UTC",
        target_timezone="Europe/Berlin",
    )
    tz_cols = [c for c in dual.columns if "Europe" in c]
    if tz_cols:
        print(f"  Added column: {tz_cols[0]}")
        print(f"    Original: {dual['systime'].iloc[0]}")
        print(f"    Berlin:   {dual[tz_cols[0]].iloc[0]}")

    # Revert to UTC
    reverted = TimezoneShift.revert_to_original_timezone(
        shifted.copy(), "systime", "UTC"
    )
    print(f"  Reverted to UTC: {reverted['systime'].iloc[0]}")

    # Calculate time difference between two columns
    # Both columns must carry a time zone: systime is UTC, made explicit here.
    dual2 = dual.copy()
    dual2["systime"] = dual2["systime"].dt.tz_localize("UTC")
    if tz_cols:
        diff_seconds = TimezoneShift.calculate_time_difference(
            dual2, "systime", tz_cols[0]
        )
        print(
            f"  Same instants in UTC and Berlin, difference (s): {diff_seconds.head(3).tolist()}"
        )
    print()


def demo_timestamp_converter():
    """Demonstrate high-precision timestamp conversion."""
    print("=" * 72)
    print("8. TIMESTAMP CONVERTER")
    print("=" * 72)

    # Create a small DataFrame with epoch nanosecond timestamps
    epoch_ns = [
        1751356800000000000,  # 2025-07-01 00:00:00 UTC
        1751356860000000000,  # +60 seconds
        1751356920000000000,  # +120 seconds
    ]
    ts_df = pd.DataFrame(
        {
            "systime": epoch_ns,
            "uuid": "test_sensor",
            "value_double": [1.0, 2.0, 3.0],
        }
    )
    print(f"\nRaw epoch nanoseconds (first): {ts_df['systime'].iloc[0]}")

    # Convert from nanoseconds to datetime in UTC
    converted_utc = TimestampConverter.convert_to_datetime(
        ts_df.copy(), columns=["systime"], unit="ns", timezone="UTC"
    )
    print(f"  After ns -> UTC datetime: {converted_utc['systime'].iloc[0]}")

    # Convert from nanoseconds directly to America/Los_Angeles
    converted_pac = TimestampConverter.convert_to_datetime(
        ts_df.copy(), columns=["systime"], unit="ns", timezone="America/Los_Angeles"
    )
    print(f"  After ns -> America/Los_Angeles:   {converted_pac['systime'].iloc[0]}")

    # Demonstrate seconds-based conversion
    epoch_s = [ts / 1e9 for ts in epoch_ns]
    ts_df_s = pd.DataFrame(
        {
            "systime": epoch_s,
            "uuid": "test_sensor",
            "value_double": [1.0, 2.0, 3.0],
        }
    )
    converted_s = TimestampConverter.convert_to_datetime(
        ts_df_s.copy(), columns=["systime"], unit="s", timezone="Europe/London"
    )
    print(f"  After s -> Europe/London: {converted_s['systime'].iloc[0]}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = create_packaging_line_data()
    print(f"Generated {len(df)} rows of packaging-line IoT data\n")

    demo_numeric_filters(df)
    demo_string_filters(df)
    demo_boolean_filters(df)
    demo_datetime_filters(df)
    demo_numeric_calc(df)
    demo_lambda_processor(df)
    demo_timezone_shift(df)
    demo_timestamp_converter()

    print("=" * 72)
    print("Transform operations demo complete.")
    print("=" * 72)
