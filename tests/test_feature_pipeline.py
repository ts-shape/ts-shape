import pytest
import pandas as pd
import numpy as np
from ts_shape.features.segment_analysis.feature_pipeline import FeaturePipeline
from ts_shape.transform.filter.numeric_filter import DoubleFilter
from ts_shape.transform.filter.datetime_filter import DateTimeFilter
from ts_shape.transform.calculator.numeric_calc import IntegerCalc
from ts_shape.transform.harmonization import DataHarmonizer
from ts_shape.features.segment_analysis.segment_extractor import SegmentExtractor
from ts_shape.features.segment_analysis.segment_processor import SegmentProcessor
from ts_shape.features.segment_analysis.time_windowed_features import (
    TimeWindowedFeatureTable,
)


@pytest.fixture
def production_df():
    """Production data: order signal + 3 process parameters over 5 minutes."""
    np.random.seed(42)
    n = 300
    times = pd.date_range("2024-01-01", periods=n, freq="1s")

    frames = []

    orders = ["Order-A"] * 100 + ["Order-B"] * 100 + ["Order-A"] * 100
    frames.append(
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "order_number",
                "value_string": orders,
                "value_double": np.nan,
            }
        )
    )

    temp = np.concatenate(
        [
            50 + np.random.randn(100) * 2,
            80 + np.random.randn(100) * 2,
            50 + np.random.randn(100) * 2,
        ]
    )
    frames.append(
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "temperature",
                "value_string": "",
                "value_double": temp,
            }
        )
    )

    pressure = np.concatenate(
        [
            100 + np.random.randn(100) * 5,
            200 + np.random.randn(100) * 5,
            100 + np.random.randn(100) * 5,
        ]
    )
    frames.append(
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "pressure",
                "value_string": "",
                "value_double": pressure,
            }
        )
    )

    speed = 1000 + np.random.randn(n) * 1
    frames.append(
        pd.DataFrame(
            {
                "systime": times,
                "uuid": "speed",
                "value_string": "",
                "value_double": speed,
            }
        )
    )

    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------
# Registration validation
# ------------------------------------------------------------------


class TestRegistrationValidation:
    def test_invalid_dataframe_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            FeaturePipeline("not a dataframe")

    def test_invalid_dataframe_shows_actual_type(self):
        with pytest.raises(TypeError, match="got str"):
            FeaturePipeline("not a dataframe")

    def test_add_step_non_callable_raises(self, production_df):
        with pytest.raises(TypeError, match="callable"):
            FeaturePipeline(production_df).add_step("not_a_function")

    def test_add_lambda_step_non_callable_raises(self, production_df):
        with pytest.raises(TypeError, match="callable"):
            FeaturePipeline(production_df).add_lambda_step(42)

    def test_add_instance_step_non_class_raises(self, production_df):
        with pytest.raises(TypeError, match="must be a class"):
            FeaturePipeline(production_df).add_instance_step(
                "not_a_class", call="method"
            )

    def test_add_instance_step_non_class_hints_add_step(self, production_df):
        with pytest.raises(TypeError, match="add_step"):
            FeaturePipeline(production_df).add_instance_step(
                DoubleFilter.filter_nan_value_double, call="method"
            )

    def test_add_instance_step_missing_method_raises(self, production_df):
        with pytest.raises(AttributeError, match="no method 'nonexistent'"):
            FeaturePipeline(production_df).add_instance_step(
                DataHarmonizer, call="nonexistent"
            )

    def test_add_instance_step_missing_method_lists_available(self, production_df):
        with pytest.raises(AttributeError, match="Available methods"):
            FeaturePipeline(production_df).add_instance_step(
                DataHarmonizer, call="nonexistent"
            )


class TestSentinelValidation:
    def test_invalid_sentinel_in_add_step_raises(self, production_df):
        with pytest.raises(ValueError, match="Unknown sentinel"):
            FeaturePipeline(production_df).add_step(
                DoubleFilter.filter_nan_value_double,
                column_name="$PREV",
            )

    def test_invalid_sentinel_case_sensitive(self, production_df):
        with pytest.raises(ValueError, match="case-sensitive"):
            FeaturePipeline(production_df).add_step(
                DoubleFilter.filter_nan_value_double,
                column_name="$Prev",
            )

    def test_invalid_sentinel_in_instance_step_raises(self, production_df):
        with pytest.raises(ValueError, match="Unknown sentinel"):
            FeaturePipeline(production_df).add_instance_step(
                DataHarmonizer,
                call="pivot_to_wide",
                aggfunc="$foo",
            )

    def test_valid_sentinel_prev_accepted(self, production_df):
        # Should not raise
        FeaturePipeline(production_df).add_step(
            SegmentProcessor.apply_ranges,
            time_ranges="$prev",
        )

    def test_valid_sentinel_input_accepted(self, production_df):
        # Should not raise
        FeaturePipeline(production_df).add_step(
            SegmentProcessor.apply_ranges,
            dataframe="$input",
        )

    def test_non_sentinel_dollar_string_rejected(self, production_df):
        with pytest.raises(ValueError, match="Unknown sentinel"):
            FeaturePipeline(production_df).add_step(
                DoubleFilter.filter_nan_value_double,
                column_name="$typo",
            )

    def test_regular_strings_not_affected(self, production_df):
        # Strings that don't start with $ are fine
        pipe = FeaturePipeline(production_df).add_step(
            DoubleFilter.filter_nan_value_double,
            column_name="value_double",
        )
        assert len(pipe.steps) == 1


class TestInstanceMethodDetection:
    def test_instance_method_in_add_step_raises(self, production_df):
        """Passing DataHarmonizer.pivot_to_wide to add_step should error."""
        with pytest.raises(TypeError, match="add_instance_step"):
            FeaturePipeline(production_df).add_step(DataHarmonizer.pivot_to_wide)

    def test_instance_method_error_includes_class_name(self, production_df):
        with pytest.raises(TypeError, match="DataHarmonizer"):
            FeaturePipeline(production_df).add_step(DataHarmonizer.resample_to_uniform)

    def test_classmethod_in_add_step_accepted(self, production_df):
        # Should not raise — DoubleFilter methods are classmethods
        FeaturePipeline(production_df).add_step(DoubleFilter.filter_nan_value_double)


# ------------------------------------------------------------------
# Return type validation
# ------------------------------------------------------------------


class TestReturnTypeValidation:
    def test_lambda_returning_none_raises(self, production_df):
        with pytest.raises(RuntimeError, match="returned None"):
            (
                FeaturePipeline(production_df)
                .add_lambda_step(lambda df: None, name="bad_step")
                .run()
            )

    def test_lambda_returning_string_raises(self, production_df):
        with pytest.raises(RuntimeError, match="returned str"):
            (
                FeaturePipeline(production_df)
                .add_lambda_step(lambda df: "oops", name="bad_step")
                .run()
            )

    def test_lambda_returning_scalar_raises(self, production_df):
        with pytest.raises(RuntimeError, match="returned"):
            (
                FeaturePipeline(production_df)
                .add_lambda_step(lambda df: 42, name="bad_step")
                .run()
            )


# ------------------------------------------------------------------
# Error context
# ------------------------------------------------------------------


class TestErrorContext:
    def test_error_includes_step_number(self, production_df):
        with pytest.raises(RuntimeError, match=r"step 2/2"):
            (
                FeaturePipeline(production_df)
                .add_step(DoubleFilter.filter_nan_value_double)
                .add_lambda_step(
                    lambda df: df[["nonexistent_col"]],
                    name="bad_column_select",
                )
                .run()
            )

    def test_error_includes_step_name(self, production_df):
        with pytest.raises(RuntimeError, match="bad_column_select"):
            (
                FeaturePipeline(production_df)
                .add_step(DoubleFilter.filter_nan_value_double)
                .add_lambda_step(
                    lambda df: df[["nonexistent_col"]],
                    name="bad_column_select",
                )
                .run()
            )

    def test_error_includes_dataframe_shape(self, production_df):
        with pytest.raises(RuntimeError, match="rows x"):
            (
                FeaturePipeline(production_df)
                .add_lambda_step(
                    lambda df: df[["nonexistent_col"]],
                    name="bad_step",
                )
                .run()
            )

    def test_error_includes_columns(self, production_df):
        with pytest.raises(RuntimeError, match="Columns:"):
            (
                FeaturePipeline(production_df)
                .add_lambda_step(
                    lambda df: df[["nonexistent_col"]],
                    name="bad_step",
                )
                .run()
            )


# ------------------------------------------------------------------
# describe()
# ------------------------------------------------------------------


class TestDescribe:
    def test_describe_shows_input_shape(self, production_df):
        pipe = FeaturePipeline(production_df)
        desc = pipe.describe()
        assert "1200 rows" in desc
        assert "4 cols" in desc

    def test_describe_no_steps(self, production_df):
        desc = FeaturePipeline(production_df).describe()
        assert "no steps registered" in desc

    def test_describe_shows_step_names(self, production_df):
        desc = (
            FeaturePipeline(production_df)
            .add_step(DoubleFilter.filter_nan_value_double)
            .add_instance_step(DataHarmonizer, call="pivot_to_wide")
            .add_lambda_step(lambda df: df, name="identity")
            .describe()
        )
        assert "filter_nan_value_double" in desc
        assert "DataHarmonizer.pivot_to_wide" in desc
        assert "identity" in desc

    def test_describe_shows_step_types(self, production_df):
        desc = (
            FeaturePipeline(production_df)
            .add_step(DoubleFilter.filter_nan_value_double)
            .add_instance_step(DataHarmonizer, call="pivot_to_wide")
            .add_lambda_step(lambda df: df, name="identity")
            .describe()
        )
        assert "[step" in desc
        assert "[instance" in desc
        assert "[func" in desc

    def test_describe_shows_params(self, production_df):
        desc = (
            FeaturePipeline(production_df)
            .add_step(IntegerCalc.scale_column, column_name="value_double", factor=2)
            .describe()
        )
        assert "column_name='value_double'" in desc
        assert "factor=2" in desc

    def test_describe_shows_sentinels(self, production_df):
        desc = (
            FeaturePipeline(production_df)
            .add_step(
                SegmentProcessor.apply_ranges, dataframe="$input", time_ranges="$prev"
            )
            .describe()
        )
        assert "$input" in desc
        assert "$prev" in desc


# ------------------------------------------------------------------
# Chaining and steps property
# ------------------------------------------------------------------


class TestAddStep:
    def test_chain_returns_self(self, production_df):
        pipe = FeaturePipeline(production_df)
        result = pipe.add_step(DoubleFilter.filter_nan_value_double)
        assert result is pipe

    def test_add_instance_step_returns_self(self, production_df):
        pipe = FeaturePipeline(production_df)
        result = pipe.add_instance_step(DataHarmonizer, call="pivot_to_wide")
        assert result is pipe

    def test_add_lambda_step_returns_self(self, production_df):
        pipe = FeaturePipeline(production_df)
        result = pipe.add_lambda_step(lambda df: df.head(10), name="head_10")
        assert result is pipe

    def test_steps_property(self, production_df):
        pipe = (
            FeaturePipeline(production_df)
            .add_step(DoubleFilter.filter_nan_value_double)
            .add_lambda_step(lambda df: df, name="identity")
        )
        assert len(pipe.steps) == 2
        assert "DoubleFilter.filter_nan_value_double" in pipe.steps[0]
        assert pipe.steps[1] == "identity"


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------


class TestRun:
    def test_no_steps_returns_copy(self, production_df):
        result = FeaturePipeline(production_df).run()
        assert len(result) == len(production_df)
        assert result is not production_df

    def test_filter_nan_removes_rows(self, production_df):
        original_len = len(production_df)
        result = (
            FeaturePipeline(production_df)
            .add_step(DoubleFilter.filter_nan_value_double)
            .run()
        )
        assert len(result) < original_len
        assert result["value_double"].isna().sum() == 0

    def test_filter_dates_reduces_rows(self, production_df):
        result = (
            FeaturePipeline(production_df)
            .add_step(
                DateTimeFilter.filter_between_datetimes,
                start_datetime="2024-01-01 00:00:00",
                end_datetime="2024-01-01 00:01:00",
            )
            .run()
        )
        assert len(result) < len(production_df)
        assert len(result) > 0

    def test_calculator_step(self, production_df):
        result = (
            FeaturePipeline(production_df)
            .add_step(DoubleFilter.filter_nan_value_double)
            .add_step(IntegerCalc.scale_column, column_name="value_double", factor=2)
            .run()
        )
        original = production_df[production_df["value_double"].notna()]
        assert result["value_double"].mean() == pytest.approx(
            original["value_double"].mean() * 2, rel=0.01
        )

    def test_lambda_step(self, production_df):
        result = (
            FeaturePipeline(production_df)
            .add_lambda_step(lambda df: df[df["uuid"] == "temperature"])
            .run()
        )
        assert set(result["uuid"].unique()) == {"temperature"}

    def test_instance_step_harmonizer(self, production_df):
        result = (
            FeaturePipeline(production_df)
            .add_lambda_step(
                lambda df: df[df["uuid"].isin(["temperature", "pressure"])]
            )
            .add_instance_step(DataHarmonizer, call="pivot_to_wide")
            .run()
        )
        assert "temperature" in result.columns or "pressure" in result.columns

    def test_prev_reference(self, production_df):
        """$prev lets a step receive the output of the previous step."""
        result = (
            FeaturePipeline(production_df)
            .add_step(
                SegmentExtractor.extract_time_ranges,
                segment_uuid="order_number",
            )
            .add_step(
                SegmentProcessor.apply_ranges,
                dataframe=production_df,
                time_ranges="$prev",
                target_uuids=["temperature", "pressure", "speed"],
            )
            .run()
        )
        assert "segment_value" in result.columns
        assert len(result) > 0

    def test_input_reference(self, production_df):
        """$input lets a step receive the original constructor DataFrame."""
        result = (
            FeaturePipeline(production_df)
            .add_step(
                SegmentExtractor.extract_time_ranges,
                segment_uuid="order_number",
            )
            .add_step(
                SegmentProcessor.apply_ranges,
                dataframe="$input",
                time_ranges="$prev",
                target_uuids=["temperature", "pressure", "speed"],
            )
            .run()
        )
        assert "segment_value" in result.columns
        assert len(result) > 0

    def test_prev_reference_no_previous_raises(self, production_df):
        """$prev on the first step should raise."""
        with pytest.raises((ValueError, RuntimeError), match="no previous step result"):
            (
                FeaturePipeline(production_df)
                .add_step(
                    SegmentProcessor.apply_ranges,
                    time_ranges="$prev",
                )
                .run()
            )

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["systime", "uuid", "value_double"])
        result = (
            FeaturePipeline(empty).add_step(DoubleFilter.filter_nan_value_double).run()
        )
        assert result.empty


class TestRunSteps:
    def test_returns_dict_with_input(self, production_df):
        results = (
            FeaturePipeline(production_df)
            .add_step(DoubleFilter.filter_nan_value_double)
            .run_steps()
        )
        assert "input" in results
        assert len(results) == 2  # input + 1 step

    def test_intermediates_are_dataframes(self, production_df):
        results = (
            FeaturePipeline(production_df)
            .add_step(DoubleFilter.filter_nan_value_double)
            .add_lambda_step(lambda df: df.head(5), name="head_5")
            .run_steps()
        )
        for key, df in results.items():
            assert isinstance(df, pd.DataFrame), f"{key} is not a DataFrame"

    def test_intermediate_shapes_differ(self, production_df):
        results = (
            FeaturePipeline(production_df)
            .add_step(DoubleFilter.filter_nan_value_double)
            .add_lambda_step(lambda df: df.head(5), name="head_5")
            .run_steps()
        )
        assert len(results["input"]) == len(production_df)
        assert len(results["head_5"]) == 5

    def test_no_steps_returns_input_only(self, production_df):
        results = FeaturePipeline(production_df).run_steps()
        assert list(results.keys()) == ["input"]


# ------------------------------------------------------------------
# Full pipeline integration
# ------------------------------------------------------------------


class TestFullPipeline:
    def test_filter_segment_features(self, production_df):
        """Full pipeline: segment -> apply -> feature table."""
        result = (
            FeaturePipeline(production_df)
            .add_step(
                SegmentExtractor.extract_time_ranges,
                segment_uuid="order_number",
            )
            .add_step(
                SegmentProcessor.apply_ranges,
                dataframe=production_df,
                time_ranges="$prev",
                target_uuids=["temperature", "pressure", "speed"],
            )
            .add_step(
                TimeWindowedFeatureTable.compute,
                freq="1min",
                metrics=["mean", "std", "min", "max"],
            )
            .run()
        )
        assert "time_window" in result.columns
        wide_cols = [c for c in result.columns if "__" in c]
        assert len(wide_cols) > 0
        assert len(result) > 0

    def test_filter_segment_features_with_input_sentinel(self, production_df):
        """Full pipeline using $input instead of passing df directly."""
        result = (
            FeaturePipeline(production_df)
            .add_step(
                SegmentExtractor.extract_time_ranges,
                segment_uuid="order_number",
            )
            .add_step(
                SegmentProcessor.apply_ranges,
                dataframe="$input",
                time_ranges="$prev",
                target_uuids=["temperature", "pressure", "speed"],
            )
            .add_step(
                TimeWindowedFeatureTable.compute,
                freq="1min",
                metrics=["mean", "std", "min", "max"],
            )
            .run()
        )
        assert "time_window" in result.columns
        assert len(result) > 0

    def test_mixed_step_types(self, production_df):
        """Mix classmethod, instance, and lambda steps."""
        result = (
            FeaturePipeline(production_df)
            .add_lambda_step(
                lambda df: df[df["uuid"].isin(["temperature", "pressure"])],
                name="filter_uuids",
            )
            .add_step(DoubleFilter.filter_nan_value_double)
            .add_instance_step(DataHarmonizer, call="pivot_to_wide")
            .run()
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_output_long_format(self, production_df):
        """Use compute_long for long-format output."""
        result = (
            FeaturePipeline(production_df)
            .add_step(
                SegmentExtractor.extract_time_ranges,
                segment_uuid="order_number",
            )
            .add_step(
                SegmentProcessor.apply_ranges,
                dataframe=production_df,
                time_ranges="$prev",
                target_uuids=["temperature"],
            )
            .add_step(
                TimeWindowedFeatureTable.compute_long,
                freq="1min",
                metrics=["mean", "std"],
            )
            .run()
        )
        assert "time_window" in result.columns
        assert "uuid" in result.columns
        assert "mean" in result.columns
        assert "std" in result.columns
