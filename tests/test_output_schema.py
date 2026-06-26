"""Tests for the canonical event-output helpers in ``ts_shape.events._output``."""

import pandas as pd  # type: ignore
import pytest

from ts_shape.events._output import (
    INTERVAL_SCHEMA,
    POINT_SCHEMA,
    SUMMARY_SCHEMA,
    empty_event_df,
    finalize_interval_df,
    finalize_point_df,
    finalize_summary_df,
    validate_event_output,
)

# ---------------------------------------------------------------------------
# empty_event_df
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "schema"),
    [
        ("point", POINT_SCHEMA),
        ("interval", INTERVAL_SCHEMA),
        ("summary", SUMMARY_SCHEMA),
    ],
)
def test_empty_event_df_has_required_columns(shape, schema):
    df = empty_event_df(shape)
    assert df.empty
    assert list(df.columns) == list(schema)


def test_empty_event_df_appends_extra_cols_without_duplicates():
    df = empty_event_df("point", extra_cols=["severity", "uuid", "value"])
    # 'uuid' already required -> not duplicated; new cols appended in order.
    assert list(df.columns) == [*POINT_SCHEMA, "severity", "value"]


def test_empty_event_df_unknown_shape_raises():
    with pytest.raises(ValueError, match="Unknown event shape"):
        empty_event_df("nonsense")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# finalize_* helpers
# ---------------------------------------------------------------------------


def test_finalize_point_df_renames_time_and_broadcasts_identity():
    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    out = finalize_point_df(df, uuid="evt", source_uuid="sensor", time_col="ts")
    assert list(out.columns[:3]) == list(POINT_SCHEMA)
    assert (out["uuid"] == "evt").all()
    assert (out["source_uuid"] == "sensor").all()


def test_finalize_interval_df_computes_duration():
    df = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01 00:00:00"]),
            "end": pd.to_datetime(["2024-01-01 00:01:00"]),
        }
    )
    out = finalize_interval_df(df, uuid="evt", source_uuid="sensor")
    assert out["duration_seconds"].iloc[0] == 60.0
    assert list(out.columns[:5]) == list(INTERVAL_SCHEMA)


def test_finalize_interval_df_requires_start_end():
    with pytest.raises(ValueError, match="requires"):
        finalize_interval_df(pd.DataFrame({"x": [1]}), uuid="e", source_uuid="s")


def test_finalize_summary_df_optional_identity():
    df = pd.DataFrame(
        {
            "start": pd.to_datetime(["2024-01-01 00:00:00"]),
            "end": pd.to_datetime(["2024-01-01 01:00:00"]),
            "mean": [1.0],
        }
    )
    out = finalize_summary_df(df)
    assert out["duration_seconds"].iloc[0] == 3600.0
    assert list(out.columns[:3]) == list(SUMMARY_SCHEMA)


# ---------------------------------------------------------------------------
# validate_event_output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["point", "interval", "summary"])
def test_validate_event_output_accepts_finalized_frames(shape):
    # An empty canonical frame must satisfy its own contract.
    df = empty_event_df(shape)
    assert validate_event_output(df, shape) is df


def test_validate_event_output_rejects_missing_columns():
    df = pd.DataFrame({"systime": []})  # missing uuid / source_uuid
    with pytest.raises(ValueError, match="missing required column"):
        validate_event_output(df, "point")


def test_validate_event_output_rejects_non_dataframe():
    with pytest.raises(TypeError, match="DataFrame"):
        validate_event_output([1, 2, 3], "point")  # type: ignore[arg-type]


def test_validate_event_output_unknown_shape():
    with pytest.raises(ValueError, match="Unknown event shape"):
        validate_event_output(pd.DataFrame(), "bogus")  # type: ignore[arg-type]
