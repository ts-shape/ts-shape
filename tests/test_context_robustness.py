"""Regression tests for ValueMapper, ContextEnricher and DataIntegratorHybrid."""

from __future__ import annotations

import pandas as pd
import pytest

from ts_shape.context.value_mapping import ValueMapper
from ts_shape.errors import ColumnNotFoundError, DataQualityWarning
from ts_shape.loader.combine.integrator import DataIntegratorHybrid
from ts_shape.loader.context.context_enricher import ContextEnricher

# ---------- ValueMapper ----------------------------------------------------


def _frame(codes) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=len(codes), freq="1min"),
            "code": codes,
        }
    )


def _mapper(tmp_path, df, table, **kw):
    path = tmp_path / "map.csv"
    table.to_csv(path, index=False)
    return ValueMapper(
        dataframe=df,
        mapping_file=str(path),
        map_column=kw.pop("map_column", "code"),
        mapping_key_column=kw.pop("key", "key"),
        mapping_value_column=kw.pop("value", "name"),
        **kw,
    )


def test_map_values_keeps_column_when_key_column_has_same_name(tmp_path):
    table = pd.DataFrame({"code": ["A", "B"], "name": ["Alpha", "Beta"]})
    out = _mapper(tmp_path, _frame(["A", "B"]), table, key="code").map_values()
    assert out["code"].tolist() == ["Alpha", "Beta"]


def test_unmapped_codes_keep_value_and_warn(tmp_path):
    table = pd.DataFrame({"key": ["A"], "name": ["Alpha"]})
    with pytest.warns(DataQualityWarning, match="2 value"):
        out = _mapper(tmp_path, _frame(["A", "X", "Y"]), table).map_values()
    assert out["code"].tolist() == ["Alpha", "X", "Y"]


def test_duplicate_mapping_keys_raise(tmp_path):
    table = pd.DataFrame({"key": ["A", "A"], "name": ["Alpha", "Other"]})
    with pytest.raises(ValueError, match="duplicate keys"):
        _mapper(tmp_path, _frame(["A"]), table).map_values()


def test_same_key_and_value_column_raises(tmp_path):
    table = pd.DataFrame({"key": ["A"], "name": ["Alpha"]})
    with pytest.raises(ValueError, match="two different columns"):
        _mapper(tmp_path, _frame(["A"]), table, value="key").map_values()


def test_int_codes_match_string_keys(tmp_path):
    table = pd.DataFrame({"key": ["1", "2"], "name": ["run", "idle"]})
    path = tmp_path / "map.json"
    table.to_json(path)
    mapper = ValueMapper(
        _frame([1, 2, 1]),
        str(path),
        "code",
        "key",
        "name",
        file_type="json",
    )
    assert mapper.map_values()["code"].tolist() == ["run", "idle", "run"]


def test_other_columns_named_like_mapping_columns_survive(tmp_path):
    df = _frame(["A"]).assign(name="keep me")
    table = pd.DataFrame({"key": ["A"], "name": ["Alpha"]})
    out = _mapper(tmp_path, df, table).map_values()
    assert out["name"].tolist() == ["keep me"]
    assert out["code"].tolist() == ["Alpha"]


def test_value_mapper_column_name_is_deprecated(tmp_path):
    table = pd.DataFrame({"key": ["A"], "name": ["Alpha"]})
    with pytest.warns(DeprecationWarning, match="time_column"):
        _mapper(tmp_path, _frame(["A"]), table, column_name="systime")


# ---------- ContextEnricher ------------------------------------------------


@pytest.fixture
def ts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=4, freq="1min"),
            "uuid": ["a", "a", "b", "b"],
            "value_string": ["0", "1", "0", "1"],
        }
    )


def test_enrich_calls_chain(ts):
    meta = pd.DataFrame({"uuid": ["a", "b"], "unit": ["C", "bar"]})
    tol = pd.DataFrame({"uuid": ["a", "b"], "low_limit": [1, 2], "high_limit": [3, 4]})
    enricher = ContextEnricher(ts)
    enricher.enrich_with_metadata(meta)
    enricher.enrich_with_tolerances(tol)
    out = enricher.get_enriched_dataframe()
    assert {"unit", "low_limit", "high_limit"} <= set(out.columns)
    assert len(out) == len(ts)


def test_duplicate_metadata_uuid_raises(ts):
    meta = pd.DataFrame({"uuid": ["a", "a"], "unit": ["C", "F"]})
    with pytest.raises(ValueError, match="duplicate key"):
        ContextEnricher(ts).enrich_with_metadata(meta)


def test_colliding_columns_raise_naming_them(ts):
    meta = pd.DataFrame({"uuid": ["a", "b"], "value_string": ["x", "y"]})
    with pytest.raises(ValueError, match="value_string"):
        ContextEnricher(ts).enrich_with_metadata(meta)


def test_numeric_uuid_matches_string_metadata():
    df = pd.DataFrame({"systime": pd.date_range("2024", periods=2), "uuid": [1, 2]})
    meta = pd.DataFrame({"uuid": ["1", "2"], "unit": ["C", "bar"]})
    out = ContextEnricher(df).enrich_with_metadata(meta)
    assert out["unit"].tolist() == ["C", "bar"]
    assert out["uuid"].tolist() == [1, 2]


def test_mapping_matches_integer_codes():
    df = pd.DataFrame({"uuid": ["s", "s"], "value_integer": [0, 1]})
    mapping = pd.DataFrame(
        {"uuid": ["s", "s"], "raw_value": ["0", "1"], "mapped_value": ["off", "on"]}
    )
    out = ContextEnricher(df).enrich_with_mapping(mapping, value_column="value_integer")
    assert out["mapped_value"].tolist() == ["off", "on"]


def test_enricher_old_uuid_keyword_is_deprecated(ts):
    meta = pd.DataFrame({"uuid": ["a", "b"], "unit": ["C", "bar"]})
    with pytest.warns(DeprecationWarning, match="metadata_uuid_column"):
        ContextEnricher(ts).enrich_with_metadata(meta, metadata_uuid_col="uuid")


def test_enricher_missing_column_raises(ts):
    with pytest.raises(ColumnNotFoundError):
        ContextEnricher(ts).enrich_with_tolerances(pd.DataFrame({"uuid": ["a"]}))


# ---------- DataIntegratorHybrid -------------------------------------------


def _signal(uuid, tz=None):
    return pd.DataFrame(
        {
            "systime": pd.date_range("2024-01-01", periods=2, freq="1h", tz=tz),
            "uuid": [uuid, uuid],
            "value_double": [1.0, 2.0],
        }
    )


def test_uuid_filter_applies_without_metadata():
    out = DataIntegratorHybrid.combine_data(
        timeseries_sources=[_signal("a"), _signal("b")], uuids=["b"]
    )
    assert set(out["uuid"]) == {"b"}


def test_conflicting_metadata_rows_raise():
    with pytest.raises(ValueError, match="disagree"):
        DataIntegratorHybrid.combine_data(
            timeseries_sources=[_signal("a")],
            metadata_sources=[
                pd.DataFrame({"uuid": ["a"], "unit": ["C"]}),
                pd.DataFrame({"uuid": ["a"], "unit": ["F"]}),
            ],
        )


def test_metadata_sources_complement_each_other():
    out = DataIntegratorHybrid.combine_data(
        timeseries_sources=[_signal("a")],
        metadata_sources=[
            pd.DataFrame({"uuid": ["a"], "unit": ["C"]}),
            pd.DataFrame({"uuid": ["a"], "area": ["z1"]}),
        ],
    )
    assert len(out) == 2
    assert out["unit"].tolist() == ["C", "C"]
    assert out["area"].tolist() == ["z1", "z1"]


def test_mixed_timezone_sources_are_converted_to_utc():
    with pytest.warns(DataQualityWarning, match="UTC"):
        out = DataIntegratorHybrid.combine_data(
            timeseries_sources=[_signal("a"), _signal("b", tz="Europe/Berlin")]
        )
    assert str(out["systime"].dt.tz) == "UTC"
    assert out["systime"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")


def test_unsupported_source_warns():
    with pytest.warns(DataQualityWarning, match="unsupported"):
        DataIntegratorHybrid.combine_data(timeseries_sources=[_signal("a"), 42])


def test_empty_result_has_standard_columns():
    with pytest.warns(DataQualityWarning, match="no timeseries"):
        out = DataIntegratorHybrid.combine_data(timeseries_sources=[])
    assert list(out.columns) == [
        "systime",
        "uuid",
        "value_bool",
        "value_integer",
        "value_double",
        "value_string",
        "is_delta",
    ]
    assert out.empty
