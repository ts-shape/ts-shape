"""Metadata JSON shapes, the integrator's merge paths and YAML lambda rules."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ts_shape.errors import DataQualityWarning
from ts_shape.loader.combine.integrator import DataIntegratorHybrid
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader

# --------------------------------------------------------------------------
# MetadataJsonLoader input shapes
# --------------------------------------------------------------------------

RECORDS = [
    {"uuid": "u1", "label": "temp", "config": {"unit": "degC", "max": 90}},
    {"uuid": "u2", "label": "pressure", "config": '{"unit": "bar"}'},
]


def test_list_of_records_with_a_json_string_config():
    df = MetadataJsonLoader(RECORDS).to_df()
    assert df.loc["u2", "unit"] == "bar"
    assert df.loc["u1", "max"] == 90


def test_list_skips_non_objects_with_a_warning():
    with pytest.warns(DataQualityWarning, match="1 metadata record"):
        df = MetadataJsonLoader([*RECORDS, "garbage"]).to_df()
    assert list(df.index) == ["u1", "u2"]


def test_columnar_lists():
    data = {
        "uuid": ["u1", "u2"],
        "label": ["temp", "pressure"],
        "config": [{"unit": "degC"}, None],
    }
    df = MetadataJsonLoader(data).to_df()
    assert df.loc["u1", "unit"] == "degC"
    assert pd.isna(df.loc["u2", "unit"])


def test_columnar_lists_of_different_lengths():
    data = {"uuid": ["u1", "u2"], "label": ["temp"], "config": [{}, {}]}
    with pytest.raises(ValueError, match="Column lengths differ"):
        MetadataJsonLoader(data)
    lenient = MetadataJsonLoader(data, strict=False).to_df()
    assert list(lenient.index) == ["u1"]  # cut to the shortest column


def test_columnar_without_label_needs_strict_false():
    data = {"uuid": ["u1"], "config": [{"unit": "bar"}]}
    with pytest.raises(ValueError, match="missing \\['label'\\]"):
        MetadataJsonLoader(data)
    df = MetadataJsonLoader(data, strict=False).to_df()
    assert df.loc["u1", "label"] is None


def test_columnar_dicts_with_numeric_string_keys_keep_their_order():
    data = {
        "uuid": {"10": "u10", "2": "u2"},
        "label": {"10": "ten", "2": "two"},
        "config": {"10": {"v": 10}, "2": "plain text"},
    }
    df = MetadataJsonLoader(data).to_df()
    assert list(df.index) == ["u2", "u10"]
    assert df.loc["u2", "value"] == "plain text"  # a non-JSON string is kept as value


def test_columnar_dicts_without_label_are_filled_when_lenient():
    data = {"uuid": {"0": "u1"}, "config": {"0": {"x": 1}}}
    df = MetadataJsonLoader(data, strict=False).to_df()
    assert df.loc["u1", "x"] == 1


def test_single_record_dict_and_scalar_config():
    df = MetadataJsonLoader({"uuid": "u1", "label": "temp", "config": 42}).to_df()
    assert df.loc["u1", "value"] == 42
    df = MetadataJsonLoader({"uuid": "u1", "label": "t", "config": "[1, 2]"}).to_df()
    assert df.loc["u1", "value"] == [1, 2]


def test_config_key_colliding_with_a_column_is_prefixed():
    df = MetadataJsonLoader(
        [{"uuid": "u1", "label": "temp", "config": {"label": "x"}}]
    ).to_df()
    assert df.loc["u1", "label"] == "temp"
    assert df.loc["u1", "config.label"] == "x"


def test_unsupported_and_empty_inputs():
    with pytest.raises(TypeError, match="Unsupported JSON structure"):
        MetadataJsonLoader(42)
    assert MetadataJsonLoader(42, strict=False).to_df().empty
    with pytest.raises(ValueError, match="metadata is empty"):
        MetadataJsonLoader([])
    empty = MetadataJsonLoader({}, strict=False).to_df()
    assert empty.empty and empty.index.name == "uuid"


def test_strict_checks_on_records():
    with pytest.raises(ValueError, match="no label in strict mode"):
        MetadataJsonLoader([{"uuid": "u1", "config": {}}])
    with pytest.raises(ValueError, match="missing UUIDs"):
        MetadataJsonLoader([{"label": "x", "config": {}}])
    with pytest.raises(ValueError, match="not unique"):
        MetadataJsonLoader([RECORDS[0], RECORDS[0]])


def test_lenient_drops_missing_and_duplicate_uuids():
    data = [RECORDS[0], {"label": "orphan"}, {**RECORDS[0], "label": "again"}]
    df = MetadataJsonLoader(data, strict=False).to_df()
    assert list(df.index) == ["u1"]
    assert df.loc["u1", "label"] == "temp"


def test_lookups_and_files(tmp_path):
    path = tmp_path / "signals.json"
    path.write_text(json.dumps(RECORDS), encoding="utf-8")
    meta = MetadataJsonLoader.from_file(str(path))
    assert meta.get_by_uuid("u2")["label"] == "pressure"
    assert meta.get_by_uuid("nope") is None
    assert meta.get_by_label("temp")["unit"] == "degC"
    assert meta.get_by_label("nope") is None
    assert meta.list_uuids() == ["u1", "u2"]
    assert meta.list_labels() == ["temp", "pressure"]
    assert list(meta.filter_by_uuid(["u2"]).index) == ["u2"]
    assert list(meta.filter_by_label(["temp"]).index) == ["u1"]
    assert len(meta.head(1)) == 1
    assert meta.to_df(copy=False) is meta.df
    ts = pd.DataFrame({"uuid": ["u1", "u3"], "value_double": [1.0, 2.0]}).set_index(
        "uuid"
    )
    assert list(meta.join_with(ts).index) == ["u1"]


# --------------------------------------------------------------------------
# DataIntegratorHybrid merge paths
# --------------------------------------------------------------------------

TS = pd.DataFrame(
    {
        "systime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:01"]),
        "uuid": ["u1", "u2"],
        "value_double": [1.0, 2.0],
    }
)


class TimeseriesSource:
    def fetch_data_as_dataframe(self):
        return TS


class MetadataSource:
    def fetch_metadata(self):
        return pd.DataFrame({"uuid": ["u1"], "unit": ["degC"]}).set_index("uuid")


def test_combine_reads_source_objects_and_index_keys():
    out = DataIntegratorHybrid.combine_data(
        timeseries_sources=[TimeseriesSource()], metadata_sources=[MetadataSource()]
    )
    units = out.set_index("uuid")["unit"]
    assert units["u1"] == "degC"
    assert pd.isna(units["u2"])
    inner = DataIntegratorHybrid.combine_data(
        [TS], [MetadataSource()], merge_how="inner"
    )
    assert inner["uuid"].tolist() == ["u1"]


def test_combine_skips_unsupported_sources():
    with pytest.warns(DataQualityWarning, match="unsupported timeseries source"):
        out = DataIntegratorHybrid.combine_data([TS, object()])
    assert len(out) == 2
    with pytest.warns(DataQualityWarning, match="unsupported metadata source"):
        out = DataIntegratorHybrid.combine_data([TS], [object()])
    assert "unit" not in out.columns


def test_combine_without_timeseries_gives_the_standard_empty_frame():
    with pytest.warns(DataQualityWarning, match="no timeseries data"):
        out = DataIntegratorHybrid.combine_data([], [MetadataSource()])
    assert out.empty
    assert {"systime", "uuid", "value_double"} <= set(out.columns)


def test_combine_metadata_key_in_a_multiindex():
    meta = pd.DataFrame({"site": ["A"], "uuid": ["u2"], "unit": ["bar"]}).set_index(
        ["site", "uuid"]
    )
    out = DataIntegratorHybrid.combine_data([TS], [meta])
    assert out.loc[out["uuid"] == "u2", "unit"].item() == "bar"


def test_combine_join_key_missing():
    meta = pd.DataFrame({"signal": ["u1"], "unit": ["degC"]})
    with pytest.warns(DataQualityWarning) as caught:
        out = DataIntegratorHybrid.combine_data([TS], [meta])
    messages = " ".join(str(w.message) for w in caught)
    assert "not found in columns or index" in messages
    assert "cannot merge" in messages and "metadata" in messages
    assert list(out.columns) == list(TS.columns)

    no_key = TS.drop(columns="uuid")
    with pytest.warns(DataQualityWarning) as caught:
        DataIntegratorHybrid.combine_data([no_key], uuids=["u1"])
    assert any("cannot filter by uuids" in str(w.message) for w in caught)


def test_combine_merges_duplicate_metadata_rows():
    a = pd.DataFrame({"uuid": ["u1"], "unit": ["degC"], "area": [None]})
    b = pd.DataFrame({"uuid": ["u1"], "unit": ["degC"], "area": ["hall 1"]})
    out = DataIntegratorHybrid.combine_data([TS], [a, b], uuids=["u1"])
    assert out[["unit", "area"]].iloc[0].tolist() == ["degC", "hall 1"]


def test_combine_mixed_timezones_are_converted_to_utc():
    aware = TS.assign(systime=TS["systime"].dt.tz_localize("Europe/Berlin"))
    with pytest.warns(DataQualityWarning, match="converted to UTC"):
        out = DataIntegratorHybrid.combine_data([TS, aware])
    assert str(out["systime"].dt.tz) == "UTC"
    assert len(out) == 4


# --------------------------------------------------------------------------
# lambda rules from YAML
# --------------------------------------------------------------------------

RULE_YAML = """
rules:
  - id: coverage_hot
    class_name: LambdaCoverageDemo
    method_name: hot
    pack: maintenance
    shape: point
    archetype: threshold
    template: "maintenance.coverage.hot"
    produces_objects: asset
    trigger:
      expression: "temp > 80"
      group_by: line
    standard_attrs:
      ts_shape:method: lambda_threshold
      ts_shape:direction: above
"""


def test_load_yaml_registers_and_runs(tmp_path):
    pytest.importorskip("yaml")
    from ts_shape.eventlog.lambda_rules import load_yaml, unregister_lambda_rule

    path = tmp_path / "rules.yaml"
    path.write_text(RULE_YAML, encoding="utf-8")
    try:
        (detector,) = load_yaml(path)
        assert detector.spec.trigger.group_by == ("line",)
        assert detector.spec.produces_objects == ("asset",)
    finally:
        unregister_lambda_rule("LambdaCoverageDemo", "hot")


def test_load_yaml_requires_a_rules_list(tmp_path):
    pytest.importorskip("yaml")
    from ts_shape.eventlog.lambda_rules import load_yaml

    path = tmp_path / "rules.yaml"
    path.write_text("dsl_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level 'rules:'"):
        load_yaml(path)
