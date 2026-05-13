import pandas as pd  # type: ignore
from ts_shape.context.value_mapping import ValueMapper


def test_value_mapper_csv_and_json(tmp_path):
    df = pd.DataFrame({"code": ["A", "B", "C"], "other": [1, 2, 3]})

    # CSV mapping
    csv_path = tmp_path / "map.csv"
    csv_df = pd.DataFrame({"key": ["A", "B", "C"], "name": ["Alpha", "Beta", "Gamma"]})
    csv_df.to_csv(csv_path, index=False)

    vm_csv = ValueMapper(
        dataframe=df,
        mapping_file=str(csv_path),
        map_column="code",
        mapping_key_column="key",
        mapping_value_column="name",
        file_type="csv",
    )
    out_csv = vm_csv.map_values()
    assert out_csv["code"].tolist() == ["Alpha", "Beta", "Gamma"]

    # JSON mapping
    json_path = tmp_path / "map.json"
    json_df = pd.DataFrame({"key": ["A", "B"], "name": ["A1", "B1"]})
    json_df.to_json(json_path)

    vm_json = ValueMapper(
        dataframe=df,
        mapping_file=str(json_path),
        map_column="code",
        mapping_key_column="key",
        mapping_value_column="name",
        file_type="json",
    )
    out_json = vm_json.map_values()
    assert out_json["code"].tolist()[:2] == ["A1", "B1"]
