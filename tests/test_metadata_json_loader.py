import pandas as pd  # type: ignore
from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader


def test_from_list_of_records_and_accessors():
    records = [
        {"uuid": "u1", "label": "L1", "config": {"a": 1, "b": 2}},
        {"uuid": "u2", "label": "L2", "config": {"a": 3}},
    ]
    loader = MetadataJsonLoader(records)
    df = loader.to_df()
    assert list(df.index) == ["u1", "u2"]
    assert "config.a" in df.columns or "a" in df.columns
    assert loader.get_by_uuid("u1")["label"] == "L1"
    assert loader.get_by_label("L2")["label"] == "L2"
    assert set(loader.list_uuids()) == {"u1", "u2"}
    assert set(loader.list_labels()) == {"L1", "L2"}

    joined = loader.join_with(pd.DataFrame({"x": [10, 20]}, index=["u1", "u2"]))
    assert "x" in joined.columns


def test_from_columnar_dicts_and_filters():
    columnar = {
        "uuid": ["a", "b", "c"],
        "label": ["A", "B", "C"],
        "config": [{}, {}, {}],
    }
    loader = MetadataJsonLoader(columnar)
    df = loader.to_df()
    assert list(df.index) == ["a", "b", "c"]

    f1 = loader.filter_by_uuid(["a", "c"])
    assert list(f1.index) == ["a", "c"]

    f2 = loader.filter_by_label(["B"])
    assert list(f2["label"]) == ["B"]


def test_from_indexed_dicts_and_strict_validation():
    indexed = {
        "uuid": {"0": "x", "1": "x"},  # duplicate to trigger strict
        "label": {"0": "X", "1": "Y"},
        "config": {"0": {}, "1": {}},
    }
    try:
        MetadataJsonLoader(indexed, strict=True)
        assert False, "Expected ValueError for duplicate UUIDs in strict mode"
    except ValueError:
        pass

    # Non-strict should drop dupes
    loader2 = MetadataJsonLoader(indexed, strict=False)
    assert loader2.to_df().shape[0] == 1
