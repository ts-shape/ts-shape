import pandas as pd  # type: ignore
from ts_shape.loader.combine.integrator import DataIntegratorHybrid


class _TSObj:
    def __init__(self, df):
        self._df = df

    def fetch_data_as_dataframe(self):
        return self._df


class _MetaObj:
    def __init__(self, df):
        self._df = df

    def fetch_metadata(self):
        return self._df


def test_integrator_with_dataframes_and_objects_and_filter():
    ts1 = pd.DataFrame({"uuid": ["a", "b"], "value": [1, 2]})
    ts2 = pd.DataFrame({"uuid": ["b", "c"], "value": [3, 4]})
    meta1 = pd.DataFrame({"uuid": ["a", "b", "c"], "label": ["A", "B", "C"]})

    combined = DataIntegratorHybrid.combine_data(
        timeseries_sources=[ts1, _TSObj(ts2)],
        metadata_sources=[
            meta1,
            _MetaObj(pd.DataFrame({"uuid": ["c"], "extra": ["E"]})),
        ],
        uuids=["b", "c"],
        join_key="uuid",
        merge_how="left",
    )

    assert not combined.empty
    assert set(combined["uuid"].unique()) == {"b", "c"}
    assert "label" in combined.columns
    assert "extra" in combined.columns
