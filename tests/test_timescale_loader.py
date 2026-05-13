import pytest
import pandas as pd  # type: ignore

pytest.importorskip("sqlalchemy")


def test_timescale_fetch_dataframe_monkeypatched(monkeypatch):
    # Patch create_engine before importing the loader class to avoid psycopg2 import
    monkeypatch.setattr(
        "ts_shape.loader.timeseries.timescale_loader.create_engine",
        lambda *args, **kwargs: None,
    )

    from ts_shape.loader.timeseries.timescale_loader import TimescaleDBDataAccess

    loader = TimescaleDBDataAccess(
        start_timestamp="2024-01-01 00:00:00",
        end_timestamp="2024-01-01 01:00:00",
        uuids=["u1", "u2"],
        db_config={"db_user": "u", "db_pass": "p", "db_host": "h", "db_name": "n"},
    )

    # Replace _fetch_data to return list-like of DataFrames for each uuid
    def fake_fetch(uuid):
        return [
            pd.DataFrame({"uuid": [uuid], "systime": [pd.Timestamp("2024-01-01")]}),
            pd.DataFrame(
                {"uuid": [uuid], "systime": [pd.Timestamp("2024-01-01 00:30:00")]}
            ),
        ]

    monkeypatch.setattr(loader, "_fetch_data", fake_fetch)
    df = loader.fetch_data_as_dataframe()
    assert set(df["uuid"]) == {"u1", "u2"}
    assert len(df) == 4
