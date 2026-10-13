"""The analytics core must work without any loader backend installed."""

import inspect
import subprocess
import sys
import textwrap

import pytest

_BLOCK = ("s3fs", "requests", "sqlalchemy", "azure", "psycopg2")


def test_core_runs_without_loader_backends():
    script = textwrap.dedent(f"""
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in {_BLOCK!r}:
                    raise ImportError(f"blocked {{name}}")
                return None

        sys.meta_path.insert(0, Block())
        import ts_shape
        df = ts_shape.make_timeseries(["sensor:temp"], n_points=200, n_outliers=2)
        ts_shape.profile(df)
        ts_shape.OutlierDetectionEvents(df, value_column="value_double").detect_outliers_zscore()
        try:
            ts_shape.S3ProxyDataAccess(start_timestamp="2024-01-01", end_timestamp="2024-01-02",
                                       uuids=["a"], s3_config={{}})
        except ImportError as exc:
            assert "ts-shape[s3]" in str(exc), exc
        else:
            raise AssertionError("expected an ImportError naming the extra")
        print("ok")
        """)
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


@pytest.mark.parametrize(
    ("module", "attr", "cls", "extra"),
    [
        (
            "ts_shape.loader.metadata.metadata_api_loader",
            "requests",
            "DatapointAPI",
            "api",
        ),
        (
            "ts_shape.loader.timeseries.energy_api_loader",
            "requests",
            "EnergyAPILoader",
            "api",
        ),
        (
            "ts_shape.loader.metadata.metadata_db_loader",
            "create_engine",
            "DatapointDB",
            "sql",
        ),
        (
            "ts_shape.loader.timeseries.timescale_loader",
            "create_engine",
            "TimescaleDBDataAccess",
            "sql",
        ),
    ],
)
def test_missing_backend_names_the_extra(monkeypatch, module, attr, cls, extra):
    mod = pytest.importorskip(module)
    monkeypatch.setattr(mod, attr, None)
    target = getattr(mod, cls)
    required = [
        n
        for n, p in inspect.signature(target).parameters.items()
        if p.default is inspect.Parameter.empty and p.kind is not p.VAR_KEYWORD
    ]
    with pytest.raises(ImportError, match=rf"ts-shape\[{extra}\]"):
        target(**{n: "x" for n in required})  # the guard runs first
