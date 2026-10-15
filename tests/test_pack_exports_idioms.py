"""Pack exports, typing idioms and the pandas 2.x mixed-offset time parse."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

# ---- exports and idioms ------------------------------------------------------


def test_pack_exports():
    import ts_shape.events.production as production
    import ts_shape.features as features
    import ts_shape.signals.infer as infer

    for name in ("FlowMetricsEvents", "LineBalancingEvents", "RuntimeAccountingEvents"):
        assert name in production.__all__
        assert getattr(production, name) is getattr(__import__("ts_shape"), name)
    for name in (
        "PatternRecognition",
        "DescriptiveFeatures",
        "TimeWindowedFeatureTable",
    ):
        assert name in features.__all__
        assert getattr(features, name).__name__ == name
    assert {"parse_times", "signal_groups"} <= set(infer.__all__)


def test_alternate_constructors_return_self_type():
    import typing

    from ts_shape.loader.metadata.metadata_json_loader import MetadataJsonLoader

    class Sub(MetadataJsonLoader):
        pass

    assert typing.get_type_hints(MetadataJsonLoader.from_str)["return"] is typing.Self
    assert type(Sub.from_str('{"u1": {"label": "a"}}', strict=False)) is Sub


def test_parse_times_mixed_offsets_become_utc():
    """pandas 2.x returns objects (with a FutureWarning), pandas 3 raises."""
    from ts_shape.errors import DataQualityWarning
    from ts_shape.signals.infer import parse_times

    values = pd.Series(["2024-03-30 12:00+01:00", "2024-03-31 12:00+02:00"], name="t")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        with pytest.warns(DataQualityWarning, match="mixes UTC offsets"):
            out = parse_times(values)
    assert str(out.dt.tz) == "UTC"
    assert list(out.dt.hour) == [11, 10]
