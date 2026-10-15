"""Base construction and the text helpers give what the pandas calls give.

``Base`` sorts, counts NaT and duplicates on integer codes, skips work an
already-sorted, already-datetime frame does not need, and never converts a
whole uuid column to text. These tests pin the results to the plain pandas
computations they replace.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ts_shape.utils.base import Base
from ts_shape.utils.normalize import (
    normalize_signal_frame,
    text_keys,
    text_mask,
    unique_text,
)
from ts_shape.utils.signal_checks import _present_uuids

T = pd.date_range("2025-01-01", periods=6, freq="1min")


def _random_frame(seed: int, n: int = 60) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    uuids = ["b", "a", "c"] + ([None] if seed % 2 else [])
    df = pd.DataFrame(
        {
            "systime": r.choice(T, n),
            "uuid": pd.Series(list(r.choice(np.array(uuids, dtype=object), n))),
            "value_double": r.normal(size=n),
        },
        index=r.permutation(n) + 10,
    )
    if seed % 3 == 0:
        df.loc[df.index[:4], "systime"] = pd.NaT
    return df


def _reference(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """What Base computed before: sort_values, isna().sum(), duplicated()."""
    keys = ["systime", "uuid"] if "uuid" in df.columns else ["systime"]
    out = df.sort_values(by=keys, kind="stable")
    return out, int(out["systime"].isna().sum()), int(out.duplicated(keys).sum())


def _construct(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        frame = Base(df).dataframe
    return frame, [str(w.message) for w in rec]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("uuid_dtype", ["object", "category", "string"])
def test_sort_and_counts_match_pandas(seed, uuid_dtype):
    df = _random_frame(seed)
    df["uuid"] = df["uuid"].astype(uuid_dtype)
    expected, nat, dups = _reference(df)
    frame, messages = _construct(df)
    pd.testing.assert_index_equal(frame.index, expected.index)
    assert frame["value_double"].tolist() == expected["value_double"].tolist()
    assert any(f"contains {dups} duplicate" in m for m in messages) == (dups > 0)
    assert any(f"contains {nat} missing" in m for m in messages) == (0 < nat < len(df))


def test_tz_aware_times_sort_by_instant():
    df = _random_frame(4)
    df["systime"] = df["systime"].dt.tz_localize("Europe/Berlin")
    expected, _, _ = _reference(df)
    frame, _ = _construct(df)
    pd.testing.assert_index_equal(frame.index, expected.index)


@pytest.mark.filterwarnings("ignore::ts_shape.errors.DataQualityWarning")
def test_sorted_input_is_kept_and_not_shared():
    df = _random_frame(1).sort_values(["systime", "uuid"], kind="stable")
    snapshot = df.copy()
    frame = Base(df).dataframe
    pd.testing.assert_index_equal(frame.index, df.index)
    frame.loc[frame.index[0], "value_double"] = 1e9
    pd.testing.assert_frame_equal(df, snapshot)


def test_mixed_type_uuids_take_the_pandas_path():
    df = pd.DataFrame(
        {"systime": T[[1, 0, 0, 0]], "uuid": [1, "1", 1.0, "0"], "value_double": 1.0}
    )
    expected, _, dups = _reference(df)
    frame, messages = _construct(df)
    pd.testing.assert_index_equal(frame.index, expected.index)
    assert any(f"contains {dups} duplicate" in m for m in messages) == (dups > 0)


def test_unique_text_matches_astype_str():
    cases = [
        pd.Series(["a", None, "b", "a"], dtype=object),
        pd.Series(["a", None, "b"], dtype="string"),
        pd.Series(["a", "b", "a"], dtype="category"),
        pd.Series([1, 2, 2]),
        pd.Series([1.5, np.nan]),
        pd.Series([1, 1.0, "1", True], dtype=object),
    ]
    for s in cases:
        assert unique_text(s) == set(s.dropna().astype(str).unique())


def test_text_mask_matches_astype_str():
    cases = [
        (pd.Series(["a", None, "b", "a"], dtype=object), "a"),
        (pd.Series(["a", None, "b"], dtype="string"), "b"),
        (pd.Series(["a", "b", "a"], dtype="category"), "a"),
        (pd.Series([1, 2, 2]), "2"),
        (pd.Series([1.0, 2.0]), "1"),
        (pd.Series([1, 1.0, "1"], dtype=object), "1"),
        (pd.Series(["a", "b"]), "zz"),
    ]
    for s, text in cases:
        expected = ((s.astype(str) == text) & s.notna()).to_numpy()
        assert text_mask(s, text).tolist() == expected.tolist()


def test_text_keys_match_astype_str():
    for s in (
        pd.Series(["b", "a", "b"]),
        pd.Series([3, 1, 3]),
        pd.Series(["a", "b"], dtype="category"),
        pd.Series([1, 1.0, "1"], dtype=object),
    ):
        assert text_keys(s).tolist() == s.astype(str).to_numpy().tolist()


@pytest.mark.parametrize("dtype", ["string", "string[pyarrow]", "str"])
def test_string_columns_become_object_with_nan(dtype):
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(
        {
            "systime": T[:3],
            "uuid": pd.array(["a", None, "a"], dtype=dtype),
            "value_string": pd.array(["x", "y", None], dtype=dtype),
        }
    )
    out = normalize_signal_frame(df.copy())
    for column in ("uuid", "value_string"):
        expected = df[column].astype(object).where(df[column].notna(), np.nan)
        assert out[column].dtype == object
        pd.testing.assert_series_equal(out[column], expected)
        assert out[column].isna().tolist() == df[column].isna().tolist()


def test_present_uuids_follow_the_current_frame():
    class Holder:
        pass

    holder = Holder()
    frame = pd.DataFrame({"uuid": ["a", "b"]})
    holder.dataframe = frame
    assert _present_uuids(holder, frame, "uuid") == {"a", "b"}
    other = pd.DataFrame({"uuid": ["c"]})
    assert _present_uuids(holder, other, "uuid") == {"c"}


@pytest.mark.filterwarnings("ignore::ts_shape.errors.DataQualityWarning")
def test_base_remembers_uuids_for_the_signal_checks():
    df = _random_frame(2)
    base = Base(df)
    assert _present_uuids(base, base.dataframe, "uuid") == set(
        df["uuid"].dropna().astype(str)
    )
