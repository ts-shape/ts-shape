"""Every detector call gives the results and warnings it gave before round 18.

Round 18 lets ``Base`` keep only the requested signal rows of a detector
that reads nothing else (the warnings still come from the whole frame).
This test runs every ``(class, method)`` of the registry on the
:mod:`tests.invariants.harness` frames (plus a categorical uuid column, an
``object`` uuid column, missing timestamps and a frame without one of the
signals) and compares a digest of each result, of the warnings recorded
during the call (category, text and the file they point at), of the error
raised, and of ``get_dataframe()`` / ``repr()`` of the detector, with the
digests captured from the code before the change.

The digests depend on the pandas version (dtypes and reprs differ), so they
are pinned per ``major.minor``; a version without digests is skipped.
Regenerate (only when a result is meant to change) with::

    PYTHONPATH=src python -m tests.invariants.test_signal_rows_equivalence --update
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tests.invariants import harness

BASELINE = Path(__file__).with_name("signal_rows_baseline.json")
PANDAS = ".".join(pd.__version__.split(".")[:2])
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@functools.cache
def _variants() -> dict[str, pd.DataFrame]:
    frames = dict(harness._cached_inputs())
    base = frames["base"]
    frames["empty"] = base.iloc[0:0]
    category = base.copy()
    category["uuid"] = category["uuid"].astype("category")
    frames["category"] = category
    obj = base.copy()
    obj["uuid"] = obj["uuid"].astype(object)
    frames["object_uuid"] = obj
    nat = harness.shuffled(base)
    nat.loc[nat.index[::50], "systime"] = pd.NaT
    frames["nat"] = nat
    frames["no_cont"] = base[base["uuid"] != harness.CONT].reset_index(drop=True)
    return frames


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return _ADDRESS.sub("0x", repr(value))


def _canon(obj: Any) -> Any:
    """A JSON-able, deterministic description of ``obj``."""
    if isinstance(obj, pd.DataFrame):
        return [
            "frame",
            [_text(c) for c in obj.columns],
            [str(d) for d in obj.dtypes],
            str(obj.index.dtype),
            _text(obj.index.tolist()),
            [_canon_values(obj.iloc[:, i]) for i in range(obj.shape[1])],
            _text(dict(obj.attrs)),
        ]
    if isinstance(obj, pd.Series):
        return [
            "series",
            _text(obj.name),
            str(obj.dtype),
            _text(obj.index.tolist()),
            _canon_values(obj),
        ]
    if isinstance(obj, np.ndarray):
        return ["array", str(obj.dtype), _text(obj.tolist())]
    if isinstance(obj, dict):
        items = sorted(((_text(k), _canon(v)) for k, v in obj.items()), key=str)
        return ["dict", items]
    if isinstance(obj, (set, frozenset)):
        return ["set", sorted(_text(v) for v in obj)]
    if isinstance(obj, (list, tuple)):
        return [type(obj).__name__, [_canon(v) for v in obj]]
    return ["value", type(obj).__name__, _text(obj)]


def _canon_values(series: pd.Series) -> str:
    return _text(series.tolist())


def _warning_record(w: warnings.WarningMessage) -> list[str]:
    return [
        w.category.__name__,
        _ADDRESS.sub("0x", str(w.message)),
        os.path.basename(str(w.filename)),
    ]


def _call(cls: Any, method: str, df: pd.DataFrame) -> Any:
    """Description of ``cls(df, ...).method(...)``: result, warnings, state."""
    try:
        c_kwargs, m_kwargs, is_cm = harness.build_kwargs(cls, method, df)
    except harness.Unsupported as exc:
        return ["unsupported", str(exc)]
    out: list[Any] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if is_cm:
                obj = None
                result = getattr(cls, method)(df, **m_kwargs)
            else:
                obj = cls(df, **c_kwargs)
                out.append(["init warnings", [_warning_record(w) for w in caught]])
                result = getattr(obj, method)(**m_kwargs)
            out.append(["result", _canon(result)])
        except Exception as exc:  # noqa: BLE001 - the error is part of the result
            obj = None
            out.append(["error", type(exc).__name__, _ADDRESS.sub("0x", str(exc))])
        out.append(["warnings", [_warning_record(w) for w in caught]])
    if obj is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.append(["get_dataframe", _canon(obj.get_dataframe())])
            out.append(["repr", _text(obj)])
    return out


def _digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def pair_digests(pair: tuple[str, str]) -> dict[str, str]:
    """``{variant: digest}`` of one registry method on every input."""
    import ts_shape

    cls_name, method = pair
    cls = getattr(ts_shape, cls_name)
    return {
        name: _digest(_call(cls, method, frame.copy()))
        for name, frame in sorted(_variants().items())
    }


def digest_of(per_variant: dict[str, str]) -> str:
    return _digest(sorted(per_variant.items()))


def compute(pairs: list[tuple[str, str]]) -> dict[str, dict[str, str]]:
    jobs = harness._jobs()
    if jobs <= 1 or len(pairs) < 40:
        values = [pair_digests(p) for p in pairs]
    else:
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
            values = list(pool.map(pair_digests, pairs, chunksize=4))
    return {f"{c}.{m}": v for (c, m), v in zip(pairs, values, strict=True)}


def load() -> dict[str, dict[str, str]]:
    if not BASELINE.exists():
        return {}
    return dict(json.loads(BASELINE.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_results_and_warnings_match_the_pre_round18_baseline():
    pinned = load().get(PANDAS)
    if pinned is None:
        pytest.skip(f"no digests pinned for pandas {PANDAS}")
    entries = harness.entries()
    assert set(pinned) == {f"{c}.{m}" for c, m in entries}
    actual = compute(entries)
    changed = sorted(
        key
        for key, per_variant in actual.items()
        if digest_of(per_variant) != pinned[key]
    )
    assert not changed, f"results or warnings changed for {len(changed)}: {changed}"


if __name__ == "__main__":  # pragma: no cover - maintenance entry point
    results = compute(harness.entries())
    if "--dump" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--dump") + 1])
        out.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
        print(f"per-variant digests written: {out}")
    if "--update" in sys.argv:
        data = load()
        data[PANDAS] = {key: digest_of(v) for key, v in sorted(results.items())}
        BASELINE.write_text(
            json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"digests for pandas {PANDAS} written: {BASELINE}")
