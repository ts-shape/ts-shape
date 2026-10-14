"""Registry-wide invariant harness for every ts-shape detector method.

Walks every ``(class, method)`` pair of
:data:`ts_shape.eventlog.taxonomy.REGISTRY`, synthesises constructor and
method arguments from parameter names, and checks three invariants on a
small synthetic plant dataset:

``empty``
    An empty frame (correct columns, zero rows) must not raise.
``tz``
    With tz-aware (Europe/Berlin) ``systime`` the call must succeed and no
    datetime column of the result may come back tz-naive (a stripped tz
    shifts every timestamp by the UTC offset).
``extra_uuid``
    Appending rows of an unrelated uuid must not change the result.
``shape``
    The result is a DataFrame with at least one column, carrying the
    canonical columns of the shape the taxonomy declares for the method
    (:mod:`ts_shape.events._output`): ``systime`` / ``uuid`` /
    ``source_uuid`` for points, ``start`` / ``end`` / ``duration_seconds`` /
    ``uuid`` / ``source_uuid`` for intervals, ``start`` / ``end`` /
    ``duration_seconds`` for summaries; ``duration_seconds`` is numeric.
``empty_schema``
    On empty input the result has the same columns as on real data, so code
    written against the normal output also works when nothing was found.
``dtypes``
    The same data with pandas nullable dtypes, with pyarrow dtypes, and with
    numbers stored as text gives the same result as with NumPy dtypes.

Each check yields a status string: ``"ok"``, ``"fail: <reason>"``, or
``"n/a: <reason>"`` (the harness could not build a valid call -- e.g. a
method that needs the output of another method). The current statuses are
pinned in ``baseline.json``: the test fails when a status changes in either
direction, so regressions are caught and fixes must update the baseline
(a ratchet). Regenerate with::

    python -m tests.invariants.harness --update
"""

from __future__ import annotations

import inspect
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import ts_shape
from ts_shape.eventlog.taxonomy import REGISTRY

BASELINE = Path(__file__).with_name("baseline.json")
BERLIN = "Europe/Berlin"
CHECKS = ("empty", "tz", "extra_uuid", "shape", "empty_schema", "dtypes")

CONT, CONT2, STATE, COUNTER, SETPOINT, TEXT = (
    "sig:cont",
    "sig:cont2",
    "sig:state",
    "sig:counter",
    "sig:setpoint",
    "sig:text",
)
NOISE = "zz:unrelated"

_VALUE_COLUMNS = ["value_double", "value_integer", "value_bool", "value_string"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def plant_frame(tz: str | None = None) -> pd.DataFrame:
    """Six signals over six hours at one-minute resolution."""
    rng = np.random.default_rng(42)
    n = 360
    t = pd.date_range("2024-03-30 22:00", periods=n, freq="1min")
    if tz is not None:
        t = t.tz_localize("UTC").tz_convert(tz)
    counter = np.cumsum(rng.integers(0, 3, n))
    counter[200:] -= counter[199]
    state = (np.arange(n) // 30) % 3 != 0
    rows = [
        pd.DataFrame(
            {"systime": t, "uuid": CONT, "value_double": 50 + rng.normal(0, 2, n)}
        ),
        pd.DataFrame(
            {"systime": t, "uuid": CONT2, "value_double": 20 + rng.normal(0, 1, n)}
        ),
        pd.DataFrame({"systime": t, "uuid": STATE, "value_bool": state}),
        pd.DataFrame({"systime": t, "uuid": COUNTER, "value_integer": counter}),
        pd.DataFrame(
            {
                "systime": t[::60],
                "uuid": SETPOINT,
                "value_double": np.tile([50.0, 55.0], 3),
            }
        ),
        pd.DataFrame(
            {
                "systime": t,
                "uuid": TEXT,
                "value_string": np.where(state, "Running", "Stopped"),
            }
        ),
    ]
    df = pd.concat(rows, ignore_index=True)
    for col in _VALUE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["is_delta"] = True
    return df.sort_values(["systime", "uuid"]).reset_index(drop=True)


def with_noise(df: pd.DataFrame) -> pd.DataFrame:
    noise = df[df["uuid"] == CONT].copy()
    noise["uuid"] = NOISE
    noise["value_double"] = noise["value_double"] * 7 + 1000
    return pd.concat([df, noise], ignore_index=True).sort_values(["systime", "uuid"])


# ---------------------------------------------------------------------------
# Argument synthesis
# ---------------------------------------------------------------------------

_UUID_HINTS = (
    (("setpoint", "sp_", "target_uuid", "reference"), SETPOINT),
    (
        (
            "counter",
            "count",
            "part",
            "scrap",
            "reject",
            "good",
            "total",
            "output",
            "production",
        ),
        COUNTER,
    ),
    (
        (
            "state",
            "run",
            "status",
            "mode",
            "alarm",
            "machine",
            "door",
            "running",
            "tower",
        ),
        STATE,
    ),
    (
        (
            "reason",
            "order",
            "batch",
            "recipe",
            "product",
            "operator",
            "string",
            "text",
            "id_uuid",
        ),
        TEXT,
    ),
)


def _uuid_for(name: str) -> str:
    low = name.lower()
    for keys, uuid in _UUID_HINTS:
        if any(k in low for k in keys):
            return uuid
    return CONT


#: Constructor arguments the harness cannot infer from names alone (one of
#: several optional parameters must be given).
INIT_OVERRIDES: dict[str, dict[str, Any]] = {
    "OutlierDetectionEvents": {"signal_uuids": [CONT]},
    "CapabilityTrendingEvents": {"upper_spec": 60.0, "lower_spec": 40.0},
    "ToleranceDeviationEvents": {
        "tolerance_column": "value_double",
        "actual_column": "value_double",
        "upper_tolerance_uuid": SETPOINT,
        "actual_uuid": CONT,
        "event_uuid": "evt",
    },
    "AssetHealthIndexEvents": {
        "asset_id": "asset:1",
        "signals": {CONT: "cont", CONT2: "cont2"},
    },
    "ProductionScheduleAdherenceEvents": {"schedule": lambda df: _schedule(df)},
    "EnvironmentalCompensationEvents": {"env_uuid": CONT2},
}

#: Method arguments that are themselves data (built from the input frame).
METHOD_OVERRIDES: dict[str, dict[str, Any]] = {
    "ReliabilityEvents": {"hours": [1.0, 10.0, 100.0]},
    "RootCauseCorrelationEvents": {
        "trigger_events": lambda df: _trigger_events(df),
        "candidate_uuids": [CONT, CONT2],
        "candidate_uuid": CONT,
    },
    "EnvironmentalCompensationEvents": {"valid_range": (15.0, 25.0)},
    "OperatingStateEvents": {"state": 1},
    "TimeToThresholdEvents": {
        "thresholds": {"warn": 55.0, "alarm": 60.0},
        "lower_bound": 40.0,
        "upper_bound": 60.0,
    },
}


def _span(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    times = pd.to_datetime(df["systime"]) if len(df) else plant_frame()["systime"]
    return times.min(), times.max()


def _schedule(df: pd.DataFrame) -> pd.DataFrame:
    t0, t1 = _span(df)
    mid = t0 + (t1 - t0) / 2
    return pd.DataFrame(
        {
            "start": [t0, mid],
            "end": [mid, t1],
            "planned_qty": [100, 100],
            "order_id": ["o1", "o2"],
        }
    )


def _trigger_events(df: pd.DataFrame) -> pd.DataFrame:
    t0, t1 = _span(df)
    times = [t0 + (t1 - t0) * f for f in (0.25, 0.5, 0.75)]
    return pd.DataFrame({"systime": times, "uuid": "trigger", "source_uuid": STATE})


def _resolve(value: Any, df: pd.DataFrame) -> Any:
    return value(df) if callable(value) else value


class Unsupported(Exception):
    """The harness cannot synthesise a valid value for a parameter."""


def _value_for(name: str, param: inspect.Parameter) -> Any:
    low = name.lower()
    ann = str(param.annotation).lower()
    if low in ("dataframe", "df", "data"):
        raise Unsupported("dataframe handled by caller")
    if "dict" in ann or "mapping" in ann:
        if "factor" in low:
            return {CONT: 0.4, CONT2: 0.2}
        if "level" in low:
            return {"warn": 52.0, "alarm": 55.0}
        if "uuid" in low or "station" in low:
            return {STATE: "S1", CONT: "S2"}  # uuid -> station name
        raise Unsupported(f"no rule for mapping parameter {name!r}")
    if "uuid" in low:
        if "list" in ann or low.endswith("uuids") or "sequence" in ann:
            return [CONT, CONT2]
        if "part_id" in low or "order" in low:
            return TEXT
        if "trigger" in low or "cycle" in low:
            return STATE
        return _uuid_for(low)
    if low.endswith("_column") or low == "column":
        return "value_double"
    # Numeric thresholds before durations: "threshold" contains "hold".
    if "threshold" in low or "limit" in low or "delta" in low:
        return 1.0
    if "cycle_time" in low and "float" in ann:
        return 30.0
    if low in ("window", "freq", "frequency", "period", "resample", "baseline_window"):
        return "1h"
    if any(k in low for k in ("duration", "hold", "gap", "lookahead", "time")):
        return "5min"
    if low.startswith(("min_", "max_")):
        return 1.0
    if low in ("high", "upper", "usl", "upper_spec", "target", "target_value"):
        return 55.0
    if low in ("low", "lower", "lsl", "lower_spec"):
        return 45.0
    if any(k in low for k in ("rate", "factor", "level", "value", "sigma")):
        return 1.0
    raise Unsupported(f"no rule for required parameter {name!r}")


def _required(fn: Any, skip: int) -> list[tuple[str, inspect.Parameter]]:
    params = list(inspect.signature(fn).parameters.items())[skip:]
    return [
        (n, p)
        for n, p in params
        if p.default is inspect.Parameter.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def build_call(cls: Any, method: str, df: pd.DataFrame) -> Any:
    """Return a zero-argument callable that runs ``cls(df, ...).method(...)``."""
    raw = inspect.getattr_static(cls, method)
    is_cm = isinstance(raw, (classmethod, staticmethod))
    fn = getattr(cls, method)

    overrides = METHOD_OVERRIDES.get(cls.__name__, {})
    m_kwargs = {}
    m_params = _required(fn, 0 if is_cm else 1)
    if is_cm:
        if not m_params or m_params[0][0] not in ("dataframe", "df", "data"):
            raise Unsupported("classmethod without a dataframe first parameter")
        m_params = m_params[1:]
    for name, param in m_params:
        m_kwargs[name] = (
            _resolve(overrides[name], df)
            if name in overrides
            else _value_for(name, param)
        )

    if is_cm:
        return lambda: fn(df, **m_kwargs)

    c_kwargs = {
        k: _resolve(v, df) for k, v in INIT_OVERRIDES.get(cls.__name__, {}).items()
    }
    for name, param in _required(cls.__init__, 2):
        if name not in c_kwargs:
            c_kwargs[name] = _value_for(name, param)

    def call() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = cls(df, **c_kwargs)
            return getattr(obj, method)(**m_kwargs)

    return call


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _short(exc: BaseException) -> str:
    msg = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {msg[:80]}"


def _run(cls: Any, method: str, df: pd.DataFrame) -> tuple[str, Any]:
    try:
        call = build_call(cls, method, df)
    except Unsupported as exc:
        return f"n/a: {exc}", None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", call()
    except ImportError as exc:
        return f"n/a: optional dependency missing ({exc.name or exc})", None
    except Exception as exc:  # noqa: BLE001 - the harness records every failure
        return f"fail: {_short(exc)}", None


def _frames(result: Any) -> list[pd.DataFrame]:
    if isinstance(result, pd.DataFrame):
        return [result]
    if isinstance(result, pd.Series):
        return [result.to_frame()]
    if isinstance(result, dict):
        return [v for v in result.values() if isinstance(v, pd.DataFrame)]
    return []


#: Calendar-date columns are legitimately tz-naive.
_DATE_COLUMNS = {"date", "production_day", "day"}


def _naive_datetime_columns(result: Any) -> list[str]:
    bad = []
    for frame in _frames(result):
        for col in frame.columns:
            if str(col) in _DATE_COLUMNS:
                continue
            s = frame[col]
            if (
                pd.api.types.is_datetime64_any_dtype(s)
                and getattr(s.dt, "tz", None) is None
            ):
                bad.append(str(col))
            elif s.dtype == object and len(s.dropna()):
                first = s.dropna().iloc[0]
                if isinstance(first, pd.Timestamp) and first.tzinfo is None:
                    bad.append(str(col))
        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is None:
            bad.append("<index>")
    return bad


def _same(a: Any, b: Any) -> bool:
    fa, fb = _frames(a), _frames(b)
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb, strict=True):
        x = x.drop(columns=[c for c in x.columns if "uuid" in str(c)], errors="ignore")
        y = y.drop(columns=[c for c in y.columns if "uuid" in str(c)], errors="ignore")
        x, y = x.reset_index(drop=True), y.reset_index(drop=True)
        # Timestamps computed from floats may differ by float noise (ns).
        for col in x.columns:
            if col in y.columns and pd.api.types.is_datetime64_any_dtype(x[col]):
                if pd.api.types.is_datetime64_any_dtype(y[col]) and len(x) == len(y):
                    close = (x[col] - y[col]).abs() <= pd.Timedelta("1us")
                    if bool((close | (x[col].isna() & y[col].isna())).all()):
                        y[col] = x[col]
        try:
            pd.testing.assert_frame_equal(
                x.reset_index(drop=True),
                y.reset_index(drop=True),
                check_dtype=False,
                check_like=True,
                atol=1e-9,
            )
        except (AssertionError, TypeError, ValueError):
            return False
    if not fa and a != b:  # scalars / dicts of scalars
        try:
            return bool(a == b)
        except Exception:  # noqa: BLE001
            return False
    return True


_REQUIRED = {
    "point": ("systime", "uuid", "source_uuid"),
    "interval": ("start", "end", "duration_seconds", "uuid", "source_uuid"),
    "summary": ("start", "end", "duration_seconds"),
}


def _shape_problem(result: Any, shape: str) -> str | None:
    if not isinstance(result, pd.DataFrame):
        return f"returns {type(result).__name__}, not a DataFrame"
    if len(result.columns) == 0:
        return "DataFrame without columns"
    missing = [c for c in _REQUIRED.get(shape, ()) if c not in result.columns]
    if missing:
        return f"{shape} output missing {missing}"
    if "duration_seconds" in result.columns and len(result):
        if not pd.api.types.is_numeric_dtype(result["duration_seconds"]):
            return "duration_seconds is not numeric"
    return None


def dtype_variants(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The same data in the dtype backends users load it with."""
    text = df.copy()
    for col in ("value_double", "value_integer"):
        text[col] = [None if pd.isna(v) else repr(v) for v in df[col]]
    return {
        "nullable": df.convert_dtypes(dtype_backend="numpy_nullable"),
        "pyarrow": df.convert_dtypes(dtype_backend="pyarrow"),
        "text numbers": text,
    }


def check(cls_name: str, method: str) -> dict[str, str]:
    cls = getattr(ts_shape, cls_name)
    base = plant_frame()
    status: dict[str, str] = {}

    empty = base.iloc[0:0]
    status["empty"], empty_result = _run(cls, method, empty)

    tz_status, tz_result = _run(cls, method, plant_frame(BERLIN))
    if tz_status == "ok":
        naive = _naive_datetime_columns(tz_result)
        tz_status = f"fail: tz stripped in {sorted(set(naive))[:4]}" if naive else "ok"
    status["tz"] = tz_status

    plain_status, plain = _run(cls, method, base)
    shape = REGISTRY[(cls_name, method)].shape
    if plain_status != "ok":
        status["shape"] = status["empty_schema"] = plain_status
    else:
        problem = _shape_problem(plain, shape)
        status["shape"] = f"fail: {problem}" if problem else "ok"
        if status["empty"] != "ok":
            status["empty_schema"] = status["empty"]
        elif not isinstance(plain, pd.DataFrame) or not isinstance(
            empty_result, pd.DataFrame
        ):
            status["empty_schema"] = "fail: not a DataFrame"
        else:
            extra = sorted(
                set(map(str, plain.columns)) - set(map(str, empty_result.columns))
            )
            status["empty_schema"] = (
                f"fail: empty result lacks {extra[:4]}" if extra else "ok"
            )
    if plain_status != "ok":
        status["extra_uuid"] = plain_status
    else:
        noisy_status, noisy = _run(cls, method, with_noise(base))
        if noisy_status != "ok":
            status["extra_uuid"] = noisy_status
        else:
            status["extra_uuid"] = (
                "ok" if _same(plain, noisy) else "fail: result changed"
            )
    if plain_status != "ok":
        status["dtypes"] = plain_status
    else:
        status["dtypes"] = "ok"
        for name, variant in dtype_variants(base).items():
            v_status, v_result = _run(cls, method, variant)
            if v_status != "ok":
                status["dtypes"] = f"{v_status} ({name})"
                break
            if not _same(plain, v_result):
                status["dtypes"] = f"fail: result changed ({name})"
                break
    return status


def entries() -> list[tuple[str, str]]:
    return sorted(REGISTRY)


def run_all() -> dict[str, dict[str, str]]:
    return {f"{c}.{m}": check(c, m) for c, m in entries()}


def load_baseline() -> dict[str, dict[str, str]]:
    if not BASELINE.exists():
        return {}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def _summary(results: dict[str, dict[str, str]]) -> str:
    lines = []
    for chk in CHECKS:
        values = [r[chk] for r in results.values()]
        ok = sum(v == "ok" for v in values)
        fail = sum(v.startswith("fail") for v in values)
        na = sum(v.startswith("n/a") for v in values)
        lines.append(f"{chk:>11}: {ok} ok, {fail} fail, {na} n/a")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - maintenance entry point
    results = run_all()
    print(_summary(results))
    if "--update" in sys.argv:
        BASELINE.write_text(
            json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"baseline written: {BASELINE}")
