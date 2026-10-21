"""Scale benchmarks for ts-shape detectors (not part of the test suite).

Every (case, n_rows) pair runs in its own Python process, so peak memory
is measured per case and one case running out of memory does not take the
others down. Each process is capped with ``resource.setrlimit(RLIMIT_AS)``
and a wall-clock timeout.

Usage::

    python benchmarks/run.py                          # default cases, 1M rows
    python benchmarks/run.py --rows 1e6 1e7 --cases base machine_state.detect_run_idle
    python benchmarks/run.py --src /path/to/other/checkout/src --label before
    python benchmarks/run.py --list

The synthetic frame is long format (``systime``, ``uuid``, ``value_double``,
``value_integer``, ``value_bool``, ``value_string``) with five signals
interleaved on a 1 s grid: a run/idle state (bool), a production counter
(integer), two analog signals (double) and a part id (string). Every value
column is filled on every row.

Reported per case: ``seconds`` (the timed call only; generating the data and
any ``setup`` step are excluded), ``peak_rss_mb`` (high-water mark during the
timed call, reset through ``/proc/self/clear_refs`` on Linux), ``base_rss_mb``
(after generating the data, before the timed call) and their difference
``delta_rss_mb``. Results go to a JSON file and are printed as a
markdown table.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

STATE = "machine:state"
COUNTER = "machine:counter"
TEMP = "process:temp"
PRESSURE = "process:pressure"
PART = "part:id"
UUIDS = (STATE, COUNTER, TEMP, PRESSURE, PART)


def make_frame(
    n_rows: int, seed: int = 0, uuid_dtype: str = "default", slim: bool = False
) -> pd.DataFrame:
    """A long-format frame of ``n_rows`` rows, sorted by (systime, uuid).

    Args:
        n_rows: Total number of rows (spread evenly over the five signals).
        seed: Random seed.
        uuid_dtype: ``"default"`` (whatever pandas infers from Python strings:
            ``str`` on pandas 3, ``object`` on pandas 2), ``"object"`` or
            ``"category"``.
        slim: Only ``systime``, ``uuid``, ``value_double`` and ``value_bool``
            (for runs at 100M rows).
    """
    rng = np.random.default_rng(seed)
    k = len(UUIDS)
    step = np.arange(n_rows, dtype=np.int64) // k
    systime = pd.to_datetime("2025-01-01") + pd.to_timedelta(step, unit="s")
    names = np.array(sorted(UUIDS), dtype=object)
    uuid_values = names[np.arange(n_rows) % k]
    if uuid_dtype == "object":
        uuid: Any = pd.Series(uuid_values, dtype=object)
    elif uuid_dtype == "category":
        uuid = pd.Categorical(uuid_values)
    else:
        # The dtype pandas infers for Python strings, without building
        # n_rows Python objects first.
        uuid = pd.Series(names.tolist()).take(np.arange(n_rows) % k).array
    # Run/idle state with runs of 1..60 samples.
    lengths = rng.integers(1, 60, size=n_rows // 2 + 2)
    state = np.repeat(np.arange(len(lengths)) % 2 == 0, lengths)[:n_rows]
    walk = np.cumsum(rng.normal(0.0, 0.05, n_rows)) + 50.0
    if slim:
        return pd.DataFrame(
            {
                "systime": systime,
                "uuid": uuid,
                "value_double": walk,
                "value_bool": state,
            }
        )
    counter = np.cumsum(state.astype(np.int64))
    parts = np.array(["P-100", "P-200", "P-300"], dtype=object)
    part = parts[(step // 3600) % len(parts)]
    return pd.DataFrame(
        {
            "systime": systime,
            "uuid": uuid,
            "value_double": walk,
            "value_integer": counter,
            "value_bool": state,
            "value_string": part,
        }
    )


Case = tuple[Callable[[pd.DataFrame], Any], Callable[[Any], Any]]


def _cases() -> dict[str, Case]:
    """Headline cases: ``name -> (setup(df) -> ctx, run(ctx))``."""
    import ts_shape
    from ts_shape.utils.base import Base

    def ident(df: pd.DataFrame) -> pd.DataFrame:
        return df

    def run_idle_frame(df: pd.DataFrame) -> pd.DataFrame:
        return ts_shape.MachineStateEvents(df, STATE).detect_run_idle()

    return {
        "base": (ident, lambda df: Base(df)),
        "machine_state.__init__": (
            ident,
            lambda df: ts_shape.MachineStateEvents(df, STATE),
        ),
        "machine_state.detect_run_idle": (
            ident,
            lambda df: ts_shape.MachineStateEvents(df, STATE).detect_run_idle(),
        ),
        "micro_stop.detect_micro_stops": (
            ident,
            lambda df: ts_shape.MicroStopEvents(df, STATE).detect_micro_stops(),
        ),
        "part_production.production_by_part": (
            ident,
            lambda df: ts_shape.PartProductionEvents(df).production_by_part(
                PART, COUNTER
            ),
        ),
        "outlier.detect_outliers_zscore": (
            ident,
            lambda df: ts_shape.OutlierDetectionEvents(
                df, "value_double", signal_uuids=[TEMP]
            ).detect_outliers_zscore(),
        ),
        "signal_quality.detect_missing_data": (
            ident,
            lambda df: ts_shape.SignalQualityEvents(df, TEMP).detect_missing_data(),
        ),
        "rate_of_change.detect_rapid_change": (
            ident,
            lambda df: ts_shape.RateOfChangeEvents(df, TEMP).detect_rapid_change(
                threshold=0.1
            ),
        ),
        "stability_index.stability_score": (
            ident,
            lambda df: ts_shape.ProcessStabilityIndexEvents(
                df, PRESSURE
            ).stability_score(),
        ),
        "degradation.detect_trend_degradation": (
            ident,
            lambda df: ts_shape.DegradationDetectionEvents(
                df, TEMP
            ).detect_trend_degradation(),
        ),
        "threshold_monitoring.time_above_threshold": (
            ident,
            lambda df: ts_shape.ThresholdMonitoringEvents(
                df, TEMP
            ).time_above_threshold(threshold=50.0),
        ),
        "signals.profile": (ident, lambda df: ts_shape.profile(df)),
        "eventlog.to_event_log": (
            run_idle_frame,
            lambda out: ts_shape.to_event_log(
                out, detector="MachineStateEvents.detect_run_idle"
            ),
        ),
    }


DEFAULT_CASES = (
    "base",
    "machine_state.detect_run_idle",
    "micro_stop.detect_micro_stops",
    "part_production.production_by_part",
    "outlier.detect_outliers_zscore",
    "signal_quality.detect_missing_data",
    "rate_of_change.detect_rapid_change",
    "stability_index.stability_score",
    "degradation.detect_trend_degradation",
    "signals.profile",
    "eventlog.to_event_log",
)


def _reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS mark (Linux), so it covers the timed call."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def _rss_mb() -> float:
    """Peak resident set size since the last reset (MB).

    ``VmHWM`` from ``/proc/self/status`` where available, else the
    process-lifetime ``ru_maxrss`` (which then includes building the data).
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


def child(case: str, n_rows: int, uuid_dtype: str, slim: bool) -> dict[str, Any]:
    """Run one case in this process and return its measurements."""
    import warnings

    warnings.simplefilter("ignore")
    df = make_frame(n_rows, uuid_dtype=uuid_dtype, slim=slim)
    setup, run = _cases()[case]
    ctx = setup(df)
    base_rss = _current_rss_mb()
    reset = _reset_peak_rss()
    start = time.perf_counter()
    run(ctx)
    seconds = time.perf_counter() - start
    peak = _rss_mb()
    return {
        "case": case,
        "n_rows": n_rows,
        "status": "ok",
        "seconds": round(seconds, 3),
        "base_rss_mb": round(base_rss, 1),
        "peak_rss_mb": round(peak, 1),
        "delta_rss_mb": round(peak - base_rss, 1),
        "input_mb": round(df.memory_usage(deep=False).sum() / 2**20, 1),
        "peak_since_data": reset,
        "pandas": pd.__version__,
    }


def _limit(max_gb: float) -> Callable[[], None]:
    def apply() -> None:
        cap = int(max_gb * 2**30)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    return apply


def run_one(
    case: str,
    n_rows: int,
    *,
    src: Path,
    max_gb: float,
    timeout: float,
    uuid_dtype: str,
    slim: bool = False,
) -> dict[str, Any]:
    """Run one case in a fresh, memory-capped subprocess."""
    env = {**os.environ, "PYTHONPATH": str(src)}
    # One thread per BLAS pool, so concurrent benchmarks do not compete.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(var, "1")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        case,
        "--n",
        str(n_rows),
        "--uuid-dtype",
        uuid_dtype,
    ] + (["--slim"] if slim else [])
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limit(max_gb),
        )
    except subprocess.TimeoutExpired:
        return {"case": case, "n_rows": n_rows, "status": f"timeout>{timeout:.0f}s"}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode == 0 and lines:
        return dict(json.loads(lines[-1]))
    err = (proc.stderr.strip().splitlines() or ["?"])[-1]
    if "MemoryError" in proc.stderr or "bad_alloc" in proc.stderr:
        err = f"out of memory (cap {max_gb:g} GB)"
    return {
        "case": case,
        "n_rows": n_rows,
        "status": f"error: {err[:120]}",
        "wall_seconds": round(time.perf_counter() - start, 1),
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    """The results as a markdown table."""
    head = "| case | rows | status | seconds | peak RSS MB | delta RSS MB |"
    out = [head, "|---|---:|---|---:|---:|---:|"]
    for r in rows:
        out.append(
            f"| {r['case']} | {r['n_rows']:,} | {r['status']} | "
            f"{r.get('seconds', '')} | {r.get('peak_rss_mb', '')} | "
            f"{r.get('delta_rss_mb', '')} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--rows", nargs="+", type=float, default=[1e6])
    parser.add_argument("--max-gb", type=float, default=12.0, help="RLIMIT_AS cap")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds")
    parser.add_argument(
        "--src", type=Path, default=ROOT / "src", help="ts_shape source dir"
    )
    parser.add_argument(
        "--uuid-dtype", choices=("default", "object", "category"), default="default"
    )
    parser.add_argument(
        "--slim", action="store_true", help="only systime/uuid/value_double/bool"
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON output file")
    parser.add_argument("--label", default="", help="label stored in the JSON")
    parser.add_argument("--list", action="store_true", help="list the cases")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        result = child(args.child, args.n, args.uuid_dtype, args.slim)
        print(json.dumps(result), flush=True)
        return 0
    if args.list:
        sys.path.insert(0, str(args.src))
        print("\n".join(_cases()))
        return 0

    rows = []
    for n in (int(x) for x in args.rows):
        for case in args.cases:
            result = run_one(
                case,
                n,
                src=args.src,
                max_gb=args.max_gb,
                timeout=args.timeout,
                uuid_dtype=args.uuid_dtype,
                slim=args.slim,
            )
            result["label"] = args.label
            rows.append(result)
            print(json.dumps(result), file=sys.stderr, flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
