"""Synthetic continuous line with stops (no machine cycle).

Backs :func:`ts_shape.datasets.make_line_stops`. A line runs and counts
produced metres; it stops now and then for one of a few causes, each with
its own trace (an alarm that rises just before, or nothing). The run state
is a bool the profiler must discover; there is no cycle to cut.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

#: (cause, alarm signal or "", median stop minutes, share of stops)
LINE_CAUSES: tuple[tuple[str, str, float, float], ...] = (
    ("web_break", "alarm_break", 12.0, 0.4),
    ("die_pressure", "alarm_pressure", 25.0, 0.3),
    ("unknown", "", 6.0, 0.3),
)


def generate(
    *,
    duration: str = "48h",
    seed: int = 0,
    start: str | pd.Timestamp = "2026-01-05 06:00:00",
    mean_run_minutes: float = 90.0,
    count_period: str = "10s",
    prefix: str = "line",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the line; see :func:`ts_shape.datasets.make_line_stops`."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    total = pd.Timedelta(duration).total_seconds()
    step = pd.Timedelta(count_period).total_seconds()
    t = 0.0
    run_rows: list[tuple[float, bool]] = [(0.0, True)]
    alarm_rows: dict[str, list[tuple[float, bool]]] = {
        a: [(0.0, False)] for _, a, _, _ in LINE_CAUSES if a
    }
    count_rows: list[tuple[float, int]] = []
    count = 0
    truth = []
    shares = np.array([c[3] for c in LINE_CAUSES])
    while t < total:
        run = float(rng.exponential(mean_run_minutes * 60.0))
        end = min(t + run, total)
        for at in np.arange(t + step, end, step):
            count += 1
            count_rows.append((float(at), count))
        t = end
        if t >= total:
            break
        cause, alarm, median, _ = LINE_CAUSES[
            int(rng.choice(len(LINE_CAUSES), p=shares))
        ]
        stop = median * 60.0 * float(rng.lognormal(0.0, 0.3))
        if alarm:
            alarm_rows[alarm].append((t - float(rng.uniform(1.0, 5.0)), True))
            alarm_rows[alarm].append((t + stop, False))
        run_rows.append((t, False))
        run_rows.append((t + stop, True))
        truth.append({"start": t, "end": t + stop, "cause": cause})
        t += stop
    frames = [
        _rows(t0, f"{prefix}:running", run_rows, "value_bool"),
        _rows(t0, f"{prefix}:produced_count", count_rows, "value_integer"),
    ]
    for name, rows in alarm_rows.items():
        frames.append(_rows(t0, f"{prefix}:{name}", sorted(rows), "value_bool"))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["systime", "uuid"], kind="stable").reset_index(drop=True)
    truth_frame = pd.DataFrame(truth, columns=["start", "end", "cause"])
    for column in ("start", "end"):
        truth_frame[column] = t0 + pd.to_timedelta(truth_frame[column], unit="s")
    truth_frame["duration_seconds"] = (
        truth_frame["end"] - truth_frame["start"]
    ).dt.total_seconds()
    return frame, truth_frame


def _rows(
    t0: pd.Timestamp, uuid: str, rows: Sequence[tuple[float, Any]], column: str
) -> pd.DataFrame:
    seconds = np.round(np.array([r[0] for r in rows], dtype=float), 3)
    df = pd.DataFrame(
        {
            "systime": t0 + pd.to_timedelta(seconds, unit="s"),
            "uuid": uuid,
            "value_double": np.nan,
            "value_integer": pd.array([pd.NA] * len(rows), dtype="Int64"),
            "value_bool": pd.array([pd.NA] * len(rows), dtype="boolean"),
            "value_string": pd.Series([None] * len(rows), dtype=object),
            "is_delta": True,
        }
    )
    values = [r[1] for r in rows]
    if column == "value_integer":
        df[column] = pd.array(values, dtype="Int64")
    elif column == "value_bool":
        df[column] = pd.array(values, dtype="boolean")
    else:
        df[column] = values
    return df
