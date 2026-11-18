"""Synthetic PLC with a step chain, handshakes and known cycle disturbances.

Backs :func:`ts_shape.datasets.make_plc_cycles`. One machine runs a step
chain ``10 .. 80``; every cycle drives the handshake signals, an analog
pressure and a part counter the way a PLC historian records them (one row per
change for discrete signals, one row per ``sample_period`` for the analog and
the heartbeat). The disturbances whose effect cycle time profiling must find
are injected on purpose and returned as ground truth:

* camera NOK with re-inspection (a long cycle, extra pulses inside step 50),
* aborted cycles that skip steps 40-70 (a short cycle),
* breaks during which the chain waits in step 10 (a pause),
* a second article that is slower in step 40 (a variant),
* a bimodal wait in step 60 (a conveyor that is not always ready).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

#: Nominal step durations in seconds (sum: 45 s).
STEP_SECONDS: dict[int, float] = {
    10: 2.0,  # load
    20: 3.0,  # clamp
    30: 4.0,  # robot handshake
    40: 12.0,  # process
    50: 5.0,  # camera inspection
    60: 8.0,  # process 2 (+ bimodal conveyor wait)
    70: 3.0,  # unclamp
    80: 8.0,  # unload
}
#: Extra seconds of step 40 for the slow article.
SLOW_ARTICLE_SECONDS = 6.0
#: Seconds one camera re-inspection adds.
RETRY_SECONDS = 12.0
#: Extra seconds of step 60 when the conveyor is not ready.
CONVEYOR_WAIT_SECONDS = 2.5


def generate(
    *,
    n_cycles: int = 300,
    seed: int = 0,
    start: str | pd.Timestamp = "2026-01-05 06:00:00",
    articles: Sequence[str] = ("A", "B"),
    article_block: int = 60,
    retry_probability: float = 0.06,
    abort_probability: float = 0.02,
    pause_every: int = 100,
    pause_minutes: float = 15.0,
    sample_period: str = "1s",
    heartbeat: bool = True,
    prefix: str = "plc",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the PLC; see :func:`ts_shape.datasets.make_plc_cycles`."""
    if n_cycles < 1:
        raise ValueError("n_cycles must be at least 1")
    if not articles:
        raise ValueError("articles must name at least one article")
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    period = pd.to_timedelta(sample_period).total_seconds()
    if period <= 0:
        raise ValueError("sample_period must be positive")

    step_rows: list[tuple[float, int]] = []
    bool_rows: dict[str, list[tuple[float, bool]]] = {
        "clamp_closed": [],
        "robot_ready": [],
        "camera_done": [],
        "camera_nok": [],
    }
    counter_rows: list[tuple[float, int]] = []
    article_rows: list[tuple[float, str]] = []
    clamp_spans: list[tuple[float, float]] = []
    truth: list[dict[str, object]] = []

    def pulse(name: str, at: float, width: float = 0.3) -> None:
        bool_rows[name].append((at, True))
        bool_rows[name].append((at + width, False))

    def article_of(p: int) -> int:
        return (p // max(article_block, 1)) % len(articles)

    t = 0.0
    counter = 1000
    passes: list[dict[str, object]] = []
    boundaries: list[float] = []
    article_rows.append((0.0, str(articles[article_of(0)])))
    # One pass runs the chain 10 .. 80. The cycle boundary is the counter
    # increment in step 80 (unload): cycle i runs from boundary i to boundary
    # i + 1, so its disturbances are those of pass i + 1.
    for p in range(n_cycles + 1):
        index = article_of(p)
        pause = pause_every > 0 and p > 1 and (p - 1) % pause_every == 0
        retries = 0
        if rng.random() < retry_probability:
            retries = 1 if rng.random() < 0.7 else 2
        aborted = (not retries) and (not pause) and rng.random() < abort_probability
        noise = {s: float(rng.lognormal(0.0, 0.04)) for s in STEP_SECONDS}

        # Step 10: load (a break waits here).
        step_rows.append((t, 10))
        wait = 0.0
        if pause:
            wait = pause_minutes * 60.0 * float(rng.uniform(0.8, 1.2))
        t += STEP_SECONDS[10] * noise[10] + wait
        # Step 20: clamp.
        step_rows.append((t, 20))
        bool_rows["clamp_closed"].append((t + 0.5, True))
        clamp_on = t + 0.5
        t += STEP_SECONDS[20] * noise[20]
        # Step 30: robot handshake.
        step_rows.append((t, 30))
        bool_rows["robot_ready"].append((t + 1.0, True))
        t += STEP_SECONDS[30] * noise[30]
        if not aborted:
            # Step 40: process (every second article is slower).
            step_rows.append((t, 40))
            bool_rows["robot_ready"].append((t + 0.5, False))
            extra = SLOW_ARTICLE_SECONDS if index % 2 == 1 else 0.0
            t += (STEP_SECONDS[40] + extra) * noise[40]
            # Step 50: camera, with re-inspection after a NOK.
            step_rows.append((t, 50))
            shot = t + STEP_SECONDS[50] * noise[50] - 0.6
            for _ in range(retries):
                pulse("camera_done", shot)
                pulse("camera_nok", shot + 0.1)
                shot += RETRY_SECONDS
            pulse("camera_done", shot)
            t += STEP_SECONDS[50] * noise[50] + retries * RETRY_SECONDS
            # Step 60: process 2, the conveyor is not always ready.
            step_rows.append((t, 60))
            conveyor = CONVEYOR_WAIT_SECONDS if rng.random() < 0.4 else 0.0
            t += STEP_SECONDS[60] * noise[60] + conveyor
            # Step 70: unclamp.
            step_rows.append((t, 70))
            bool_rows["clamp_closed"].append((t + 0.5, False))
            clamp_spans.append((clamp_on, t + 0.5))
            t += STEP_SECONDS[70] * noise[70]
        else:
            # Aborted: the robot hands back, the clamp opens, unload at once.
            bool_rows["robot_ready"].append((t, False))
            bool_rows["clamp_closed"].append((t + 0.2, False))
            clamp_spans.append((clamp_on, t + 0.2))
            t += 0.4
        # Step 80: unload; the counter counts the part (cycle boundary).
        step_rows.append((t, 80))
        counter += 1
        counter_rows.append((t + 1.0, counter))
        boundaries.append(t + 1.0)
        nxt = article_of(p + 1)
        if nxt != index:
            # The next order is loaded before the part is counted.
            article_rows.append((t + 0.5, str(articles[nxt])))
        kind = (
            "pause"
            if pause
            else "long" if retries else "short" if aborted else "normal"
        )
        passes.append(
            {
                "kind": kind,
                "article": str(articles[index]),
                "retry_count": retries,
                "pause_seconds": wait,
            }
        )
        t += STEP_SECONDS[80] * noise[80]

    for i in range(n_cycles):
        truth.append(
            {"start": boundaries[i], "end": boundaries[i + 1], **passes[i + 1]}
        )

    end_t = t
    frames = [
        _discrete(t0, f"{prefix}:step", step_rows, "value_integer"),
        _discrete(t0, f"{prefix}:part_counter", counter_rows, "value_integer"),
        _discrete(t0, f"{prefix}:article", article_rows, "value_string"),
    ]
    for name, rows in bool_rows.items():
        initial = [(0.0, False)] if rows and rows[0][0] > 0 else []
        frames.append(_discrete(t0, f"{prefix}:{name}", initial + rows, "value_bool"))

    grid = np.arange(0.0, end_t, period)
    closed = np.zeros(len(grid), dtype=bool)
    for on, off in clamp_spans:
        closed |= (grid >= on) & (grid < off)
    pressure = np.where(closed, 180.0, 5.0) + rng.normal(0.0, 2.0, len(grid))
    frames.append(_sampled(t0, f"{prefix}:pressure", grid, pressure, "value_double"))
    if heartbeat:
        alive = (np.floor(grid / period).astype(np.int64) % 2) == 0
        frames.append(_sampled(t0, f"{prefix}:heartbeat", grid, alive, "value_bool"))

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(["systime", "uuid"], kind="stable").reset_index(drop=True)

    truth_frame = pd.DataFrame(truth)
    for col in ("start", "end"):
        seconds = np.round(truth_frame[col].to_numpy(dtype=float), 3)
        truth_frame[col] = t0 + pd.to_timedelta(seconds, unit="s")
    truth_frame["duration_seconds"] = (
        truth_frame["end"] - truth_frame["start"]
    ).dt.total_seconds()
    truth_frame = truth_frame[
        [
            "start",
            "end",
            "duration_seconds",
            "kind",
            "article",
            "retry_count",
            "pause_seconds",
        ]
    ]
    return frame, truth_frame


_COLUMNS = ("value_double", "value_integer", "value_bool", "value_string")


def _empty_values(n: int) -> dict[str, object]:
    return {
        "value_double": np.full(n, np.nan),
        "value_integer": pd.array([pd.NA] * n, dtype="Int64"),
        "value_bool": pd.array([pd.NA] * n, dtype="boolean"),
        "value_string": pd.Series([None] * n, dtype=object),
    }


def _frame(
    times: pd.DatetimeIndex, uuid: str, values: object, column: str
) -> pd.DataFrame:
    n = len(times)
    data: dict[str, object] = {"systime": times, "uuid": uuid}
    data.update(_empty_values(n))
    if column == "value_integer":
        data[column] = pd.array(list(values), dtype="Int64")  # type: ignore[call-overload]
    elif column == "value_bool":
        data[column] = pd.array(list(values), dtype="boolean")  # type: ignore[call-overload]
    elif column == "value_string":
        data[column] = pd.Series(list(values), dtype=object)  # type: ignore[call-overload]
    else:
        data[column] = np.asarray(values, dtype=float)
    data["is_delta"] = column != "value_double"
    frame = pd.DataFrame(data)
    return frame[["systime", "uuid", *_COLUMNS, "is_delta"]]


def _discrete(
    t0: pd.Timestamp, uuid: str, rows: Sequence[tuple[float, object]], column: str
) -> pd.DataFrame:
    rows = sorted(rows, key=lambda r: r[0])
    seconds = np.round(np.array([r[0] for r in rows], dtype=float), 3)
    times = pd.DatetimeIndex(t0 + pd.to_timedelta(seconds, unit="s"))
    return _frame(times, uuid, [r[1] for r in rows], column)


def _sampled(
    t0: pd.Timestamp, uuid: str, grid: np.ndarray, values: np.ndarray, column: str
) -> pd.DataFrame:
    times = pd.DatetimeIndex(t0 + pd.to_timedelta(np.round(grid, 3), unit="s"))
    frame = _frame(times, uuid, values, column)
    frame["is_delta"] = False
    return frame
