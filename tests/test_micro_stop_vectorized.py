"""MicroStopEvents.detect_micro_stops matches its former per-row loop."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from ts_shape import MicroStopEvents


def _former_detect_micro_stops(
    ev: MicroStopEvents, max_duration: str = "30s", min_duration: str = "0s"
) -> pd.DataFrame:
    """The iterrows implementation replaced for speed, kept as the reference."""
    cols = [
        "start",
        "end",
        "duration_seconds",
        "uuid",
        "source_uuid",
        "duration",
        "preceding_run_duration",
    ]
    intervals = ev._intervalize()
    if intervals.empty:
        return pd.DataFrame(columns=cols)
    max_td = pd.to_timedelta(max_duration)
    min_td = pd.to_timedelta(min_duration)
    events: list[dict[str, Any]] = []
    for i, row in intervals.iterrows():
        if row["state"] != "idle":
            continue
        dur = row["duration"]
        if dur > max_td or dur < min_td:
            continue
        preceding_run = pd.Timedelta(0)
        if i > 0:
            prev = intervals.iloc[i - 1]
            if prev["state"] == "run":
                preceding_run = prev["duration"]
        events.append(
            {
                "start": row["start"],
                "end": row["end"],
                "duration_seconds": (row["end"] - row["start"]).total_seconds(),
                "uuid": ev.event_uuid,
                "source_uuid": ev.run_state_uuid,
                "duration": dur,
                "preceding_run_duration": preceding_run,
            }
        )
    return pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    unit = ["ns", "us", "ms"][seed % 3]
    tz = [None, "Europe/Berlin", "UTC"][seed % 3]
    steps = np.cumsum(rng.integers(1, 40, n))
    t = (pd.Timestamp("2024-03-31", tz=tz) + pd.to_timedelta(steps, unit="s")).as_unit(
        unit
    )
    values: Any = rng.random(n) < 0.6
    if seed % 5 == 0:
        values = values.astype(object)
        values[rng.random(n) < 0.1] = None
    return pd.DataFrame({"systime": t, "uuid": "run", "value_bool": values})


@pytest.mark.filterwarnings("ignore::ts_shape.errors.DataQualityWarning")
@pytest.mark.parametrize("seed", range(24))
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"max_duration": "20s"}, {"max_duration": "60s", "min_duration": "5s"}],
)
def test_matches_former_loop(seed: int, kwargs: dict[str, str]) -> None:
    ev = MicroStopEvents(_frame(seed), "run")
    got = ev.detect_micro_stops(**kwargs)
    expected = _former_detect_micro_stops(ev, **kwargs)
    pd.testing.assert_frame_equal(got, expected, check_exact=True)
