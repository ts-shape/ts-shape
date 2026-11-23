"""Synthetic control loops: setpoint, process value and controller output.

Backs :func:`ts_shape.datasets.make_control_loops`. Each loop is a first
order plus dead time plant under a PI controller with its own tuning:
``good`` (settles quickly, little overshoot), ``oscillating`` (gain far too
high) or ``sluggish`` (gain far too low). The setpoint steps now and then.
Distractor signals look like parts of a loop but are none: a held setting
nobody follows, a drifting ambient temperature and a valve that moves on its
own. Signal names carry no role the profiler may use.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ts_shape._line_stops import _rows

#: quality -> (gain factor on the good Kp, integral time factor)
TUNINGS: dict[str, tuple[float, float]] = {
    "good": (1.0, 1.0),
    "oscillating": (7.0, 0.2),
    "sluggish": (0.3, 3.0),
}


def generate(
    *,
    loops: Sequence[str] = ("good", "oscillating", "sluggish"),
    duration: str = "8h",
    seed: int = 0,
    start: str | pd.Timestamp = "2026-01-05 06:00:00",
    sample_period: str = "2s",
    mean_step_minutes: float = 20.0,
    distractors: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the loops; see :func:`ts_shape.datasets.make_control_loops`."""
    unknown = [q for q in loops if q not in TUNINGS]
    if unknown:
        raise ValueError(
            f"make_control_loops: unknown loop quality {unknown}; "
            f"known: {list(TUNINGS)}"
        )
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    dt = pd.Timedelta(sample_period).total_seconds()
    n = int(pd.Timedelta(duration).total_seconds() // dt) + 1
    t = np.arange(n) * dt
    frames = []
    truth = []
    for i, quality in enumerate(loops):
        tag = f"tic{101 + i}"
        gain = float(rng.uniform(0.8, 1.5))  # PV units per % output
        tau = float(rng.uniform(40.0, 90.0))  # s
        dead = float(rng.uniform(4.0, 10.0))  # s
        base = gain * float(rng.uniform(35.0, 65.0))  # needs 35-65 % output
        # Good PI (lambda tuning, lambda = tau): Kp = tau / (K (lambda + dead)).
        kp_good = tau / (gain * (tau + dead))
        kp_factor, ti_factor = TUNINGS[quality]
        kp = kp_good * kp_factor
        ti = tau * ti_factor
        sp = np.empty(n)
        level = base
        next_step = float(rng.exponential(mean_step_minutes * 60.0)) + 300.0
        for k in range(n):
            if t[k] >= next_step:
                level = base + float(rng.choice([-1, 1])) * float(rng.uniform(5, 20))
                next_step = (
                    t[k] + 300.0 + float(rng.exponential(mean_step_minutes * 60.0))
                )
            sp[k] = level
        # Simulate at a fine step for stability, sample every dt.
        sub = max(1, int(np.ceil(dt / 0.5)))
        h = dt / sub
        delay = max(1, int(round(dead / h)))
        u_hist = np.full(delay, base / gain)
        y = base
        integ = base / gain
        pv = np.empty(n)
        op = np.empty(n)
        disturbance = 0.0
        for k in range(n):
            for _ in range(sub):
                e = sp[k] - y
                integ += kp * h / ti * e
                integ = float(np.clip(integ, 0.0, 100.0))  # anti-windup
                u = float(np.clip(kp * e + integ, 0.0, 100.0))
                u_hist = np.roll(u_hist, -1)
                u_hist[-1] = u
                disturbance += float(rng.normal(0.0, 0.02)) - 0.001 * disturbance
                y += h / tau * (gain * u_hist[0] - y) + h / tau * disturbance
            pv[k] = y + float(rng.normal(0.0, 0.05))
            op[k] = u
        frames.append(_rows(t0, f"plc:{tag}_sp", list(zip(t, sp)), "value_double"))
        frames.append(_rows(t0, f"plc:{tag}_pv", list(zip(t, pv)), "value_double"))
        frames.append(_rows(t0, f"plc:{tag}_op", list(zip(t, op)), "value_double"))
        truth.append(
            {
                "loop": tag,
                "quality": quality,
                "setpoint_uuid": f"plc:{tag}_sp",
                "actual_uuid": f"plc:{tag}_pv",
                "output_uuid": f"plc:{tag}_op",
                "kp": kp,
                "ti_seconds": ti,
                "plant_gain": gain,
                "plant_tau_seconds": tau,
                "dead_time_seconds": dead,
                "step_count": int(np.count_nonzero(np.diff(sp))),
            }
        )
    if distractors:
        held = np.repeat(rng.choice([80.0, 90.0, 100.0], size=n // 900 + 1), 900)[:n]
        ambient = 21.0 + np.cumsum(rng.normal(0.0, 0.01, n))
        valve = np.clip(50.0 + np.cumsum(rng.normal(0.0, 0.3, n)), 0.0, 100.0)
        frames.append(
            _rows(t0, "plc:line_speed_set", list(zip(t, held)), "value_double")
        )
        frames.append(
            _rows(t0, "plc:ambient_temp", list(zip(t, ambient)), "value_double")
        )
        frames.append(
            _rows(t0, "plc:bypass_valve", list(zip(t, valve)), "value_double")
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["is_delta"] = False
    frame = frame.sort_values(["systime", "uuid"], kind="stable").reset_index(drop=True)
    return frame, pd.DataFrame(truth)
