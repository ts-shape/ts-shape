"""Data for the charts on the docs landing page, computed with ts-shape.

Run from the repository root after changing an example::

    python scripts/landing_examples.py

Each example is a short code snippet. The script runs exactly that code on
ts-shape's simulated data and stores the snippet together with the numbers
the chart needs in ``website/src/data/landing-examples.json``, so the page
shows the code that produced each chart. The file is committed: the site
build does not run Python for it.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # the sources next to this script

OUT = ROOT / "website" / "src" / "data" / "landing-examples.json"

SPC = """\
import pandas as pd
import ts_shape
from ts_shape import StatisticalProcessControlEvents

df = pd.concat([
    ts_shape.make_timeseries(["oven:temp_ref"], n_points=200, seed=1),
    ts_shape.make_timeseries(["oven:temp"], n_points=480, drift=1.5, seed=2),
])
spc = StatisticalProcessControlEvents(df, value_column="value_double",
    tolerance_uuid="oven:temp_ref", actual_uuid="oven:temp", event_uuid="oven:spc")
limits = spc.control_limits()
violations = spc.detect_violations()
"""

LINE = """\
import ts_shape
from ts_shape import MachineStateEvents, OEEEvents

df, _ = ts_shape.make_line_stops(duration="18h", seed=3)
states = MachineStateEvents(df, run_state_uuid="line:running").detect_run_idle()
oee = OEEEvents(df).oee(run_state_uuid="line:running",
    counter_uuid="line:produced_count", ideal_cycle_time=8.0)
"""

LOOPS = """\
import ts_shape
from ts_shape import ControlLoopProfileEvents

df, _ = ts_shape.make_control_loops(duration="8h", seed=1)
loops = ControlLoopProfileEvents(df)   # finds the SP / PV / OP triples itself
register = loops.loop_register()       # one row per loop, with a diagnosis
"""

ENERGY = """\
import ts_shape
from ts_shape import EnergyEfficiencyEvents
from ts_shape.utils.counters import counter_increments

df, _ = ts_shape.make_plc_cycles(n_cycles=700, seed=1, energy=True)
meter = df["uuid"] == "plc:energy_kwh"   # a cumulative kWh meter
df.loc[meter, "value_double"] = counter_increments(df.loc[meter, "value_double"])
sec = EnergyEfficiencyEvents(df).specific_energy_consumption(
    "plc:energy_kwh", "plc:part_counter", window="30min")
"""

DRIVERS = """\
import ts_shape
from ts_shape import CrossSignalAnalytics

df, _ = ts_shape.make_control_loops(loops=("good",), duration="8h", seed=1)
signals = CrossSignalAnalytics(df)    # long format in, one column per signal
tests = {c: signals.granger_causality(c, "plc:tic101_pv", max_lag=10)
         for c in signals.signals if c != "plc:tic101_pv"}
"""


def _run(code: str) -> dict:
    scope: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "<landing example>", "exec"), scope)  # noqa: S102
    return scope


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


def spc_example() -> dict:
    s = _run(SPC)
    df, limits, violations = s["df"], s["limits"].iloc[0], s["violations"]
    series = df[df["uuid"] == "oven:temp"].sort_values("systime")
    points = [
        [_iso(t), round(float(v), 3)]
        for t, v in zip(series["systime"], series["value_double"], strict=True)
    ]
    hits = violations.sort_values("systime")
    out_of_limits = hits[hits["rule"] == "rule_1"]
    run_rules = hits[hits["rule"] != "rule_1"].drop_duplicates("systime")
    # Stretches flagged by the run rules (2, 3, 5, 6, ...): consecutive flagged
    # samples, allowing gaps of up to 5 samples, drawn as bands.
    step = series["systime"].diff().median()
    spans: list[list[pd.Timestamp]] = []
    for t in run_rules["systime"]:
        if spans and t - spans[-1][1] <= 6 * step:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return {
        "code": SPC,
        "points": points,
        "mean": round(float(limits["mean"]), 3),
        "ucl": round(float(limits["3sigma_upper"]), 3),
        "lcl": round(float(limits["3sigma_lower"]), 3),
        "out_of_limits": [
            [_iso(t), round(float(v), 3)]
            for t, v in zip(
                out_of_limits["systime"], out_of_limits["value"], strict=True
            )
        ],
        "run_rules": [
            [_iso(t), round(float(v), 3), r]
            for t, v, r in zip(
                run_rules["systime"], run_rules["value"], run_rules["rule"], strict=True
            )
        ],
        "run_spans": [[_iso(a), _iso(b)] for a, b in spans],
        "first_run_rule": _iso(run_rules["systime"].min()),
        "rule_counts": {k: int(v) for k, v in hits["rule"].value_counts().items()},
    }


def line_example() -> dict:
    s = _run(LINE)
    states, oee = s["states"], s["oee"].iloc[0]
    idle = states[states["state"] == "idle"]
    return {
        "code": LINE,
        "intervals": [
            [_iso(a), _iso(b), st]
            for a, b, st in zip(
                states["start"], states["end"], states["state"], strict=True
            )
            if b > a
        ],
        "oee_pct": round(float(oee["oee_pct"]), 1),
        "availability_pct": round(float(oee["availability_pct"]), 1),
        "performance_pct": round(float(oee["performance_pct"]), 1),
        "quality_pct": round(float(oee["quality_pct"]), 1),
        "stop_count": int(len(idle)),
        "stopped_minutes": round(float(idle["duration_seconds"].sum()) / 60, 1),
        "longest_stop_minutes": round(float(idle["duration_seconds"].max()) / 60, 1),
    }


def loops_example() -> dict:
    s = _run(LOOPS)
    df, register = s["df"], s["register"]
    steps = s["loops"].loop_steps()
    loops = []
    for row in register.sort_values("loop").itertuples():
        sp = df[df["uuid"] == row.setpoint_uuid].set_index("systime")["value_double"]
        pv = df[df["uuid"] == row.actual_uuid].set_index("systime")["value_double"]
        changes = sp[sp.diff().abs() > 0]
        # the largest step with 10 quiet minutes after it
        best = None
        for t, new in changes.items():
            after = changes[(changes.index > t)]
            if len(after) and after.index[0] - t < pd.Timedelta("10min"):
                continue
            old = float(sp[sp.index < t].iloc[-1])
            if best is None or abs(new - old) > abs(best[2] - best[1]):
                best = (t, old, float(new))
        t, old, new = best
        window = pv[
            (pv.index >= t - pd.Timedelta("1min"))
            & (pv.index <= t + pd.Timedelta("10min"))
        ]
        window = window.iloc[::3]
        step = steps[(steps["loop"] == row.loop) & (steps["start"] == t)]
        loops.append(
            {
                "name": row.loop.split(":", 1)[-1].removesuffix("_sp").upper(),
                "diagnosis": row.diagnosis,
                "steps": int(row.step_count),
                "overshoot_pct": round(float(row.median_overshoot_pct), 1),
                "settle_seconds": round(float(row.median_settle_seconds)),
                "settled_pct": round(float(row.settled_share_ratio) * 100),
                "step_overshoot_pct": (
                    round(float(step["overshoot_pct"].iloc[0]), 1)
                    if len(step) and pd.notna(step["overshoot_pct"].iloc[0])
                    else None
                ),
                # seconds since the step, response in % of the step (0 old, 100 new)
                "response": [
                    [
                        round((ts - t).total_seconds()),
                        round((float(v) - old) / (new - old) * 100, 1),
                    ]
                    for ts, v in window.items()
                ],
            }
        )
    return {"code": LOOPS, "loops": loops}


def energy_example() -> dict:
    s = _run(ENERGY)
    sec = s["sec"]
    sec = sec[sec["total_output"] >= 15]  # drop the half-empty last window
    values = sec["sec"].astype(float)
    best = float(values.min())
    return {
        "code": ENERGY,
        "windows": [
            [_iso(t), round(float(v), 3), int(n), round(float(e), 2)]
            for t, v, n, e in zip(
                sec["start"],
                sec["sec"],
                sec["total_output"],
                sec["total_energy"],
                strict=True,
            )
        ],
        "median_kwh_per_part": round(float(values.median()), 3),
        "best_kwh_per_part": round(best, 3),
        "worst_kwh_per_part": round(float(values.max()), 3),
        "worst_window": _iso(sec.loc[values.idxmax(), "start"]),
        "parts": int(sec["total_output"].sum()),
        "kwh": round(float(sec["total_energy"].sum()), 1),
        # energy above the best window's rate, for the parts made
        "excess_kwh": round(
            float((sec["total_energy"] - best * sec["total_output"]).sum()), 1
        ),
    }


def drivers_example() -> dict:
    s = _run(DRIVERS)
    names = {
        "plc:tic101_op": "Valve output (OP)",
        "plc:tic101_sp": "Setpoint (SP)",
        "plc:ambient_temp": "Ambient temperature",
        "plc:bypass_valve": "Bypass valve",
        "plc:line_speed_set": "Line speed setting",
    }
    rows = [
        {
            "uuid": c,
            "label": names.get(c, c),
            "f_statistic": round(float(r["f_statistic"]), 1),
            "p_value_adjusted": float(f"{float(r['p_value_adjusted']):.3g}"),
            "lag": int(r["optimal_lag"]),
            "causal": bool(r["is_causal"]),
        }
        for c, r in s["tests"].items()
    ]
    rows.sort(key=lambda r: r["f_statistic"], reverse=True)
    return {"code": DRIVERS, "target": "plc:tic101_pv", "signals": rows}


def build() -> dict:
    return {
        "line": line_example(),
        "loops": loops_example(),
        "spc": spc_example(),
        "energy": energy_example(),
        "drivers": drivers_example(),
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
