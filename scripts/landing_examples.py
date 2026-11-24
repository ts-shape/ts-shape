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

CYCLES = """\
import ts_shape

frame, _ = ts_shape.make_plc_cycles(n_cycles=300, seed=1)
profile = ts_shape.profile_cycles(frame)   # no signal mapping
register = profile.tables["contribution_register"]
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


def _step_label(event: str) -> str:
    # "plc:step=60" -> "step 60", "plc:camera_done↓" -> "camera done ↓"
    name = event.split(":", 1)[-1].replace("step=", "step ").replace("_", " ")
    for raw, nice in (("↓", " off"), ("↑", " on"), (">", " high"), ("<", " low")):
        name = name.replace(raw, nice)
    return " ".join(name.split())


def cycles_example() -> dict:
    s = _run(CYCLES)
    register, summary = s["register"], s["profile"].summary
    variant = summary["variants"][0]
    rows = register[register["variant"] == variant["variant"]]
    top = rows.sort_values("potential_seconds", ascending=False).head(6)
    return {
        "code": CYCLES,
        "variant": variant["variant"],
        "median_seconds": round(float(variant["median_seconds"]), 1),
        "target_seconds": round(float(variant["target_seconds"]), 1),
        "throughput_gain_pct": round(float(variant["throughput_gain_pct"]), 1),
        "cycle_count": int(summary["cycle_count"]),
        "segments": [
            {
                "label": f"{_step_label(a)} → {_step_label(b)}",
                "potential_seconds": round(float(p), 2),
                "median_seconds": round(float(m), 2),
                "p10_seconds": round(float(q), 2),
            }
            for a, b, p, m, q in zip(
                top["from_event"],
                top["to_event"],
                top["potential_seconds"],
                top["median_seconds"],
                top["p10_seconds"],
                strict=True,
            )
        ],
    }


def build() -> dict:
    return {"spc": spc_example(), "line": line_example(), "cycles": cycles_example()}


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
