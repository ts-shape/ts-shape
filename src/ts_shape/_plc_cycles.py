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
#: Stop causes of ``stop_causes``: (name, trace, median stop seconds). The
#: trace is the signal the cause leaves (``alarm_1``, ``alarm_2``, a held
#: door + manual mode, or nothing); a cause without a trace is unexplained.
STOP_CAUSES: tuple[tuple[str, str, float], ...] = (
    ("jam", "alarm_1", 120.0),
    ("material_low", "alarm_2", 300.0),
    ("manual_fix", "door", 200.0),
    ("unknown", "", 90.0),
)
#: Changeover steps of ``changeover_steps``: (name, median seconds).
CHANGEOVER_STEPS: tuple[tuple[str, float], ...] = (
    ("unlock_tool", 60.0),
    ("change_fixture", 150.0),
    ("lock_tool", 60.0),
    ("verify", 40.0),
)
#: Share of changeovers in which the fixture is prepared while A still runs.
EXTERNAL_FIXTURE_SHARE = 0.3
#: Pressure curve faults of ``curve_faults``.
CURVE_FAULTS = ("low_plateau", "late_rise", "spike_in_hold")
#: Mould temperature: target while running, ambient, time constants (s).
MOLD_TARGET, MOLD_AMBIENT, MOLD_TAU_HEAT, MOLD_TAU_COOL = 220.0, 60.0, 180.0, 900.0
#: kWh per second while processing (steps 40-60) and while idle.
ENERGY_PROCESS_KW, ENERGY_IDLE_KW = 30.0, 4.0


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
    parts_per_cycle: int = 1,
    reported_cycle_time: bool = False,
    alt_sequence_probability: float = 0.0,
    shape_fault_probability: float = 0.0,
    parallel_lane: bool = False,
    poll_period: str | None = None,
    stop_causes: int = 0,
    stop_probability: float = 0.0,
    changeover_steps: bool = False,
    rampup_cycles: int = 0,
    handshake_partner: bool = False,
    order_size: int = 0,
    scrap_probability: float = 0.0,
    setting_changes: Sequence[tuple[int, float]] = (),
    energy: bool = False,
    curve_faults: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate the PLC; see :func:`ts_shape.datasets.make_plc_cycles`."""
    if not 0 <= stop_causes <= len(STOP_CAUSES):
        raise ValueError(f"stop_causes must be 0..{len(STOP_CAUSES)}")
    unknown_faults = set(curve_faults or {}) - set(CURVE_FAULTS)
    if unknown_faults:
        raise ValueError(
            f"unknown curve_faults {sorted(unknown_faults)}; known: {list(CURVE_FAULTS)}"
        )
    if parts_per_cycle < 1:
        raise ValueError("parts_per_cycle must be at least 1")
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
    clamp_spans: list[tuple[float, float, float]] = []
    fault_spans: list[tuple[float, float, str]] = []
    handshake_waits: list[float] = []
    reported_rows: list[tuple[float, float]] = []
    truth: list[dict[str, object]] = []
    # Optional features draw from their own generator, so the default data
    # (and the main generator's sequence) stay the same.
    xrng = np.random.default_rng([seed, 7919])
    extra_bool: dict[str, list[tuple[float, bool]]] = {}
    mode_rows: list[tuple[float, int]] = [(0.0, 1)]
    fixture_rows: list[tuple[float, str]] = [(0.0, "F-0")]
    order_rows: list[tuple[float, str]] = []
    good_rows: list[tuple[float, int]] = []
    bad_rows: list[tuple[float, int]] = []
    override_rows: list[tuple[float, float]] = []
    energy_rows: list[tuple[float, float]] = []
    stops: list[tuple[float, float]] = []  # machine not producing
    settings = {int(i): float(v) for i, v in setting_changes}
    override = 100.0
    good = bad = 0
    kwh = 0.0
    ramp_left = 0
    pending_changeover = False
    fixture_ready = False
    fixture_no = 0
    faults = dict(curve_faults or {})

    def flag(name: str, at: float, value: bool) -> None:
        extra_bool.setdefault(name, []).append((at, value))

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
        # Optional disturbances draw from the generator only when enabled, so
        # the default data stay the same.
        swapped = (
            alt_sequence_probability > 0
            and not aborted
            and bool(rng.random() < alt_sequence_probability)
        )
        shape_fault = (
            shape_fault_probability > 0
            and not aborted
            and bool(rng.random() < shape_fault_probability)
        )
        level = 130.0 if shape_fault else 180.0
        fault = "low_plateau" if shape_fault else ""
        if faults and not aborted and not shape_fault:
            for name, probability in faults.items():
                if xrng.random() < probability:
                    fault = name
                    break
            if fault == "low_plateau":
                level = 130.0
        stop_cause = ""
        stop_seconds = 0.0
        if stop_causes and stop_probability > 0 and not pause and p > 0:
            if xrng.random() < stop_probability:
                name, _trace, median = STOP_CAUSES[int(xrng.integers(stop_causes))]
                stop_cause = name
                stop_seconds = median * float(xrng.lognormal(0.0, 0.3))
        changeover_seconds = 0.0
        startup = ramp_left > 0
        if p in settings:
            override = settings[p]
            override_rows.append((t, override))
        if order_size > 0 and p % order_size == 0:
            order_rows.append((t + 0.2, f"45000{p // order_size + 1:05d}"))

        pass_start = t
        # Step 10: load (a break waits here).
        step_rows.append((t, 10))
        wait = 0.0
        if pause:
            wait = pause_minutes * 60.0 * float(rng.uniform(0.8, 1.2))
        load_end = t + STEP_SECONDS[10] * noise[10]
        if pending_changeover:
            # The changeover runs in step 10 of the first B pass.
            pending_changeover = False
            c = load_end
            flag("setup_mode", c, True)
            for name, median in CHANGEOVER_STEPS:
                seconds = median * float(xrng.lognormal(0.0, 0.25))
                if name == "change_fixture" and fixture_ready:
                    seconds = 10.0  # prepared while A was still running
                elif name == "change_fixture":
                    fixture_no += 1
                    fixture_rows.append((c + 0.5 * seconds, f"F-{fixture_no}"))
                if name in ("unlock_tool", "lock_tool"):
                    flag(f"{name}", c + 1.0, True)
                    flag(f"{name}", c + 1.3, False)
                c += seconds
            flag("setup_mode", c, False)
            changeover_seconds = c - load_end
            fixture_ready = False
            stops.append((load_end, c))
            ramp_left = rampup_cycles
            startup = False
        if stop_cause:
            trace = dict((n, tr) for n, tr, _ in STOP_CAUSES)[stop_cause]
            s0 = load_end + changeover_seconds
            if trace.startswith("alarm"):
                flag(trace, s0 - float(xrng.uniform(0.5, 3.0)), True)
                flag(trace, s0 + stop_seconds, False)
            elif trace == "door":
                mode_rows.append((s0 - float(xrng.uniform(0.5, 2.0)), 2))
                flag("door_open", s0 + 1.0, True)
                flag("door_open", s0 + stop_seconds - 1.0, False)
                mode_rows.append((s0 + stop_seconds, 1))
            stops.append((s0, s0 + stop_seconds))
            ramp_left = rampup_cycles
        if pause:
            stops.append((load_end, load_end + wait))
            ramp_left = rampup_cycles
        t = load_end + changeover_seconds + stop_seconds + wait
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
            ramp = 1.0
            if startup and rampup_cycles > 0:
                # The first cycles after a stop run slower (cold mould).
                ramp = 1.0 + 0.6 * ramp_left / rampup_cycles
            t += (STEP_SECONDS[40] + extra) * noise[40] * ramp * (100.0 / override)

            def camera(t: float) -> float:
                # Step 50: camera, with re-inspection after a NOK.
                step_rows.append((t, 50))
                shot = t + STEP_SECONDS[50] * noise[50] - 0.6
                for _ in range(retries):
                    pulse("camera_done", shot)
                    pulse("camera_nok", shot + 0.1)
                    shot += RETRY_SECONDS
                pulse("camera_done", shot)
                return t + STEP_SECONDS[50] * noise[50] + retries * RETRY_SECONDS

            def process2(t: float) -> float:
                # Step 60: process 2, the conveyor is not always ready.
                step_rows.append((t, 60))
                conveyor = CONVEYOR_WAIT_SECONDS if rng.random() < 0.4 else 0.0
                if handshake_partner:
                    # Four-phase handshake with the conveyor: request, wait
                    # for the acknowledge, drop the request, drop the ack.
                    req = t + 0.5
                    delay = 0.4 + conveyor + float(xrng.lognormal(-1.5, 0.4))
                    flag("req_conveyor", req, True)
                    flag("ack_conveyor", req + delay, True)
                    flag("req_conveyor", req + delay + 0.2, False)
                    flag("ack_conveyor", req + delay + 0.5, False)
                    handshake_waits.append(delay)
                return t + STEP_SECONDS[60] * noise[60] + conveyor

            # An alternative route runs process 2 before the camera.
            for block in (process2, camera) if swapped else (camera, process2):
                t = block(t)
            # Step 70: unclamp.
            step_rows.append((t, 70))
            bool_rows["clamp_closed"].append((t + 0.5, False))
            clamp_spans.append((clamp_on, t + 0.5, level))
            fault_spans.append((clamp_on, t + 0.5, fault))
            t += STEP_SECONDS[70] * noise[70]
        else:
            # Aborted: the robot hands back, the clamp opens, unload at once.
            bool_rows["robot_ready"].append((t, False))
            bool_rows["clamp_closed"].append((t + 0.2, False))
            clamp_spans.append((clamp_on, t + 0.2, level))
            fault_spans.append((clamp_on, t + 0.2, ""))
            t += 0.4
        # Step 80: unload; the counter counts the part (cycle boundary).
        step_rows.append((t, 80))
        counter += parts_per_cycle
        counter_rows.append((t + 1.0, counter))
        boundaries.append(t + 1.0)
        nok = False
        if scrap_probability > 0:
            rate = scrap_probability
            rate *= 5.0 if startup else 1.0
            rate *= 10.0 if fault else 1.0
            rate *= 3.0 if stop_cause.startswith(("jam", "material")) else 1.0
            nok = bool(xrng.random() < min(rate, 0.9))
            bad += 1 if nok else 0
            good += parts_per_cycle - (1 if nok else 0)
            good_rows.append((t + 1.0, good))
            if nok:
                bad_rows.append((t + 1.0, bad))
        if energy:
            busy = t - pass_start - changeover_seconds - stop_seconds - wait
            idle = changeover_seconds + stop_seconds + wait
            kwh += (
                ENERGY_PROCESS_KW * busy * (override / 100.0) + ENERGY_IDLE_KW * idle
            ) / 3600.0
            energy_rows.append((t + 1.0, round(kwh, 3)))
        if ramp_left > 0 and startup:
            ramp_left -= 1
        if reported_cycle_time and len(boundaries) >= 2:
            # The controller publishes the duration of the cycle just ended.
            reported_rows.append((t + 2.0, round(boundaries[-1] - boundaries[-2], 1)))
        nxt = article_of(p + 1)
        if nxt != index:
            # The next order is loaded before the part is counted.
            article_rows.append((t + 0.5, str(articles[nxt])))
            if changeover_steps:
                pending_changeover = True
                if xrng.random() < EXTERNAL_FIXTURE_SHARE:
                    # The fixture is prepared while A is still running.
                    fixture_ready = True
                    fixture_no += 1
                    fixture_rows.append((t - 20.0, f"F-{fixture_no}"))
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
                "parts": parts_per_cycle,
                "sequence": "swapped" if swapped else "standard",
                "shape_fault": shape_fault,
                "stop_cause": stop_cause,
                "stop_seconds": stop_seconds,
                "changeover_seconds": changeover_seconds,
                "startup": startup,
                "nok": nok,
                "order_id": order_rows[-1][1] if order_rows else "",
                "setting_value": override,
                "curve_fault": fault,
                "handshake_wait_seconds": (
                    handshake_waits[-1] if handshake_waits else np.nan
                ),
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
    pressure = np.full(len(grid), 5.0)
    for (on, off, level), (_on, _off, fault) in zip(
        clamp_spans, fault_spans or [(0, 0, "")] * len(clamp_spans), strict=True
    ):
        if fault == "late_rise":
            on += 2.0
        pressure[(grid >= on) & (grid < off)] = level
        if fault == "spike_in_hold":
            mid = on + 0.5 * (off - on)
            pressure[(grid >= mid) & (grid < mid + 1.0)] = 260.0
    pressure = pressure + rng.normal(0.0, 2.0, len(grid))
    frames.append(_sampled(t0, f"{prefix}:pressure", grid, pressure, "value_double"))
    if heartbeat:
        alive = (np.floor(grid / period).astype(np.int64) % 2) == 0
        frames.append(_sampled(t0, f"{prefix}:heartbeat", grid, alive, "value_bool"))

    for name, rows in sorted(extra_bool.items()):
        rows = sorted(rows)
        initial = [(0.0, False)] if rows and rows[0][0] > 0 else []
        frames.append(_discrete(t0, f"{prefix}:{name}", initial + rows, "value_bool"))
    if stop_causes and len(mode_rows) > 1:
        frames.append(
            _discrete(t0, f"{prefix}:mode", sorted(mode_rows), "value_integer")
        )
    if changeover_steps and len(fixture_rows) > 1:
        frames.append(
            _discrete(t0, f"{prefix}:fixture_id", sorted(fixture_rows), "value_string")
        )
    if order_rows:
        frames.append(_discrete(t0, f"{prefix}:order_no", order_rows, "value_string"))
    if scrap_probability > 0:
        frames.append(
            _discrete(t0, f"{prefix}:good_counter", good_rows, "value_integer")
        )
        initial_bad = [(0.0, 0)]
        frames.append(
            _discrete(
                t0, f"{prefix}:bad_counter", initial_bad + bad_rows, "value_integer"
            )
        )
    if override_rows:
        override_all = (
            override_rows
            if override_rows[0][0] == 0.0
            else [(0.0, 100.0), *override_rows]
        )
        frames.append(
            _discrete(t0, f"{prefix}:speed_override", override_all, "value_double")
        )
    if energy_rows:
        frames.append(
            _discrete(t0, f"{prefix}:energy_kwh", energy_rows, "value_double")
        )
    if rampup_cycles > 0:
        frames.append(
            _sampled(
                t0,
                f"{prefix}:mold_temp",
                grid,
                _mold(grid, stops, xrng),
                "value_double",
            )
        )

    if reported_cycle_time and reported_rows:
        frames.append(
            _discrete(t0, f"{prefix}:last_cycle_time", reported_rows, "value_double")
        )
    if parallel_lane:
        frames.extend(_lane(t0, f"{prefix}:lane2", end_t, rng))

    frame = pd.concat(frames, ignore_index=True)
    if poll_period is not None:
        frame = _poll(frame, t0, end_t, pd.to_timedelta(poll_period).total_seconds())
    frame = frame.sort_values(["systime", "uuid"], kind="stable").reset_index(drop=True)

    optional = {
        "stop_cause": stop_causes > 0,
        "stop_seconds": stop_causes > 0,
        "changeover_seconds": changeover_steps,
        "startup": rampup_cycles > 0,
        "nok": scrap_probability > 0,
        "order_id": order_size > 0,
        "setting_value": bool(settings),
        "curve_fault": bool(faults),
        "handshake_wait_seconds": handshake_partner,
    }
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
            "parts",
            "sequence",
            "shape_fault",
        ]
        + [c for c, on in optional.items() if on]
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


#: Period of the independent second lane, as a multiple of the nominal cycle.
LANE_PERIOD_FACTOR = 1.13


def _lane(
    t0: pd.Timestamp, prefix: str, end_t: float, rng: np.random.Generator
) -> list[pd.DataFrame]:
    """A second, independent machine on the same controller (a parallel lane).

    It runs at a slightly different rate, so its phase drifts through the
    main cycle.
    """
    nominal = sum(STEP_SECONDS.values()) * LANE_PERIOD_FACTOR
    t, count = 3.0, 500
    counter: list[tuple[float, object]] = []
    clamp: list[tuple[float, object]] = [(0.0, False)]
    while t < end_t:
        length = nominal * float(rng.lognormal(0.0, 0.03))
        clamp.append((t + 0.1 * length, True))
        clamp.append((t + 0.7 * length, False))
        count += 1
        counter.append((t + 0.9 * length, count))
        t += length
    return [
        _discrete(t0, f"{prefix}_counter", counter, "value_integer"),
        _discrete(t0, f"{prefix}_clamp", clamp, "value_bool"),
    ]


def _poll(
    frame: pd.DataFrame, t0: pd.Timestamp, end_t: float, period: float
) -> pd.DataFrame:
    """Every signal read at one fixed poll, as a polling OPC UA client does."""
    if period <= 0:
        raise ValueError("poll_period must be positive")
    grid = pd.DatetimeIndex(
        t0 + pd.to_timedelta(np.arange(0.0, end_t, period), unit="s")
    ).as_unit("ns")
    parts = []
    for uuid, rows in frame.groupby("uuid", sort=True):
        rows = rows.sort_values("systime", kind="stable")
        rows = rows.assign(systime=pd.DatetimeIndex(rows["systime"]).as_unit("ns"))
        probe = pd.DataFrame({"systime": grid})
        polled = pd.merge_asof(probe, rows.drop(columns="uuid"), on="systime")
        polled = polled[polled["is_delta"].notna()].assign(uuid=uuid, is_delta=False)
        parts.append(polled)
    out = pd.concat(parts, ignore_index=True)
    for column, dtype in (("value_integer", "Int64"), ("value_bool", "boolean")):
        out[column] = out[column].astype(dtype)
    return out[["systime", "uuid", *_COLUMNS, "is_delta"]]


def _mold(
    grid: np.ndarray, stops: list[tuple[float, float]], rng: np.random.Generator
) -> np.ndarray:
    """Mould temperature: heats towards its target while producing, cools in stops."""
    stopped = np.zeros(len(grid), dtype=bool)
    for a, b in stops:
        stopped[(grid >= a) & (grid < b)] = True
    temp = np.empty(len(grid))
    value = MOLD_TARGET
    dt = float(grid[1] - grid[0]) if len(grid) > 1 else 1.0
    for i, idle in enumerate(stopped):
        if idle:
            value += (MOLD_AMBIENT - value) * dt / MOLD_TAU_COOL
        else:
            value += (MOLD_TARGET - value) * dt / MOLD_TAU_HEAT
        temp[i] = value
    return temp + rng.normal(0.0, 0.5, len(grid))
