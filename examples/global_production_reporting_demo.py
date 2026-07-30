"""Global production reporting across a fleet of differently-wired machines.

Run me::

    python examples/global_production_reporting_demo.py

The scenario is the one that makes fleet reporting hard: four machines that all
make parts, and no two of which say so the same way.

============  ==========================  ==========================  ================
Machine       Part counters               Machine status              Cycle time
============  ==========================  ==========================  ================
PRESS-01      16-bit total + NOK, wraps   PackML state number         published, ms
CELL-07       part-done pulse, 2 cavities boolean run flag            counter deltas
OVEN-02       incremental (+n per event)  one-hot lamps, overlapping   part-present flag
LINE-03       one total counter, resets   nothing                     nothing
============  ==========================  ==========================  ================

Every one of those differences is configuration, not code. The report below is
written once and produces one comparable frame for all four -- plus the
diagnostics that say which numbers to trust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ts_shape.events.production.signals import (
    ContextBinding,
    ContextSpec,
    CounterKind,
    CounterMode,
    CounterSpec,
    CycleMode,
    CycleSpec,
    FleetProductionReport,
    FleetSpec,
    MachineSpec,
    MachineState,
    Rollover,
    StatusMode,
    StatusSpec,
    TimeUnit,
)

BASE = pd.Timestamp("2025-03-03 06:00:00")
HORIZON = BASE + pd.Timedelta("8h")
STEPS = 97  # five-minute samples across an eight-hour shift
RNG = np.random.default_rng(7)


def rows(uuid: str, times, values, column: str) -> pd.DataFrame:
    """One signal in the standard ts-shape schema."""
    return pd.DataFrame({"systime": times, "uuid": uuid, column: values})


def grid(n: int = STEPS) -> pd.DatetimeIndex:
    return pd.date_range(BASE, periods=n, freq="5min")


# ---------------------------------------------------------------------------
# Synthetic fleet data -- each machine wired differently on purpose
# ---------------------------------------------------------------------------
def make_data() -> pd.DataFrame:
    t = grid()
    frames: list[pd.DataFrame] = []

    # -- PRESS-01: a 16-bit totalizer that wraps mid-shift, plus a NOK counter,
    #    a PackML state, and a cycle time published in milliseconds.
    total = 65500 + np.arange(STEPS) * 20  # crosses 65536 after ~2 samples
    frames += [
        rows("p1.total", t, (total % 65536).astype("int64"), "value_integer"),
        rows("p1.nok", t, (np.arange(STEPS) // 12).astype("int64"), "value_integer"),
        rows(
            "p1.packml",
            t,
            # Execute, a fault around midday, a changeover, then Execute again.
            [6] * 40 + [9] * 6 + [15] * 4 + [6] * 47,
            "value_integer",
        ),
        rows("p1.cycle_ms", t, RNG.normal(15000, 400, STEPS).round(), "value_double"),
        rows(
            "p1.order",
            [BASE, BASE + pd.Timedelta("4h")],
            ["A-4711", "A-4712"],
            "value_string",
        ),
        rows(
            "p1.part",
            [BASE, BASE + pd.Timedelta("4h")],
            ["  000815 ", "000816"],  # padded and spaced, as controllers write it
            "value_string",
        ),
        rows(
            "p1.customer_part",
            [BASE, BASE + pd.Timedelta("4h")],
            ["BOSCH-9920", "BOSCH-9921"],
            "value_string",
        ),
    ]

    # -- CELL-07: a part-done pulse where one pulse means two parts (twin
    #    cavity), a boolean run flag that flickers, and a defect code.
    pulse = np.zeros(STEPS, dtype=bool)
    pulse[::2] = True
    run = np.ones(STEPS, dtype=bool)
    run[60:66] = False
    frames += [
        rows("c7.part_done", t, pulse, "value_bool"),
        rows(
            "c7.defect_code",
            t,
            np.where(np.arange(STEPS) % 23 == 0, 4, 0),
            "value_integer",
        ),
        rows("c7.running", t, run, "value_bool"),
        # A one-scan flicker that debouncing must absorb.
        rows(
            "c7.running",
            [BASE + pd.Timedelta("90min"), BASE + pd.Timedelta("90min1s")],
            [False, True],
            "value_bool",
        ),
        rows("c7.order", [BASE], ["A-4711"], "value_string"),
        # The serial is written a moment *after* the part it belongs to.
        rows(
            "c7.serial",
            [ts + pd.Timedelta("3s") for ts in t[::2]],
            [f"SN-{i:04d}" for i in range(len(t[::2]))],
            "value_string",
        ),
    ]

    # -- OVEN-02: sends its quantity per event, reports state as three separate
    #    lamps (which overlap), and only has a part-present flag.
    frames += [
        rows("o2.qty", t, RNG.integers(0, 4, STEPS), "value_integer"),
        rows("o2.lamp_green", t, np.arange(STEPS) % 40 < 34, "value_bool"),
        rows(
            "o2.lamp_red",
            t,
            (np.arange(STEPS) % 40 >= 34) & (np.arange(STEPS) % 40 < 37),
            "value_bool",
        ),
        # Green stays on during a fault -- the drive bit and the alarm word.
        rows("o2.lamp_yellow", t, np.arange(STEPS) % 40 >= 37, "value_bool"),
        rows(
            "o2.part_present",
            # A part is present for 110 s of every 5-minute slot.
            pd.DatetimeIndex(np.repeat(t.to_numpy(), 2))
            + pd.to_timedelta(np.tile([0, 110], STEPS), unit="s"),
            np.tile([True, False], STEPS),
            "value_bool",
        ),
        rows(
            "o2.batch",
            [BASE, BASE + pd.Timedelta("5h")],
            ["B-22-07", "B-22-08"],
            "value_string",
        ),
    ]

    # -- LINE-03: one counter, reset at the shift break, and nothing else. The
    #    badly instrumented machine every fleet has.
    line = np.arange(STEPS) * 3
    line[48:] = np.arange(STEPS - 48) * 3  # reset to zero halfway
    frames += [
        rows("l3.total", t, line.astype("int64"), "value_integer"),
        # A tag that does not exist in the data: a mis-mapped configuration.
        # (Configured on the machine below, deliberately absent here.)
    ]

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# The fleet configuration -- data, not code
# ---------------------------------------------------------------------------
def make_fleet() -> FleetSpec:
    press = MachineSpec(
        machine_id="PRESS-01",
        attributes={"plant": "Ingolstadt", "line": "L1", "machine_type": "press"},
        counters=(
            CounterSpec(
                uuid="p1.total",
                kind=CounterKind.TOTAL,
                mode=CounterMode.CUMULATIVE,
                rollover_at=Rollover.UINT16,
                max_plausible_delta=500,
            ),
            CounterSpec(
                uuid="p1.nok", kind=CounterKind.NOK, mode=CounterMode.CUMULATIVE
            ),
        ),
        status=StatusSpec(
            mode=StatusMode.PACKML, uuid="p1.packml", stale_after="30min"
        ),
        cycle=CycleSpec(
            mode=CycleMode.EXPLICIT,
            uuid="p1.cycle_ms",
            unit=TimeUnit.AUTO,  # infers milliseconds from magnitude
            min_seconds=5,
            max_seconds=60,
            gate_states=(MachineState.DOWN, MachineState.SETUP),
        ),
        context=(
            ContextSpec(
                name="order_number", uuid="p1.order", valid_for="12h", upper=True
            ),
            ContextSpec(name="part_number", uuid="p1.part", strip_leading_zeros=True),
            ContextSpec(name="customer_part_number", uuid="p1.customer_part"),
        ),
        ideal_cycle_seconds=14.0,
    )

    cell = MachineSpec(
        machine_id="CELL-07",
        attributes={"plant": "Ingolstadt", "line": "L2", "machine_type": "assembly"},
        counters=(
            CounterSpec(
                uuid="c7.part_done",
                kind=CounterKind.OK,
                mode=CounterMode.PULSE,
                parts_per_count=2,  # twin cavity
            ),
            CounterSpec(
                uuid="c7.defect_code",
                kind=CounterKind.NOK,
                mode=CounterMode.DEFECT_CODE,
                ok_values=("0",),
                trigger_uuid="c7.part_done",
            ),
        ),
        status=StatusSpec(
            mode=StatusMode.BOOLEAN, uuid="c7.running", min_duration="10s"
        ),
        cycle=CycleSpec(
            mode=CycleMode.COUNTER_DELTA, uuid="c7.part_done", max_seconds=1800
        ),
        context=(
            ContextSpec(name="order_number", uuid="c7.order", valid_for="12h"),
            ContextSpec(
                name="serial_number",
                uuid="c7.serial",
                binding=ContextBinding.NEXT_KNOWN,  # published after the part
                valid_for="30s",
                zero_is_sentinel=True,
            ),
        ),
        ideal_cycle_seconds=280.0,
    )

    oven = MachineSpec(
        machine_id="OVEN-02",
        attributes={"plant": "Ingolstadt", "line": "L2", "machine_type": "oven"},
        counters=(
            CounterSpec(
                uuid="o2.qty", kind=CounterKind.OK, mode=CounterMode.INCREMENTAL
            ),
        ),
        status=StatusSpec(
            mode=StatusMode.ONE_HOT,
            state_map={
                "o2.lamp_green": MachineState.PRODUCING,
                "o2.lamp_red": MachineState.DOWN,
                "o2.lamp_yellow": MachineState.IDLE,
            },
        ),
        cycle=CycleSpec(
            mode=CycleMode.PART_PRESENT, uuid="o2.part_present", max_seconds=600
        ),
        context=(ContextSpec(name="batch_number", uuid="o2.batch"),),
        ideal_cycle_seconds=100.0,
    )

    line = MachineSpec(
        machine_id="LINE-03",
        attributes={"plant": "Regensburg", "line": "L9", "machine_type": "line"},
        counters=(CounterSpec(uuid="l3.total", kind=CounterKind.TOTAL),),
        # Configured but absent from the data -- signal_coverage() will say so.
        context=(ContextSpec(name="order_number", uuid="l3.order"),),
    )

    return FleetSpec(machines=(press, cell, oven, line))


def show(title: str, frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    view = frame[[c for c in columns if c in frame.columns]] if columns else frame
    if view.empty:
        print("(no rows)")
        return
    print(view.to_string(index=False, max_rows=20))


def main() -> None:
    data = make_data()
    fleet = make_fleet()
    report = FleetProductionReport(fleet, data, until=HORIZON)

    print(report)
    print(f"\n{len(data):,} readings across {data['uuid'].nunique()} signals.")
    print(
        f"Fleet reads {len(fleet.signal_uuids)} of them; grouping is available by "
        f"{list(fleet.attribute_keys)}."
    )

    # 1. The commissioning question first: is the mapping right? At fleet scale
    #    a wrong tag is far more common than a wrong metric.
    coverage = report.signal_coverage()
    show(
        "1. Signal coverage -- configured signals that never reported",
        coverage[~coverage["present"]],
        ["machine_id", "role", "uuid", "present", "readings"],
    )

    # 2. What the normalizers had to interpret.
    show(
        "2. Data-quality findings across the fleet",
        report.anomalies(),
        ["machine_id", "source_uuid", "category", "anomaly", "severity", "occurrences"],
    )

    # 3. The core fact table: canonical quantities per machine and hour, with
    #    identical columns for four incompatible encodings.
    show(
        "3. Production per machine and hour (canonical OK / NOK / total)",
        report.production_by_window("2h"),
        [
            "start",
            "machine_id",
            "line",
            "qty_ok",
            "qty_nok",
            "qty_total",
            "good_rate_pct",
            "nok_ppm",
        ],
    )

    # 4. Machine state, normalised from PackML, a boolean and three lamps.
    show(
        "4. Machine state and availability per machine",
        report.status_by_window(None),
        [
            "machine_id",
            "producing_seconds",
            "down_seconds",
            "setup_seconds",
            "idle_seconds",
            "unknown_seconds",
            "availability_pct",
        ],
    )

    # 5. Cycle time, normalised from milliseconds, counter gaps and a
    #    part-present flag -- per part, so the numbers are comparable.
    show(
        "5. Cycle time per machine",
        report.cycle_by_window("4h"),
        [
            "start",
            "machine_id",
            "cycles",
            "mean_seconds",
            "median_seconds",
            "p95_seconds",
        ],
    )

    # 6. Production context: the same quantities per order, part and customer.
    show(
        "6. Production per order and part number",
        report.production_by_context(["order_number", "part_number"]),
        [
            "machine_id",
            "order_number",
            "part_number",
            "qty_ok",
            "qty_nok",
            "qty_total",
            "good_rate_pct",
        ],
    )

    # 7. OEE from canonical inputs. Unknown factors stay unknown -- LINE-03
    #    measures no quality and has no status signal, so it has no OEE.
    show(
        "7. OEE factors per machine (NaN = not measured, never assumed)",
        report.oee("1D"),
        [
            "machine_id",
            "qty_total",
            "producing_seconds",
            "availability_pct",
            "performance_pct",
            "quality_pct",
            "oee_pct",
        ],
    )

    # 8. The fleet league table, with instrumentation facts next to the
    #    performance figures so a low number can be read correctly.
    show(
        "8. Machine summary",
        report.machine_summary(),
        [
            "machine_id",
            "plant",
            "qty_ok",
            "qty_nok",
            "qty_total",
            "good_rate_pct",
            "availability_pct",
            "cycles",
            "mean_cycle_seconds",
            "counter_anomalies",
            "unmapped_state_values",
        ],
    )

    show(
        "9. Roll-up by plant (rates recomputed from the sums)",
        report.fleet_summary(group_by=["plant"]),
    )

    # 10. Drill down with the very objects the report used.
    press = report.normalizers("PRESS-01")
    assert press.counters is not None and press.context is not None
    show(
        "10. PRESS-01 drill-down: counter health",
        press.counters.diagnostics(),
        [
            "source_uuid",
            "kind",
            "mode",
            "readings",
            "monotonic",
            "resets",
            "rollovers",
            "implausible",
            "quantity",
        ],
    )
    show(
        "10b. PRESS-01 drill-down: context coverage",
        press.context.coverage(),
        [
            "dimension",
            "source_uuid",
            "binding",
            "readings",
            "distinct_values",
            "fill_pct",
        ],
    )
    assert press.cycles is not None
    print(
        f"\nPRESS-01 cycle-time unit inferred as "
        f"{press.cycles.resolved_unit().value!r}; "
        f"{len(press.cycles.rejected())} candidate cycle(s) rejected "
        f"(breaks and gated states)."
    )

    cell = report.normalizers("CELL-07")
    assert cell.context is not None
    show(
        "11. CELL-07 serial numbers bound to the parts that carry them",
        cell.context.attach(
            cell.counters.count_events() if cell.counters else pd.DataFrame()
        ).head(6),
        ["systime", "kind", "quantity", "order_number", "serial_number"],
    )

    print(
        "\nEvery frame above has one schema for four machines that share nothing "
        "but making parts.\nThe configuration that made it possible round-trips "
        "through plain dicts:\n"
        f"  fleet.to_dict() -> {len(str(fleet.to_dict())):,} characters of "
        "reviewable JSON."
    )


if __name__ == "__main__":
    main()
