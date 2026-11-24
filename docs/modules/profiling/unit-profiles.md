---
tags: [cycle-time, machine-state, process-control]
---

# Unit profilers (mapping-free)

The [cycle profiler](cycle-profiling.md) answers *where does the cycle time
go?* from the raw signals of one PLC, without naming any signal. The unit
profilers ask the same kind of question about other **units of analysis** and
run them through the same steps:

> units → learned standard → classify → register (potential = median − P10) →
> causes → evaluability → one record per unit

| Profiler | Unit | Question |
|---|---|---|
| `StopProfileEvents` | a stop / micro-stop | Why does the machine stop, which cause costs the most? |
| `ChangeoverProfileEvents` | a variant change | Which changeover step to shorten first (SMED)? |
| `StartupProfileEvents` | the restart after a stop | How long until normal running, what holds it back? |
| `OrderProfileEvents` | a production order | Which orders run slow or scrap the most? |
| `ScrapProfileEvents` | a cycle with a bad part | What precedes a bad part (scrap Pareto without reason codes)? |
| `HandshakeProfileEvents` | a request → answer wait | How long does the machine wait on its partners? |
| `CurveSignatureEvents` | the curve of an analog signal in one cycle | Which curve shapes deviate, and what do they cost? |
| `SettingImpactEvents` | a change of a held setting | Did that parameter change help? |
| `ControlLoopProfileEvents` | a control loop (setpoint step) | Which loops oscillate or are sluggish? |

Every profiler takes `cycle_profile=` to reuse one computed
`CycleProfileEvents`, has an `evaluation_window()` that says whether it can be
trusted (`ok`, `too_few_units`, `below_resolution`, `no_candidate_signal: ...`,
`upstream_not_evaluable: ...`) and a `summary()` of headline numbers.

## Run them all

`profile_units` prepares the signals once, computes the cycle profile once and
runs every profiler on top of it:

```python
import ts_shape
from ts_shape.events.profiling.units import profile_units

frame, truth = ts_shape.make_plc_cycles(
    n_cycles=400, seed=3, stop_causes=4, stop_probability=0.08,
    changeover_steps=True, article_block=40, rampup_cycles=4,
)
result = profile_units(frame)
print(result.summary["stops"]["top_signature"])
stops = result["stops"].tables["stop_signatures"]
```

`result.save(folder)` writes one subfolder per profiler (`cycles/`, `stops/`,
...); `UnitProfilesResult.load(folder)` reads them back.

## Stops

A `pause` cycle is a stop, a `long` cycle a micro-stop; the lost time is the
duration beyond the variant's median cycle. On a continuous line without a
cycle, the stops are the off-periods of a discovered run signal (a state value
during which a counter keeps counting). Gaps in which no signal logged anything
for hours are data holes: unknown time, not stops.

The cause of a stop is learned, not configured: the events in a window before
each stop are compared with the same windows in normal running (Fisher exact
test, lift ≥ 3, q < 0.05). The earliest such event before a stop is its
signature (*first-out*); without one, a discrete state enriched during stops
(door open, manual mode); otherwise `unexplained`. Analog crossings during a
stop (a cooling mould) are consequences and never signatures; an event present
at every stop (the run signal falling) marks the stop itself and is not a cause.

```python
from ts_shape import StopProfileEvents

stops = StopProfileEvents(frame)
register = stops.stop_signatures()      # lost time, MTTR, MTBF, potential
records = stops.stop_records()          # one record per stop with its signature
```

## Changeovers and start-ups

A changeover runs from the last normal cycle of the old variant to the start of
`ramp_cycles` normal cycles of the new one. Events much more frequent in
changeovers than in running are steps; those in most changeovers form the
standard sequence. A changing id (fixture number) counts as one step
(`<signal>Δ`). A step that has also been done while still producing is tagged
`external_possible`.

A start-up follows a stop or a changeover and lasts until `ramp_cycles`
consecutive normal cycles. Each analog signal's running band is learned from
steady running; `startup_signals()` shows how long each needs to get back into
it and which one is last.

```python
from ts_shape import ChangeoverProfileEvents, StartupProfileEvents

changeovers = ChangeoverProfileEvents(frame)
steps = changeovers.changeover_view()
startups = StartupProfileEvents(frame)
slowest = startups.startup_signals()
```

## Orders and scrap

Which counter is the total, the good and the bad count is learned, not named:
`good + bad = total` must hold per cycle (`counter_roles()`); equal counters
are aliases, a counter with fractional increments is energy-like. A count
reported at a cycle boundary belongs to the cycle that just ended.

The order signal is an identifier held over many cycles
(`identify_objects`), else the variant signal. Per order: cycles, run and stop
time, speed against the variant's best (P10) cycle, good and bad parts and,
with a `plan` (`order_id`, `planned_quantity`), the attainment.

Every cycle with a bad part is a NOK unit. Its context -- a stop in the cycle,
after a stop or a changeover, a long or short cycle, an analog per-cycle
deviation, a curve-shape anomaly of the cycle profile, and every rare discrete
event in the cycle -- is compared with the context of good cycles. The result
is a scrap Pareto without reason codes:

```python
from ts_shape import OrderProfileEvents, ScrapProfileEvents

frame, truth = ts_shape.make_plc_cycles(
    n_cycles=600, seed=4, order_size=40, scrap_probability=0.02,
    rampup_cycles=4, stop_causes=2, stop_probability=0.05,
)
orders = OrderProfileEvents(frame).order_records()
pareto = ScrapProfileEvents(frame).scrap_pareto()
```

## Handshakes

Request/answer pairs are found from the signals' edges alone: two boolean
signals of different sources whose edges follow each other (≥ 95 % both ways)
with a delay well below the request interval. A four-phase pair (request ↑,
answer ↑, request ↓, answer ↓) is told from a two-phase one. A wait with a
fixed delay is a timer (`waits_on = fixed`); a wait that varies is a wait on
the partner (`waits_on = partner`). `handshake_register()` ranks the waits by
potential and shows their share of, and correlation with, the cycle time.

```python
from ts_shape import HandshakeProfileEvents

frame, truth = ts_shape.make_plc_cycles(n_cycles=300, seed=2, handshake_partner=True)
pairs = HandshakeProfileEvents(frame).handshake_pairs()
```

## Curve signatures

Per variant and analog signal a golden curve (median and P10–P90 band) is
learned over the cycle, time-warped between the discrete milestones of the
standard cycle so that a slower step does not read as a fault. Each cycle gets
a deviation score (mean plus peak residual), its deviating phase and
direction. Anomalous cycles are clustered into fault patterns
(`curve_patterns()`, per signal over all variants, at least 3 cycles) with
their long-cycle share and lost time. A run of off-band cycles right after a
pause or variant change is a warm-up (`after_restart`, counted in the summary
as `warmup_cycle_count`), not a fault pattern.

```python
from ts_shape import CurveSignatureEvents

frame, truth = ts_shape.make_plc_cycles(
    n_cycles=400, seed=5, articles=("A",),
    curve_faults={"low_plateau": 0.04, "late_rise": 0.04},
)
patterns = CurveSignatureEvents(frame).curve_patterns()
```

## Setting changes

A held setting is a numeric signal that changes rarely and holds its value for
many cycles. Every change is a before/after test on the cycle time, every
segment of the standard cycle, the bad-part rate and the energy per cycle
(Hodges–Lehmann shift, Cliff's delta, Mann–Whitney with BH q-values). Only
cycles of the variant running at the change are compared. Confounders
(another setting change, a variant change or a pause at the change, a trend
before it) lower the confidence. A setting that only changes with the variant
is a recipe parameter and is not tested. `setting_register()` gives the
sensitivity (Theil–Sen) and the best value seen.

:::note

On a machine without a separate article signal the cycle profiler may
take a held setting for the variant signal; it is then not tested here.

:::

```python
from ts_shape import SettingImpactEvents

frame, truth = ts_shape.make_plc_cycles(
    n_cycles=400, seed=1, setting_changes=[(150, 120.0), (280, 90.0)], energy=True,
)
impact = SettingImpactEvents(frame).setting_impact()
```

## Control loops

`ControlLoopProfileEvents` (in `ts_shape.events.engineering`) needs no machine
cycle. Setpoints are piecewise-constant signals that step now and then; each
is paired one-to-one with the continuous signal that follows its steps and
tracks it. A signal that reacts *at once* to a setpoint step is the
controller output (the process value reacts only after its dead time). Every
loop is measured with `SetpointChangeEvents` (settling, overshoot,
oscillation per step) and `ControlLoopHealthEvents` (IAE, output saturation),
diagnosed `oscillating`, `sluggish` or `ok`, and ranked by health (rank 1 =
worst).

```python
from ts_shape import ControlLoopProfileEvents

frame, truth = ts_shape.make_control_loops(seed=1)
register = ControlLoopProfileEvents(frame).loop_register()
```

## Adding a profiler

A profiler is one class. It subclasses `_UnitProfiler` (units cut from the
cycle profile) or `_ProfilerBase` (no cycles needed, like the control loops)
from `ts_shape.events.profiling._base` and declares in its class
body:

<!-- no-run -->
```python
class ToolWearProfileEvents(_UnitProfiler):
    PROFILE_KIND = "tool_wear"            # its name in profile_units
    PROFILE_TABLES = ("wear_records", "wear_register")
    PROFILE_ORDER = 45                    # after orders, before scrap
    PROFILE_OPTIONS = ()                  # profile_units keyword arguments it takes
```

`profile_units` then finds it, runs it in order and saves its tables plus
`evaluation_window`. The base gives `_point_records` (the shared record
columns), `_evaluation_row` (the shared `REASON_*` rules), `_summary_head`,
`memo` caching and `_signals()` (every signal's readings, read once). Its
event-log rules go into `ts_shape.eventlog.profiler_rules.PROFILER_RULES`
(`evaluation_window` is added automatically); exports and the package
docstring list are the remaining hand-written places.
