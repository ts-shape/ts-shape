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

Every profiler takes `cycle_profile=` to reuse one computed
`CycleProfileEvents`, has an `evaluation_window()` that says whether it can be
trusted (`ok`, `too_few_units`, `below_resolution`, `no_candidate_signal: ...`,
`upstream_not_evaluable: ...`) and a `summary()` of headline numbers.

## Run them all

`profile_units` prepares the signals once, computes the cycle profile once and
runs every profiler on top of it:

```python
import ts_shape
from ts_shape.unit_profiles import profile_units

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
