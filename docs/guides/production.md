# Production and OEE

How to get machine states, downtime, OEE, shift figures and part traceability
from run-state, counter and reason signals. The classes live in
`ts_shape.events.production`; each has its own page in the
[module reference](../modules/index.md).

## Example data

One line over 24 hours, sampled every minute: a boolean run state, a
cumulative part counter, a cumulative reject (NOK) counter, the good-part
counter and a stop-reason string.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
t = pd.date_range("2024-06-03 06:00", periods=24 * 60, freq="1min", tz="Europe/Berlin")
running = rng.random(len(t)) > 0.02                      # a short stop now and then
running[600:660] = False                                 # one long stop
parts = np.cumsum(running * 2)                           # 2 parts per running minute
nok = np.cumsum(running & (rng.random(len(t)) < 0.03))   # about 3 % rejects
reason = np.where(running, "", rng.choice(["material", "tooling"], len(t)))

def signal(uuid, column, values):
    return pd.DataFrame({"systime": t, "uuid": uuid, column: values})

df = pd.concat([
    signal("line1:running", "value_bool", running),
    signal("line1:parts", "value_integer", parts),
    signal("line1:nok", "value_integer", nok),
    signal("line1:ok", "value_integer", parts - nok),
    signal("line1:stop_reason", "value_string", reason),
], ignore_index=True)
```

Counters may reset (shift start, PLC restart, rollover). All counter-based
methods handle that by default (`handle_resets=True`) and report the number of
resets in a `resets` column.

## Machine states and downtime

```python
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.events.production.downtime_tracking import DowntimeEvents
from ts_shape.events.production.micro_stop_detection import MicroStopEvents

intervals = MachineStateEvents(df, run_state_uuid="line1:running").detect_run_idle(min_duration="5min")

downtime = DowntimeEvents(df)
by_shift = downtime.downtime_by_shift("line1:running", running_value=True)
by_reason = downtime.downtime_by_reason(
    "line1:running", "line1:stop_reason", stopped_value=False, value_column_state="value_bool"
)

micro_stops = MicroStopEvents(df, run_state_uuid="line1:running").detect_micro_stops(max_duration="3min")
```

- `detect_run_idle` returns one interval per run or idle period; intervals
  shorter than `min_duration` are dropped.
- `downtime_by_shift` returns downtime, uptime and `availability_pct` per
  shift. The run state can be a boolean (`running_value=True`) or a string
  state in `value_string` (`running_value="Running"`).
- `downtime_by_reason` is a Pareto of stop reasons by occurrences and
  duration.

Shifts default to three 8-hour shifts starting at 06:00 (`shift_1` to
`shift_3`). Pass `shift_definitions={"early": ("06:00", "14:00"), ...}` to the
constructor of the shift-based classes to change them. A night shift that
crosses midnight is counted on the day it starts (`production_day=True`).

## OEE

```python
from ts_shape.events.production.oee_calculator import OEEEvents

oee = OEEEvents(df).oee(
    "line1:running",
    "line1:parts",
    ideal_cycle_time=25.0,          # seconds per part at full speed
    total_uuid="line1:parts",
    reject_uuid="line1:nok",
)
oee[["start", "availability_pct", "performance_pct", "quality_pct", "oee_pct"]]
```

One row per calendar day. Availability is run time over planned time,
performance is actual over ideal parts in the run time, quality is good over
total parts. The basis of each percentage is in the same row (`run_seconds`,
`planned_seconds`, `ideal_parts`, `good_parts`, ...). Pass
`planned_time_hours` for a fixed planned time per day, and `max_gap` so that a
logging gap is not booked as run time. `availability`, `performance` and `quality` are also available on their own.

## Parts, quality and shift reports

```python
from ts_shape.events.production.shift_reporting import ShiftReportEvents
from ts_shape.events.production.quality_tracking import QualityEvents
from ts_shape.events.production.shift_handover import ShiftHandoverEvents

production = ShiftReportEvents(df).shift_production("line1:parts")
quality = QualityEvents(df).nok_by_shift("line1:ok", "line1:nok")

report = ShiftHandoverEvents(df).generate_report(
    "line1:parts", "line1:ok", "line1:nok", "line1:running",
    running_value=True,
    targets={"shift_1": 900, "shift_2": 900, "shift_3": 900},
)
```

`generate_report` combines production against target, quality and
availability per shift in one table. `highlight_issues` lists every shift
metric below its threshold (by default 95 % of the production target, 98 %
quality, 90 % availability).

## Which class answers which question

| Question | Class | Page |
|---|---|---|
| How many parts per hour or per part number? | `PartProductionEvents`, `LineThroughputEvents` | [Part tracking](../modules/production/part-tracking.md), [Line throughput](../modules/production/line-throughput.md) |
| How long are the cycles, which ones are slow? | `CycleTimeEvents` | [Cycle time](../modules/production/cycle-time.md) |
| Where is speed lost against the target cycle time? | `PerformanceLossEvents` | [Performance loss](../modules/production/performance-loss.md) |
| Which stops are long, which are micro stops? | `LongDowntimeEvents`, `MicroStopEvents` | [Long downtime](../modules/production/long-downtime.md), [Micro stops](../modules/production/micro-stops.md) |
| Which station limits the line? | `BottleneckDetectionEvents`, `LineBalancingEvents` | [Bottleneck](../modules/production/bottleneck-detection.md), [Line balancing](../modules/production/line-balancing.md) |
| Is a station blocked or starved? | `FlowConstraintEvents` | [Flow constraints](../modules/production/flow-constraints.md) |
| WIP, throughput and lead time (Little's law)? | `FlowMetricsEvents` | [Flow metrics](../modules/production/flow-metrics.md) |
| When are product changes, how long are setups? | `ChangeoverEvents`, `SetupTimeEvents` | [Changeover](../modules/production/changeover.md), [Setup time](../modules/production/setup-time.md) |
| Scrap, rework, NOK by reason and cost? | `ScrapEvents`, `ReworkEvents`, `QualityEvents` | [Scrap](../modules/production/scrap-tracking.md), [Rework](../modules/production/rework-tracking.md), [Quality tracking](../modules/production/quality-tracking.md) |
| Actual against target per shift or day? | `TargetEvents` | [Target tracking](../modules/production/target-tracking.md) |
| Actual against a production schedule or work orders? | `ProductionScheduleAdherenceEvents` | [Schedule adherence](../modules/production/schedule-adherence.md) |
| Output and quality per operator? | `OperatorPerformanceEvents` | [Operator performance](../modules/production/operator-performance.md) |
| Weekly and monthly roll-ups? | `PeriodSummaryEvents` | [Period summary](../modules/production/period-summary.md) |
| Batches, their durations and yield? | `BatchTrackingEvents` | [Batch tracking](../modules/production/batch-tracking.md) |
| Alarm floods, chattering and standing alarms? | `AlarmManagementEvents` | [Alarm management](../modules/production/alarm-management.md) |
| How often does a pump or valve cycle? | `DutyCycleEvents` | [Duty cycle](../modules/production/duty-cycle.md) |
| Operating hours and starts? | `RuntimeAccountingEvents` | [Runtime accounting](../modules/production/runtime-accounting.md) |
| Machine states without a run/idle flag, from current or power? | `OperatingStateEvents` | [Operating state](../modules/production/operating-state.md) |

## Traceability

Three classes follow parts through stations, depending on how the part id is
recorded:

| Class | Signals |
|---|---|
| `ValueTraceabilityEvents` | one part-id string signal per station |
| `RoutingTraceabilityEvents` | one part-id signal plus an integer routing-step signal |
| `MultiProcessTraceabilityEvents` | one part-id signal per process, optional handover signals between parallel cells |

```python
from ts_shape.events.production.order_traceability import ValueTraceabilityEvents

t = pd.date_range("2024-06-03 06:00", periods=5, freq="10min", tz="UTC")
trace_df = pd.DataFrame({
    "systime": list(t[:3]) + list(t[2:5]),
    "uuid": ["press:part_id"] * 3 + ["weld:part_id"] * 3,
    "value_string": ["P-001", "P-002", "P-003"] * 2,
})
trace = ValueTraceabilityEvents(trace_df, station_uuids={"press:part_id": "Press", "weld:part_id": "Weld"})
timeline = trace.build_timeline()   # one row per part and station visit
lead_times = trace.lead_time()      # first station to last station, per part
```

With one sample per visit the dwell time is 0; `build_timeline(infer_dwell_from_next=True)`
estimates it from the next sample. See [Value traceability](../modules/production/order-traceability.md),
[Routing traceability](../modules/production/routing-traceability.md) and
[Multi-process traceability](../modules/production/multi-process-traceability.md).
