# Object analytics

[Object detection](objects.md) tells you when each serial, carrier or batch was
present at which station. The functions in `ts_shape.eventlog.analysis` turn those
presences into the numbers a line is run by: lead time, dwell per station, work in
process (WIP), carrier loop times and parts per object, plus enter / leave events
that give a serial-centric process-mining log real traces.

A small line with two stations, each with a serial reader:

```python
import pandas as pd
from ts_shape.eventlog import ObjectSpec, object_intervals

t0 = pd.Timestamp("2025-03-10 06:00", tz="UTC")


def reader(uuid, station, minutes, ids):
    return pd.DataFrame({
        "systime": [t0 + pd.Timedelta(minutes=m) for m in minutes],
        "uuid": uuid, "source_uuid": station, "value_string": ids,
    })


df = pd.concat([
    reader("st1:serial", "st1", [0, 10, 20, 40], ["S1", "S2", "S3", ""]),
    reader("st2:serial", "st2", [12, 25, 40, 45], ["S1", "S2", "S3", ""]),
], ignore_index=True)
specs = [ObjectSpec("st1:serial", "serial"), ObjectSpec("st2:serial", "serial")]
intervals = object_intervals(df, specs)
print(intervals[["oid", "start", "end", "source_uuid"]])
```

S3 leaves station 1 at 06:40, the instant it enters station 2. The empty readings end
the last serial at each station.

## Input and stations

Every function takes the interval table of `object_intervals` or an `EventLog`:

- a log made from `presence_events` (below), alone or concatenated with other logs:
  the presences are rebuilt from its enter / leave events;
- else the `active` / `released` lifecycle in `object_changes` of the
  `detect_objects` log. Those presences have no station.

The **station** of a presence is its `source_uuid`, or its id signal `uuid` when it
has none (one reader per station). Rows without an `oid`, `type`, `start` or `end`
are skipped; a presence that ends before it starts raises `ValueError`.

Times keep the zone of the input. The interval table is in UTC; pass `tz=` to
`object_kpis` and `wip_over_time` for local windows: a daily window on a DST day then
lasts 23 or 25 hours. Naive times are read as UTC when converted.

## Enter and leave events

`presence_events(intervals, object_type=None)` gives one `enter` and one `leave`
point event per presence, with `systime`, `oid`, `type`, `event`, `station`,
`presence_index` (0-based per object), `uuid` and `source_uuid`. At one instant the
events of an object follow its presences: S3 leaves station 1 before it enters
station 2.

```python
from ts_shape.eventlog import presence_events, to_event_log, to_event_log_xes

events = presence_events(intervals)
log = to_event_log(events, detector="eventlog.presence_events")
xes = to_event_log_xes(log, case_object_type="serial")
print(xes.loc[xes["case:concept:name"] == "S3", ["concept:name", "time:timestamp"]])
```

`to_event_log` names the activity `production.presence.<event>.<station>` and links
each event to its object (`S3` of type `serial`) and to its station as `asset`, so
every serial is a trace: enter station 1, leave it, enter station 2, leave it. The
same log goes to OCEL with `to_event_log_ocel`.

## Lead time and dwell

`object_kpis(intervals, object_type=..., window=None, tz=None)` returns an
`ObjectKPIs` with three tables:

- `objects`: one row per object with `start` (first enter), `end` (last leave),
  `lead_time_seconds`, `presence_count`, `station_count` and `dwell_seconds` (the
  sum of its presence durations).
- `stations`: one row per object and station, in order of arrival, with `start`,
  `end`, `presence_count` and `dwell_seconds`.
- `throughput`: with `window=`, one row per window with `completed_count` (objects
  whose last leave falls in the window) and their `mean_lead_time_seconds` (NaN when
  none completed). `None` without `window`.

```python
from ts_shape.eventlog import object_kpis

kpis = object_kpis(intervals, object_type="serial", window="15min")
print(kpis.objects[["oid", "lead_time_seconds", "presence_count", "dwell_seconds"]])
print(kpis.stations[["oid", "station", "dwell_seconds"]])
print(kpis.throughput[["start", "completed_count", "mean_lead_time_seconds"]])
```

S1 takes 25 minutes from entering station 1 to leaving station 2, of which it spends
23 at a station: the 2 minutes between 06:10 and 06:12 it is between stations. An
object still present at the end of the data ends there. Presences at two stations
at once both count in `dwell_seconds`, so dwell can exceed the lead time.

## Work in process

`wip_over_time(intervals, object_type=..., window=..., tz=None)` gives, per window,
`object_seconds` (the presence time of all objects in the window), `wip_mean`
(`object_seconds / duration_seconds`, the time-weighted average) and `wip_max` (the
most objects present at once).

```python
from ts_shape.eventlog import wip_over_time

wip = wip_over_time(intervals, object_type="serial", window="15min")
print(wip[["start", "object_seconds", "wip_mean", "wip_max"]])
```

An object counts once while it is at two stations, and a hand-over does not count it
twice. The windows cover the data from the first enter to the last leave and are
averaged over their whole length.

## Carrier loops

`carrier_loops(intervals, object_type="carrier", start_station=None)` measures
carriers that circulate: a loop is the time between two successive arrivals at the
start station (by default the station each carrier is first seen at).

```python
from ts_shape.eventlog import carrier_loops


def seen(carrier, station, start, end):
    return {"oid": carrier, "type": "carrier", "uuid": station,
            "start": t0 + pd.Timedelta(minutes=start),
            "end": t0 + pd.Timedelta(minutes=end)}


carriers = pd.DataFrame([
    seen("C1", "load", 0, 3), seen("C1", "unload", 5, 8),
    seen("C1", "load", 20, 23), seen("C1", "unload", 26, 29),
    seen("C1", "load", 43, 46),
])
loops = carrier_loops(carriers)
print(loops.loops[["oid", "station", "loop_index", "duration_seconds"]])
print(loops.carriers)
```

`loops.loops` has one row per loop (`start` is the arrival, `end` the next arrival);
`loops.carriers` one row per carrier with `loop_count` and `mean_loop_seconds`.

## Parts per object

`attach_counts(intervals, dataframe, counter_uuid, object_type=..., value_column=
"value_integer", station=None)` books a counter's production to the objects present
at the counter's station (its `source_uuid`, or `station=`).

```python
from ts_shape.eventlog import attach_counts

counter = pd.DataFrame({
    "systime": [t0 + pd.Timedelta(minutes=m) for m in [12, 18, 30, 32, 40, 47]],
    "uuid": "st2:good", "source_uuid": "st2",
    "value_integer": [100, 106, 112, 2, 6, 9],
})
print(attach_counts(intervals, counter, "st2:good", object_type="serial"))
```

- The increment at a reading was produced since the reading before. It is shared
  between the objects present in that time, in proportion to the time each was
  present: the 6 parts of 06:18-06:30 go 3.5 to S1 (7 minutes) and 2.5 to S2
  (5 minutes). Objects present at once share an increment, never more than all of it.
- Resets follow the library's counter rule
  ([`counter_increments`](../../reference/ts_shape/utils/counters.md)): a reading
  lower than the one before is a reset, and its value is the production since the
  reset. At 06:32 the counter drops from 112 to 2: 2 parts for S2.
- Production while no object is present is reported in a last row with `oid` `<NA>`:
  S3 leaves at 06:45, so 2/7 of the 3 parts counted at 06:47 are unattributed.

The quantities add up to the counter's total.

## Traceability classes

`ValueTraceabilityEvents`, `RoutingTraceabilityEvents` and
`MultiProcessTraceabilityEvents` build a station timeline per item themselves.
Their `to_intervals(object_type="serial")` returns that timeline as an interval
table with the columns of `object_intervals` plus `station` (the station name, which
the functions on this page use as the station), so every function here takes it:

```python
from ts_shape.events.production import ValueTraceabilityEvents

trace = ValueTraceabilityEvents(
    df, station_uuids={"st1:serial": "Station 1", "st2:serial": "Station 2"}
)
for end in ("last_sample", "next_change"):
    kpis = object_kpis(trace.to_intervals(interval_end=end), object_type="serial")
    print(end)
    print(kpis.objects[["oid", "lead_time_seconds", "dwell_seconds"]])
```

The two settings of `interval_end` differ in where a visit ends:

- `"last_sample"` (the default of the classes, so their results stay as they were):
  at the item's last sample at the station. These readers write each serial once, on
  arrival, so every visit lasts 0 seconds and the lead time runs from the first
  arrival to the last arrival: 12 minutes for S1 (06:00 to 06:12).
- `"next_change"`: at the station's next different reading (the next serial or an
  empty reading), the way `object_intervals` holds an id until it changes; a
  station's last visit is held until the last timestamp of the station signals. With
  it, `to_intervals` gives the same presences as `object_intervals` above, and S1
  again takes 25 minutes with 23 at a station.

`to_intervals` keeps the class's own reading of the signals: every non-empty value
is an id (misreads are not filtered, unlike `ObjectSpec.exclude`), and
`MultiProcessTraceabilityEvents` also splits a visit at a time gap.

The classes' event logs link to their objects too: in `to_event_log`, the item column
of a timeline, lead time or status frame binds objects of type `serial`, the station
column objects of type `station`. The event ids stay what they were without these
links.

```python
log = to_event_log(trace.build_timeline(), detector="ValueTraceabilityEvents.build_timeline")
print(log.objects.sort_values(["ocel:type", "ocel:oid"]).to_string(index=False))
```

A frame from `to_intervals` converts with `detector="<class>.to_intervals"`: one
interval event per visit, linked to its serial, its station and its signal.
