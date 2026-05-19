# Event Handling — Visual Overview

!!! note
    This page zooms into the events layer. For the full library map
    with every package, class and detector method — searchable and
    clickable — see [Architecture](architecture.md).

How ts-shape turns raw signals into events. Three layers, four archetypes, one canonical event log.

Every detection in ts-shape — whether it comes from one of the 290 built-in detector methods or a [user-authored lambda rule](lambda-rules.md) — passes through the same three-layer flow:

1. **Signals** — raw timeseries DataFrame columns (booleans, floats, categoricals).
2. **Events** — rows in the canonical `EventLog` produced by a detector method.
3. **Rule** — the code or YAML that defines *when* signals trigger an event, classified into one of four **archetypes**: `threshold`, `interval`, `aggregate`, `static`.

The archetype determines the event's *shape* (point / interval / summary / static), which standard attributes the rule must populate (e.g. `ts_shape:method`, `ts_shape:lifecycle_state`), and which OCEL object types are auto-extracted.

The four diagrams below show one representative scenario per archetype. Read each as a stack: raw signals on top, derived events in the middle, rule definition on the bottom. Dashed links read **rule → signal it watches**; solid links read **rule → event it produces**.

---

## Archetype 1 — `threshold` (point shape)

Per-row check: a single sample compared against a reference. Outliers, SPC violations, tolerance deviations, exceedances. One True row → one point event.

**Representative scenario.** Tool wear: emit one event per torque sample that exceeds 75 Nm while the tool is running.

```mermaid
flowchart TB
    subgraph SIG["① Signals — raw DataFrame columns"]
        direction LR
        S1["<b>torque</b><br/>72 · 74 · <b>79.5</b> · <b>86</b> · 73 · <b>95</b> · 68"]
        S2["<b>state</b><br/>run · run · run · run · run · run · idle"]
        S3["<b>source_uuid</b><br/>asset-A everywhere"]
    end
    subgraph EVT["② Events — canonical EventLog rows"]
        direction LR
        E1["⚠ high_torque @ t₃<br/>severity=warn · value=79.5"]
        E2["🛑 high_torque @ t₄<br/>severity=critical · value=86"]
        E3["🛑 high_torque @ t₆<br/>severity=critical · value=95"]
    end
    subgraph RULE["③ Rule — threshold archetype"]
        R["<b>OutlierDetectionEvents.detect_outliers_iqr</b><br/>(built-in)<br/>— or —<br/><b>LambdaToolWear.high_torque</b><br/>(YAML rule)<br/><br/>expression: torque &gt; 75 &amp; state == 'run'<br/>activity template: maintenance.tool.high_torque<br/>required standard_attrs: ts_shape:method, ts_shape:direction"]
    end
    R -.watches.-> S1
    R -.watches.-> S2
    R ==>|emits| E1
    R ==>|emits| E2
    R ==>|emits| E3
    S3 -.auto-extract→ asset object.-> E1
    style SIG fill:#0f2a3d,stroke:#38bdf8,color:#e0f2fe
    style EVT fill:#3d2a0f,stroke:#fbbf24,color:#fef3c7
    style RULE fill:#1a3a2e,stroke:#34d399,color:#d1fae5
    style E1 fill:#78350f,color:#fef3c7
    style E2 fill:#7f1d1d,color:#fee2e2
    style E3 fill:#7f1d1d,color:#fee2e2
```

**Required `standard_attrs` for `threshold`:** `ts_shape:method`, `ts_shape:direction`.

---

## Archetype 2 — `interval` (interval shape)

Contiguous runs of True samples are coalesced (per group) into one event with `start`, `end`, duration. Optional `min_duration_s` filter rejects short blips (hysteresis).

**Representative scenario.** Sustained hot-bearing windows per machine: ignore single-sample spikes, flag any window of ≥ 30 s where `bearing_temp_c > 85`.

```mermaid
flowchart TB
    subgraph SIG["① Signals — multi-asset stream"]
        direction LR
        SA["<b>bearing_temp_c</b> @ asset-A<br/>82 · 86 · 88 · 87 · 84 · 83 · 90 · 82"]
        SB["<b>bearing_temp_c</b> @ asset-B<br/>80 · 81 · 82 · 80 · 89 · 90 · 91 · 92"]
        SU["<b>source_uuid</b><br/>used as group_by key"]
    end
    subgraph MASK["② Mask + coalesce per group"]
        direction LR
        MA["asset-A run 1: 3 samples · 45 s ✓"]
        MA2["asset-A run 2: 1 sample · 0 s ✗ drop"]
        MB["asset-B run: 4 samples · 45 s ✓"]
    end
    subgraph EVT["③ Events — interval rows"]
        direction LR
        EA["🟥 hot_window @ asset-A<br/>start=t₁ · end=t₃ · 45 s · mean=87"]
        EB["🟥 hot_window @ asset-B<br/>start=t₄ · end=t₆ · 45 s · mean=90"]
    end
    subgraph RULE["④ Rule — interval archetype"]
        R["<b>MachineStateEvents.detect_run_idle</b> (built-in)<br/>— or —<br/><b>LambdaBearing.hot_window</b> (YAML rule)<br/><br/>expression: bearing_temp_c &gt; 85<br/>min_duration_s: 30 · group_by: [source_uuid]<br/>activity template: maintenance.bearing.hot<br/>required standard_attrs: ts_shape:lifecycle_state"]
    end
    R -.watches.-> SA
    R -.watches.-> SB
    R -.group by.-> SU
    SA --> MA
    SA --> MA2
    SB --> MB
    MA ==> EA
    MB ==> EB
    MA2 -. dropped .-x EA
    R --> MA
    style SIG fill:#0f2a3d,stroke:#38bdf8,color:#e0f2fe
    style MASK fill:#2a1a3d,stroke:#a78bfa,color:#ede9fe
    style EVT fill:#3d2a0f,stroke:#fbbf24,color:#fef3c7
    style RULE fill:#1a3a2e,stroke:#34d399,color:#d1fae5
    style EA fill:#7f1d1d,color:#fee2e2
    style EB fill:#7f1d1d,color:#fee2e2
    style MA2 fill:#3f3f46,color:#a1a1aa,stroke-dasharray: 4 3
```

**Required `standard_attrs` for `interval`:** `ts_shape:lifecycle_state`.

---

## Archetype 3 — `aggregate` (summary shape)

Window-based statistics. One row per period × group: per-shift OEE, per-day production, hourly cycle-time stats. The event timestamp marks the period end; `ts_shape:start_timestamp` marks the period start.

**Representative scenario.** OEE by shift: roll cycle counts, downtime, and reject counts into one summary row per (shift × asset).

```mermaid
flowchart TB
    subgraph SIG["① Signals — multi-source"]
        direction LR
        SC["<b>cycle_count</b><br/>1 · 1 · 1 · 0 · 1 · 1 · …"]
        SR["<b>reject_count</b><br/>0 · 0 · 1 · 0 · 0 · 0 · …"]
        SS["<b>state</b> (run/idle/down)"]
        SP["<b>part_id, shift_id</b><br/>(context columns)"]
    end
    subgraph AGG["② Aggregation window"]
        direction LR
        W1["shift_id = A · 08:00–16:00<br/>parts=412 · rejects=7 · downtime=18min"]
        W2["shift_id = B · 16:00–00:00<br/>parts=389 · rejects=4 · downtime=22min"]
    end
    subgraph EVT["③ Events — summary rows"]
        direction LR
        EW1["📊 oee_shift @ shift-A<br/>availability=0.95 · performance=0.91 · quality=0.98<br/>OEE=0.85 · sample_count=412"]
        EW2["📊 oee_shift @ shift-B<br/>availability=0.93 · performance=0.89 · quality=0.99<br/>OEE=0.82 · sample_count=389"]
    end
    subgraph RULE["④ Rule — aggregate archetype"]
        R["<b>OEECalculator.calculate_oee</b> (built-in)<br/>— or —<br/><b>PartProductionTracking.daily_production_summary</b><br/><br/>group_by: [shift_id, asset]<br/>window: per shift<br/>activity template: production.oee.shift_summary<br/>required standard_attrs: ts_shape:sample_count"]
    end
    R -.watches.-> SC
    R -.watches.-> SR
    R -.watches.-> SS
    R -.group by.-> SP
    SC --> W1
    SR --> W1
    SS --> W1
    SC --> W2
    W1 ==> EW1
    W2 ==> EW2
    style SIG fill:#0f2a3d,stroke:#38bdf8,color:#e0f2fe
    style AGG fill:#2a1a3d,stroke:#a78bfa,color:#ede9fe
    style EVT fill:#3d2a0f,stroke:#fbbf24,color:#fef3c7
    style RULE fill:#1a3a2e,stroke:#34d399,color:#d1fae5
    style EW1 fill:#064e3b,color:#d1fae5
    style EW2 fill:#064e3b,color:#d1fae5
```

**Required `standard_attrs` for `aggregate`:** `ts_shape:sample_count`.

---

## Archetype 4 — `static` (static shape)

No natural time axis. Reference data, snapshots, top-N tables, gauge R&R results. One event per row, timestamped at the export moment so it still fits the canonical schema.

**Representative scenario.** SPC control-limit calculation: produce a single snapshot per signal capturing UCL/LCL/centre line for downstream rule evaluation.

```mermaid
flowchart TB
    subgraph SIG["① Signals — historical baseline window"]
        direction LR
        SH["<b>value_double</b><br/>(100 historical samples for asset-A · torque)"]
    end
    subgraph CALC["② Control-limit calculation"]
        C["mean = 42.1<br/>std = 1.4<br/>UCL = mean + 3σ = 46.3<br/>LCL = mean − 3σ = 37.9"]
    end
    subgraph EVT["③ Events — static rows"]
        direction LR
        ES["📐 spc_limits @ asset-A · torque<br/>method=xbar_r · UCL=46.3 · LCL=37.9 · mean=42.1<br/>sample_count=100"]
    end
    subgraph RULE["④ Rule — static archetype"]
        R["<b>StatisticalProcessControlRuleBased.calculate_control_limits</b><br/>— or —<br/><b>GaugeRepeatabilityEvents.repeatability</b><br/><br/>method: xbar_r / nelson / westgard<br/>activity template: quality.spc.control_limits<br/>required standard_attrs: ts_shape:method"]
    end
    R -.watches.-> SH
    SH --> C
    C ==> ES
    style SIG fill:#0f2a3d,stroke:#38bdf8,color:#e0f2fe
    style CALC fill:#2a1a3d,stroke:#a78bfa,color:#ede9fe
    style EVT fill:#3d2a0f,stroke:#fbbf24,color:#fef3c7
    style RULE fill:#1a3a2e,stroke:#34d399,color:#d1fae5
    style ES fill:#1e3a8a,color:#dbeafe
```

**Required `standard_attrs` for `static`:** `ts_shape:method`.

---

## Reading the diagrams

| Visual cue | Meaning |
|---|---|
| **Blue subgraph** | Raw signals — DataFrame columns the detector reads. |
| **Purple subgraph** | Intermediate computation (mask, coalescing, aggregation window). Not always present. |
| **Amber subgraph** | Events — rows in the canonical `EventLog`. Color of an event node hints at severity. |
| **Green subgraph** | The rule itself — code (built-in detector class.method) or YAML (lambda rule). |
| **Dashed arrow** | Rule *watches* this signal — it appears as a column reference in the trigger or as a parameter to the detector constructor. |
| **Solid bold arrow** | Rule *emits* this event row. |
| **Dotted X arrow** | Filtered out (e.g., interval shorter than `min_duration_s`). |

---

## Where this fits in the rest of the library

For the full module map — including loaders, transforms, features, and how the event layer plugs into them — see the architecture chart in [Concept](../concept.md#full-library-architecture). For the canonical event-log schema events ultimately land in, see [Event Log (XES & OCEL)](eventlog.md). For the user-authored path that emits straight into this same flow, see [Lambda Rules](lambda-rules.md).
