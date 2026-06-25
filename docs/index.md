---
hide:
  - navigation
  - toc
---

<style>
.md-typeset h1 { display: none; }
</style>

<div class="tx-hero" markdown>

# **ts-shape**

<div class="tx-hero__tagline">
From raw industrial timeseries to OEE, SPC, and process-mining-ready event logs — in pure pandas.
</div>

<p class="tx-hero__subtext">
A composable Python toolkit for loading, shaping, and analysing manufacturing &amp; IoT signals.
<strong>DataFrame in, DataFrame out</strong> — across loaders, transforms, features, and 290+ event detectors.
</p>

<div class="tx-hero__badges">
<a href="https://pypi.org/project/ts-shape/"><img src="https://img.shields.io/pypi/v/ts-shape.svg" alt="PyPI"></a>
<a href="https://pepy.tech/projects/ts-shape"><img src="https://static.pepy.tech/badge/ts-shape" alt="Downloads"></a>
<a href="https://pypi.org/project/ts-shape/"><img src="https://img.shields.io/pypi/pyversions/ts-shape.svg" alt="Python"></a>
<a href="license.md"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</div>

[Get Started](user_guide/installation.md){ .md-button .md-button--primary }
[See Pipelines](pipelines/index.md){ .md-button }
[GitHub](https://github.com/ts-shape/ts-shape){ .md-button }

```bash
pip install ts-shape
```

</div>

---

<div class="tx-section-heading" markdown>

## Why ts-shape?

</div>

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **DataFrame-First**

    ---

    Every operation takes and returns a Pandas DataFrame. No proprietary
    formats, no lock-in — drop straight into any notebook or pipeline.

-   :material-factory:{ .lg .middle } **290+ Detectors, 8 Packs**

    ---

    OEE, SPC, cycle times, downtime, traceability, energy, maintenance —
    production use cases, batteries included.

-   :material-cloud-sync:{ .lg .middle } **Multi-Source Loading**

    ---

    Parquet, S3, Azure Blob, TimescaleDB behind one interface. Vectorised,
    chunked, concurrent.

-   :material-sitemap:{ .lg .middle } **Process-Mining Native**

    ---

    Every detector normalizes into a canonical **OCEL 2.0 / XES** event log —
    ready for pm4py, Celonis, or Disco.

</div>

---

<div class="tx-section-heading" markdown>

## Signals to event logs, in four steps

</div>

```python
from ts_shape.loader.timeseries.parquet_loader import ParquetLoader
from ts_shape.events.production.machine_state import MachineStateEvents
from ts_shape.eventlog import to_event_log, to_event_log_ocel

# 1. Load raw signals
df = ParquetLoader.load_by_uuids("data/", ["machine-state"], "2024-01-01", "2024-01-31")

# 2. Detect events
intervals = MachineStateEvents(df, run_state_uuid="machine-state").detect_run_idle(min_duration="30s")

# 3. Build the canonical event log, then 4. export OCEL 2.0
log = to_event_log(intervals, detector="MachineStateEvents.detect_run_idle")
tables = to_event_log_ocel(log)
```

Any detector's output flows into the **same canonical event log** — that is what keeps the library working end to end.

---

<div class="tx-section-heading" markdown>

## Explore the docs

</div>

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quick Start**

    ---

    Install and run your first pipeline in minutes.

    [:octicons-arrow-right-24: Get started](user_guide/quick_start.md)

-   :material-code-tags:{ .lg .middle } **Guides**

    ---

    Topic-focused guides from data acquisition to shift reports.

    [:octicons-arrow-right-24: See guides](guides/index.md)

-   :material-pipe:{ .lg .middle } **Pipelines**

    ---

    End-to-end workflows from raw signals to production KPIs.

    [:octicons-arrow-right-24: View pipelines](pipelines/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    Complete auto-generated API documentation.

    [:octicons-arrow-right-24: Browse API](reference/index.md)

</div>

---

<div align="center" markdown>

**MIT License** — Built for the timeseries community

</div>
