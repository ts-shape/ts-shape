# Event Log Demo

Demonstrates the canonical OCEL 2.0 / XES-shaped event log produced by the `ts_shape.eventlog` package. Two detectors (machine state + outlier) are normalized into one `EventLog`, concatenated, and exported to both XES-flat and OCEL 2.0 tables.

**Run it:** `python examples/eventlog_demo.py`

**Modules demonstrated:** `to_event_log`, `concat`, `to_flat_df`, `to_ocel_tables`, `MachineStateEvents`, `OutlierDetectionEvents`

**Related guides:** [Event Log: pm4py-shaped output for process mining](../guides/eventlog.md)

---

```python
--8<-- "examples/eventlog_demo.py"
```
