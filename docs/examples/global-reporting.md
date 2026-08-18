# Global Reporting Demo

Consolidates three plants on three continents into one report: facility master
data and its validation, local production days across timezones (DST included),
coverage checks, weighted global and regional rollups, benchmarking with
opportunity sizing, rate-versus-mix variance, currency normalisation, and the
management scorecard.

**Run it:** `python examples/global_reporting_demo.py`

**Modules demonstrated:** `FacilityRegistry`, `FacilityCalendar`, `ReportingCoverage`, `FleetRollup`, `MetricSpec`, `FacilityBenchmark`, `VarianceBridge`, `MetricNormalizer`, `GlobalScorecard`

**Related guides:** [Global Reporting](../guides/global-reporting.md) | [Shift Reports & KPIs](../guides/reporting.md)

---

```python
--8<-- "examples/global_reporting_demo.py"
```
