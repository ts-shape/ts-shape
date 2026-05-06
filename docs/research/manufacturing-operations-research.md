# Manufacturing Operations: Market Research, Gap Analysis & Proposed Methods

**Date:** May 2026  
**Scope:** Common manufacturing industry challenges vs. procedures implemented in ts-shape, with proposals for additional methods.

---

## 1. Market Research: Top Manufacturing Operations Issues

The following issues represent the most frequently cited operational challenges across discrete, process, and hybrid manufacturing industries (sourced from industry surveys by Deloitte, McKinsey, LNS Research, and Manufacturing Leadership Council, 2024–2026).

---

### 1.1 Unplanned Equipment Downtime

**Industry Data:**
- Unplanned downtime costs manufacturers an estimated **$50 billion per year** globally (Siemens/Aberdeen Research).
- 82% of companies have experienced at least one unplanned downtime event in the past three years (Vanson Bourne, 2024).
- Average cost per hour of unplanned downtime: **$260,000** in automotive; **$490,000** in semiconductor.

**Root Causes Cited:**
- Lack of predictive/prescriptive maintenance programs
- Reactive-only maintenance cultures
- Poor sensor coverage and data collection gaps
- Insufficient vibration, thermal, and oil-analysis monitoring
- No remaining useful life (RUL) forecasting

---

### 1.2 Poor Overall Equipment Effectiveness (OEE)

**Industry Data:**
- World-class OEE is considered **85%+**; the average manufacturer operates at **60%** (MESA International).
- Performance losses (speed losses, micro-stops) account for **~36%** of total OEE losses.
- Availability losses (unplanned downtime, changeovers) account for **~32%** of OEE losses.
- Quality losses (defects, rework, scrap) account for **~32%** of OEE losses.

**Root Causes Cited:**
- No real-time OEE visibility at the machine level
- Changeover times not systematically tracked or reduced (SMED)
- Micro-stops not captured (below threshold for formal downtime logging)
- No shift-to-shift or line-to-line OEE benchmarking

---

### 1.3 Quality Defects and Scrap

**Industry Data:**
- Manufacturing companies spend **20–30% of revenue** on quality-related costs (Cost of Poor Quality — CoPQ), according to ASQ.
- First Pass Yield (FPY) below 90% is reported by 43% of manufacturers (LNS Research Quality Management Survey).
- Defect detection at end-of-line is 10x more expensive than detection at source (Rule of Ten).
- SPC adoption remains below **40%** in SME manufacturers despite proven ROI.

**Root Causes Cited:**
- No real-time Statistical Process Control (SPC) at the machine level
- Inadequate measurement system analysis (Gauge R&R)
- Sensor drift going undetected
- No automated tolerance deviation alerting
- Process capability (Cpk) not monitored continuously

---

### 1.4 Supply Chain Disruptions and Inventory Imbalances

**Industry Data:**
- 73% of manufacturers experienced significant supply chain disruptions in 2024–2025 (Gartner).
- Stockouts and overstocking cost manufacturers **$1.1 trillion** annually worldwide.
- Average lead time variability increased by **34%** post-2020 and has not fully recovered.
- Demand forecast accuracy below 75% is reported by 55% of manufacturers.

**Root Causes Cited:**
- No real-time inventory level monitoring with early warning
- Lead time anomalies not detected until they cause production stoppages
- Demand spikes and seasonality patterns not automatically identified
- Lack of multi-tier supply chain visibility

---

### 1.5 Energy Waste and Sustainability Compliance

**Industry Data:**
- Energy represents **8–12%** of total manufacturing costs on average (U.S. DOE).
- Manufacturers face increasing regulatory pressure: EU's CSRD, EPA Subpart W, ISO 50001.
- 60% of manufacturers cite energy cost reduction as a top 5 operational priority (McKinsey, 2025).
- Industrial energy waste is estimated at **30%** of total consumption due to inefficiencies.

**Root Causes Cited:**
- No real-time energy consumption monitoring by machine or process area
- Energy efficiency KPIs not integrated with production KPIs
- Inability to correlate energy spikes with production events (setpoint changes, startups)
- No automated reporting for regulatory compliance (Scope 1/2/3)

---

### 1.6 Workforce Productivity and Skills Gaps

**Industry Data:**
- 2.1 million manufacturing jobs could go unfilled by 2030 due to skills gaps (Deloitte/NAM).
- 67% of manufacturers cite "finding qualified workers" as their #1 challenge.
- Operator-to-operator variability accounts for **15–25%** of output quality variation in labor-intensive processes.
- Knowledge capture from retiring workers is cited as critical by 71% of plant managers.

**Root Causes Cited:**
- Operator performance not systematically tracked
- No automated shift handover reports to preserve production knowledge
- No benchmarking of operator performance vs. target
- Training gaps not linked to quality/output data

---

### 1.7 Process Variability and Lack of Process Control

**Industry Data:**
- 68% of manufacturers report process variability as a key driver of waste (APQC).
- Setpoint deviations are the #1 root cause in process manufacturing defect investigations (ARC Advisory).
- Control loop health is only formally assessed in **25%** of plants.
- Disturbance recovery times are rarely measured despite direct impact on yield.

**Root Causes Cited:**
- No automated setpoint change detection
- Control loop performance not monitored (oscillation, offset, dead band)
- Process stability not continuously scored
- No detection of operating range violations

---

### 1.8 Alarm Flood and Alarm Fatigue

**Industry Data:**
- The average control room operator acknowledges **300+ alarms per shift** in process industries (EEMUA 191).
- 40% of configured alarms are "nuisance alarms" that require no action (ISA-18.2 benchmarks).
- Alarm floods (>10 alarms/10 minutes) occur in 65% of plants at least monthly.
- Alarm fatigue contributes to human error and near-miss incidents.

**Root Causes Cited:**
- No alarm rationalization or chattering detection
- Standing alarms not identified or cleared
- No systematic alarm KPI reporting (average alarm rate, flood events)

---

### 1.9 Traceability and Regulatory Compliance

**Industry Data:**
- Product recalls cost manufacturers an average of **$10 million per event** (Marsh & McLennan).
- 55% of manufacturers in regulated industries (pharma, food, auto) cite traceability as a top 5 compliance risk.
- FDA 21 CFR Part 11 / EUDRALEX Annex 11 require electronic batch records in pharma.
- ISO/TS 16949 and IATF 16949 mandate part-level traceability in automotive.

**Root Causes Cited:**
- No end-to-end part or batch traceability across production stations
- Batch genealogy not captured in real time
- No routing deviation detection

---

### 1.10 Production Scheduling and Bottleneck Management

**Industry Data:**
- Production bottlenecks reduce throughput by an average of **20–35%** in mixed-model assembly (Theory of Constraints research).
- 58% of manufacturers cannot identify their current production bottleneck in real time (LNS).
- Line balancing and takt time adherence are manually managed in 70% of SME facilities.
- Changeover inefficiency contributes to **12–18%** of available production time loss.

**Root Causes Cited:**
- No automated bottleneck identification on production lines
- Throughput and takt time deviations not monitored in real time
- Flow constraints (blocked/starved) not distinguished from true downtime
- Changeover optimization not data-driven

---

## 2. Mapping: Industry Issues vs. Implemented Procedures in ts-shape

The table below maps each identified industry challenge to the corresponding ts-shape modules and methods.

| # | Manufacturing Issue | ts-shape Coverage | Coverage Level |
|---|---|---|---|
| 1.1 | Unplanned Equipment Downtime | `DegradationDetectionEvents`, `FailurePredictionEvents`, `VibrationAnalysisEvents` | **Partial** |
| 1.2 | Poor OEE | `OEECalculator`, `MachineStateEvents`, `MicroStopEvents`, `ChangeoverEvents`, `LineThroughputEvents` | **Strong** |
| 1.3 | Quality Defects & Scrap | `StatisticalProcessControlRuleBased`, `OutlierDetectionEvents`, `ToleranceDeviationEvents`, `CapabilityTrendingEvents`, `GaugeRepeatabilityEvents`, `ScrapTracking`, `QualityTracking` | **Strong** |
| 1.4 | Supply Chain Disruptions | `InventoryMonitoringEvents`, `LeadTimeAnalysisEvents`, `DemandPatternEvents` | **Partial** |
| 1.5 | Energy Waste | `ConsumptionAnalysisEvents`, `EfficiencyTrackingEvents` | **Partial** |
| 1.6 | Workforce Productivity | `OperatorPerformanceTracking`, `ShiftReporting`, `ShiftHandoverReport`, `TargetTracking` | **Good** |
| 1.7 | Process Variability | `SetpointChangeEvents`, `ControlLoopHealthEvents`, `ProcessStabilityIndex`, `OperatingRangeEvents`, `DisturbanceRecoveryEvents` | **Strong** |
| 1.8 | Alarm Flood & Fatigue | `AlarmManagementEvents` (chattering, standing alarms, flood detection) | **Strong** |
| 1.9 | Traceability & Compliance | `ValueTraceabilityEvents`, `OrderTraceabilityEvents`, `RoutingTraceabilityEvents`, `MultiProcessTraceabilityEvents`, `BatchTrackingEvents` | **Strong** |
| 1.10 | Bottleneck Management | `BottleneckDetectionEvents`, `FlowConstraintEvents`, `LineThroughputEvents`, `ChangeoverEvents` | **Strong** |

### Coverage Legend
- **Strong** — dedicated, production-ready implementation covering the core use cases
- **Good** — solid coverage with minor gaps in depth or breadth  
- **Partial** — foundational functionality present but significant capability gaps remain
- **Missing** — no current implementation

---

## 3. Detailed Gap Analysis

### 3.1 Gap: Predictive Maintenance Depth (Issue 1.1)

**What is implemented:**
- `DegradationDetectionEvents`: trend, variance increase, level shift, health score (0–100)
- `FailurePredictionEvents`: linear/exponential RUL estimation with failure threshold
- `VibrationAnalysisEvents`: RMS, amplitude, kurtosis, crest factor

**What is missing:**
- **Thermal imaging / temperature-based degradation**: No integration of thermal sensor analysis for bearing overheating, motor winding faults, or lubrication failures
- **Oil analysis / contamination signals**: No lubrication health monitoring module
- **Multi-signal anomaly fusion**: Degradation events are per-signal; no fusion of vibration + temperature + current draw into a single health index per asset
- **Failure mode classification**: RUL is estimated but the *type* of failure (bearing fault vs. imbalance vs. misalignment) is not classified
- **Prescriptive maintenance recommendations**: No mapping from health score to specific maintenance action

---

### 3.2 Gap: Supply Chain Visibility (Issue 1.4)

**What is implemented:**
- `InventoryMonitoringEvents`: stock level tracking, reorder point detection, stockout prediction
- `LeadTimeAnalysisEvents`: lead time calculation, anomaly detection
- `DemandPatternEvents`: demand spikes, seasonality detection

**What is missing:**
- **Multi-tier supplier risk scoring**: No supplier performance tracking (on-time delivery rate, defect rate per supplier)
- **Production-to-supply coupling**: No linkage between production schedule deviations and upstream inventory signals
- **Goods receiving quality inspection events**: No integration of incoming material quality data
- **Safety stock optimization signals**: No dynamic safety stock recommendation based on lead time variance

---

### 3.3 Gap: Energy & Sustainability (Issue 1.5)

**What is implemented:**
- `ConsumptionAnalysisEvents`: energy consumption trending
- `EfficiencyTrackingEvents`: energy efficiency KPI tracking

**What is missing:**
- **Production-normalized energy (EnPI)**: No Energy Performance Indicator calculation — energy per unit produced, per machine, per shift
- **Carbon intensity tracking**: No CO2/kWh or Scope 1/2 emissions calculation for regulatory reporting
- **Peak demand event detection**: No identification of demand spikes that trigger penalty tariffs
- **Energy baseline modeling**: No regression-based energy baseline (ISO 50001 methodology) to separate structural from behavioral waste
- **Idle energy detection**: No detection of machines consuming energy during non-production periods

---

### 3.4 Gap: Quality — Incoming & SPC Breadth (Issue 1.3)

**What is partially covered:**
- SPC covers Western Electric rules (8 rules) for univariate signals
- Gauge R&R is implemented for measurement system analysis

**What is missing:**
- **Multivariate SPC (MSPC)**: Hotelling T² and CUSUM charts for processes with correlated quality variables
- **Automated root cause suggestion**: When an SPC rule fires, no causal factor ranking is proposed (e.g., "this alarm historically correlates with tool wear signal X")
- **First Article Inspection (FAI) tracking**: No structured new-part introduction quality workflow
- **Customer complaint / field return linkage**: No feedback loop from field quality to production parameter correlation

---

## 4. Proposed New Methods

The following new modules are recommended to close the identified gaps and align ts-shape with emerging industry needs.

---

### 4.1 `AssetHealthIndexEvents` (Predictive Maintenance)
**Addresses:** Gap 3.1 — multi-signal health fusion

**Concept:** Fuse degradation signals from vibration, temperature, current, and operational hours into a single normalized asset health index (0–100) per piece of equipment. Weight signals by failure mode criticality. Emit events when health crosses configurable warning and critical thresholds.

**Key outputs:**
- `asset_health_score`: composite 0–100 index
- `dominant_failure_mode`: classification (bearing, thermal, electrical, mechanical)
- `recommended_maintenance_window`: estimated days until intervention required
- `confidence_level`: model confidence based on signal availability

**Implementation approach:**
- Extend `DegradationDetectionEvents` to accept a list of signal UUIDs per asset
- Apply configurable weighting matrix (default: equal weights)
- Output single event row per asset per evaluation window
- No external ML dependency — use weighted scoring and threshold logic consistent with existing architecture

---

### 4.2 `IdleEnergyDetectionEvents` (Energy)
**Addresses:** Gap 3.3 — idle energy detection

**Concept:** Cross-reference machine state (from `MachineStateEvents`) with energy consumption data to detect and quantify energy consumed during idle/off-shift periods. Emit events for machines consuming above baseline energy while in non-production state.

**Key outputs:**
- `idle_energy_kwh`: energy consumed during idle intervals
- `idle_duration_minutes`: corresponding idle duration
- `idle_fraction`: idle energy as fraction of total consumption
- `estimated_annual_waste_kwh`: annualized idle waste extrapolation

**Implementation approach:**
- Accept machine state DataFrame (from `MachineStateEvents`) and energy signal DataFrame
- Merge on time intervals using `pd.merge_asof` or interval join
- Compute energy integrals (trapz) over idle windows

---

### 4.3 `EnergyPerformanceIndicatorEvents` (Energy)
**Addresses:** Gap 3.3 — production-normalized energy (EnPI)

**Concept:** Calculate energy consumed per unit of production output, tracking EnPI trends and deviations. Supports ISO 50001 energy management system requirements. Compares current EnPI vs. baseline period EnPI.

**Key outputs:**
- `enpi_value`: kWh per unit produced
- `enpi_vs_baseline_pct`: % deviation from baseline EnPI
- `production_volume`: units produced in the period
- `energy_consumed_kwh`: total energy in the period
- `enpi_trend`: improving / stable / degrading

**Implementation approach:**
- Accept energy consumption DataFrame and production count DataFrame
- Aggregate to configurable periods (shift, daily, weekly)
- Compute linear regression baseline from historical window
- Flag periods where EnPI exceeds baseline + configurable sigma threshold

---

### 4.4 `CarbonIntensityEvents` (Energy / Sustainability)
**Addresses:** Gap 3.3 — Scope 1/2 emissions tracking

**Concept:** Convert energy consumption and fuel usage signals to CO2-equivalent emissions using configurable emission factors (kgCO2e/kWh for electricity by grid region; kgCO2e/m³ for natural gas, etc.). Supports CSRD and ISO 14064 reporting.

**Key outputs:**
- `scope1_kgco2e`: direct emissions (gas, diesel, process emissions)
- `scope2_kgco2e`: indirect electricity emissions
- `total_kgco2e`: combined
- `intensity_kgco2e_per_unit`: carbon intensity per unit produced
- `emission_factor_used`: audit trail for the factor applied

**Implementation approach:**
- Accept energy/fuel signal DataFrames + emission factor configuration (JSON/dict)
- Multiply consumption by emission factors
- Aggregate over configurable reporting periods

---

### 4.5 `SupplierPerformanceEvents` (Supply Chain)
**Addresses:** Gap 3.2 — supplier risk scoring

**Concept:** Track and score supplier delivery performance based on goods receiving events: on-time delivery rate, quantity accuracy, incoming quality rejection rate, and lead time variance per supplier.

**Key outputs:**
- `supplier_id`: supplier identifier
- `on_time_delivery_rate_pct`: % deliveries on time
- `quantity_accuracy_pct`: % orders with correct quantities
- `incoming_quality_rejection_rate_pct`: % lots rejected at receiving
- `avg_lead_time_days` / `lead_time_cv`: mean and coefficient of variation
- `supplier_risk_score`: composite score (0–100, higher = more risk)

**Implementation approach:**
- Accept goods-receiving event DataFrame (expected vs. actual delivery timestamps, quantities, quality status)
- Compute rolling statistics per supplier UUID over configurable lookback window
- Emit events when supplier risk score crosses warning/critical thresholds

---

### 4.6 `MultivariateProcessControlEvents` (Quality)
**Addresses:** Gap 3.4 — multivariate SPC

**Concept:** Detect out-of-control conditions in processes with multiple correlated quality variables using Hotelling T² statistic. Complements the existing univariate `StatisticalProcessControlRuleBased` for processes where individual signals appear in-control but their joint distribution shifts.

**Key outputs:**
- `t2_statistic`: Hotelling T² value
- `t2_ucl`: upper control limit (chi-squared or F-distribution based)
- `out_of_control`: boolean flag
- `contribution_scores`: per-variable contribution to T² (for root cause identification)
- `phase`: Phase I (historical) or Phase II (monitoring)

**Implementation approach:**
- Accept wide-format DataFrame (one column per quality variable, one row per sample)
- Estimate mean vector and covariance matrix from Phase I data
- Compute T² for each new sample: `T² = (x - x̄)ᵀ Σ⁻¹ (x - x̄)`
- Use `np.linalg.inv` for covariance inversion (no external ML dependency)
- Emit event rows with start/end timestamps aligned to existing event schema

---

### 4.7 `SafetyEventDetectionEvents` (Production — New Domain)
**Addresses:** Safety risk management — currently entirely absent from ts-shape

**Concept:** Monitor safety-critical signals for near-miss conditions: pressure relief valve activations, emergency stop events, interlock trips, and safety system bypasses. This is a frequently requested capability as manufacturing operations digitize safety monitoring.

**Key outputs:**
- `safety_event_type`: e-stop, interlock_trip, relief_valve, bypass_active
- `duration_seconds`: duration of the safety condition
- `zone_id`: production zone / safety area affected
- `is_repeated`: whether this safety event repeats within a configurable window (pattern detection)
- `severity`: low / medium / high / critical

**Implementation approach:**
- Accept boolean and discrete-value signal DataFrames
- Define safety event types via configuration (signal UUID + trigger condition)
- Apply edge detection and duration measurement (consistent with `MachineStateEvents` approach)
- Repeated event detection using sliding window count

---

### 4.8 `ProductionScheduleAdherenceEvents` (Production)
**Addresses:** Schedule vs. actual tracking — currently absent

**Concept:** Compare planned production schedules (imported as a target DataFrame) against actual production output to detect schedule deviations, catch-up pressure events, and early/late completion patterns.

**Key outputs:**
- `planned_qty`: scheduled production quantity for the period
- `actual_qty`: actual quantity produced
- `schedule_adherence_pct`: actual / planned × 100
- `deviation_units`: actual − planned
- `deviation_type`: ahead / on_track / behind / critical_behind
- `recovery_required_rate`: units/hour needed to recover by end of shift

**Implementation approach:**
- Accept planned schedule DataFrame (period start/end, planned quantity, order ID) and actual `PartProductionTracking` output
- Merge on time periods and compute adherence metrics
- Emit events for deviations exceeding configurable thresholds

---

### 4.9 `RootCauseCorrelationEvents` (Correlation — Enhancement)
**Addresses:** Gap 3.4 — automated root cause suggestion for quality and downtime events

**Concept:** When a quality excursion or downtime event is detected, automatically rank potential causal signals by their temporal correlation with the event onset. Extends existing `AnomalyCorrelationEvents` with an event-driven, directed analysis mode.

**Key outputs:**
- `trigger_event_type`: the event that triggered the search (SPC violation, downtime, etc.)
- `trigger_uuid`: signal UUID of the triggering event
- `causal_candidates`: list of `{uuid, correlation_coefficient, lag_seconds, p_value}`
- `top_cause_uuid`: UUID of highest-ranked causal candidate
- `top_cause_lag_seconds`: typical lead time of cause before effect

**Implementation approach:**
- Accept trigger event DataFrame + full signal DataFrame for the lookback window
- For each candidate signal, compute cross-correlation with the trigger signal over configurable lag range
- Filter by statistical significance (p-value threshold)
- Return ranked candidates — no ML required, pure signal processing

---

## 5. Priority Roadmap

Based on industry impact (frequency × cost) and implementation effort, the following priority order is recommended:

| Priority | Proposed Module | Business Driver | Effort Estimate |
|---|---|---|---|
| P1 | `AssetHealthIndexEvents` | Downtime cost reduction — highest ROI category | Medium |
| P1 | `IdleEnergyDetectionEvents` | Quick win — energy waste is immediately measurable | Low |
| P1 | `ProductionScheduleAdherenceEvents` | Closes OEE-to-scheduling gap; common customer request | Low |
| P2 | `EnergyPerformanceIndicatorEvents` | ISO 50001 / sustainability reporting demand | Medium |
| P2 | `SafetyEventDetectionEvents` | Regulatory & insurance pressure; new market segment | Medium |
| P2 | `RootCauseCorrelationEvents` | Amplifies value of all existing event detectors | Medium |
| P3 | `MultivariateProcessControlEvents` | High value in process manufacturing; more complex | High |
| P3 | `CarbonIntensityEvents` | CSRD compliance deadline pressure — EU market | Medium |
| P3 | `SupplierPerformanceEvents` | Requires external data (goods receiving); integration dependency | High |

---

## 6. Summary

ts-shape is a **highly capable manufacturing analytics platform** with strong coverage of OEE, quality/SPC, process engineering, alarm management, traceability, and bottleneck analysis — all areas confirmed by market research as top manufacturing challenges.

**Strongest areas** relative to industry needs:
- OEE decomposition and micro-stop detection
- Statistical process control (8 Western Electric rules + Cp/Cpk trending)
- Traceability across multi-process topologies
- Alarm management (chattering, standing alarms, ISA-18.2 KPIs)
- Process engineering (control loop health, setpoint analysis, disturbance recovery)

**Key gaps** to address:
1. **Predictive maintenance** lacks multi-signal health fusion and failure mode classification
2. **Energy management** lacks production-normalized EnPI, idle energy detection, and carbon reporting
3. **Supply chain** lacks supplier performance scoring and incoming quality integration
4. **Quality** lacks multivariate SPC for correlated variable processes
5. **Safety monitoring** is entirely absent — a growing market requirement
6. **Schedule adherence** tracking is not yet implemented

The nine proposed modules (§4) would close these gaps with minimal external dependencies, consistent with ts-shape's existing architecture philosophy (pandas/numpy/scipy only, DataFrame-in/DataFrame-out).
