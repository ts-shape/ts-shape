// Sidebars, one per navbar section. Converted from the former mkdocs.yml nav.
import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  getStarted: [
    {type: "doc", id: "index", label: "Overview"},
    {type: "doc", id: "user_guide/installation", label: "Installation"},
    {type: "doc", id: "user_guide/quick_start", label: "Quick start"}
  ],
  concepts: [
    {type: "doc", id: "concept", label: "Data model and layers"},
    {type: "doc", id: "guides/signals", label: "Signal semantics and profiling"},
    {type: "doc", id: "naming", label: "Naming and output conventions"}
  ],
  guides: [
    {type: "doc", id: "guides/loading", label: "Loading data"},
    {type: "doc", id: "guides/transforms", label: "Transforms and statistics"},
    {type: "doc", id: "guides/feature-extraction", label: "Cycles and segments"},
    {type: "doc", id: "guides/production", label: "Production and OEE"},
    {type: "doc", id: "guides/quality", label: "Quality and SPC"},
    {type: "doc", id: "guides/engineering", label: "Process engineering"},
    {type: "doc", id: "guides/maintenance", label: "Maintenance"},
    {type: "doc", id: "guides/energy", label: "Energy"},
    {type: "doc", id: "guides/context", label: "Context and enrichment"},
    {type: "doc", id: "guides/ml-features", label: "Features for ML"},
    {type: "doc", id: "guides/pipeline-builder", label: "Pipelines"},
    {type: "doc", id: "guides/chunked-processing", label: "Chunked processing"},
    {
      type: "category",
      label: "Event log (XES and OCEL)",
      items: [
        {type: "doc", id: "guides/eventlog/index", label: "Overview"},
        {type: "doc", id: "guides/eventlog/objects", label: "Object detection"},
        {type: "doc", id: "guides/eventlog/genealogy", label: "Genealogy"},
        {type: "doc", id: "guides/eventlog/object-analytics", label: "Object analytics"},
        {type: "doc", id: "guides/object-identification", label: "Object identification"},
        {type: "doc", id: "guides/object-variants", label: "Object signal variants"},
        {type: "doc", id: "guides/eventlog/taxonomy", label: "Labeling and taxonomy"},
        {type: "doc", id: "guides/eventlog/adapters", label: "Adapters"}
      ]
    },
    {type: "doc", id: "guides/lambda-rules", label: "Lambda rules"},
    {type: "doc", id: "examples", label: "Example scripts"}
  ],
  modules: [
    {type: "doc", id: "modules/index", label: "Overview"},
    {
      type: "category",
      label: "Quality",
      link: {type: "generated-index", slug: "/modules/quality/", description: "Outliers, control charts, capability, drift, gauge R&R and measurement uncertainty."},
      items: [
        {type: "doc", id: "modules/quality/outlier-detection", label: "Outlier Detection"},
        {type: "doc", id: "modules/quality/spc", label: "Statistical Process Control"},
        {type: "doc", id: "modules/quality/tolerance-deviation", label: "Tolerance Deviation"},
        {type: "doc", id: "modules/quality/anomaly-classification", label: "Anomaly Classification"},
        {type: "doc", id: "modules/quality/signal-quality", label: "Signal Quality"},
        {type: "doc", id: "modules/quality/sensor-drift", label: "Sensor Drift"},
        {type: "doc", id: "modules/quality/multi-sensor-validation", label: "Multi-Sensor Validation"},
        {type: "doc", id: "modules/quality/gauge-rr", label: "Gauge R&R"},
        {type: "doc", id: "modules/quality/capability-trending", label: "Capability Trending"},
        {type: "doc", id: "modules/quality/change-point", label: "Change-Point Detection"},
        {type: "doc", id: "modules/quality/multivariate-spc", label: "Multivariate SPC (Hotelling T²)"},
        {type: "doc", id: "modules/quality/calibration-validation", label: "Calibration Validation"},
        {type: "doc", id: "modules/quality/measurement-uncertainty", label: "Measurement Uncertainty (GUM)"},
        {type: "doc", id: "modules/quality/value-distribution", label: "Value Distribution"},
        {type: "doc", id: "modules/quality/sampling-resolution", label: "Sampling & Resolution"},
        {type: "doc", id: "modules/quality/data-gap-analysis", label: "Data Gap Analysis"},
        {type: "doc", id: "modules/quality/environmental-compensation", label: "Environmental Compensation"}
      ]
    },
    {
      type: "category",
      label: "Production",
      link: {type: "generated-index", slug: "/modules/production/", description: "Machine and line states, cycle times, downtime, OEE, traceability and reporting."},
      items: [
        {
          type: "category",
      label: "Event Detection",
      link: {type: "generated-index", slug: "/modules/production/event-detection/", description: "Machine and operating states, throughput, changeovers and flow constraints."},
          items: [
            {type: "doc", id: "modules/production/machine-state", label: "Machine State"},
            {type: "doc", id: "modules/production/operating-state", label: "Operating State"},
            {type: "doc", id: "modules/production/line-throughput", label: "Line Throughput"},
            {type: "doc", id: "modules/production/changeover", label: "Changeover"},
            {type: "doc", id: "modules/production/flow-constraints", label: "Flow Constraints"},
            {type: "doc", id: "modules/production/continuous-process-alignment", label: "Continuous Process Alignment"}
          ]
        },
        {
          type: "category",
      label: "Daily Tracking",
      link: {type: "generated-index", slug: "/modules/production/daily-tracking/", description: "Parts, cycle times, unit profiles, downtime, quality and shift reports."},
          items: [
            {type: "doc", id: "modules/production/part-tracking", label: "Part Tracking"},
            {type: "doc", id: "modules/production/cycle-time", label: "Cycle Time"},
            {type: "doc", id: "modules/production/cycle-profiling", label: "Cycle Time Profiling"},
            {type: "doc", id: "modules/production/unit-profiles", label: "Unit Profilers"},
            {type: "doc", id: "modules/production/downtime", label: "Downtime"},
            {type: "doc", id: "modules/production/quality-tracking", label: "Quality Tracking"},
            {type: "doc", id: "modules/production/shift-reporting", label: "Shift Reporting"}
          ]
        },
        {
          type: "category",
      label: "OEE & Advanced",
      link: {type: "generated-index", slug: "/modules/production/oee-and-advanced/", description: "OEE, alarms, batches, bottlenecks, micro-stops, line balancing and flow metrics."},
          items: [
            {type: "doc", id: "modules/production/oee-calculator", label: "OEE Calculator"},
            {type: "doc", id: "modules/production/alarm-management", label: "Alarm Management"},
            {type: "doc", id: "modules/production/batch-tracking", label: "Batch Tracking"},
            {type: "doc", id: "modules/production/bottleneck-detection", label: "Bottleneck Detection"},
            {type: "doc", id: "modules/production/micro-stops", label: "Micro-Stop Detection"},
            {type: "doc", id: "modules/production/duty-cycle", label: "Duty Cycle"},
            {type: "doc", id: "modules/production/line-balancing", label: "Line Balancing"},
            {type: "doc", id: "modules/production/flow-metrics", label: "Flow Metrics (Little's Law)"},
            {type: "doc", id: "modules/production/long-downtime", label: "Long Downtime"},
            {type: "doc", id: "modules/production/runtime-accounting", label: "Runtime Accounting"}
          ]
        },
        {
          type: "category",
      label: "Traceability",
      link: {type: "generated-index", slug: "/modules/production/traceability/", description: "Follow values, orders and parts through routings and processes."},
          items: [
            {type: "doc", id: "modules/production/order-traceability", label: "Value Traceability"},
            {type: "doc", id: "modules/production/routing-traceability", label: "Routing Traceability"},
            {type: "doc", id: "modules/production/multi-process-traceability", label: "Multi-Process Traceability"}
          ]
        },
        {
          type: "category",
      label: "Setup & Operators",
      link: {type: "generated-index", slug: "/modules/production/setup-and-operators/", description: "Setup times (SMED), operator performance and rework."},
          items: [
            {type: "doc", id: "modules/production/setup-time", label: "Setup Time (SMED)"},
            {type: "doc", id: "modules/production/operator-performance", label: "Operator Performance"},
            {type: "doc", id: "modules/production/rework-tracking", label: "Rework Tracking"}
          ]
        },
        {
          type: "category",
      label: "Performance & Reporting",
      link: {type: "generated-index", slug: "/modules/production/performance-and-reporting/", description: "Performance loss, scrap, targets, schedule adherence and period summaries."},
          items: [
            {type: "doc", id: "modules/production/performance-loss", label: "Performance Loss"},
            {type: "doc", id: "modules/production/scrap-tracking", label: "Scrap Tracking"},
            {type: "doc", id: "modules/production/target-tracking", label: "Target Tracking"},
            {type: "doc", id: "modules/production/schedule-adherence", label: "Schedule Adherence"},
            {type: "doc", id: "modules/production/shift-handover", label: "Shift Handover"},
            {type: "doc", id: "modules/production/period-summary", label: "Period Summary"}
          ]
        }
      ]
    },
    {
      type: "category",
      label: "Engineering",
      link: {type: "generated-index", slug: "/modules/engineering/", description: "Setpoints, start-ups, thresholds, steady state, process windows and control loops."},
      items: [
        {type: "doc", id: "modules/engineering/setpoint-events", label: "Setpoint Events"},
        {type: "doc", id: "modules/engineering/startup-detection", label: "Startup Detection"},
        {type: "doc", id: "modules/engineering/threshold-monitoring", label: "Threshold Monitoring"},
        {type: "doc", id: "modules/engineering/rate-of-change", label: "Rate of Change"},
        {type: "doc", id: "modules/engineering/steady-state", label: "Steady State Detection"},
        {type: "doc", id: "modules/engineering/signal-comparison", label: "Signal Comparison"},
        {type: "doc", id: "modules/engineering/operating-range", label: "Operating Range"},
        {type: "doc", id: "modules/engineering/warmup-analysis", label: "Warm-Up Analysis"},
        {type: "doc", id: "modules/engineering/process-windows", label: "Process Windows"},
        {type: "doc", id: "modules/engineering/control-loop-health", label: "Control Loop Health"},
        {type: "doc", id: "modules/engineering/disturbance-recovery", label: "Disturbance Recovery"},
        {type: "doc", id: "modules/engineering/material-balance", label: "Material Balance"},
        {type: "doc", id: "modules/engineering/process-stability", label: "Process Stability Index"},
        {type: "doc", id: "modules/engineering/time-to-threshold", label: "Time to Threshold"}
      ]
    },
    {
      type: "category",
      label: "Maintenance",
      link: {type: "generated-index", slug: "/modules/maintenance/", description: "Degradation, failure prediction, reliability (MTBF / MTTR / Weibull), vibration and asset health."},
      items: [
        {type: "doc", id: "modules/maintenance/degradation-detection", label: "Degradation Detection"},
        {type: "doc", id: "modules/maintenance/failure-prediction", label: "Failure Prediction"},
        {type: "doc", id: "modules/maintenance/reliability", label: "Reliability (MTBF / MTTR / Weibull)"},
        {type: "doc", id: "modules/maintenance/vibration-analysis", label: "Vibration Analysis"},
        {type: "doc", id: "modules/maintenance/asset-health-index", label: "Asset Health Index"}
      ]
    },
    {
      type: "category",
      label: "Energy",
      link: {type: "generated-index", slug: "/modules/energy/", description: "Power integration, consumption, efficiency, EnPI, idle energy and carbon intensity."},
      items: [
        {type: "doc", id: "modules/energy/power-integration", label: "Power Integration"},
        {type: "doc", id: "modules/energy/consumption-analysis", label: "Consumption Analysis"},
        {type: "doc", id: "modules/energy/efficiency-tracking", label: "Efficiency Tracking"},
        {type: "doc", id: "modules/energy/energy-performance-indicator", label: "Energy Performance Indicator (EnPI)"},
        {type: "doc", id: "modules/energy/idle-energy-detection", label: "Idle Energy"},
        {type: "doc", id: "modules/energy/carbon-intensity", label: "Carbon Intensity"}
      ]
    },
    {
      type: "category",
      label: "Correlation",
      link: {type: "generated-index", slug: "/modules/correlation/", description: "Correlations between signals, anomalies and root causes."},
      items: [
        {type: "doc", id: "modules/correlation/signal-correlation", label: "Signal Correlation"},
        {type: "doc", id: "modules/correlation/anomaly-correlation", label: "Anomaly Correlation"},
        {type: "doc", id: "modules/correlation/root-cause-correlation", label: "Root Cause Correlation"}
      ]
    },
    {
      type: "category",
      label: "Supply Chain",
      link: {type: "generated-index", slug: "/modules/supply-chain/", description: "Inventory, demand patterns and lead times."},
      items: [
        {type: "doc", id: "modules/supplychain/inventory-monitoring", label: "Inventory Monitoring"},
        {type: "doc", id: "modules/supplychain/demand-patterns", label: "Demand Patterns"},
        {type: "doc", id: "modules/supplychain/lead-time-analysis", label: "Lead Time Analysis"}
      ]
    },
    {
      type: "category",
      label: "Development",
      link: {type: "generated-index", slug: "/modules/development/", description: "Design of experiments, design space, golden batches, recipe phases and parameter ranking."},
      items: [
        {type: "doc", id: "modules/development/design-of-experiments", label: "Design of Experiments"},
        {type: "doc", id: "modules/development/design-space", label: "Design Space"},
        {type: "doc", id: "modules/development/golden-batch", label: "Golden Batch Deviation"},
        {type: "doc", id: "modules/development/recipe-phase-adherence", label: "Recipe Phase Adherence"},
        {type: "doc", id: "modules/development/critical-parameter-ranking", label: "Critical Parameter Ranking"}
      ]
    }
  ],
  project: [
    {type: "doc", id: "migration", label: "Migration and deprecations"},
    {type: "doc", id: "changelog", label: "Changelog"},
    {type: "doc", id: "roadmap", label: "Roadmap"},
    {type: "doc", id: "contributing", label: "Contributing"},
    {type: "doc", id: "insiders/development", label: "Development"},
    {type: "doc", id: "license", label: "License"}
  ],
  api: [
    {type: "doc", id: "reference/index", label: "Overview"},
    {type: "autogenerated", dirName: "reference/ts_shape"}
  ]
};

export default sidebars;
