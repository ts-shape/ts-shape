"""Production Events and Tracking

This module contains both event detection and daily tracking tools for production:

Event Detection Classes:
- MachineStateEvents: Run/idle intervals and transition points from a boolean state signal.
  - detect_run_idle: Intervalize run/idle with optional min duration.
  - transition_events: Point events on idle→run and run→idle changes.
  - detect_rapid_transitions: Identify suspicious rapid state changes.
  - state_quality_metrics: Return quality metrics for the state data.

- LongDowntimeEvents: Detect idle/stopped intervals exceeding a minimum duration, and
  count production events between consecutive long-downtime boundaries.
  - detect_long_downtime: Find downtime segments >= min_gap (default 3 h).
  - count_events_between_gaps: Count/sum/transitions in each inter-gap production window.

- LineThroughputEvents: Throughput metrics and takt adherence.
  - count_parts: Parts per fixed window from a counter uuid.
  - takt_adherence: Cycle time violations vs. a takt time.
  - cycle_quality_check: Enhanced cycle detection with quality validation.
  - throughput_oee: Calculate Overall Equipment Effectiveness (OEE) metrics.
  - throughput_trends: Throughput trend with moving averages and degradation flags.

- ChangeoverEvents: Product/recipe changes and end-of-changeover derivation.
  - detect_changeover: Point events at product value changes.
  - changeover_window: End via fixed window or stable band metrics.
  - changeover_quality_metrics: Compute quality metrics for changeovers.

- FlowConstraintEvents: Blocked/starved intervals between upstream/downstream run signals.
  - blocked_events: Upstream running while downstream not consuming.
  - starved_events: Downstream running while upstream not supplying.
  - flow_constraint_analytics: Blocked and starved events with severity and summary statistics.

Daily Production Tracking Classes:
- PartProductionEvents: Track production quantities by part number.
  - production_by_part: Production quantity per time window.
  - daily_production_summary: Daily totals by part.
  - production_totals: Totals over date ranges.
  - detect_resets: Find the points in time where the counter was reset.

- CycleTimeEvents: Analyze cycle times by part number.
  - cycle_time_by_part: Calculate cycle times.
  - cycle_time_statistics: Statistical analysis (min/avg/max/std).
  - detect_slow_cycles: Anomaly detection.
  - cycle_time_trend: Trend analysis.
  - hourly_cycle_time_summary: Hourly summary of cycle times by part.

- ShiftReportEvents: Shift-based performance analysis.
  - shift_production: Production per shift.
  - shift_comparison: Compare shift performance.
  - shift_targets: Target vs actual analysis.
  - best_and_worst_shifts: Performance ranking.

- DowntimeEvents: Machine availability and downtime analysis.
  - downtime_by_shift: Downtime and availability per shift.
  - downtime_by_reason: Root cause analysis.
  - top_downtime_reasons: Pareto analysis (80/20 rule).
  - availability_trend: Track availability over time.

- QualityEvents: NOK (defective parts) and quality metrics.
  - nok_by_shift: NOK parts and First Pass Yield per shift.
  - quality_by_part: Quality metrics by part number.
  - nok_by_reason: Defect type analysis.
  - daily_quality_summary: Daily quality rollup.

OEE and Advanced Analytics:
- OEEEvents: Overall Equipment Effectiveness (Availability x Performance x Quality).
  - availability: Availability % from run/idle intervals.
  - performance: Actual vs ideal throughput.
  - quality: Good parts / total parts.
  - oee: Combined daily OEE metric.

- AlarmManagementEvents: ISA-18.2 style alarm analysis.
  - alarm_frequency: Alarm activations per time window.
  - alarm_duration_statistics: Min/avg/max/total duration of alarm ON states.
  - chattering_detection: Detect nuisance chattering alarms.
  - standing_alarms: Identify alarms that stay active too long.

- BatchTrackingEvents: Batch/recipe production tracking.
  - detect_batches: Detect batch start/end from value changes.
  - batch_duration_statistics: Duration statistics per batch type.
  - batch_yield: Production quantity per batch.
  - batch_transition_matrix: Batch-to-batch transition frequencies.

- BottleneckDetectionEvents: Identify production line bottlenecks.
  - station_utilization: Per-station uptime percentage per window.
  - detect_bottleneck: Identify bottleneck station per window.
  - shifting_bottleneck: Track when the bottleneck moves.
  - throughput_constraint_summary: Summary statistics.

- MicroStopEvents: Detect brief idle intervals that accumulate into losses.
  - detect_micro_stops: Find idle intervals shorter than max_duration.
  - micro_stop_frequency: Count micro-stops per window.
  - micro_stop_impact: Time lost to micro-stops per window.
  - micro_stop_patterns: Group micro-stops by hour-of-day.

- DutyCycleEvents: Analyze on/off patterns from boolean signals.
  - duty_cycle_by_window: On-time percentage per window.
  - on_off_intervals: List every on/off interval with duration.
  - cycle_count: Transition counts per window.
  - excessive_cycling: Flag windows with too many transitions.

Traceability:
- ValueTraceabilityEvents: Trace any shared identifier across multiple stations.
  - build_timeline: Full timeline of every identifier at every station.
  - lead_time: End-to-end lead time per identifier.
  - current_status: Last-known station for each identifier.
  - station_dwell_statistics: Dwell-time stats per station.
  (OrderTraceabilityEvents is a deprecated alias.)

- RoutingTraceabilityEvents: Trace item routing using ID + state/routing signal.
  The ``state_map`` argument maps signal values (PLC steps, station codes) to
  station names.
  - build_routing_timeline: Correlate ID signal with state signal.
  - lead_time: End-to-end lead time per item.
  - station_statistics: Dwell-time stats per station/step.
  - routing_paths: Most common routing path sequences.

- ContinuousProcessAlignmentEvents: Align multi-station readings on a continuous line
  to a common material reference time using speed-based transport lag compensation.
  Methods: align_to_reference, segment_by_cut, lag_profile, alignment_quality.

Flow and Line Balancing:
- FlowMetricsEvents: WIP, throughput, lead time and Little's Law from entry/exit signals.
  - wip_over_time: Time-weighted WIP per window.
  - throughput: Units out per window.
  - lead_time: FIFO-matched entry -> exit lead time.
  - flow_summary: Little's Law check and Process Cycle Efficiency.

- LineBalancingEvents: Line balancing and takt analysis from per-station cycle signals.
  - station_cycle_times: Cycle time statistics per station and window.
  - balance_metrics: Balance efficiency, balance delay and smoothness index.
  - yamazumi: Station-loading table against takt.

- RuntimeAccountingEvents: Operating-hours accounting from a boolean run signal.
  - runtime_summary: Run and idle time, utilization, starts, longest run.
  - runtime_by_window: Run time per window.
  - operating_hours_meter: Cumulative operating hours.
  - run_uuid: Deprecated read-only alias of ``run_state_uuid``.

- MultiProcessTraceabilityEvents: Multi-station topology with parallel paths and handovers.
  - build_timeline: Full timeline with parallel-station awareness.
  - lead_time: End-to-end lead time with parallel flag.
  - parallel_activity: Detect overlapping station intervals per item.
  - handover_log: Extract and correlate handover events with item IDs.
  - station_statistics: Per-station/cell dwell-time stats.
  - routing_paths: Path frequency analysis with parallel flag.

Setup, Rework, and Operator Tracking:
- SetupTimeEvents: Analyze changeover/setup durations for SMED improvement.
  - setup_durations: List every setup interval with duration.
  - setup_by_product: Setup time by product transition (from → to).
  - setup_statistics: Overall setup time stats (count, avg, median, std, % of available time).
  - setup_trend: Track setup time improvement over time.

- OperatorPerformanceEvents: Compare production output and quality across operators.
  - production_by_operator: Parts produced per operator.
  - operator_efficiency: Operator efficiency vs per-shift target.
  - quality_by_operator: Quality metrics (FPY) per operator.
  - operator_comparison: Ranked operator performance comparison.

- ReworkEvents: Track parts requiring rework (re-processing).
  - rework_by_shift: Rework count per shift.
  - rework_by_reason: Rework by reason code.
  - rework_rate: Rework rate as % of total production.
  - rework_cost: Convert rework counts to monetary cost.
  - rework_trend: Rework trend over time.

Performance and Target Tracking:
- PerformanceLossEvents: Track speed losses against target cycle times.
  - performance_by_shift: Performance % per shift.
  - slow_periods: Identify windows below target performance.
  - performance_trend: Performance trend over time.

- ScrapEvents: Track material scrap and waste.
  - scrap_by_shift: Scrap quantity per shift.
  - scrap_by_reason: Scrap by reason code.
  - scrap_cost: Convert scrap to monetary cost.
  - scrap_trend: Scrap trend over time.

- TargetEvents: Compare any metric to targets.
  - compare_to_target: Actual vs per-shift targets.
  - target_achievement_summary: Daily target achievement.
  - target_hit_rate: Percentage of days meeting targets.

- ProductionScheduleAdherenceEvents: Planned schedule vs actual production.
  - adherence_by_period: Planned vs actual and recovery rate per period.
  - deviation_events: Events for periods breaching the deviation bands.
  - adherence_summary: Roll-up across the scheduled horizon.

- ShiftHandoverEvents: Automated shift handover report generation.
  - generate_report: Full shift report with production, quality, downtime.
  - highlight_issues: Auto-identify metrics below thresholds.
  - from_shift_data: Build a handover report from pre-computed shift-level DataFrames.

- PeriodSummaryEvents: Weekly/monthly summary aggregation.
  - weekly_summary: Roll up daily metrics to weekly.
  - monthly_summary: Roll up daily metrics to monthly.
  - compare_periods: Period-over-period comparison.
  - from_daily_data: Roll up a pre-computed daily DataFrame to weekly or monthly.

- OperatingStateEvents: Infer operating states from continuous signals.
  - infer_states: Cluster the signal into N operating states.
  - state_duration_statistics: Duration statistics per inferred state.
  - unusual_state_sequence: Detect unexpected state transitions.
  - dwell_time_violation: Find intervals exceeding max allowed duration in a state.
"""

# Event Detection Classes
from .alarm_management import AlarmManagementEvents
from .batch_tracking import BatchTrackingEvents
from .bottleneck_detection import BottleneckDetectionEvents
from .changeover import ChangeoverEvents

# Traceability
from .continuous_process_alignment import ContinuousProcessAlignmentEvents
from .cycle_time_tracking import (
    CycleTimeEvents,
    CycleTimeTracking,  # deprecated alias
)
from .downtime_tracking import (
    DowntimeEvents,
    DowntimeTracking,  # deprecated alias
)
from .duty_cycle import DutyCycleEvents
from .flow_constraints import FlowConstraintEvents
from .flow_metrics import FlowMetricsEvents
from .line_balancing import LineBalancingEvents
from .line_throughput import LineThroughputEvents
from .long_downtime_events import LongDowntimeEvents
from .machine_state import MachineStateEvents
from .micro_stop_detection import MicroStopEvents
from .multi_process_traceability import MultiProcessTraceabilityEvents

# OEE and Advanced Analytics
from .oee_calculator import (
    OEECalculator,  # deprecated alias
    OEEEvents,
)

# Setup, Rework, and Operator Tracking
from .operating_state import OperatingStateEvents
from .operator_performance import (
    OperatorPerformanceEvents,
    OperatorPerformanceTracking,  # deprecated alias
)
from .order_traceability import OrderTraceabilityEvents, ValueTraceabilityEvents

# Daily Production Tracking Classes
from .part_tracking import (
    PartProductionEvents,
    PartProductionTracking,  # deprecated alias
)

# Performance, Target, and Reporting
from .performance_loss import (
    PerformanceLossEvents,
    PerformanceLossTracking,  # deprecated alias
)
from .period_summary import (
    PeriodSummary,  # deprecated alias
    PeriodSummaryEvents,
)
from .quality_tracking import (
    QualityEvents,
    QualityTracking,  # deprecated alias
)
from .rework_tracking import (
    ReworkEvents,
    ReworkTracking,  # deprecated alias
)
from .routing_traceability import RoutingTraceabilityEvents
from .runtime_accounting import RuntimeAccountingEvents
from .schedule_adherence import ProductionScheduleAdherenceEvents
from .scrap_tracking import (
    ScrapEvents,
    ScrapTracking,  # deprecated alias
)
from .setup_time_tracking import (
    SetupTimeEvents,
    SetupTimeTracking,  # deprecated alias
)
from .shift_handover import (
    ShiftHandoverEvents,
    ShiftHandoverReport,  # deprecated alias
)
from .shift_reporting import (
    ShiftReportEvents,
    ShiftReporting,  # deprecated alias
)
from .target_tracking import (
    TargetEvents,
    TargetTracking,  # deprecated alias
)

__all__ = [
    "OperatingStateEvents",
    # Event Detection
    "MachineStateEvents",
    "LongDowntimeEvents",
    "LineThroughputEvents",
    "ChangeoverEvents",
    "FlowConstraintEvents",
    # Daily Production Tracking
    "PartProductionEvents",
    "PartProductionTracking",  # deprecated alias
    "CycleTimeEvents",
    "CycleTimeTracking",  # deprecated alias
    "ShiftReportEvents",
    "ShiftReporting",  # deprecated alias
    "DowntimeEvents",
    "DowntimeTracking",  # deprecated alias
    "QualityEvents",
    "QualityTracking",  # deprecated alias
    # OEE and Advanced Analytics
    "OEEEvents",
    "OEECalculator",  # deprecated alias
    "AlarmManagementEvents",
    "BatchTrackingEvents",
    # Bottleneck, Micro-Stop, and Duty Cycle Analysis
    "BottleneckDetectionEvents",
    "MicroStopEvents",
    "DutyCycleEvents",
    # Traceability
    "ContinuousProcessAlignmentEvents",
    "ValueTraceabilityEvents",
    "OrderTraceabilityEvents",  # deprecated alias
    "RoutingTraceabilityEvents",
    "MultiProcessTraceabilityEvents",
    # Flow and line balancing
    "FlowMetricsEvents",
    "LineBalancingEvents",
    "RuntimeAccountingEvents",
    # Setup, Rework, and Operator Tracking
    "SetupTimeEvents",
    "SetupTimeTracking",  # deprecated alias
    "OperatorPerformanceEvents",
    "OperatorPerformanceTracking",  # deprecated alias
    "ReworkEvents",
    "ReworkTracking",  # deprecated alias
    # Performance, Target, and Reporting
    "PerformanceLossEvents",
    "PerformanceLossTracking",  # deprecated alias
    "ProductionScheduleAdherenceEvents",
    "ScrapEvents",
    "ScrapTracking",  # deprecated alias
    "TargetEvents",
    "TargetTracking",  # deprecated alias
    "ShiftHandoverEvents",
    "ShiftHandoverReport",  # deprecated alias
    "PeriodSummaryEvents",
    "PeriodSummary",  # deprecated alias
]
