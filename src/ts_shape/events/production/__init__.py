"""Production Events and Tracking

This module contains both event detection and daily tracking tools for production:

Event Detection Classes:
- MachineStateEvents: Run/idle intervals and transition points from a boolean state signal.
  - detect_run_idle: Intervalize run/idle with optional min duration.
  - transition_events: Point events on idle→run and run→idle changes.

- LongDowntimeEvents: Detect idle/stopped intervals exceeding a minimum duration, and
  count production events between consecutive long-downtime boundaries.
  - detect_long_downtime: Find downtime segments >= min_gap (default 3 h).
  - count_events_between_gaps: Count/sum/transitions in each inter-gap production window.

- LineThroughputEvents: Throughput metrics and takt adherence.
  - count_parts: Parts per fixed window from a counter uuid.
  - takt_adherence: Cycle time violations vs. a takt time.

- ChangeoverEvents: Product/recipe changes and end-of-changeover derivation.
  - detect_changeover: Point events at product value changes.
  - changeover_window: End via fixed window or stable band metrics.

- FlowConstraintEvents: Blocked/starved intervals between upstream/downstream run signals.
  - blocked_events: Upstream running while downstream not consuming.
  - starved_events: Downstream running while upstream not supplying.

Daily Production Tracking Classes:
- PartProductionTracking: Track production quantities by part number.
  - production_by_part: Production quantity per time window.
  - daily_production_summary: Daily totals by part.
  - production_totals: Totals over date ranges.

- CycleTimeTracking: Analyze cycle times by part number.
  - cycle_time_by_part: Calculate cycle times.
  - cycle_time_statistics: Statistical analysis (min/avg/max/std).
  - detect_slow_cycles: Anomaly detection.
  - cycle_time_trend: Trend analysis.

- ShiftReporting: Shift-based performance analysis.
  - shift_production: Production per shift.
  - shift_comparison: Compare shift performance.
  - shift_targets: Target vs actual analysis.
  - best_and_worst_shifts: Performance ranking.

- DowntimeTracking: Machine availability and downtime analysis.
  - downtime_by_shift: Downtime and availability per shift.
  - downtime_by_reason: Root cause analysis.
  - top_downtime_reasons: Pareto analysis (80/20 rule).
  - availability_trend: Track availability over time.

- QualityTracking: NOK (defective parts) and quality metrics.
  - nok_by_shift: NOK parts and First Pass Yield per shift.
  - quality_by_part: Quality metrics by part number.
  - nok_by_reason: Defect type analysis.
  - daily_quality_summary: Daily quality rollup.

OEE and Advanced Analytics:
- OEECalculator: Overall Equipment Effectiveness (Availability x Performance x Quality).
  - calculate_availability: Availability % from run/idle intervals.
  - calculate_performance: Actual vs ideal throughput.
  - calculate_quality: Good parts / total parts.
  - calculate_oee: Combined daily OEE metric.

- AlarmManagementEvents: ISA-18.2 style alarm analysis.
  - alarm_frequency: Alarm activations per time window.
  - alarm_duration_stats: Min/avg/max/total duration of alarm ON states.
  - chattering_detection: Detect nuisance chattering alarms.
  - standing_alarms: Identify alarms that stay active too long.

- BatchTrackingEvents: Batch/recipe production tracking.
  - detect_batches: Detect batch start/end from value changes.
  - batch_duration_stats: Duration statistics per batch type.
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
  - duty_cycle_per_window: On-time percentage per window.
  - on_off_intervals: List every on/off interval with duration.
  - cycle_count: Transition counts per window.
  - excessive_cycling: Flag windows with too many transitions.

Traceability:
- ValueTraceabilityEvents: Trace any shared identifier across multiple stations.
  - build_timeline: Full timeline of every identifier at every station.
  - lead_time: End-to-end lead time per identifier.
  - current_status: Last-known station for each identifier.
  - station_dwell_statistics: Dwell-time stats per station.
  (OrderTraceabilityEvents is a backwards-compatible alias.)

- RoutingTraceabilityEvents: Trace item routing using ID + state/routing signal.
  - state_map: Maps signal values (PLC steps, station codes) to station names.
  - build_routing_timeline: Correlate ID signal with state signal.
  - lead_time: End-to-end lead time per item.
  - station_statistics: Dwell-time stats per station/step.
  - routing_paths: Most common routing path sequences.

- ContinuousProcessAlignmentEvents: Align multi-station readings on a continuous line
  to a common material reference time using speed-based transport lag compensation.
  Methods: align_to_reference, segment_by_cut, lag_profile, alignment_quality.

- MultiProcessTraceabilityEvents: Multi-station topology with parallel paths and handovers.
  - build_timeline: Full timeline with parallel-station awareness.
  - lead_time: End-to-end lead time with parallel flag.
  - parallel_activity: Detect overlapping station intervals per item.
  - handover_log: Extract and correlate handover events with item IDs.
  - station_statistics: Per-station/cell dwell-time stats.
  - routing_paths: Path frequency analysis with parallel flag.

Setup, Rework, and Operator Tracking:
- SetupTimeTracking: Analyze changeover/setup durations for SMED improvement.
  - setup_durations: List every setup interval with duration.
  - setup_by_product: Setup time by product transition (from → to).
  - setup_statistics: Overall setup time stats (count, avg, median, std, % of available time).
  - setup_trend: Track setup time improvement over time.

- OperatorPerformanceTracking: Compare production output and quality across operators.
  - production_by_operator: Parts produced per operator.
  - operator_efficiency: Operator efficiency vs per-shift target.
  - quality_by_operator: Quality metrics (FPY) per operator.
  - operator_comparison: Ranked operator performance comparison.

- ReworkTracking: Track parts requiring rework (re-processing).
  - rework_by_shift: Rework count per shift.
  - rework_by_reason: Rework by reason code.
  - rework_rate: Rework rate as % of total production.
  - rework_cost: Convert rework counts to monetary cost.
  - rework_trend: Rework trend over time.

Performance and Target Tracking:
- PerformanceLossTracking: Track speed losses against target cycle times.
  - performance_by_shift: Performance % per shift.
  - slow_periods: Identify windows below target performance.
  - performance_trend: Performance trend over time.

- ScrapTracking: Track material scrap and waste.
  - scrap_by_shift: Scrap quantity per shift.
  - scrap_by_reason: Scrap by reason code.
  - scrap_cost: Convert scrap to monetary cost.
  - scrap_trend: Scrap trend over time.

- TargetTracking: Compare any metric to targets.
  - compare_to_target: Actual vs per-shift targets.
  - target_achievement_summary: Daily target achievement.
  - target_hit_rate: Percentage of days meeting targets.

- ShiftHandoverReport: Automated shift handover report generation.
  - generate_report: Full shift report with production, quality, downtime.
  - highlight_issues: Auto-identify metrics below thresholds.

- PeriodSummary: Weekly/monthly summary aggregation.
  - weekly_summary: Roll up daily metrics to weekly.
  - monthly_summary: Roll up daily metrics to monthly.
  - compare_periods: Period-over-period comparison.
"""

# Event Detection Classes
from .machine_state import MachineStateEvents
from .long_downtime_events import LongDowntimeEvents
from .line_throughput import LineThroughputEvents
from .changeover import ChangeoverEvents
from .flow_constraints import FlowConstraintEvents

# Daily Production Tracking Classes
from .part_tracking import PartProductionTracking
from .cycle_time_tracking import CycleTimeTracking
from .shift_reporting import ShiftReporting
from .downtime_tracking import DowntimeTracking
from .quality_tracking import QualityTracking

# OEE and Advanced Analytics
from .oee_calculator import OEECalculator
from .alarm_management import AlarmManagementEvents
from .batch_tracking import BatchTrackingEvents
from .bottleneck_detection import BottleneckDetectionEvents
from .micro_stop_detection import MicroStopEvents
from .duty_cycle import DutyCycleEvents

# Traceability
from .continuous_process_alignment import ContinuousProcessAlignmentEvents
from .order_traceability import ValueTraceabilityEvents, OrderTraceabilityEvents
from .routing_traceability import RoutingTraceabilityEvents
from .multi_process_traceability import MultiProcessTraceabilityEvents

# Setup, Rework, and Operator Tracking
from .setup_time_tracking import SetupTimeTracking
from .operator_performance import OperatorPerformanceTracking
from .rework_tracking import ReworkTracking

# Performance, Target, and Reporting
from .performance_loss import PerformanceLossTracking
from .scrap_tracking import ScrapTracking
from .target_tracking import TargetTracking
from .shift_handover import ShiftHandoverReport
from .period_summary import PeriodSummary

__all__ = [
    # Event Detection
    "MachineStateEvents",
    "LongDowntimeEvents",
    "LineThroughputEvents",
    "ChangeoverEvents",
    "FlowConstraintEvents",
    # Daily Production Tracking
    "PartProductionTracking",
    "CycleTimeTracking",
    "ShiftReporting",
    "DowntimeTracking",
    "QualityTracking",
    # OEE and Advanced Analytics
    "OEECalculator",
    "AlarmManagementEvents",
    "BatchTrackingEvents",
    # Bottleneck, Micro-Stop, and Duty Cycle Analysis
    "BottleneckDetectionEvents",
    "MicroStopEvents",
    "DutyCycleEvents",
    # Traceability
    "ContinuousProcessAlignmentEvents",
    "ValueTraceabilityEvents",
    "OrderTraceabilityEvents",  # backwards-compatible alias
    "RoutingTraceabilityEvents",
    "MultiProcessTraceabilityEvents",
    # Setup, Rework, and Operator Tracking
    "SetupTimeTracking",
    "OperatorPerformanceTracking",
    "ReworkTracking",
    # Performance, Target, and Reporting
    "PerformanceLossTracking",
    "ScrapTracking",
    "TargetTracking",
    "ShiftHandoverReport",
    "PeriodSummary",
]
