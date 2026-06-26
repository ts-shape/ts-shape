"""Engineering Events

Detectors for engineering-related patterns over shaped timeseries.

Classes:
- SetpointChangeEvents: Detect setpoint changes and compute response KPIs.
  - detect_setpoint_steps: Point events where |Δsetpoint| ≥ min_delta and holds for min_hold.
  - detect_setpoint_ramps: Intervals where |dS/dt| ≥ min_rate for at least min_duration.
  - detect_setpoint_changes: Unified table of steps and ramps with standardized columns.
  - time_to_settle: Time until |actual − setpoint| ≤ tol for a hold duration within a window.
  - overshoot_metrics: Peak overshoot magnitude/percent and time-to-peak after a change.

- StartupDetectionEvents: Detect startup intervals from thresholds or slope.
  - detect_startup_by_threshold: Rising threshold crossing with minimum dwell above threshold.
  - detect_startup_by_slope: Intervals with sustained positive slope ≥ min_slope for min_duration.

- ThresholdMonitoringEvents: Multi-level threshold monitoring with hysteresis.
  - multi_level_threshold: Intervals exceeding warning/alarm/critical levels.
  - threshold_with_hysteresis: Alarm entry/exit with separate high/low thresholds.
  - time_above_threshold: Time and percentage above threshold per window.
  - threshold_exceedance_trend: Track exceedance frequency over time.

- RateOfChangeEvents: Detect rapid changes and step jumps.
  - detect_rapid_change: Flag intervals where rate exceeds threshold.
  - rate_statistics: Per-window rate of change statistics.
  - detect_step_changes: Sudden value jumps within a short duration.

- SteadyStateDetectionEvents: Identify steady-state vs transient periods.
  - detect_steady_state: Intervals where rolling std stays below threshold.
  - detect_transient_periods: Intervals where signal is changing.
  - steady_state_statistics: Summary of steady vs transient time.
  - steady_state_value_bands: Operating band (mean ± std) per steady interval.

- SignalComparisonEvents: Compare two signals and detect divergence.
  - detect_divergence: Intervals where |actual - reference| > tolerance.
  - deviation_statistics: Per-window MAE, max error, RMSE, bias.
  - tracking_error_trend: Whether deviation is growing or shrinking.
  - correlation_windows: Per-window Pearson correlation.

- OperatingRangeEvents: Analyze signal operating envelope and range.
  - operating_envelope: Per-window min/max/mean/percentiles.
  - detect_regime_change: Detect shifts in the operating point.
  - time_in_range: Percentage of time within a defined range.
  - value_distribution: Histogram of signal values.

- WarmUpCoolDownEvents: Detect and characterize warm-up / cool-down curves.
  - detect_warmup: Rising ramp intervals.
  - detect_cooldown: Falling ramp intervals.
  - warmup_consistency: Compare warm-up durations and rates.
  - time_to_target: Time from ramp start until target value is reached.

- ProcessWindowEvents: Time-windowed process statistics and shift detection.
  - windowed_statistics: Per-window count, mean, std, min, max, percentiles.
  - detect_mean_shift: Flag windows where mean shifts significantly.
  - detect_variance_change: Flag windows where variance changes.
  - window_comparison: Compare each window to overall baseline.

- ControlLoopHealthEvents: Continuous PID/control loop health assessment.
  - error_integrals: Per-window IAE, ISE, ITAE, bias.
  - detect_oscillation: Sustained oscillation in the error signal.
  - output_saturation: Valve pegged at limits.
  - loop_health_summary: Shift-level report card.

- DisturbanceRecoveryEvents: Detect external upsets and measure recovery.
  - detect_disturbances: Intervals where signal deviates from baseline.
  - recovery_time: Time until signal returns to baseline after disturbance.
  - disturbance_frequency: Count disturbances per window.
  - before_after_comparison: Compare process stats before vs after disturbance.

- MaterialBalanceEvents: Check whether inputs and outputs balance.
  - balance_check: Per-window sum(inputs) vs sum(outputs).
  - imbalance_trend: Track whether imbalance is growing or shrinking.
  - detect_balance_exceedance: Sustained imbalance events.
  - contribution_breakdown: Each signal's share of total input/output.

- ProcessStabilityIndex: Single 0-100 stability score per shift/day.
  - stability_score: Composite 0-100 score per window.
  - score_trend: Is stability improving or degrading?
  - worst_periods: N worst-scoring windows.
  - stability_comparison: Compare each window to best-observed.
"""

from .control_loop_health import ControlLoopHealthEvents  # re-export
from .disturbance_recovery import DisturbanceRecoveryEvents  # re-export
from .material_balance import MaterialBalanceEvents  # re-export
from .operating_range import OperatingRangeEvents  # re-export
from .process_stability_index import ProcessStabilityIndex  # re-export
from .process_window import ProcessWindowEvents  # re-export
from .rate_of_change import RateOfChangeEvents  # re-export
from .setpoint_events import SetpointChangeEvents  # re-export
from .signal_comparison import SignalComparisonEvents  # re-export
from .startup_events import StartupDetectionEvents  # re-export
from .steady_state_detection import SteadyStateDetectionEvents  # re-export
from .threshold_monitoring import ThresholdMonitoringEvents  # re-export
from .warmup_analysis import WarmUpCoolDownEvents  # re-export

__all__ = [
    "SetpointChangeEvents",
    "StartupDetectionEvents",
    "ThresholdMonitoringEvents",
    "RateOfChangeEvents",
    "SteadyStateDetectionEvents",
    "SignalComparisonEvents",
    "OperatingRangeEvents",
    "WarmUpCoolDownEvents",
    "ProcessWindowEvents",
    "ControlLoopHealthEvents",
    "DisturbanceRecoveryEvents",
    "MaterialBalanceEvents",
    "ProcessStabilityIndex",
]
