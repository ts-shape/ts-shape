"""Quality Events

Detectors for quality-related events: outliers, statistical process control,
and tolerance deviations over time series.

Classes:
- OutlierDetectionEvents: Detect and group outlier events in a time series.
  - detect_outliers_zscore: Detect outliers using Z-score thresholding and group nearby points.
  - detect_outliers_iqr: Detect outliers using IQR bounds and group nearby points.
  - detect_outliers_isolation_forest: Detects outliers using sklearn's IsolationForest algorithm.
  - detect_outliers_mad: Detects outliers using the Median Absolute Deviation (MAD) method.

- StatisticalProcessControlEvents: Apply Western Electric rules to actual values
  using tolerance context to flag control-limit violations.
  - control_limits: Compute mean and ±1/±2/±3 standard-deviation bands from tolerance rows.
  - detect_violations: Apply selected rules and emit event rows for violations.
  - rule_1: One point beyond the 3-sigma control limits.
  - rule_2: Nine consecutive points on one side of the mean.
  - rule_3: Six consecutive points steadily increasing or decreasing.
  - rule_4: Fourteen consecutive points alternating up and down.
  - rule_5: Two of three consecutive points near the control limit (between 2 and 3 sigma).
  - rule_6: Four of five consecutive points near the control limit (between 1 and 2 sigma).
  - rule_7: Fifteen consecutive points within 1 sigma of the mean.
  - rule_8: Eight consecutive points on both sides of the mean within 1 sigma.
  - detect_rule_violations: Apply the selected Western Electric rules in one vectorized pass.
  - dynamic_control_limits: Calculate dynamic control limits that adapt over time.
  - detect_cusum_shifts: Detect process shifts using CUSUM (Cumulative Sum) control chart.
  - interpret_violations: Add human-readable interpretations to rule violations.

- ToleranceDeviationEvents: Flag intervals where actual values cross/compare against
  tolerance settings and group them into start/end events.
  - detect_deviations: Build grouped deviation events with event UUIDs.
  - capability_indices: Calculate process capability indices (Cp, Cpk, Pp, Ppk).

- AnomalyClassificationEvents: Classify anomaly types in numeric signals.
  - classify_anomalies: Detect and classify by type (spike/drift/oscillation/flatline/level_shift).
  - detect_flatline: Signal stuck at constant value.
  - detect_oscillation: Periodic instability detection.
  - detect_drift: Short-term slope-based drift events.

- SignalQualityEvents: Signal data quality monitoring.
  - detect_missing_data: Find gaps exceeding expected sampling frequency.
  - sampling_regularity: Inter-sample interval statistics per window.
  - detect_out_of_range: Flag values outside physical/expected bounds.
  - data_completeness: Percentage of expected samples received per window.

- CapabilityTrendingEvents: Track process capability over rolling time windows.
  - capability_over_time: Cp/Cpk/Pp/Ppk per time window.
  - detect_capability_drop: Alert when Cpk falls below threshold.
  - capability_forecast: Extrapolate Cpk trend to predict threshold breach.
  - yield_estimate: Estimated yield, DPMO, and sigma level per window.

- SensorDriftEvents: Detect calibration drift in inline sensors.
  - detect_zero_drift: Track mean offset from baseline per window.
  - detect_span_drift: Track measurement sensitivity changes over time.
  - drift_trend: Rolling linear trend analysis on signal statistics.
  - calibration_health: Composite health score per window.

- MultiSensorValidationEvents: Cross-validate redundant inline sensors.
  - detect_disagreement: Flag windows where sensor spread exceeds threshold.
  - pairwise_bias: Mean difference between each sensor pair per window.
  - consensus_score: Per-window measurement consensus across sensors.
  - detect_outlier_sensor: Find the sensor furthest from the group.
  - sensor_uuids: Deprecated read-only alias of ``signal_uuids``.

- GaugeRepeatabilityEvents: Measurement System Analysis (Gauge R&R).
  - repeatability: Equipment Variation (EV) per part.
  - reproducibility: Appraiser Variation (AV) across operators.
  - gauge_rr_summary: Full Gauge R&R table with %GRR and ndc.
  - measurement_bias: Compare measurements to known reference values.

- ChangePointEvents: Offline change-point detection (PELT, binary segmentation).
  - detect_change_points: Level (and spread) shifts with before/after stats.
  - segment_statistics: Stable segments between change points.

- MultivariateSPCEvents: Hotelling T² multivariate SPC.
  - hotelling_t2: T² with Phase I / Phase II control limits.
  - contributions: Per-variable T² decomposition.
  - detect_out_of_control: Out-of-control runs with the top contributor.

- MeasurementUncertaintyEvents: GUM-based measurement uncertainty estimation.
  - type_a_uncertainty: Statistical (Type A) uncertainty per window.
  - type_b_uncertainty: Uncertainty from external sources (calibration, resolution).
  - combined_uncertainty: Combined and expanded uncertainty (u_c, U).
  - uncertainty_budget: Contribution breakdown by source.

- EnvironmentalCompensationEvents: Environmental effect compensation.
  - detect_correlation: Pearson correlation with environmental signal per window.
  - sensitivity: Environmental sensitivity coefficient estimation.
  - compensated_signal: Apply linear compensation model.
  - detect_env_exceedance: Flag out-of-range environmental conditions.

- CalibrationValidationEvents: Multi-point calibration validation.
  - linearity_assessment: Residual analysis at calibration points.
  - calibration_curve: Polynomial fit to calibration data.
  - detect_recalibration_need: Monitor bias/drift for recal triggers.
  - calibration_stability: Bias stability scoring over time.

- SamplingResolutionEvents: Sampling and resolution analysis.
  - detect_quantization: Detect A/D quantization in measurements.
  - effective_resolution: Estimate effective vs theoretical resolution.
  - detect_aliasing: Check for Nyquist violations via FFT.
  - sampling_jitter: Measure timing regularity of sampling.

- DataGapAnalysisEvents: Analyze gaps and coverage in signal data.
  - detect_gaps: Locate all gaps longer than a threshold.
  - gap_summary: Aggregate statistics across all gaps.
  - coverage_by_period: Data coverage percentage per time window.
  - interpolation_candidates: Gaps small enough to interpolate safely.

- ValueDistributionEvents: Examine signal distribution over time.
  - detect_mode_changes: Detect shifts between distinct operating modes.
  - detect_bimodal: Test whether the signal has a bimodal distribution.
  - normality_windows: Flag time windows with non-normal distributions.
  - percentile_tracking: Track selected percentiles over time windows.
"""

# This module re-exports the deprecated class aliases on purpose; using them
# anywhere else is still reported by mypy's "deprecated" check.
# mypy: disable-error-code="deprecated"

import importlib as _importlib
from typing import TYPE_CHECKING

# Explicit imports for type checkers and IDEs only (so ``from ts_shape import X``
# is typed as X, not object); at runtime __getattr__ below imports lazily.
# tests/typing/test_lazy_exports.py keeps this block in sync with the map.
if TYPE_CHECKING:
    from ts_shape.events.quality.anomaly_classification import (
        AnomalyClassificationEvents as AnomalyClassificationEvents,
    )
    from ts_shape.events.quality.calibration_validation import (
        CalibrationValidationEvents as CalibrationValidationEvents,
    )
    from ts_shape.events.quality.capability_trending import (
        CapabilityTrendingEvents as CapabilityTrendingEvents,
    )
    from ts_shape.events.quality.change_point import (
        ChangePointEvents as ChangePointEvents,
    )
    from ts_shape.events.quality.data_gap_analysis import (
        DataGapAnalysisEvents as DataGapAnalysisEvents,
    )
    from ts_shape.events.quality.environmental_compensation import (
        EnvironmentalCompensationEvents as EnvironmentalCompensationEvents,
    )
    from ts_shape.events.quality.gauge_repeatability import (
        GaugeRepeatabilityEvents as GaugeRepeatabilityEvents,
    )
    from ts_shape.events.quality.measurement_uncertainty import (
        MeasurementUncertaintyEvents as MeasurementUncertaintyEvents,
    )
    from ts_shape.events.quality.multi_sensor_validation import (
        MultiSensorValidationEvents as MultiSensorValidationEvents,
    )
    from ts_shape.events.quality.multivariate_spc import (
        MultivariateSPCEvents as MultivariateSPCEvents,
    )
    from ts_shape.events.quality.outlier_detection import (
        OutlierDetectionEvents as OutlierDetectionEvents,
    )
    from ts_shape.events.quality.sampling_resolution import (
        SamplingResolutionEvents as SamplingResolutionEvents,
    )
    from ts_shape.events.quality.sensor_drift import (
        SensorDriftEvents as SensorDriftEvents,
    )
    from ts_shape.events.quality.signal_quality import (
        SignalQualityEvents as SignalQualityEvents,
    )
    from ts_shape.events.quality.statistical_process_control import (
        StatisticalProcessControlEvents as StatisticalProcessControlEvents,
    )
    from ts_shape.events.quality.statistical_process_control import (
        StatisticalProcessControlRuleBased as StatisticalProcessControlRuleBased,
    )
    from ts_shape.events.quality.tolerance_deviation import (
        ToleranceDeviationEvents as ToleranceDeviationEvents,
    )
    from ts_shape.events.quality.value_distribution import (
        ValueDistributionEvents as ValueDistributionEvents,
    )

from ts_shape import _LAZY

# Classes are imported on first access, so ``import ts_shape.events.quality``
# stays cheap and optional dependencies (e.g. scikit-learn for the isolation
# forest) are only needed by the class that uses them.
_CLASSES = {
    name: module for name, module in _LAZY.items() if module.startswith(__name__ + ".")
}

__all__ = sorted(_CLASSES)


def __getattr__(name: str) -> object:
    if name in _CLASSES:
        return getattr(_importlib.import_module(_CLASSES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CLASSES))
