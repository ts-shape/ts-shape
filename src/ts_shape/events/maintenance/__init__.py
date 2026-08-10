"""Maintenance Events

Detectors for maintenance-related patterns: degradation trends, failure
prediction, and vibration analysis for predictive maintenance on
manufacturing/industrial IoT time series data.

Classes:
- AssetHealthIndexEvents: Fuse several degradation signals into one asset
  health index, weighted by failure-mode criticality.
  - health_index: Weighted 0-100 asset score per evaluation window.
  - health_events: Point events on warning/critical threshold crossings.
  - signal_contributions: Per-signal share of the asset's health loss.
  - maintenance_forecast: Days until the index reaches a critical level.

- DegradationDetectionEvents: Detect trend degradation, variance increases,
  level shifts, and compute composite health scores.
  - detect_trend_degradation: Rolling linear regression slope detection.
  - detect_variance_increase: Rolling variance vs baseline comparison.
  - detect_level_shift: CUSUM-like permanent mean shift detection.
  - health_score: Composite 0-100 score from drift, variance, and trend.

- FailurePredictionEvents: Predict remaining useful life and escalating
  failure patterns.
  - remaining_useful_life: Linear extrapolation to failure threshold.
  - detect_exceedance_pattern: Rolling threshold exceedance frequency.
  - time_to_threshold: Rate-of-change based threshold ETA.

- VibrationAnalysisEvents: Vibration signal analysis for rotating equipment.
  - detect_rms_exceedance: Rolling RMS vs baseline threshold.
  - detect_amplitude_growth: Peak-to-peak amplitude tracking.
  - bearing_health_indicators: Kurtosis and crest factor per window.
"""

from .asset_health_index import AssetHealthIndexEvents
from .degradation_detection import DegradationDetectionEvents
from .failure_prediction import FailurePredictionEvents
from .vibration_analysis import VibrationAnalysisEvents

__all__ = [
    "AssetHealthIndexEvents",
    "DegradationDetectionEvents",
    "FailurePredictionEvents",
    "VibrationAnalysisEvents",
]
