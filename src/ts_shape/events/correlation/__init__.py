"""Correlation Events

Cross-signal correlation analysis for detecting related anomalies
and patterns across multiple timeseries signals.

Classes:
- SignalCorrelationEvents: Analyze correlations between signals.
  - rolling_correlation: Time-windowed Pearson correlation between two signals.
  - correlation_breakdown: Detect periods where normally correlated signals diverge.
  - lag_correlation: Cross-correlation with time lag analysis.

- AnomalyCorrelationEvents: Correlate anomaly events across signals.
  - coincident_anomalies: Find anomalies occurring simultaneously across signals.
  - cascade_detection: Detect anomaly cascades (signal A anomaly followed by B).
  - root_cause_ranking: Rank signals by how often their anomalies precede others.
"""

from .anomaly_correlation import AnomalyCorrelationEvents
from .signal_correlation import SignalCorrelationEvents

__all__ = [
    "SignalCorrelationEvents",
    "AnomalyCorrelationEvents",
]
