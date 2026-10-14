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

- RootCauseCorrelationEvents: Rank candidate causes for events another
  detector already found (directed, event-driven analysis).
  - rank_causes: Ranked causal candidates with lead time and p-value.
  - top_cause: Single-row verdict naming the best-supported candidate.
  - lag_profile: Correlation-vs-lag curve for one candidate signal.
"""

from .anomaly_correlation import AnomalyCorrelationEvents
from .root_cause_correlation import RootCauseCorrelationEvents
from .signal_correlation import SignalCorrelationEvents

__all__ = [
    "AnomalyCorrelationEvents",
    "RootCauseCorrelationEvents",
    "SignalCorrelationEvents",
]
