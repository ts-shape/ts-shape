import warnings
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.events._windows import labeled_window_summary
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import REMOVAL, caller_stacklevel, keyword_only

# Divisors for Type B uncertainty distributions (GUM standard)
_DIVISORS = {
    "rectangular": np.sqrt(3),
    "triangular": np.sqrt(6),
    "normal": 2.0,  # default coverage factor k=2
    "u-shaped": np.sqrt(2),
}


class MeasurementUncertaintyEvents(Base):
    """Quality: GUM-based Measurement Uncertainty Estimation

    Compute Type A (statistical) and Type B (external sources)
    uncertainty, combined and expanded uncertainty, and uncertainty
    budgets per ISO/IEC Guide 98-3 (GUM).

    Methods:
    - type_a_uncertainty: Statistical uncertainty from repeated measurements.
    - type_b_uncertainty: Uncertainty from calibration certs, resolution, etc.
    - combined_uncertainty: Root-sum-of-squares combination with expansion.
    - uncertainty_budget: Contribution breakdown by source.

    ``part_uuid`` is deprecated: it has no effect (no method groups by
    part) and will be removed in the next major release; passing it emits a
    ``DeprecationWarning``.
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        value_column: str = "value_double",
        part_uuid: str | None = None,
        event_uuid: str = "quality:uncertainty",
        time_column: str = "systime",
    ) -> None:
        """Initialize the uncertainty estimator.

        Args:
            dataframe: Long-format timeseries frame (one row per sample, ``uuid``
                identifies the signal).
            signal_uuid: UUID of the measurement signal.
            value_column: Column holding the signal values.
            part_uuid: Deprecated; has no effect and warns when given.
            event_uuid: UUID stamped on emitted events.
            time_column: Timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.signal_uuid = signal_uuid
        self.value_column = value_column
        self.part_uuid = part_uuid
        self.event_uuid = event_uuid
        self.time_column = time_column

        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.signal[self.time_column] = pd.to_datetime(self.signal[self.time_column])

        # Nothing groups by part; the parameter only warns.
        if self.part_uuid is not None:
            warnings.warn(
                "MeasurementUncertaintyEvents(): part_uuid has no effect and "
                f"will be removed in {REMOVAL}",
                DeprecationWarning,
                stacklevel=caller_stacklevel(),
            )

    @keyword_only(1)
    @twin_columns({"n_samples": "sample_count"})
    def type_a_uncertainty(self, window: str = "8h") -> pd.DataFrame:
        """Statistical (Type A) uncertainty from repeated measurements.

        u_A = std / sqrt(n) per time window.

        Args:
            window: Resample window size.

        Returns:
            Summary DataFrame with columns: start, end, duration_seconds
            (the window bounds), window_start (the resample label; a
            deprecated twin of ``start``), mean, std, n_samples,
            type_a_uncertainty.

            Canonical twin columns: ``sample_count`` (after ``n_samples``); each holds
            the same values as the deprecated column right before it, which is removed
            in the next major release.
        """
        cols = ["window_start", "mean", "std", "n_samples", "type_a_uncertainty"]
        if self.signal.empty:
            return labeled_window_summary([], cols, window)

        sig = self.signal[[self.time_column, self.value_column]].copy()
        sig = sig.set_index(self.time_column)

        events: list[dict[str, Any]] = []
        for ts, group in sig.resample(window):
            vals = group[self.value_column].dropna()
            n = len(vals)
            if n < 2:
                continue

            mean = float(vals.mean())
            std = float(vals.std())
            u_a = std / np.sqrt(n)

            events.append(
                {
                    "window_start": ts,
                    "mean": float(mean),
                    "std": float(std),
                    "n_samples": n,
                    "type_a_uncertainty": float(u_a),
                }
            )

        return labeled_window_summary(events, cols, window)

    @staticmethod
    def type_b_uncertainty(sources: list[dict[str, Any]]) -> pd.DataFrame:
        """Type B uncertainty from external sources (GUM method).

        Args:
            sources: List of dicts, each with:
                - 'source': str (name, e.g. 'calibration', 'resolution')
                - 'half_width': float (half-width of distribution)
                - 'distribution': str ('rectangular', 'triangular',
                  'normal', 'u-shaped')

        Returns:
            DataFrame with columns: source, distribution, half_width,
            divisor, type_b_uncertainty.
        """
        cols = ["source", "distribution", "half_width", "divisor", "type_b_uncertainty"]
        if not sources:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []
        for s in sources:
            dist = s.get("distribution", "rectangular")
            divisor = _DIVISORS.get(dist, _DIVISORS["rectangular"])
            half_width = float(s["half_width"])
            u_b = half_width / divisor

            events.append(
                {
                    "source": s["source"],
                    "distribution": dist,
                    "half_width": half_width,
                    "divisor": float(divisor),
                    "type_b_uncertainty": float(u_b),
                }
            )

        return pd.DataFrame(events, columns=cols)

    @keyword_only(1)
    def combined_uncertainty(
        self,
        window: str = "8h",
        type_b_sources: list[dict[str, Any]] | None = None,
        coverage_factor: float = 2.0,
    ) -> pd.DataFrame:
        """Combined and expanded uncertainty (GUM).

        u_c = sqrt(u_A² + sum(u_Bi²))
        U = k × u_c

        Args:
            window: Resample window size.
            type_b_sources: Type B sources (same format as type_b_uncertainty).
            coverage_factor: k value for expanded uncertainty (default 2.0
                for ~95.45% coverage).

        Returns:
            Summary DataFrame with columns: start, end, duration_seconds
            (the window bounds), window_start (the resample label; a
            deprecated twin of ``start``), type_a, type_b_combined,
            combined_uncertainty, expanded_uncertainty, coverage_factor.
        """
        cols = [
            "window_start",
            "type_a",
            "type_b_combined",
            "combined_uncertainty",
            "expanded_uncertainty",
            "coverage_factor",
        ]

        type_a_df = self.type_a_uncertainty(window=window)
        if type_a_df.empty:
            return labeled_window_summary([], cols, window)

        # Compute combined Type B
        type_b_combined = 0.0
        if type_b_sources:
            type_b_df = self.type_b_uncertainty(type_b_sources)
            type_b_combined = float(
                np.sqrt((type_b_df["type_b_uncertainty"] ** 2).sum())
            )

        events: list[dict[str, Any]] = []
        for _, row in type_a_df.iterrows():
            u_a = row["type_a_uncertainty"]
            u_c = np.sqrt(u_a**2 + type_b_combined**2)
            u_expanded = coverage_factor * u_c

            events.append(
                {
                    "window_start": row["window_start"],
                    "type_a": float(u_a),
                    "type_b_combined": float(type_b_combined),
                    "combined_uncertainty": float(u_c),
                    "expanded_uncertainty": float(u_expanded),
                    "coverage_factor": coverage_factor,
                }
            )

        return labeled_window_summary(events, cols, window)

    @keyword_only(1)
    @twin_columns({"pct_contribution": "contribution_pct"})
    def uncertainty_budget(
        self,
        window: str = "8h",
        type_b_sources: list[dict[str, Any]] | None = None,
    ) -> pd.DataFrame:
        """Uncertainty budget — contribution breakdown by source.

        Shows which uncertainty source dominates (repeatability vs
        resolution vs calibration, etc).

        Args:
            window: Resample window size (uses mean Type A across windows).
            type_b_sources: Type B sources.

        Returns:
            DataFrame with columns: source, uncertainty, pct_contribution.

            Canonical twin columns: ``contribution_pct`` (after ``pct_contribution``);
            each holds the same values as the deprecated column right before it, which
            is removed in the next major release.
        """
        cols = ["source", "uncertainty", "pct_contribution"]

        type_a_df = self.type_a_uncertainty(window=window)
        if type_a_df.empty:
            return pd.DataFrame(columns=cols)

        # Average Type A across windows
        mean_u_a = float(type_a_df["type_a_uncertainty"].mean())

        entries: list[dict[str, Any]] = [
            {"source": "repeatability (Type A)", "uncertainty": mean_u_a}
        ]

        if type_b_sources:
            type_b_df = self.type_b_uncertainty(type_b_sources)
            for _, row in type_b_df.iterrows():
                entries.append(
                    {
                        "source": f"{row['source']} (Type B)",
                        "uncertainty": float(row["type_b_uncertainty"]),
                    }
                )

        # Compute variance contributions
        total_variance = sum(e["uncertainty"] ** 2 for e in entries)
        if total_variance <= 0:
            return pd.DataFrame(columns=cols)

        for e in entries:
            e["pct_contribution"] = float(
                (e["uncertainty"] ** 2 / total_variance) * 100
            )
            e["uncertainty"] = float(e["uncertainty"])

        return pd.DataFrame(entries, columns=cols)
