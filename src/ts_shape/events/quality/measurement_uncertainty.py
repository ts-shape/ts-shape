from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

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
    """

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
        super().__init__(dataframe, column_name=time_column)
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

        # If part_uuid provided, extract part identifiers
        if self.part_uuid is not None:
            self._parts = (
                self.dataframe[self.dataframe["uuid"] == self.part_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            self._parts[self.time_column] = pd.to_datetime(
                self._parts[self.time_column]
            )
        else:
            self._parts = None

    def type_a_uncertainty(self, window: str = "8h") -> pd.DataFrame:
        """Statistical (Type A) uncertainty from repeated measurements.

        u_A = std / sqrt(n) per time window.

        Args:
            window: Resample window size.

        Returns:
            DataFrame with columns: window_start, mean, std, n_samples,
            type_a_uncertainty.
        """
        cols = ["window_start", "mean", "std", "n_samples", "type_a_uncertainty"]
        if self.signal.empty:
            return pd.DataFrame(columns=cols)

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
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "n_samples": n,
                    "type_a_uncertainty": round(u_a, 8),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

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
                    "divisor": round(divisor, 6),
                    "type_b_uncertainty": round(u_b, 8),
                }
            )

        return pd.DataFrame(events, columns=cols)

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
            DataFrame with columns: window_start, type_a, type_b_combined,
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

        type_a_df = self.type_a_uncertainty(window)
        if type_a_df.empty:
            return pd.DataFrame(columns=cols)

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
                    "type_a": round(u_a, 8),
                    "type_b_combined": round(type_b_combined, 8),
                    "combined_uncertainty": round(u_c, 8),
                    "expanded_uncertainty": round(u_expanded, 8),
                    "coverage_factor": coverage_factor,
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

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
        """
        cols = ["source", "uncertainty", "pct_contribution"]

        type_a_df = self.type_a_uncertainty(window)
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
            e["pct_contribution"] = round(
                (e["uncertainty"] ** 2 / total_variance) * 100, 2
            )
            e["uncertainty"] = round(e["uncertainty"], 8)

        return pd.DataFrame(entries, columns=cols)
