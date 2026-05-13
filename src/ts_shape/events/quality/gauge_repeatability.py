import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)

# d2 constants for subgroup sizes 2-10 (AIAG MSA reference)
_D2 = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
}


class GaugeRepeatabilityEvents(Base):
    """Quality: Gauge Repeatability & Reproducibility (Gauge R&R)

    Measurement System Analysis (MSA) for inline sensors. Evaluates
    repeatability (within-part, same sensor) and reproducibility
    (across operators/stations) using repeated measurements of
    reference parts or check standards.

    Methods:
    - repeatability: Equipment Variation (EV) per part.
    - reproducibility: Appraiser Variation (AV) across operators.
    - gauge_rr_summary: Full Gauge R&R table with %GRR and ndc.
    - measurement_bias: Compare measurements to known reference values.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        signal_uuid: str,
        *,
        part_column: str = "value_string",
        value_column: str = "value_double",
        operator_column: Optional[str] = None,
        event_uuid: str = "quality:gauge_rr",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.signal_uuid = signal_uuid
        self.part_column = part_column
        self.value_column = value_column
        self.operator_column = operator_column
        self.event_uuid = event_uuid
        self.time_column = time_column

        # Extract signal data
        self.signal = (
            self.dataframe[self.dataframe["uuid"] == self.signal_uuid]
            .copy()
            .sort_values(self.time_column)
        )

    def repeatability(self, n_trials: Optional[int] = None) -> pd.DataFrame:
        """Equipment Variation (EV) — within-part variation.

        Groups measurements by part, computes range-based or pooled
        standard deviation estimate of repeatability.

        Args:
            n_trials: Expected number of trials per part. If None,
                auto-detected from data.

        Returns:
            DataFrame with columns: part, mean, range, repeatability_std, EV.
        """
        cols = ["part", "mean", "range", "repeatability_std", "EV"]
        if self.signal.empty or self.part_column not in self.signal.columns:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        grouped = self.signal.groupby(self.part_column)

        ranges = []
        for part, group in grouped:
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue

            part_mean = float(vals.mean())
            part_range = float(vals.max() - vals.min())
            part_std = float(vals.std())
            trials = n_trials if n_trials else len(vals)

            # EV using range/d2 method (AIAG)
            d2 = _D2.get(min(trials, 10), _D2[10])
            ev = part_range / d2

            ranges.append(part_range)
            events.append(
                {
                    "part": part,
                    "mean": round(part_mean, 6),
                    "range": round(part_range, 6),
                    "repeatability_std": round(part_std, 6),
                    "EV": round(ev, 6),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def reproducibility(self) -> pd.DataFrame:
        """Appraiser Variation (AV) — between-operator variation.

        Groups by operator_column, computes variation of part means
        across operators. Requires operator_column to be set.

        Returns:
            DataFrame with columns: operator, mean, reproducibility_std, AV.
        """
        cols = ["operator", "mean", "reproducibility_std", "AV"]
        if self.operator_column is None:
            return pd.DataFrame(columns=cols)
        if self.signal.empty or self.operator_column not in self.signal.columns:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []

        # Per-operator mean across all parts
        op_means = []
        for op, group in self.signal.groupby(self.operator_column):
            vals = group[self.value_column].dropna()
            if vals.empty:
                continue
            op_mean = float(vals.mean())
            op_means.append(op_mean)
            events.append(
                {
                    "operator": op,
                    "mean": round(op_mean, 6),
                    "reproducibility_std": 0.0,
                    "AV": 0.0,
                }
            )

        if len(op_means) < 2:
            return (
                pd.DataFrame(events, columns=cols)
                if events
                else pd.DataFrame(columns=cols)
            )

        # AV = range of operator means / d2
        op_range = max(op_means) - min(op_means)
        d2 = _D2.get(min(len(op_means), 10), _D2[10])
        av = op_range / d2
        repro_std = float(np.std(op_means, ddof=1))

        for e in events:
            e["reproducibility_std"] = round(repro_std, 6)
            e["AV"] = round(av, 6)

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def gauge_rr_summary(self, tolerance_range: Optional[float] = None) -> pd.DataFrame:
        """Full Gauge R&R summary table.

        Computes EV, AV, GRR, PV, TV, %GRR, %PV, and number of
        distinct categories (ndc).

        Args:
            tolerance_range: Total tolerance range (USL - LSL). If
                provided, %GRR is also computed relative to tolerance.

        Returns:
            DataFrame with single row: EV, AV, GRR, PV, TV, pct_GRR,
            pct_PV, ndc, and optionally pct_GRR_tolerance.
        """
        cols = ["EV", "AV", "GRR", "PV", "TV", "pct_GRR", "pct_PV", "ndc"]
        if tolerance_range is not None:
            cols.append("pct_GRR_tolerance")

        if self.signal.empty or self.part_column not in self.signal.columns:
            return pd.DataFrame(columns=cols)

        # Compute EV (average across parts)
        rep = self.repeatability()
        if rep.empty:
            return pd.DataFrame(columns=cols)
        ev = float(rep["EV"].mean())

        # Compute AV
        repro = self.reproducibility()
        if not repro.empty and repro["AV"].iloc[0] > 0:
            av = float(repro["AV"].iloc[0])
        else:
            av = 0.0

        # GRR
        grr = np.sqrt(ev**2 + av**2)

        # PV (Part Variation) — std of part means × correction
        part_means = rep["mean"].values
        if len(part_means) >= 2:
            pv_range = float(part_means.max() - part_means.min())
            d2 = _D2.get(min(len(part_means), 10), _D2[10])
            pv = pv_range / d2
        else:
            pv = 0.0

        # TV (Total Variation)
        tv = np.sqrt(grr**2 + pv**2)

        # Percentages
        pct_grr = (grr / tv * 100) if tv > 0 else 0.0
        pct_pv = (pv / tv * 100) if tv > 0 else 0.0

        # ndc (number of distinct categories)
        ndc = int(np.floor(1.41 * (pv / grr))) if grr > 0 else 0

        result = {
            "EV": round(ev, 6),
            "AV": round(av, 6),
            "GRR": round(grr, 6),
            "PV": round(pv, 6),
            "TV": round(tv, 6),
            "pct_GRR": round(pct_grr, 2),
            "pct_PV": round(pct_pv, 2),
            "ndc": ndc,
        }

        if tolerance_range is not None and tolerance_range > 0:
            result["pct_GRR_tolerance"] = round(grr / tolerance_range * 100, 2)

        return pd.DataFrame([result])

    def measurement_bias(self, reference_values: Dict[str, float]) -> pd.DataFrame:
        """Compare average measurements to known reference values.

        Args:
            reference_values: Dict mapping part identifier to its true
                reference value, e.g. {"part_A": 10.0, "part_B": 20.0}.

        Returns:
            DataFrame with columns: part, measured_mean, reference,
            bias, bias_pct.
        """
        cols = ["part", "measured_mean", "reference", "bias", "bias_pct"]
        if self.signal.empty or self.part_column not in self.signal.columns:
            return pd.DataFrame(columns=cols)

        events: List[Dict[str, Any]] = []
        for part, group in self.signal.groupby(self.part_column):
            if part not in reference_values:
                continue
            vals = group[self.value_column].dropna()
            if vals.empty:
                continue

            measured_mean = float(vals.mean())
            ref = reference_values[part]
            bias = measured_mean - ref
            bias_pct = (bias / ref * 100) if ref != 0 else 0.0

            events.append(
                {
                    "part": part,
                    "measured_mean": round(measured_mean, 6),
                    "reference": ref,
                    "bias": round(bias, 6),
                    "bias_pct": round(bias_pct, 4),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
