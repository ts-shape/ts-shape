import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._output import twin_columns
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only

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

# d2* for a single range (g = 1) of m values: AIAG MSA K2 = 1/d2* (appraisers)
# and K3 = 1/d2* (parts). The large-g d2 above would overstate AV and PV.
_D2_STAR_G1 = {
    2: 1.41421,
    3: 1.91155,
    4: 2.23887,
    5: 2.48124,
    6: 2.67253,
    7: 2.82981,
    8: 2.96288,
    9: 3.07794,
    10: 3.17905,
    11: 3.26909,
    12: 3.35016,
    13: 3.42378,
    14: 3.49116,
    15: 3.55333,
}


def _d2_star(m: int) -> float:
    """d2* (g = 1) for a range of ``m`` values; ``m > 15`` uses ``m = 15``."""
    return _D2_STAR_G1[min(max(m, 2), 15)]


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
        operator_column: str | None = None,
        event_uuid: str = "quality:gauge_rr",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
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

    @keyword_only(1)
    def repeatability(self, n_trials: int | None = None) -> pd.DataFrame:
        """Equipment Variation (EV) — within-part variation.

        Groups measurements by part, computes range-based or pooled
        standard deviation estimate of repeatability.

        When ``operator_column`` is set, ranges and standard deviations are
        taken within each operator x part cell and averaged per part (AIAG
        average-and-range method). Pooling every operator's trials into one
        range folded operator bias (reproducibility) into EV.

        Args:
            n_trials: Expected number of trials per part. If None,
                auto-detected from data.

        Returns:
            DataFrame with columns: part, mean, range, repeatability_std, EV.
        """
        cols = ["part", "mean", "range", "repeatability_std", "EV"]
        if self.signal.empty or self.part_column not in self.signal.columns:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []
        grouped = self.signal.groupby(self.part_column)

        ranges = []
        by_operator = (
            self.operator_column is not None
            and self.operator_column in self.signal.columns
        )
        for part, group in grouped:
            vals = group[self.value_column].dropna()
            if len(vals) < 2:
                continue

            part_mean = float(vals.mean())
            if by_operator:
                cells = [
                    cell[self.value_column].dropna()
                    for _, cell in group.groupby(self.operator_column)
                ]
                cells = [c for c in cells if len(c) >= 2]
                if not cells:
                    continue
                part_range = float(np.mean([c.max() - c.min() for c in cells]))
                part_std = float(np.sqrt(np.mean([c.var() for c in cells])))
                cell_trials = round(np.mean([len(c) for c in cells]))
                trials = n_trials if n_trials else cell_trials
            else:
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
                    "mean": float(part_mean),
                    "range": float(part_range),
                    "repeatability_std": float(part_std),
                    "EV": float(ev),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    def reproducibility(self) -> pd.DataFrame:
        """Appraiser Variation (AV) — between-operator variation.

        Groups by operator_column, computes variation of part means
        across operators. Requires operator_column to be set.

        AIAG average-and-range method: ``AV = sqrt((X_diff * K2)^2 -
        EV^2 / (n * r))`` with ``X_diff`` the range of operator means,
        ``K2 = 1/d2*`` for the number of operators, ``n`` parts and ``r``
        trials. The ``EV^2 / (n * r)`` term removes the repeatability noise
        already contained in the operator means; a negative radicand gives
        ``AV = 0``.

        Returns:
            DataFrame with columns: operator, mean, reproducibility_std, AV.
        """
        cols = ["operator", "mean", "reproducibility_std", "AV"]
        if self.operator_column is None:
            return pd.DataFrame(columns=cols)
        if self.signal.empty or self.operator_column not in self.signal.columns:
            return pd.DataFrame(columns=cols)

        events: list[dict[str, Any]] = []

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
                    "mean": float(op_mean),
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

        op_range = max(op_means) - min(op_means)
        av_raw_sq = (op_range / _d2_star(len(op_means))) ** 2
        rep = self.repeatability()
        if rep.empty:
            ev_noise = 0.0
        else:
            n_parts = len(rep)
            cells = self.signal.groupby([self.part_column, self.operator_column])[
                self.value_column
            ].count()
            trials = float(cells[cells > 0].mean())
            ev_noise = float(rep["EV"].mean()) ** 2 / (n_parts * trials)
        av = float(np.sqrt(max(av_raw_sq - ev_noise, 0.0)))
        repro_std = float(np.std(op_means, ddof=1))

        for e in events:
            e["reproducibility_std"] = float(repro_std)
            e["AV"] = float(av)

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )

    @keyword_only(1)
    @twin_columns(
        {
            "pct_GRR": "GRR_pct",
            "pct_PV": "PV_pct",
            "pct_GRR_tolerance": "GRR_tolerance_pct",
        }
    )
    def gauge_rr_summary(self, tolerance_range: float | None = None) -> pd.DataFrame:
        """Full Gauge R&R summary table.

        Computes EV, AV, GRR, PV, TV, %GRR, %PV, and number of
        distinct categories (ndc).

        Args:
            tolerance_range: Total tolerance range (USL - LSL). If
                provided, %GRR is also computed relative to tolerance.

        Returns:
            DataFrame with single row: EV, AV, GRR, PV, TV, pct_GRR,
            pct_PV, ndc, and optionally pct_GRR_tolerance.

            Canonical twin columns: ``GRR_pct`` (after ``pct_GRR``), ``PV_pct`` (after
            ``pct_PV``), ``GRR_tolerance_pct`` (after ``pct_GRR_tolerance``); each holds
            the same values as the deprecated column right before it, which is removed
            in the next major release.
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

        # PV (Part Variation): range of part means x K3 (= 1/d2*, AIAG)
        part_means = rep["mean"].values
        if len(part_means) >= 2:
            pv_range = float(part_means.max() - part_means.min())
            pv = pv_range / _d2_star(len(part_means))
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
            "EV": float(ev),
            "AV": float(av),
            "GRR": float(grr),
            "PV": float(pv),
            "TV": float(tv),
            "pct_GRR": float(pct_grr),
            "pct_PV": float(pct_pv),
            "ndc": ndc,
        }

        if tolerance_range is not None and tolerance_range > 0:
            result["pct_GRR_tolerance"] = float(grr / tolerance_range * 100)

        return pd.DataFrame([result])

    def measurement_bias(self, reference_values: dict[str, float]) -> pd.DataFrame:
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

        events: list[dict[str, Any]] = []
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
                    "measured_mean": float(measured_mean),
                    "reference": ref,
                    "bias": float(bias),
                    "bias_pct": float(bias_pct),
                }
            )

        return (
            pd.DataFrame(events, columns=cols) if events else pd.DataFrame(columns=cols)
        )
