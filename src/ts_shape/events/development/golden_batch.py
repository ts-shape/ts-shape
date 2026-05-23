"""Golden-batch deviation: compare new batch trajectories to a reference.

In batch process development a *golden batch* is a canonical run whose
trajectory delivered the target product. Comparing later batches to that
trajectory -- per timestep, per phase, or as a whole-trajectory shape
distance -- is a workhorse diagnostic in pharma, food, and specialty
chemicals.

This detector supports three comparison modes:

* ``pointwise`` -- resample both batches to a common normalised batch
  time and report the maximum signed residual.
* ``area``      -- numerical integral of the absolute residual using the
  trapezoid rule. Captures cumulative deviation.
* ``dtw``       -- pure-numpy dynamic time warping distance with a Sakoe
  -Chiba band. Captures trajectory *shape* even when batches run at
  different paces. No external DTW library required.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class GoldenBatchDeviationEvents(Base):
    """Quantify deviation of a new batch from a golden reference batch."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        signal_uuid: str,
        *,
        event_uuid: str = "dev:golden_batch_deviation",
        value_column: str = "value_double",
        time_column: str = "systime",
        n_resample: int = 256,
    ) -> None:
        super().__init__(reference_df, column_name=time_column)
        self.signal_uuid = signal_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        if n_resample < 4:
            raise ValueError("n_resample must be >= 4 for a useful comparison")
        self.n_resample = n_resample
        self._validate_uuid(self.dataframe, signal_uuid)

        ref = (
            self.dataframe[self.dataframe["uuid"] == signal_uuid][
                [self.time_column, self.value_column]
            ]
            .copy()
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )
        if ref.empty:
            raise ValueError(
                f"No samples found for reference signal_uuid={signal_uuid!r}"
            )
        self._ref_resampled = self._resample_to_grid(
            ref[self.time_column], ref[self.value_column].to_numpy(dtype=float)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resample_to_grid(self, times: pd.Series, values: np.ndarray) -> np.ndarray:
        """Resample a batch trace to ``n_resample`` evenly-spaced batch-time points.

        Batch time is normalised to [0, 1]; the trace is linearly
        interpolated onto an evenly spaced grid so two batches of
        different wall-clock durations are directly comparable.
        """
        t = pd.to_datetime(times).to_numpy()
        if len(t) < 2:
            return np.full(self.n_resample, float(values[0]) if len(values) else np.nan)
        t_sec = (t - t[0]) / np.timedelta64(1, "s")
        total = float(t_sec[-1])
        if total <= 0:
            return np.full(self.n_resample, float(values[0]))
        norm = t_sec / total
        grid = np.linspace(0.0, 1.0, self.n_resample)
        return np.interp(grid, norm, values)

    @staticmethod
    def _dtw_distance(a: np.ndarray, b: np.ndarray, band: int) -> float:
        """Dynamic time warping distance with a Sakoe-Chiba band.

        Pure-numpy O(n * band) implementation. Returns the *normalised*
        DTW distance (total cost divided by warping-path length).
        """
        n, m = len(a), len(b)
        if n == 0 or m == 0:
            return float("nan")
        band = max(int(band), abs(n - m) + 1)
        inf = np.inf
        prev = np.full(m + 1, inf, dtype=float)
        curr = np.full(m + 1, inf, dtype=float)
        prev[0] = 0.0
        for i in range(1, n + 1):
            curr[:] = inf
            j_lo = max(1, i - band)
            j_hi = min(m, i + band)
            for j in range(j_lo, j_hi + 1):
                cost = abs(a[i - 1] - b[j - 1])
                curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
            prev, curr = curr, prev
        total = prev[m]
        if not np.isfinite(total):
            return float("nan")
        return float(total / (n + m))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        batch_df: pd.DataFrame,
        *,
        mode: str = "pointwise",
        dtw_band_frac: float = 0.1,
    ) -> pd.DataFrame:
        """Compare a single new batch to the stored golden reference.

        Args:
            batch_df: Long-form DataFrame containing the candidate batch's
                signal. The configured ``signal_uuid`` is used to extract
                the trace.
            mode: ``"pointwise"``, ``"area"``, or ``"dtw"``.
            dtw_band_frac: Sakoe-Chiba band width as a fraction of the
                resampled grid length. Only used for ``mode="dtw"``.

        Returns:
            Interval-shape DataFrame with one row, columns: ``start``,
            ``end``, ``duration_seconds``, ``uuid``, ``mode``,
            ``deviation_score``, ``max_abs_residual``,
            ``batch_time_at_max``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "mode",
            "deviation_score",
            "max_abs_residual",
            "batch_time_at_max",
        ]
        if batch_df.empty:
            return pd.DataFrame(columns=cols)
        sel = batch_df[batch_df["uuid"] == self.signal_uuid][
            [self.time_column, self.value_column]
        ].copy()
        if sel.empty:
            return pd.DataFrame(columns=cols)
        sel = sel.sort_values(self.time_column).reset_index(drop=True)

        candidate = self._resample_to_grid(
            sel[self.time_column], sel[self.value_column].to_numpy(dtype=float)
        )
        residual = candidate - self._ref_resampled
        max_abs = float(np.max(np.abs(residual)))
        argmax = int(np.argmax(np.abs(residual)))
        batch_time_at_max = float(argmax / (self.n_resample - 1))

        if mode == "pointwise":
            score = max_abs
        elif mode == "area":
            # Trapezoid integral of |residual| on the normalised grid (0..1).
            grid = np.linspace(0.0, 1.0, self.n_resample)
            score = float(np.trapezoid(np.abs(residual), grid))
        elif mode == "dtw":
            band = max(2, int(dtw_band_frac * self.n_resample))
            score = self._dtw_distance(candidate, self._ref_resampled, band)
        else:
            raise ValueError(
                f"Unknown mode {mode!r}; choose 'pointwise', 'area', or 'dtw'."
            )

        start = pd.to_datetime(sel[self.time_column].iloc[0])
        end = pd.to_datetime(sel[self.time_column].iloc[-1])
        row: dict[str, Any] = {
            "start": start,
            "end": end,
            "duration_seconds": (end - start).total_seconds(),
            "uuid": self.event_uuid,
            "mode": mode,
            "deviation_score": score,
            "max_abs_residual": max_abs,
            "batch_time_at_max": batch_time_at_max,
        }
        return pd.DataFrame([row], columns=cols)

    def phase_breakdown(
        self,
        batch_df: pd.DataFrame,
        phase_df: pd.DataFrame,
        *,
        phase_uuid: str,
    ) -> pd.DataFrame:
        """Per-phase deviation of a batch against the golden reference.

        The reference batch is sliced into phases by the same boundaries
        applied to the candidate batch -- this assumes the phase sequence
        is consistent across runs, which is true for recipes that are
        executed phase-by-phase under recipe control.

        Args:
            batch_df: Long-form signal DataFrame for the candidate batch.
            phase_df: Long-form DataFrame containing the phase-tracking
                signal, where ``value_string`` (or the configured value
                column) carries the phase name.
            phase_uuid: UUID of the phase signal in ``phase_df``.

        Returns:
            Summary DataFrame: ``start``, ``end``, ``duration_seconds``,
            ``phase``, ``deviation_score``, ``max_abs_residual``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "phase",
            "deviation_score",
            "max_abs_residual",
        ]
        if batch_df.empty or phase_df.empty:
            return pd.DataFrame(columns=cols)

        # Determine phase intervals from the phase signal.
        phase_col = (
            "value_string" if "value_string" in phase_df.columns else self.value_column
        )
        ph = (
            phase_df[phase_df["uuid"] == phase_uuid][[self.time_column, phase_col]]
            .copy()
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )
        if ph.empty:
            return pd.DataFrame(columns=cols)

        # Build contiguous phase intervals.
        ph["_change"] = (ph[phase_col] != ph[phase_col].shift()).cumsum()
        intervals = (
            ph.groupby("_change")
            .agg(
                start=(self.time_column, "first"),
                end=(self.time_column, "last"),
                phase=(phase_col, "first"),
            )
            .reset_index(drop=True)
        )

        # Extract candidate batch trace once.
        sel = (
            batch_df[batch_df["uuid"] == self.signal_uuid][
                [self.time_column, self.value_column]
            ]
            .copy()
            .sort_values(self.time_column)
            .reset_index(drop=True)
        )
        if sel.empty:
            return pd.DataFrame(columns=cols)
        sel[self.time_column] = pd.to_datetime(sel[self.time_column])

        rows: list[dict[str, Any]] = []
        for _, ivl in intervals.iterrows():
            mask = (sel[self.time_column] >= ivl["start"]) & (
                sel[self.time_column] <= ivl["end"]
            )
            window = sel.loc[mask]
            if len(window) < 2:
                continue
            candidate = self._resample_to_grid(
                window[self.time_column],
                window[self.value_column].to_numpy(dtype=float),
            )
            residual = candidate - self._ref_resampled
            grid = np.linspace(0.0, 1.0, self.n_resample)
            rows.append(
                {
                    "start": ivl["start"],
                    "end": ivl["end"],
                    "duration_seconds": (ivl["end"] - ivl["start"]).total_seconds(),
                    "phase": ivl["phase"],
                    "deviation_score": float(np.trapezoid(np.abs(residual), grid)),
                    "max_abs_residual": float(np.max(np.abs(residual))),
                }
            )
        return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
