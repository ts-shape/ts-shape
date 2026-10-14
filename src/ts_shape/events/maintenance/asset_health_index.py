"""Asset health index: fuse several degradation signals into one score.

``DegradationDetectionEvents`` scores one signal at a time. A real asset
degrades across several of them at once -- vibration climbs while the
bearing temperature drifts and the motor current gets noisier -- and a
per-signal score cannot say *how healthy the machine is*, only how healthy
one of its sensors looks.

This detector evaluates every configured signal over the same window grid,
scores each with the same drift / variance / trend penalty model used by
:meth:`~ts_shape.events.maintenance.degradation_detection.DegradationDetectionEvents.health_score`,
and fuses the per-signal scores into a single weighted asset index. Each
signal carries a *failure mode* label, so the fused row also names the mode
that dominates the loss of health, and a confidence level that reports how
much of the configured signal weight actually had data in the window.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.utils.base import Base
from ts_shape.utils.coerce import to_numeric_checked
from ts_shape.utils.deprecation import renamed_kwargs

logger = logging.getLogger(__name__)

# Penalty split of the 100-point scale, mirroring
# DegradationDetectionEvents.health_score so a single-signal asset index and
# that detector's score stay comparable.
_DRIFT_WEIGHT = 33.3
_VAR_WEIGHT = 33.3
_SLOPE_WEIGHT = 33.4

_DEFAULT_FAILURE_MODE = "unknown"


class AssetHealthIndexEvents(Base):
    """Maintenance: multi-signal asset health fusion.

    Each asset is described by a mapping of signal UUID to a failure mode
    (and optionally a weight)::

        AssetHealthIndexEvents(
            df,
            asset_id="press-01",
            signals={
                "vib:press-01": "bearing",
                "temp:press-01": {"failure_mode": "thermal", "weight": 2.0},
                "curr:press-01": {"failure_mode": "electrical"},
            },
        )

    A bare string is shorthand for ``{"failure_mode": <str>, "weight": 1.0}``.
    Weights express failure-mode criticality: a thermal runaway that scraps
    the tool should outweigh a slow bearing drift.

    Methods:
    - health_index: fused 0-100 asset score per window.
    - health_events: point events when the index crosses warning/critical.
    - signal_contributions: per-signal breakdown of the health loss.
    - maintenance_forecast: days until the index reaches a critical level.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        asset_id: str,
        signals: Mapping[str, str | Mapping[str, Any]],
        *,
        event_uuid: str = "maint:asset_health",
        value_column: str = "value_double",
        time_column: str = "systime",
        uuid_column: str = "uuid",
    ) -> None:
        """Initialize the asset health index.

        Args:
            dataframe: Timeseries frame holding every configured signal.
            asset_id: Identifier of the asset the signals belong to. It is
                emitted as ``source_uuid`` so the event log binds the rows to
                one asset object.
            signals: Mapping of signal UUID to failure mode (string) or to a
                ``{"failure_mode": str, "weight": float}`` mapping.
            event_uuid: UUID stamped on emitted events.
            value_column: Column holding the numeric readings.
            time_column: Timestamp column.
            uuid_column: Column identifying the signal.

        Raises:
            ValueError: If ``signals`` is empty or a weight is not positive.
        """
        super().__init__(dataframe, time_column=time_column)
        if not signals:
            raise ValueError("signals must contain at least one signal UUID")

        self.asset_id = asset_id
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self.uuid_column = uuid_column
        self.signals = self._normalize_signals(signals)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_signals(
        signals: Mapping[str, str | Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Expand the shorthand signal spec into ``{uuid: {mode, weight}}``."""
        out: dict[str, dict[str, Any]] = {}
        for uuid, spec in signals.items():
            if isinstance(spec, str):
                mode, weight = spec, 1.0
            else:
                mode = str(spec.get("failure_mode", _DEFAULT_FAILURE_MODE))
                weight = float(spec.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(
                    f"weight for signal {uuid!r} must be positive, got {weight}"
                )
            out[uuid] = {"failure_mode": mode, "weight": weight}
        return out

    @property
    def _total_weight(self) -> float:
        return float(sum(s["weight"] for s in self.signals.values()))

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------

    def _signal_frame(self, uuid: str) -> pd.DataFrame:
        """Return the sorted ``(time, value)`` frame for one signal."""
        sel = self.dataframe[self.dataframe[self.uuid_column] == uuid]
        if sel.empty:
            return pd.DataFrame(columns=[self.time_column, self.value_column])
        out = sel[[self.time_column, self.value_column]].copy()
        out[self.time_column] = pd.to_datetime(out[self.time_column])
        out[self.value_column] = to_numeric_checked(
            out[self.value_column], what=f"signal {uuid!r}"
        )
        return out.dropna(subset=[self.value_column]).sort_values(self.time_column)

    @staticmethod
    def _slope_per_second(times: pd.Series, values: np.ndarray) -> float:
        """Least-squares slope of ``values`` against seconds elapsed."""
        if len(values) < 2:
            return 0.0
        secs = (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        if np.ptp(secs) == 0:
            return 0.0
        try:
            return float(np.polyfit(secs, values, 1)[0])
        except (np.linalg.LinAlgError, ValueError):  # pragma: no cover - defensive
            logger.debug("Slope fit failed for asset health window.")
            return 0.0

    def _signal_health(
        self, uuid: str, *, window: str, baseline_window: str
    ) -> pd.DataFrame:
        """Score one signal per window against its own opening baseline.

        Returns a frame indexed by window start with columns ``health``,
        ``mean_drift_pct``, ``variance_ratio``, ``trend_slope`` and ``count``.
        An empty frame means the signal had no usable data.
        """
        cols = [
            "health",
            "mean_drift_pct",
            "variance_ratio",
            "trend_slope",
            "count",
        ]
        sig = self._signal_frame(uuid)
        if sig.empty:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="start"))

        first_time = sig[self.time_column].iloc[0]
        baseline = sig.loc[
            sig[self.time_column] <= first_time + pd.to_timedelta(baseline_window),
            self.value_column,
        ]
        if len(baseline) < 2:
            # Too little history to judge drift; treat the signal as healthy
            # rather than inventing a penalty from a one-sample baseline.
            baseline_mean, baseline_var = float(baseline.mean() or 0.0), 0.0
            has_baseline = False
        else:
            baseline_mean = float(baseline.mean())
            baseline_var = float(baseline.var())
            has_baseline = True

        eps = float(np.finfo(float).eps)
        baseline_var = baseline_var if baseline_var > 0 else eps
        mean_scale = abs(baseline_mean) if baseline_mean != 0 else eps

        rows: list[dict[str, Any]] = []
        index: list[pd.Timestamp] = []
        grouped = sig.set_index(self.time_column).resample(window)
        for start, group in grouped:
            if group.empty:
                continue
            values = group[self.value_column].to_numpy(dtype=float)
            if not has_baseline or len(values) < 2:
                rows.append(
                    {
                        "health": 100.0,
                        "mean_drift_pct": 0.0,
                        "variance_ratio": 1.0,
                        "trend_slope": 0.0,
                        "count": len(values),
                    }
                )
                index.append(start)
                continue

            drift_pct = abs(float(values.mean()) - baseline_mean) / mean_scale * 100.0
            var_ratio = float(values.var(ddof=1)) / baseline_var
            slope = self._slope_per_second(
                group.index.to_series().reset_index(drop=True), values
            )

            drift_penalty = min(drift_pct / 100.0, 1.0) * _DRIFT_WEIGHT
            var_penalty = min(max(var_ratio - 1.0, 0.0) / 5.0, 1.0) * _VAR_WEIGHT
            slope_penalty = (
                min(abs(slope) * 3600.0 / (mean_scale + 1e-9), 1.0) * _SLOPE_WEIGHT
            )
            health = max(
                0.0, min(100.0, 100.0 - drift_penalty - var_penalty - slope_penalty)
            )

            rows.append(
                {
                    "health": float(health),
                    "mean_drift_pct": float(drift_pct),
                    "variance_ratio": float(var_ratio),
                    "trend_slope": float(slope),
                    "count": len(values),
                }
            )
            index.append(start)

        if not rows:
            return pd.DataFrame(columns=cols, index=pd.DatetimeIndex([], name="start"))
        return pd.DataFrame(rows, index=pd.DatetimeIndex(index, name="start"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_index(
        self,
        *,
        window: str = "1h",
        baseline_window: str = "24h",
    ) -> pd.DataFrame:
        """Fuse the configured signals into one asset health score per window.

        A signal with no data in a window is skipped rather than scored as
        zero; ``confidence_level`` reports the share of the configured weight
        that was actually available, so a score computed from one of four
        signals is visibly weaker evidence than a score from all four.

        Args:
            window: Evaluation window (e.g. ``'1h'``, ``'1D'``).
            baseline_window: Opening period of each signal used as its
                healthy reference for drift and variance.

        Returns:
            Summary-shape DataFrame: ``start``, ``end``, ``duration_seconds``,
            ``uuid``, ``source_uuid``, ``asset_id``, ``asset_health_score``,
            ``dominant_failure_mode``, ``worst_signal_uuid``,
            ``signals_available``, ``signals_expected``, ``confidence_level``,
            ``sample_count``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "asset_id",
            "asset_health_score",
            "dominant_failure_mode",
            "worst_signal_uuid",
            "signals_available",
            "signals_expected",
            "confidence_level",
            "sample_count",
        ]
        per_signal = {
            uuid: self._signal_health(
                uuid, window=window, baseline_window=baseline_window
            )
            for uuid in self.signals
        }
        per_signal = {u: f for u, f in per_signal.items() if not f.empty}
        if not per_signal:
            return pd.DataFrame(columns=cols)

        window_td = pd.to_timedelta(window)
        starts = sorted({ts for frame in per_signal.values() for ts in frame.index})

        rows: list[dict[str, Any]] = []
        for start in starts:
            weighted_sum = 0.0
            weight_sum = 0.0
            samples = 0
            worst_uuid: str | None = None
            worst_deficit = -1.0
            for uuid, frame in per_signal.items():
                if start not in frame.index:
                    continue
                health = float(frame.at[start, "health"])
                weight = float(self.signals[uuid]["weight"])
                weighted_sum += health * weight
                weight_sum += weight
                samples += int(frame.at[start, "count"])
                deficit = (100.0 - health) * weight
                if deficit > worst_deficit:
                    worst_deficit = deficit
                    worst_uuid = uuid
            if weight_sum == 0:  # pragma: no cover - guarded by the loop above
                continue

            score = weighted_sum / weight_sum
            mode = (
                self.signals[worst_uuid]["failure_mode"]
                if worst_uuid is not None and worst_deficit > 0
                else _DEFAULT_FAILURE_MODE
            )
            rows.append(
                {
                    "start": start,
                    "end": start + window_td,
                    "duration_seconds": window_td.total_seconds(),
                    "uuid": self.event_uuid,
                    "source_uuid": self.asset_id,
                    "asset_id": self.asset_id,
                    "asset_health_score": float(score),
                    "dominant_failure_mode": mode,
                    "worst_signal_uuid": worst_uuid,
                    "signals_available": int(
                        sum(1 for f in per_signal.values() if start in f.index)
                    ),
                    "signals_expected": len(self.signals),
                    "confidence_level": float(weight_sum / self._total_weight),
                    "sample_count": samples,
                }
            )

        return pd.DataFrame(rows, columns=cols)

    @renamed_kwargs(on_crossing_only="only_crossings")
    def health_events(
        self,
        *,
        window: str = "1h",
        baseline_window: str = "24h",
        warning_threshold: float = 70.0,
        critical_threshold: float = 50.0,
        only_crossings: bool = True,
    ) -> pd.DataFrame:
        """Emit an event when the fused index falls to warning or critical.

        Args:
            window: Evaluation window passed to :meth:`health_index`.
            baseline_window: Baseline period passed to :meth:`health_index`.
            warning_threshold: Score at or below which a ``warning`` fires.
            critical_threshold: Score at or below which a ``critical`` fires.
            only_crossings: When ``True`` (default) only the window that
                *enters* a severity band emits a row, so a machine sitting at
                40 for a week reports one critical, not 168. Set ``False`` to
                emit every window that sits below a threshold.

        Returns:
            Point-shape DataFrame: ``systime``, ``uuid``, ``source_uuid``,
            ``asset_id``, ``asset_health_score``, ``severity``, ``threshold``,
            ``dominant_failure_mode``, ``confidence_level``, ``direction``.

        Raises:
            ValueError: If ``critical_threshold`` exceeds ``warning_threshold``.
        """
        if critical_threshold > warning_threshold:
            raise ValueError(
                "critical_threshold must not be above warning_threshold "
                f"({critical_threshold} > {warning_threshold})"
            )
        cols = [
            "systime",
            "uuid",
            "source_uuid",
            "asset_id",
            "asset_health_score",
            "severity",
            "threshold",
            "dominant_failure_mode",
            "confidence_level",
            "direction",
        ]
        index = self.health_index(window=window, baseline_window=baseline_window)
        if index.empty:
            return pd.DataFrame(columns=cols)

        def _band(score: float) -> str:
            if score <= critical_threshold:
                return "critical"
            if score <= warning_threshold:
                return "warning"
            return "ok"

        bands = index["asset_health_score"].astype(float).map(_band)
        previous = bands.shift(fill_value="ok")

        rows: list[dict[str, Any]] = []
        for pos, (_, row) in enumerate(index.iterrows()):
            band = bands.iloc[pos]
            if band == "ok":
                continue
            if only_crossings and previous.iloc[pos] == band:
                continue
            rows.append(
                {
                    "systime": row["end"],
                    "uuid": self.event_uuid,
                    "source_uuid": self.asset_id,
                    "asset_id": self.asset_id,
                    "asset_health_score": row["asset_health_score"],
                    "severity": band,
                    "threshold": (
                        critical_threshold if band == "critical" else warning_threshold
                    ),
                    "dominant_failure_mode": row["dominant_failure_mode"],
                    "confidence_level": row["confidence_level"],
                    "direction": "below",
                }
            )

        return pd.DataFrame(rows, columns=cols)

    def signal_contributions(
        self,
        *,
        window: str = "1h",
        baseline_window: str = "24h",
    ) -> pd.DataFrame:
        """Rank the configured signals by how much health they cost the asset.

        The contribution of a signal is its mean weighted health deficit
        ``(100 - health) * weight`` over the evaluated windows, expressed as a
        share of the total deficit. A perfectly healthy asset has no deficit
        to distribute and every contribution is ``0``.

        Args:
            window: Evaluation window passed to the per-signal scorer.
            baseline_window: Baseline period passed to the per-signal scorer.

        Returns:
            Static-shape DataFrame: ``asset_id``, ``signal_uuid``,
            ``failure_mode``, ``weight``, ``mean_health``, ``min_health``,
            ``weighted_deficit``, ``contribution_pct``, ``windows``, ``rank``,
            ``method``.
        """
        cols = [
            "asset_id",
            "signal_uuid",
            "failure_mode",
            "weight",
            "mean_health",
            "min_health",
            "weighted_deficit",
            "contribution_pct",
            "windows",
            "rank",
            "method",
        ]
        rows: list[dict[str, Any]] = []
        for uuid, spec in self.signals.items():
            frame = self._signal_health(
                uuid, window=window, baseline_window=baseline_window
            )
            if frame.empty:
                continue
            weight = float(spec["weight"])
            mean_health = float(frame["health"].mean())
            rows.append(
                {
                    "asset_id": self.asset_id,
                    "signal_uuid": uuid,
                    "failure_mode": spec["failure_mode"],
                    "weight": weight,
                    "mean_health": float(mean_health),
                    "min_health": float(frame["health"].min()),
                    "weighted_deficit": float((100.0 - mean_health) * weight),
                    "windows": len(frame),
                    "method": "weighted_deficit",
                }
            )
        if not rows:
            return pd.DataFrame(columns=cols)

        out = pd.DataFrame(rows)
        total = float(out["weighted_deficit"].sum())
        out["contribution_pct"] = (
            out["weighted_deficit"] / total * 100.0 if total > 0 else 0.0
        )
        out = out.sort_values(
            ["weighted_deficit", "signal_uuid"], ascending=[False, True]
        ).reset_index(drop=True)
        out["rank"] = range(1, len(out) + 1)
        return out[cols]

    def maintenance_forecast(
        self,
        *,
        window: str = "1h",
        baseline_window: str = "24h",
        critical_threshold: float = 50.0,
        min_confidence: float = 0.5,
    ) -> pd.DataFrame:
        """Project when the asset index reaches ``critical_threshold``.

        Fits a least-squares line through the fused index and extrapolates it
        to the threshold. ``confidence`` is the fit's R²: a flat but noisy
        index nearly always slopes slightly one way, so a fit below
        ``min_confidence`` reports its slope and confidence but no horizon,
        rather than handing out a maintenance date the data cannot support.
        An index that is flat or improving likewise yields ``NaN`` for the
        horizon columns.

        Args:
            window: Evaluation window passed to :meth:`health_index`.
            baseline_window: Baseline period passed to :meth:`health_index`.
            critical_threshold: Score treated as "intervention required".
            min_confidence: Minimum R² of the trend fit before a horizon is
                projected. Pass ``0.0`` to always project.

        Returns:
            Summary-shape DataFrame with a single row: ``start``, ``end``,
            ``duration_seconds``, ``uuid``, ``source_uuid``, ``asset_id``,
            ``current_health``, ``decline_per_day``,
            ``recommended_maintenance_days``, ``forecast_time``,
            ``critical_threshold``, ``confidence``, ``sample_count``,
            ``method``.
        """
        cols = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "asset_id",
            "current_health",
            "decline_per_day",
            "recommended_maintenance_days",
            "forecast_time",
            "critical_threshold",
            "confidence",
            "sample_count",
            "method",
        ]
        index = self.health_index(window=window, baseline_window=baseline_window)
        if len(index) < 2:
            return pd.DataFrame(columns=cols)

        times = pd.to_datetime(index["end"])
        days = (times - times.iloc[0]).dt.total_seconds().to_numpy(
            dtype=float
        ) / 86400.0
        scores = index["asset_health_score"].to_numpy(dtype=float)

        slope, intercept = np.polyfit(days, scores, 1)
        fitted = slope * days + intercept
        ss_res = float(np.sum((scores - fitted) ** 2))
        ss_tot = float(np.sum((scores - scores.mean()) ** 2))
        confidence = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        current = float(scores[-1])
        decline_per_day = -float(slope)
        confidence = max(0.0, min(1.0, confidence))
        if current <= critical_threshold:
            remaining_days = 0.0
            forecast_time = times.iloc[-1]
        elif decline_per_day > 0 and confidence >= min_confidence:
            remaining_days = (current - critical_threshold) / decline_per_day
            forecast_time = times.iloc[-1] + pd.to_timedelta(remaining_days, unit="D")
        else:
            remaining_days = float("nan")
            forecast_time = pd.NaT

        out = pd.DataFrame(
            [
                {
                    "start": index["start"].iloc[0],
                    "end": index["end"].iloc[-1],
                    "duration_seconds": (
                        index["end"].iloc[-1] - index["start"].iloc[0]
                    ).total_seconds(),
                    "uuid": self.event_uuid,
                    "source_uuid": self.asset_id,
                    "asset_id": self.asset_id,
                    "current_health": float(current),
                    "decline_per_day": float(decline_per_day),
                    "recommended_maintenance_days": (
                        float(remaining_days)
                        if pd.notna(remaining_days)
                        else float("nan")
                    ),
                    "forecast_time": forecast_time,
                    "critical_threshold": critical_threshold,
                    "confidence": float(confidence),
                    "sample_count": int(index["sample_count"].sum()),
                    "method": "linear_health_extrapolation",
                }
            ],
            columns=cols,
        )
        # Keep the input's datetime dtype (and timezone) even when there is
        # no forecast (NaT).
        out["forecast_time"] = pd.Series([forecast_time], dtype=times.dtype)
        return out
