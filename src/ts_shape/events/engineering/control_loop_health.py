import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class ControlLoopHealthEvents(Base):
    """Engineering: Control Loop Health

    Continuously assess PID/control loop health from setpoint + actual pairs,
    independent of setpoint changes. Computes error integrals, detects
    oscillation in the error signal, and checks for valve saturation.

    Methods:
    - error_integrals: Per-window IAE, ISE, ITAE, bias.
    - detect_oscillation: Sustained oscillation in the error signal.
    - output_saturation: Valve pegged at limits.
    - loop_health_summary: Shift-level report card.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        setpoint_uuid: str,
        actual_uuid: str,
        *,
        output_uuid: Optional[str] = None,
        event_uuid: str = "eng:control_loop_health",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.setpoint_uuid = setpoint_uuid
        self.actual_uuid = actual_uuid
        self.output_uuid = output_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self._sp = self._load_signal(setpoint_uuid)
        self._pv = self._load_signal(actual_uuid)
        self._out = self._load_signal(output_uuid) if output_uuid else pd.DataFrame()
        self._aligned = self._align()

    def _load_signal(self, uuid: str) -> pd.DataFrame:
        sig = (
            self.dataframe[self.dataframe["uuid"] == uuid]
            .copy()
            .sort_values(self.time_column)
        )
        sig[self.time_column] = pd.to_datetime(sig[self.time_column])
        return sig

    def _align(self) -> pd.DataFrame:
        """Align SP and PV by nearest timestamp."""
        if self._sp.empty or self._pv.empty:
            return pd.DataFrame(columns=[self.time_column, "sp", "pv", "error"])

        sp = self._sp[[self.time_column, self.value_column]].rename(
            columns={self.value_column: "sp"}
        )
        pv = self._pv[[self.time_column, self.value_column]].rename(
            columns={self.value_column: "pv"}
        )
        merged = pd.merge_asof(
            sp.sort_values(self.time_column),
            pv.sort_values(self.time_column),
            on=self.time_column,
            direction="nearest",
        )
        merged["error"] = merged["pv"] - merged["sp"]
        return merged.dropna(subset=["sp", "pv"])

    def error_integrals(self, window: str = "1h") -> pd.DataFrame:
        """Per-window error integrals for control loop performance.

        Args:
            window: Resample window (e.g. '1h', '8h').

        Returns:
            DataFrame with columns: window_start, iae, ise, itae, bias,
            sample_count.
        """
        cols = ["window_start", "iae", "ise", "itae", "bias", "sample_count"]
        if self._aligned.empty:
            return pd.DataFrame(columns=cols)

        df = self._aligned.set_index(self.time_column)
        groups = df.resample(window)

        events: List[Dict[str, Any]] = []
        for window_start, group in groups:
            if len(group) < 2:
                continue
            error = group["error"]
            dt = group.index.to_series().diff().dt.total_seconds().fillna(0)
            t_rel = (group.index - group.index[0]).total_seconds()

            iae = float((error.abs() * dt).sum())
            ise = float((error**2 * dt).sum())
            itae = float((t_rel * error.abs() * dt).sum())
            bias = float(error.mean())

            events.append(
                {
                    "window_start": window_start,
                    "iae": iae,
                    "ise": ise,
                    "itae": itae,
                    "bias": bias,
                    "sample_count": len(group),
                }
            )

        return pd.DataFrame(events, columns=cols)

    def detect_oscillation(
        self,
        window: str = "5min",
        min_crossings: int = 4,
    ) -> pd.DataFrame:
        """Detect sustained oscillation in the error signal.

        Args:
            window: Analysis window size.
            min_crossings: Minimum zero-crossings to flag oscillation.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta,
            crossing_count, estimated_period_seconds, amplitude,
            damping_direction.
        """
        cols = [
            "start",
            "end",
            "uuid",
            "is_delta",
            "crossing_count",
            "estimated_period_seconds",
            "amplitude",
            "damping_direction",
        ]
        if self._aligned.empty or len(self._aligned) < 4:
            return pd.DataFrame(columns=cols)

        df = self._aligned.set_index(self.time_column)
        td = pd.Timedelta(window)
        groups = df.resample(window)

        osc_windows: List[Dict[str, Any]] = []
        for window_start, group in groups:
            if len(group) < 4:
                continue

            error = group["error"].values
            signs = np.sign(error)
            crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

            if crossings < min_crossings:
                continue

            # Estimate period from crossing intervals
            cross_idx = np.where(np.abs(np.diff(signs)) > 0)[0]
            times = group.index
            if len(cross_idx) >= 2:
                cross_times = times[cross_idx]
                intervals = (
                    np.diff(cross_times).astype("timedelta64[ms]").astype(float) / 1000
                )
                # Full period = 2 × half-period (crossing to crossing)
                est_period = float(np.mean(intervals)) * 2
            else:
                est_period = 0.0

            amplitude = float(np.max(np.abs(error)))

            # Damping: compare amplitude in first half vs second half
            mid = len(error) // 2
            amp_first = float(np.max(np.abs(error[:mid]))) if mid > 0 else 0.0
            amp_second = float(np.max(np.abs(error[mid:]))) if mid > 0 else 0.0
            if amp_second > amp_first * 1.1:
                damping = "growing"
            elif amp_second < amp_first * 0.9:
                damping = "decaying"
            else:
                damping = "sustained"

            osc_windows.append(
                {
                    "start": window_start,
                    "end": window_start + td,
                    "uuid": self.event_uuid,
                    "is_delta": False,
                    "crossing_count": crossings,
                    "estimated_period_seconds": est_period,
                    "amplitude": amplitude,
                    "damping_direction": damping,
                }
            )

        if not osc_windows:
            return pd.DataFrame(columns=cols)

        # Merge contiguous oscillating windows
        events: List[Dict[str, Any]] = []
        current = osc_windows[0].copy()
        for i in range(1, len(osc_windows)):
            w = osc_windows[i]
            if w["start"] <= current["end"]:
                current["end"] = w["end"]
                current["crossing_count"] += w["crossing_count"]
                current["amplitude"] = max(current["amplitude"], w["amplitude"])
                current["estimated_period_seconds"] = (
                    current["estimated_period_seconds"] + w["estimated_period_seconds"]
                ) / 2
                current["damping_direction"] = w["damping_direction"]
            else:
                events.append(current)
                current = w.copy()
        events.append(current)

        return pd.DataFrame(events, columns=cols)

    def output_saturation(
        self,
        high_limit: float = 100.0,
        low_limit: float = 0.0,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Detect when controller output is pegged at limits.

        Requires output_uuid in constructor.

        Returns:
            DataFrame with columns: window_start, pct_time_at_high,
            pct_time_at_low, pct_time_saturated, longest_saturation_seconds.
        """
        cols = [
            "window_start",
            "pct_time_at_high",
            "pct_time_at_low",
            "pct_time_saturated",
            "longest_saturation_seconds",
        ]
        if self._out.empty:
            return pd.DataFrame(columns=cols)

        out = self._out.set_index(self.time_column)[self.value_column]
        groups = out.resample(window)
        tol = (high_limit - low_limit) * 0.01  # 1% of range

        events: List[Dict[str, Any]] = []
        for window_start, group in groups:
            if group.empty:
                continue
            n = len(group)
            at_high = (group >= high_limit - tol).sum()
            at_low = (group <= low_limit + tol).sum()

            # Longest saturation run
            saturated = (group >= high_limit - tol) | (group <= low_limit + tol)
            if saturated.any():
                runs = (saturated != saturated.shift()).cumsum()
                sat_runs = saturated.groupby(runs).apply(
                    lambda g: (
                        (g.index[-1] - g.index[0]).total_seconds() if g.iloc[0] else 0
                    )
                )
                longest = float(sat_runs.max())
            else:
                longest = 0.0

            events.append(
                {
                    "window_start": window_start,
                    "pct_time_at_high": float(at_high / n * 100),
                    "pct_time_at_low": float(at_low / n * 100),
                    "pct_time_saturated": float((at_high + at_low) / n * 100),
                    "longest_saturation_seconds": longest,
                }
            )

        return pd.DataFrame(events, columns=cols)

    def loop_health_summary(self, window: str = "8h") -> pd.DataFrame:
        """Shift-level report card combining all loop health metrics.

        Returns:
            DataFrame with columns: window_start, iae, bias,
            oscillation_count, pct_saturated, health_grade.
        """
        cols = [
            "window_start",
            "iae",
            "bias",
            "oscillation_count",
            "pct_saturated",
            "health_grade",
        ]
        integrals = self.error_integrals(window)
        if integrals.empty:
            return pd.DataFrame(columns=cols)

        osc = self.detect_oscillation()
        sat = self.output_saturation(window=window)

        events: List[Dict[str, Any]] = []
        for _, row in integrals.iterrows():
            ws = row["window_start"]
            we = ws + pd.Timedelta(window)

            # Count oscillation events in this window
            if not osc.empty:
                osc_in = osc[(osc["start"] >= ws) & (osc["start"] < we)]
                osc_count = len(osc_in)
            else:
                osc_count = 0

            # Saturation in this window
            if not sat.empty:
                sat_row = sat[sat["window_start"] == ws]
                pct_sat = (
                    float(sat_row["pct_time_saturated"].iloc[0])
                    if not sat_row.empty
                    else 0.0
                )
            else:
                pct_sat = 0.0

            # Grade: count issues
            iae_median = (
                float(integrals["iae"].median()) if len(integrals) > 1 else row["iae"]
            )
            issues = 0
            if row["iae"] > iae_median * 1.5:
                issues += 1
            if osc_count > 0:
                issues += 1
            if pct_sat > 5.0:
                issues += 1

            grade = "good" if issues == 0 else ("fair" if issues == 1 else "poor")

            events.append(
                {
                    "window_start": ws,
                    "iae": row["iae"],
                    "bias": row["bias"],
                    "oscillation_count": osc_count,
                    "pct_saturated": pct_sat,
                    "health_grade": grade,
                }
            )

        return pd.DataFrame(events, columns=cols)
