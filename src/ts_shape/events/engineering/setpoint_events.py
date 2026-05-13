import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import Optional, List, Dict, Any, Tuple

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class SetpointChangeEvents(Base):
    """
    Detect step/ramp changes on a setpoint signal and compute follow-up KPIs
    like time-to-settle and overshoot based on an actual (process) value.

    Schema assumptions (columns):
    - uuid, sequence_number, systime, plctime, is_delta
    - value_integer, value_string, value_double, value_bool, value_bytes
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        setpoint_uuid: str,
        *,
        event_uuid: str = "setpoint_change_event",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.setpoint_uuid = setpoint_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        # isolate setpoint series and ensure proper dtypes/sort
        self.sp = (
            self.dataframe[self.dataframe["uuid"] == self.setpoint_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.sp[self.time_column] = pd.to_datetime(self.sp[self.time_column])

        # Cache for performance optimization
        self._actual_cache: Dict[str, pd.DataFrame] = {}

    def _get_actual(self, actual_uuid: str) -> pd.DataFrame:
        """
        Get and cache actual signal data for performance optimization.
        """
        if actual_uuid not in self._actual_cache:
            actual = (
                self.dataframe[self.dataframe["uuid"] == actual_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            actual[self.time_column] = pd.to_datetime(actual[self.time_column])
            self._actual_cache[actual_uuid] = actual
        return self._actual_cache[actual_uuid]

    # ---- Change detection ----
    def detect_setpoint_steps(
        self,
        min_delta: float,
        min_hold: str = "0s",
        filter_noise: bool = False,
        noise_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Point events at times where the setpoint changes by >= min_delta and the
        new level holds for at least `min_hold` (no subsequent change within that time).

        Args:
            min_delta: Minimum magnitude of change to detect
            min_hold: Minimum duration the new level must hold
            filter_noise: If True, filter out changes smaller than noise_threshold
            noise_threshold: Threshold for noise filtering (absolute value)

        Returns:
            DataFrame with columns: start, end (== start), uuid, is_delta,
            change_type='step', magnitude, prev_level, new_level.
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "change_type",
                    "magnitude",
                    "prev_level",
                    "new_level",
                ]
            )

        sp = self.sp[[self.time_column, self.value_column]].copy()

        # Apply noise filtering if requested
        if filter_noise:
            # Group consecutive values within noise_threshold and use mean
            vals = sp[self.value_column].values.copy()
            current_group = vals[0]
            for i in range(1, len(vals)):
                if abs(vals[i] - current_group) <= noise_threshold:
                    vals[i] = current_group
                else:
                    current_group = vals[i]
            sp["filtered_value"] = vals
            sp[self.value_column] = sp["filtered_value"]
            sp = sp.drop(columns=["filtered_value"])

        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        change_mask = sp["delta"].abs() >= float(min_delta)

        # hold condition: next change must be after min_hold
        change_times = sp.loc[change_mask, self.time_column]
        min_hold_td = pd.to_timedelta(min_hold)
        next_change_times = change_times.shift(-1)
        hold_ok = (
            next_change_times - change_times >= min_hold_td
        ) | next_change_times.isna()
        valid_change_times = change_times[hold_ok]

        rows: List[Dict[str, Any]] = []
        for t in valid_change_times:
            row = sp.loc[sp[self.time_column] == t].iloc[0]
            rows.append(
                {
                    "start": t,
                    "end": t,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "change_type": "step",
                    "magnitude": float(row["delta"]),
                    "prev_level": float(row["prev"]) if pd.notna(row["prev"]) else None,
                    "new_level": float(row[self.value_column]),
                }
            )

        return pd.DataFrame(rows)

    def detect_setpoint_ramps(
        self, min_rate: float, min_duration: str = "0s"
    ) -> pd.DataFrame:
        """
        Interval events where |dS/dt| >= min_rate for at least `min_duration`.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta, change_type='ramp',
            avg_rate, delta.
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "change_type",
                    "avg_rate",
                    "delta",
                ]
            )

        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["dt_s"] = sp[self.time_column].diff().dt.total_seconds()
        sp["dv"] = sp[self.value_column].diff()
        sp["rate"] = sp["dv"] / sp["dt_s"]
        rate_mask = sp["rate"].abs() >= float(min_rate)

        # group contiguous True segments
        group_id = (rate_mask != rate_mask.shift()).cumsum()
        events: List[Dict[str, Any]] = []
        min_d = pd.to_timedelta(min_duration)
        for gid, seg in sp.groupby(group_id):
            seg_mask_true = rate_mask.loc[seg.index]
            if not seg_mask_true.any():
                continue
            # boundaries
            start_time = seg.loc[seg_mask_true, self.time_column].iloc[0]
            end_time = seg.loc[seg_mask_true, self.time_column].iloc[-1]
            if (end_time - start_time) < min_d:
                continue
            avg_rate = seg.loc[seg_mask_true, "rate"].mean()
            delta = seg.loc[seg_mask_true, "dv"].sum()
            events.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "change_type": "ramp",
                    "avg_rate": float(avg_rate) if pd.notna(avg_rate) else None,
                    "delta": float(delta) if pd.notna(delta) else None,
                }
            )

        return pd.DataFrame(events)

    def detect_setpoint_changes(
        self,
        *,
        min_delta: float = 0.0,
        min_rate: Optional[float] = None,
        min_hold: str = "0s",
        min_duration: str = "0s",
    ) -> pd.DataFrame:
        """
        Unified setpoint change table (steps + ramps) with standardized columns.
        """
        steps = self.detect_setpoint_steps(min_delta=min_delta, min_hold=min_hold)
        ramps = (
            self.detect_setpoint_ramps(min_rate=min_rate, min_duration=min_duration)
            if min_rate is not None
            else pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "change_type",
                    "avg_rate",
                    "delta",
                ]
            )
        )
        # ensure uniform columns
        if not steps.empty:
            steps = steps.assign(avg_rate=None, delta=None)[
                [
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "change_type",
                    "magnitude",
                    "prev_level",
                    "new_level",
                    "avg_rate",
                    "delta",
                ]
            ]
        if not ramps.empty:
            ramps = ramps.assign(magnitude=None, prev_level=None, new_level=None)[
                [
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "change_type",
                    "magnitude",
                    "prev_level",
                    "new_level",
                    "avg_rate",
                    "delta",
                ]
            ]
        frames = [df for df in (steps, ramps) if not df.empty]
        combined = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "change_type",
                    "magnitude",
                    "prev_level",
                    "new_level",
                    "avg_rate",
                    "delta",
                ]
            )
        )
        return (
            combined.sort_values(["start", "end"]) if not combined.empty else combined
        )

    # ---- Follow-up KPIs ----
    def time_to_settle(
        self,
        actual_uuid: str,
        *,
        tol: float = 0.0,
        settle_pct: Optional[float] = None,
        hold: str = "0s",
        lookahead: str = "10m",
    ) -> pd.DataFrame:
        """
        For each setpoint change (any change), compute time until the actual signal
        is within ±`tol` of the new setpoint for a continuous duration of `hold`.

        Args:
            actual_uuid: UUID of the actual/process value signal
            tol: Absolute tolerance (used if settle_pct is None)
            settle_pct: Percentage-based tolerance (e.g., 0.02 for 2% of step magnitude)
            hold: Minimum duration the signal must stay within tolerance
            lookahead: Maximum time window to search for settling

        Returns:
            DataFrame with columns: start, uuid, is_delta, t_settle_seconds, settled.
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=["start", "uuid", "is_delta", "t_settle_seconds", "settled"]
            )

        # Use cached actual data
        actual = self._get_actual(actual_uuid)
        hold_td = pd.to_timedelta(hold)
        look_td = pd.to_timedelta(lookahead)

        # change instants
        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        change_times = sp.loc[
            sp["delta"].abs() > 0, [self.time_column, self.value_column, "delta"]
        ].reset_index(drop=True)

        rows: List[Dict[str, Any]] = []
        for _, c in change_times.iterrows():
            t0 = c[self.time_column]
            s_new = float(c[self.value_column])
            delta = float(c["delta"]) if pd.notna(c["delta"]) else 0.0

            # Calculate tolerance based on settle_pct or use absolute tol
            if settle_pct is not None and delta != 0:
                effective_tol = abs(delta) * settle_pct
            else:
                effective_tol = tol

            window = actual[
                (actual[self.time_column] >= t0)
                & (actual[self.time_column] <= t0 + look_td)
            ]
            if window.empty:
                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "t_settle_seconds": None,
                        "settled": False,
                    }
                )
                continue
            err = (window[self.value_column] - s_new).abs()
            inside = err <= effective_tol

            # time to first entry within tolerance (ignores hold)
            if inside.any():
                first_idx = inside[inside].index[0]
                t_enter = window.loc[first_idx, self.time_column]
            else:
                t_enter = None

            # determine if any contiguous inside segment satisfies hold duration
            settled = False
            if inside.any():
                gid = (inside.ne(inside.shift())).cumsum()
                for _, seg in window.groupby(gid):
                    seg_inside = inside.loc[seg.index]
                    if not seg_inside.iloc[0]:
                        continue
                    start_seg = seg[self.time_column].iloc[0]
                    end_seg = seg[self.time_column].iloc[-1]
                    if (end_seg - start_seg) >= hold_td:
                        settled = True
                        break

            rows.append(
                {
                    "start": t0,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "t_settle_seconds": (
                        (t_enter - t0).total_seconds() if t_enter is not None else None
                    ),
                    "settled": bool(settled),
                }
            )

        return pd.DataFrame(rows)

    def overshoot_metrics(
        self,
        actual_uuid: str,
        *,
        window: str = "10m",
    ) -> pd.DataFrame:
        """
        For each change, compute peak overshoot, undershoot, and oscillation metrics
        relative to the new setpoint within a lookahead window.

        Returns:
            DataFrame with columns: start, uuid, is_delta, overshoot_abs, overshoot_pct,
            t_peak_seconds, undershoot_abs, undershoot_pct, t_undershoot_seconds,
            oscillation_count, oscillation_amplitude.
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "is_delta",
                    "overshoot_abs",
                    "overshoot_pct",
                    "t_peak_seconds",
                    "undershoot_abs",
                    "undershoot_pct",
                    "t_undershoot_seconds",
                    "oscillation_count",
                    "oscillation_amplitude",
                ]
            )

        # Use cached actual data
        actual = self._get_actual(actual_uuid)
        look_td = pd.to_timedelta(window)

        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        changes = sp.loc[
            sp["delta"].abs() > 0,
            [self.time_column, self.value_column, "delta", "prev"],
        ]

        out_rows: List[Dict[str, Any]] = []
        for _, r in changes.iterrows():
            t0 = r[self.time_column]
            s_new = float(r[self.value_column])
            s_prev = float(r["prev"]) if pd.notna(r["prev"]) else s_new
            delta = float(r["delta"]) if pd.notna(r["delta"]) else 0.0
            win = actual[
                (actual[self.time_column] >= t0)
                & (actual[self.time_column] <= t0 + look_td)
            ]

            if win.empty:
                out_rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "overshoot_abs": None,
                        "overshoot_pct": None,
                        "t_peak_seconds": None,
                        "undershoot_abs": None,
                        "undershoot_pct": None,
                        "t_undershoot_seconds": None,
                        "oscillation_count": None,
                        "oscillation_amplitude": None,
                    }
                )
                continue

            err = win[self.value_column] - s_new

            # Overshoot (in direction of change)
            if delta >= 0:
                peak = err.max()
                t_peak = win.loc[err.idxmax(), self.time_column] if peak > 0 else None
                # Undershoot (opposite direction)
                undershoot_val = err.min()
                t_undershoot = (
                    win.loc[err.idxmin(), self.time_column]
                    if undershoot_val < 0
                    else None
                )
            else:
                peak = -err.min()  # magnitude for downward step
                t_peak = (
                    win.loc[err.idxmin(), self.time_column] if err.min() < 0 else None
                )
                # Undershoot (opposite direction)
                undershoot_val = -err.max()
                t_undershoot = (
                    win.loc[err.idxmax(), self.time_column] if err.max() > 0 else None
                )

            overshoot_abs = float(peak) if pd.notna(peak) and peak > 0 else 0.0
            overshoot_pct = (
                (overshoot_abs / abs(delta))
                if (delta != 0 and overshoot_abs is not None)
                else None
            )

            undershoot_abs = (
                float(abs(undershoot_val))
                if pd.notna(undershoot_val) and abs(undershoot_val) > 0
                else 0.0
            )
            undershoot_pct = (
                (undershoot_abs / abs(delta))
                if (delta != 0 and undershoot_abs is not None)
                else None
            )

            # Oscillation detection: count zero crossings
            err_sign = np.sign(err)
            sign_changes = (
                err_sign.diff() != 0
            ).sum() - 1  # -1 to exclude initial transition
            oscillation_count = max(0, int(sign_changes))

            # Oscillation amplitude: average of peak deviations
            oscillation_amplitude = float(err.abs().mean()) if len(err) > 0 else None

            out_rows.append(
                {
                    "start": t0,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "overshoot_abs": overshoot_abs if overshoot_abs > 0 else None,
                    "overshoot_pct": (
                        float(overshoot_pct)
                        if overshoot_pct is not None and overshoot_abs > 0
                        else None
                    ),
                    "t_peak_seconds": (
                        (t_peak - t0).total_seconds() if t_peak is not None else None
                    ),
                    "undershoot_abs": undershoot_abs if undershoot_abs > 0 else None,
                    "undershoot_pct": (
                        float(undershoot_pct)
                        if undershoot_pct is not None and undershoot_abs > 0
                        else None
                    ),
                    "t_undershoot_seconds": (
                        (t_undershoot - t0).total_seconds()
                        if t_undershoot is not None
                        else None
                    ),
                    "oscillation_count": (
                        oscillation_count if oscillation_count > 0 else 0
                    ),
                    "oscillation_amplitude": oscillation_amplitude,
                }
            )

        return pd.DataFrame(out_rows)

    def time_to_settle_derivative(
        self,
        actual_uuid: str,
        *,
        rate_threshold: float = 0.01,
        lookahead: str = "10m",
        hold: str = "0s",
    ) -> pd.DataFrame:
        """
        Detect settling based on rate of change (derivative) falling below threshold.
        More sensitive to when the process has truly stopped moving.

        Args:
            actual_uuid: UUID of the actual/process value signal
            rate_threshold: Maximum absolute rate of change to consider settled
            lookahead: Maximum time window to search for settling
            hold: Minimum duration the rate must stay below threshold

        Returns:
            DataFrame with columns: start, uuid, is_delta, t_settle_seconds, settled, final_rate.
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "is_delta",
                    "t_settle_seconds",
                    "settled",
                    "final_rate",
                ]
            )

        # Use cached actual data
        actual = self._get_actual(actual_uuid)
        look_td = pd.to_timedelta(lookahead)
        hold_td = pd.to_timedelta(hold)

        # change instants
        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        change_times = sp.loc[
            sp["delta"].abs() > 0, [self.time_column, self.value_column]
        ].reset_index(drop=True)

        rows: List[Dict[str, Any]] = []
        for _, c in change_times.iterrows():
            t0 = c[self.time_column]
            window = actual[
                (actual[self.time_column] >= t0)
                & (actual[self.time_column] <= t0 + look_td)
            ]

            if window.empty or len(window) < 2:
                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "t_settle_seconds": None,
                        "settled": False,
                        "final_rate": None,
                    }
                )
                continue

            # Calculate rate of change
            win_copy = window.copy()
            win_copy["dt_s"] = win_copy[self.time_column].diff().dt.total_seconds()
            win_copy["dv"] = win_copy[self.value_column].diff()
            win_copy["rate"] = (win_copy["dv"] / win_copy["dt_s"]).abs()

            # Find when rate drops below threshold
            below_threshold = win_copy["rate"] <= rate_threshold

            t_settle = None
            settled = False
            final_rate = None

            if below_threshold.any():
                # Find first entry below threshold
                first_idx = below_threshold[below_threshold].index[0]
                t_first = win_copy.loc[first_idx, self.time_column]

                # Check if it stays below threshold for hold duration
                if hold_td.total_seconds() > 0:
                    gid = (below_threshold.ne(below_threshold.shift())).cumsum()
                    for _, seg in win_copy.groupby(gid):
                        seg_below = below_threshold.loc[seg.index]
                        if not seg_below.iloc[0]:
                            continue
                        start_seg = seg[self.time_column].iloc[0]
                        end_seg = seg[self.time_column].iloc[-1]
                        if (end_seg - start_seg) >= hold_td:
                            t_settle = start_seg
                            settled = True
                            final_rate = float(seg["rate"].mean())
                            break
                else:
                    t_settle = t_first
                    settled = True
                    final_rate = float(win_copy.loc[first_idx, "rate"])

            rows.append(
                {
                    "start": t0,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "t_settle_seconds": (
                        (t_settle - t0).total_seconds()
                        if t_settle is not None
                        else None
                    ),
                    "settled": settled,
                    "final_rate": final_rate,
                }
            )

        return pd.DataFrame(rows)

    def rise_time(
        self,
        actual_uuid: str,
        *,
        start_pct: float = 0.1,
        end_pct: float = 0.9,
        lookahead: str = "10m",
    ) -> pd.DataFrame:
        """
        Compute rise time: time for actual to go from start_pct to end_pct of the setpoint change.
        Typically measured from 10% to 90% of the final value.

        Args:
            actual_uuid: UUID of the actual/process value signal
            start_pct: Starting percentage of change (e.g., 0.1 for 10%)
            end_pct: Ending percentage of change (e.g., 0.9 for 90%)
            lookahead: Maximum time window to search

        Returns:
            DataFrame with columns: start, uuid, is_delta, rise_time_seconds, reached_end.
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "is_delta",
                    "rise_time_seconds",
                    "reached_end",
                ]
            )

        # Use cached actual data
        actual = self._get_actual(actual_uuid)
        look_td = pd.to_timedelta(lookahead)

        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        changes = sp.loc[
            sp["delta"].abs() > 0,
            [self.time_column, self.value_column, "delta", "prev"],
        ].reset_index(drop=True)

        rows: List[Dict[str, Any]] = []
        for _, c in changes.iterrows():
            t0 = c[self.time_column]
            s_new = float(c[self.value_column])
            s_prev = float(c["prev"]) if pd.notna(c["prev"]) else s_new
            delta = float(c["delta"]) if pd.notna(c["delta"]) else 0.0

            window = actual[
                (actual[self.time_column] >= t0)
                & (actual[self.time_column] <= t0 + look_td)
            ]

            if window.empty or delta == 0:
                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "rise_time_seconds": None,
                        "reached_end": False,
                    }
                )
                continue

            # Calculate target levels
            start_level = s_prev + delta * start_pct
            end_level = s_prev + delta * end_pct

            # Find crossing times
            t_start = None
            t_end = None

            if delta > 0:
                # Upward step
                start_crossed = window[self.value_column] >= start_level
                end_crossed = window[self.value_column] >= end_level
            else:
                # Downward step
                start_crossed = window[self.value_column] <= start_level
                end_crossed = window[self.value_column] <= end_level

            if start_crossed.any():
                t_start = window.loc[start_crossed.idxmax(), self.time_column]
            if end_crossed.any():
                t_end = window.loc[end_crossed.idxmax(), self.time_column]

            if t_start is not None and t_end is not None:
                rise_time_sec = (t_end - t_start).total_seconds()
                reached_end = True
            else:
                rise_time_sec = None
                reached_end = False

            rows.append(
                {
                    "start": t0,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "rise_time_seconds": rise_time_sec,
                    "reached_end": reached_end,
                }
            )

        return pd.DataFrame(rows)

    def decay_rate(
        self,
        actual_uuid: str,
        *,
        lookahead: str = "10m",
        min_points: int = 5,
    ) -> pd.DataFrame:
        """
        Estimate exponential decay rate of the settling behavior.
        Fits error(t) = A * exp(-lambda * t) and returns lambda.

        Args:
            actual_uuid: UUID of the actual/process value signal
            lookahead: Time window for analysis
            min_points: Minimum number of points required for fitting

        Returns:
            DataFrame with columns: start, uuid, is_delta, decay_rate_lambda, fit_quality_r2.
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "is_delta",
                    "decay_rate_lambda",
                    "fit_quality_r2",
                ]
            )

        # Use cached actual data
        actual = self._get_actual(actual_uuid)
        look_td = pd.to_timedelta(lookahead)

        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        changes = sp.loc[
            sp["delta"].abs() > 0, [self.time_column, self.value_column]
        ].reset_index(drop=True)

        rows: List[Dict[str, Any]] = []
        for _, c in changes.iterrows():
            t0 = c[self.time_column]
            s_new = float(c[self.value_column])
            window = actual[
                (actual[self.time_column] >= t0)
                & (actual[self.time_column] <= t0 + look_td)
            ]

            if window.empty or len(window) < min_points:
                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "decay_rate_lambda": None,
                        "fit_quality_r2": None,
                    }
                )
                continue

            # Calculate error and time since change
            err = (window[self.value_column] - s_new).abs()
            t_sec = (window[self.time_column] - t0).dt.total_seconds()

            # Filter out zero or near-zero errors for log fit
            valid_mask = err > 1e-6
            if valid_mask.sum() < min_points:
                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "decay_rate_lambda": None,
                        "fit_quality_r2": None,
                    }
                )
                continue

            err_valid = err[valid_mask]
            t_valid = t_sec[valid_mask]

            try:
                # Linear fit to log(error) vs time: log(err) = log(A) - lambda*t
                log_err = np.log(err_valid)
                coeffs = np.polyfit(t_valid, log_err, 1)
                decay_lambda = -coeffs[0]  # negative slope

                # Calculate R^2
                log_err_pred = np.polyval(coeffs, t_valid)
                ss_res = np.sum((log_err - log_err_pred) ** 2)
                ss_tot = np.sum((log_err - log_err.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "decay_rate_lambda": (
                            float(decay_lambda) if decay_lambda > 0 else None
                        ),
                        "fit_quality_r2": float(r2),
                    }
                )
            except Exception:
                logger.debug("Exponential decay fit failed for window at %s", t0)
                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "decay_rate_lambda": None,
                        "fit_quality_r2": None,
                    }
                )

        return pd.DataFrame(rows)

    def oscillation_frequency(
        self,
        actual_uuid: str,
        *,
        window: str = "10m",
        min_oscillations: int = 2,
    ) -> pd.DataFrame:
        """
        Estimate the frequency of oscillations during settling.
        Counts zero crossings and estimates period.

        Args:
            actual_uuid: UUID of the actual/process value signal
            window: Time window for analysis
            min_oscillations: Minimum number of oscillations to compute frequency

        Returns:
            DataFrame with columns: start, uuid, is_delta, oscillation_freq_hz, period_seconds.
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "is_delta",
                    "oscillation_freq_hz",
                    "period_seconds",
                ]
            )

        # Use cached actual data
        actual = self._get_actual(actual_uuid)
        look_td = pd.to_timedelta(window)

        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        changes = sp.loc[
            sp["delta"].abs() > 0, [self.time_column, self.value_column]
        ].reset_index(drop=True)

        rows: List[Dict[str, Any]] = []
        for _, c in changes.iterrows():
            t0 = c[self.time_column]
            s_new = float(c[self.value_column])
            win = actual[
                (actual[self.time_column] >= t0)
                & (actual[self.time_column] <= t0 + look_td)
            ]

            if win.empty:
                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "oscillation_freq_hz": None,
                        "period_seconds": None,
                    }
                )
                continue

            # Calculate error
            err = win[self.value_column] - s_new

            # Detect zero crossings
            err_sign = np.sign(err)
            sign_changes = np.where(np.diff(err_sign) != 0)[0]
            num_crossings = len(sign_changes)

            if (
                num_crossings >= min_oscillations * 2
            ):  # each oscillation has 2 crossings
                # Calculate time between crossings
                crossing_times = win[self.time_column].iloc[sign_changes]
                time_diffs = crossing_times.diff().dt.total_seconds().dropna()

                if len(time_diffs) > 0:
                    # Average period (2 crossings = 1 period)
                    avg_half_period = time_diffs.mean()
                    period = avg_half_period * 2
                    freq = 1.0 / period if period > 0 else None

                    rows.append(
                        {
                            "start": t0,
                            "uuid": self.event_uuid,
                            "is_delta": True,
                            "oscillation_freq_hz": (
                                float(freq) if freq is not None else None
                            ),
                            "period_seconds": float(period),
                        }
                    )
                else:
                    rows.append(
                        {
                            "start": t0,
                            "uuid": self.event_uuid,
                            "is_delta": True,
                            "oscillation_freq_hz": None,
                            "period_seconds": None,
                        }
                    )
            else:
                rows.append(
                    {
                        "start": t0,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "oscillation_freq_hz": None,
                        "period_seconds": None,
                    }
                )

        return pd.DataFrame(rows)

    def control_quality_metrics(
        self,
        actual_uuid: str,
        *,
        tol: float = 0.0,
        settle_pct: Optional[float] = None,
        hold: str = "0s",
        lookahead: str = "10m",
        rate_threshold: float = 0.01,
    ) -> pd.DataFrame:
        """
        Comprehensive control quality metrics combining multiple performance indicators.

        Computes all available metrics for each setpoint change and returns them in a single DataFrame.
        This includes: settling time, rise time, overshoot, undershoot, oscillations, and decay characteristics.

        Args:
            actual_uuid: UUID of the actual/process value signal
            tol: Absolute tolerance for settling (used if settle_pct is None)
            settle_pct: Percentage-based tolerance for settling
            hold: Minimum duration to confirm settling
            lookahead: Time window for all analyses
            rate_threshold: Rate threshold for derivative-based settling

        Returns:
            DataFrame with comprehensive metrics including:
            - start, uuid, is_delta
            - t_settle_seconds, settled (from time_to_settle)
            - t_settle_derivative_seconds (from time_to_settle_derivative)
            - rise_time_seconds (from rise_time)
            - overshoot_abs, overshoot_pct (from overshoot_metrics)
            - undershoot_abs, undershoot_pct
            - oscillation_count, oscillation_amplitude, oscillation_freq_hz
            - decay_rate_lambda, fit_quality_r2
            - steady_state_error (final error in window)
        """
        if self.sp.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "uuid",
                    "is_delta",
                    "t_settle_seconds",
                    "settled",
                    "t_settle_derivative_seconds",
                    "rise_time_seconds",
                    "overshoot_abs",
                    "overshoot_pct",
                    "undershoot_abs",
                    "undershoot_pct",
                    "oscillation_count",
                    "oscillation_amplitude",
                    "oscillation_freq_hz",
                    "decay_rate_lambda",
                    "fit_quality_r2",
                    "steady_state_error",
                ]
            )

        # Compute all individual metrics using cached actual data
        settle_df = self.time_to_settle(
            actual_uuid, tol=tol, settle_pct=settle_pct, hold=hold, lookahead=lookahead
        )
        settle_deriv_df = self.time_to_settle_derivative(
            actual_uuid, rate_threshold=rate_threshold, lookahead=lookahead, hold=hold
        )
        rise_df = self.rise_time(actual_uuid, lookahead=lookahead)
        overshoot_df = self.overshoot_metrics(actual_uuid, window=lookahead)
        decay_df = self.decay_rate(actual_uuid, lookahead=lookahead)
        freq_df = self.oscillation_frequency(actual_uuid, window=lookahead)

        # Compute steady-state error
        actual = self._get_actual(actual_uuid)
        look_td = pd.to_timedelta(lookahead)

        sp = self.sp[[self.time_column, self.value_column]].copy()
        sp["prev"] = sp[self.value_column].shift(1)
        sp["delta"] = sp[self.value_column] - sp["prev"]
        changes = sp.loc[
            sp["delta"].abs() > 0, [self.time_column, self.value_column]
        ].reset_index(drop=True)

        ss_error_rows: List[Dict[str, Any]] = []
        for _, c in changes.iterrows():
            t0 = c[self.time_column]
            s_new = float(c[self.value_column])
            win = actual[
                (actual[self.time_column] >= t0)
                & (actual[self.time_column] <= t0 + look_td)
            ]

            if win.empty:
                ss_error = None
            else:
                # Use last 10% of window for steady-state
                n_points = len(win)
                last_10pct = win.iloc[int(n_points * 0.9) :]
                if len(last_10pct) > 0:
                    ss_error = float(
                        (last_10pct[self.value_column] - s_new).abs().mean()
                    )
                else:
                    ss_error = None

            ss_error_rows.append({"start": t0, "steady_state_error": ss_error})

        ss_error_df = pd.DataFrame(ss_error_rows)

        # Merge all metrics on 'start'
        result = settle_df.copy()

        # Merge settle_deriv
        result = result.merge(
            settle_deriv_df[["start", "t_settle_seconds"]].rename(
                columns={"t_settle_seconds": "t_settle_derivative_seconds"}
            ),
            on="start",
            how="left",
        )

        # Merge rise time
        result = result.merge(
            rise_df[["start", "rise_time_seconds"]],
            on="start",
            how="left",
        )

        # Merge overshoot metrics
        result = result.merge(
            overshoot_df[
                [
                    "start",
                    "overshoot_abs",
                    "overshoot_pct",
                    "undershoot_abs",
                    "undershoot_pct",
                    "oscillation_count",
                    "oscillation_amplitude",
                ]
            ],
            on="start",
            how="left",
        )

        # Merge frequency
        result = result.merge(
            freq_df[["start", "oscillation_freq_hz"]],
            on="start",
            how="left",
        )

        # Merge decay
        result = result.merge(
            decay_df[["start", "decay_rate_lambda", "fit_quality_r2"]],
            on="start",
            how="left",
        )

        # Merge steady-state error
        result = result.merge(
            ss_error_df[["start", "steady_state_error"]],
            on="start",
            how="left",
        )

        return result
