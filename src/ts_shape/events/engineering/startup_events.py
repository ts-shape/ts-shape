import logging
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from typing import List, Dict, Any, Optional

from ts_shape.utils.base import Base

logger = logging.getLogger(__name__)


class StartupDetectionEvents(Base):
    """
    Detect equipment startup intervals based on threshold crossings or
    sustained positive slope in a numeric metric (speed, temperature, etc.).

    Schema assumptions (columns):
    - uuid, sequence_number, systime, plctime, is_delta
    - value_integer, value_string, value_double, value_bool, value_bytes
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_uuid: str,
        *,
        event_uuid: str = "startup_event",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, column_name=time_column)
        self.target_uuid = target_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

        self.series = (
            self.dataframe[self.dataframe["uuid"] == self.target_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        self.series[self.time_column] = pd.to_datetime(self.series[self.time_column])

    def detect_startup_by_threshold(
        self,
        *,
        threshold: float,
        hysteresis: tuple[float, float] | None = None,
        min_above: str = "0s",
    ) -> pd.DataFrame:
        """
        Startup begins at first crossing above `threshold` (or hysteresis enter)
        and is valid only if the metric stays above the (exit) threshold for at
        least `min_above`.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta, method, threshold.
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=["start", "end", "uuid", "is_delta", "method", "threshold"]
            )

        enter_thr = threshold if hysteresis is None else hysteresis[0]
        exit_thr = threshold if hysteresis is None else hysteresis[1]
        min_above_td = pd.to_timedelta(min_above)

        s = self.series[[self.time_column, self.value_column]].copy()
        above_enter = s[self.value_column] >= enter_thr
        rising = (~above_enter.shift(fill_value=False)) & above_enter
        rise_times = s.loc[rising, self.time_column]

        events: List[Dict[str, Any]] = []
        for t0 in rise_times:
            # ensure dwell above exit threshold for min_above
            win = s[
                (s[self.time_column] >= t0) & (s[self.time_column] <= t0 + min_above_td)
            ]
            if win.empty:
                continue
            if (win[self.value_column] >= exit_thr).all():
                events.append(
                    {
                        "start": t0,
                        "end": t0 + min_above_td,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "method": "threshold",
                        "threshold": float(threshold),
                    }
                )

        return pd.DataFrame(events)

    def detect_startup_by_slope(
        self,
        *,
        min_slope: float,
        slope_window: str = "0s",
        min_duration: str = "0s",
    ) -> pd.DataFrame:
        """
        Startup intervals where per-second slope >= `min_slope` for at least
        `min_duration`. `slope_window` is accepted for API completeness but the
        current implementation uses instantaneous slope between samples.

        Returns:
            DataFrame with columns: start, end, uuid, is_delta, method, min_slope, avg_slope.
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "method",
                    "min_slope",
                    "avg_slope",
                ]
            )

        s = self.series[[self.time_column, self.value_column]].copy()
        s["dt_s"] = s[self.time_column].diff().dt.total_seconds()
        s["dv"] = s[self.value_column].diff()
        s["slope"] = s["dv"] / s["dt_s"]
        mask = s["slope"] >= float(min_slope)

        gid = (mask != mask.shift()).cumsum()
        min_d = pd.to_timedelta(min_duration)
        events: List[Dict[str, Any]] = []
        for _, seg in s.groupby(gid):
            seg_mask = mask.loc[seg.index]
            if not seg_mask.any():
                continue
            start_t = seg.loc[seg_mask, self.time_column].iloc[0]
            end_t = seg.loc[seg_mask, self.time_column].iloc[-1]
            if (end_t - start_t) < min_d:
                continue
            avg_slope = seg.loc[seg_mask, "slope"].mean()
            events.append(
                {
                    "start": start_t,
                    "end": end_t,
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "method": "slope",
                    "min_slope": float(min_slope),
                    "avg_slope": float(avg_slope) if pd.notna(avg_slope) else None,
                }
            )

        return pd.DataFrame(events)

    def detect_startup_multi_signal(
        self,
        signals: Dict[str, Dict[str, Any]],
        logic: str = "all",
        *,
        time_tolerance: str = "30s",
    ) -> pd.DataFrame:
        """
        Detect startups based on multiple signals with configurable AND/OR logic.

        Args:
            signals: Dict mapping uuid to detection config. Each config should contain:
                - 'method': 'threshold' or 'slope'
                - For threshold: 'threshold', optional 'hysteresis', 'min_above'
                - For slope: 'min_slope', optional 'slope_window', 'min_duration'
            logic: 'all' (AND - all signals must detect) or 'any' (OR - at least one)
            time_tolerance: Maximum time difference between signals for 'all' logic

        Returns:
            DataFrame with columns: start, end, uuid, is_delta, method, signals_triggered, signal_details
        """
        if logic not in ["all", "any"]:
            raise ValueError(f"logic must be 'all' or 'any', got '{logic}'")

        # Detect startups for each signal
        signal_events: Dict[str, pd.DataFrame] = {}
        for sig_uuid, config in signals.items():
            # Temporarily store original settings
            orig_uuid = self.target_uuid
            orig_series = self.series.copy()

            # Switch to the signal's uuid
            self.target_uuid = sig_uuid
            self.series = (
                self.dataframe[self.dataframe["uuid"] == sig_uuid]
                .copy()
                .sort_values(self.time_column)
            )
            self.series[self.time_column] = pd.to_datetime(
                self.series[self.time_column]
            )

            # Detect based on method
            method = config.get("method", "threshold")
            if method == "threshold":
                events = self.detect_startup_by_threshold(
                    threshold=config.get("threshold", 0),
                    hysteresis=config.get("hysteresis"),
                    min_above=config.get("min_above", "0s"),
                )
            elif method == "slope":
                events = self.detect_startup_by_slope(
                    min_slope=config.get("min_slope", 0),
                    slope_window=config.get("slope_window", "0s"),
                    min_duration=config.get("min_duration", "0s"),
                )
            else:
                raise ValueError(f"Unknown method '{method}' for signal {sig_uuid}")

            signal_events[sig_uuid] = events

            # Restore original settings
            self.target_uuid = orig_uuid
            self.series = orig_series

        # Combine events based on logic
        if logic == "any":
            # Union: any signal detecting is sufficient
            all_events = []
            for sig_uuid, events in signal_events.items():
                for _, event in events.iterrows():
                    all_events.append(
                        {
                            "start": event["start"],
                            "end": event["end"],
                            "uuid": self.event_uuid,
                            "is_delta": True,
                            "method": "multi_signal_any",
                            "signals_triggered": [sig_uuid],
                            "signal_details": {sig_uuid: event.to_dict()},
                        }
                    )
            return pd.DataFrame(all_events)

        else:  # logic == "all"
            # Intersection: all signals must detect within time_tolerance
            if not signal_events or any(df.empty for df in signal_events.values()):
                return pd.DataFrame(
                    columns=[
                        "start",
                        "end",
                        "uuid",
                        "is_delta",
                        "method",
                        "signals_triggered",
                        "signal_details",
                    ]
                )

            tolerance = pd.to_timedelta(time_tolerance)
            combined_events = []

            # Get events from first signal as reference
            first_sig = list(signal_events.keys())[0]
            for _, ref_event in signal_events[first_sig].iterrows():
                ref_start = ref_event["start"]

                # Check if all other signals have an event within tolerance
                matching_signals = {first_sig: ref_event.to_dict()}
                all_match = True

                for sig_uuid in list(signal_events.keys())[1:]:
                    sig_df = signal_events[sig_uuid]
                    # Find events within tolerance
                    matches = sig_df[
                        (sig_df["start"] >= ref_start - tolerance)
                        & (sig_df["start"] <= ref_start + tolerance)
                    ]

                    if matches.empty:
                        all_match = False
                        break

                    # Use the closest match
                    closest_idx = (matches["start"] - ref_start).abs().idxmin()
                    matching_signals[sig_uuid] = matches.loc[closest_idx].to_dict()

                if all_match:
                    # Calculate overall start/end from all matching events
                    all_starts = [
                        matching_signals[sig]["start"] for sig in matching_signals
                    ]
                    all_ends = [
                        matching_signals[sig]["end"] for sig in matching_signals
                    ]

                    combined_events.append(
                        {
                            "start": min(all_starts),
                            "end": max(all_ends),
                            "uuid": self.event_uuid,
                            "is_delta": True,
                            "method": "multi_signal_all",
                            "signals_triggered": list(matching_signals.keys()),
                            "signal_details": matching_signals,
                        }
                    )

            return pd.DataFrame(combined_events)

    def detect_startup_adaptive(
        self,
        *,
        baseline_window: str = "1h",
        sensitivity: float = 2.0,
        min_above: str = "10s",
        lookback_periods: int = 5,
    ) -> pd.DataFrame:
        """
        Detect startups using adaptive thresholds calculated from historical baseline data.

        Args:
            baseline_window: Window size for calculating baseline statistics
            sensitivity: Multiplier for standard deviation (threshold = mean + sensitivity * std)
            min_above: Minimum time the value must stay above threshold
            lookback_periods: Number of baseline periods to use for statistics

        Returns:
            DataFrame with columns: start, end, uuid, is_delta, method, adaptive_threshold,
                                    baseline_mean, baseline_std
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "method",
                    "adaptive_threshold",
                    "baseline_mean",
                    "baseline_std",
                ]
            )

        window_td = pd.to_timedelta(baseline_window)
        min_above_td = pd.to_timedelta(min_above)

        s = self.series[[self.time_column, self.value_column]].copy()
        s = s.sort_values(self.time_column).reset_index(drop=True)

        events: List[Dict[str, Any]] = []

        # Calculate rolling statistics
        s["rolling_mean"] = (
            s[self.value_column].rolling(window=lookback_periods, min_periods=1).mean()
        )
        s["rolling_std"] = (
            s[self.value_column].rolling(window=lookback_periods, min_periods=1).std()
        )

        # Calculate adaptive threshold
        s["adaptive_threshold"] = s["rolling_mean"] + sensitivity * s[
            "rolling_std"
        ].fillna(0)

        # Detect crossings above adaptive threshold
        s["above_threshold"] = s[self.value_column] >= s["adaptive_threshold"]
        s["crossing_up"] = (~s["above_threshold"].shift(fill_value=False)) & s[
            "above_threshold"
        ]

        # Find sustained periods above threshold
        for idx in s[s["crossing_up"]].index:
            t0 = s.loc[idx, self.time_column]
            threshold_at_crossing = s.loc[idx, "adaptive_threshold"]
            baseline_mean = s.loc[idx, "rolling_mean"]
            baseline_std = s.loc[idx, "rolling_std"]

            # Check if value stays above threshold for min_above duration
            win = s[
                (s[self.time_column] >= t0) & (s[self.time_column] <= t0 + min_above_td)
            ]

            if win.empty:
                continue

            # Value must stay above the threshold calculated at crossing time
            if (win[self.value_column] >= threshold_at_crossing).all():
                events.append(
                    {
                        "start": t0,
                        "end": t0 + min_above_td,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "method": "adaptive_threshold",
                        "adaptive_threshold": float(threshold_at_crossing),
                        "baseline_mean": (
                            float(baseline_mean) if pd.notna(baseline_mean) else None
                        ),
                        "baseline_std": (
                            float(baseline_std) if pd.notna(baseline_std) else None
                        ),
                    }
                )

        return pd.DataFrame(events)

    def assess_startup_quality(
        self,
        startup_events: pd.DataFrame,
        *,
        smoothness_window: int = 5,
        anomaly_threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Assess the quality of detected startup events.

        Args:
            startup_events: DataFrame of detected startup events (must have 'start' and 'end' columns)
            smoothness_window: Window size for calculating smoothness metrics
            anomaly_threshold: Z-score threshold for detecting anomalies

        Returns:
            DataFrame with quality metrics for each startup:
                - duration: Total duration of startup
                - smoothness_score: Inverse of derivative variance (higher = smoother)
                - anomaly_flags: Number of anomalous points detected
                - value_change: Total change in value during startup
                - avg_rate: Average rate of change
                - max_value: Maximum value reached
                - stability_score: Measure of how stable the final state is
        """
        if startup_events.empty or self.series.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration",
                    "smoothness_score",
                    "anomaly_flags",
                    "value_change",
                    "avg_rate",
                    "max_value",
                    "stability_score",
                ]
            )

        quality_results = []
        s = self.series[[self.time_column, self.value_column]].copy()

        for _, event in startup_events.iterrows():
            start = pd.to_datetime(event["start"])
            end = pd.to_datetime(event["end"])

            # Extract data for this startup period
            period_data = s[
                (s[self.time_column] >= start) & (s[self.time_column] <= end)
            ].copy()

            if period_data.empty or len(period_data) < 2:
                quality_results.append(
                    {
                        "start": start,
                        "end": end,
                        "duration": pd.Timedelta(0),
                        "smoothness_score": None,
                        "anomaly_flags": 0,
                        "value_change": None,
                        "avg_rate": None,
                        "max_value": None,
                        "stability_score": None,
                    }
                )
                continue

            # Calculate duration
            duration = end - start

            # Calculate derivatives for smoothness
            period_data["dt_s"] = (
                period_data[self.time_column].diff().dt.total_seconds()
            )
            period_data["derivative"] = (
                period_data[self.value_column].diff() / period_data["dt_s"]
            )

            # Smoothness: inverse of derivative variance (normalized)
            derivative_var = period_data["derivative"].var()
            smoothness_score = (
                1.0 / (1.0 + derivative_var) if pd.notna(derivative_var) else None
            )

            # Anomaly detection using z-scores
            values = period_data[self.value_column]
            z_scores = (
                np.abs((values - values.mean()) / values.std())
                if values.std() > 0
                else np.zeros(len(values))
            )
            anomaly_flags = int((z_scores > anomaly_threshold).sum())

            # Value change metrics
            value_change = values.iloc[-1] - values.iloc[0] if len(values) > 1 else 0
            avg_rate = (
                value_change / duration.total_seconds()
                if duration.total_seconds() > 0
                else 0
            )
            max_value = values.max()

            # Stability score: inverse of coefficient of variation in final 20% of period
            final_portion_size = max(1, len(period_data) // 5)
            final_values = values.iloc[-final_portion_size:]
            cv = (
                final_values.std() / final_values.mean()
                if final_values.mean() != 0
                else float("inf")
            )
            stability_score = 1.0 / (1.0 + cv) if np.isfinite(cv) else 0.0

            quality_results.append(
                {
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "smoothness_score": (
                        float(smoothness_score)
                        if smoothness_score is not None
                        else None
                    ),
                    "anomaly_flags": anomaly_flags,
                    "value_change": float(value_change),
                    "avg_rate": float(avg_rate),
                    "max_value": float(max_value),
                    "stability_score": float(stability_score),
                }
            )

        return pd.DataFrame(quality_results)

    def track_startup_phases(
        self,
        phases: List[Dict[str, Any]],
        *,
        min_phase_duration: str = "5s",
    ) -> pd.DataFrame:
        """
        Track progression through defined startup phases.

        Args:
            phases: List of phase definitions, each containing:
                - 'name': Phase name
                - 'condition': 'threshold', 'range', or 'slope'
                - For 'threshold': 'min_value' (value must be >= min_value)
                - For 'range': 'min_value' and 'max_value' (value in range)
                - For 'slope': 'min_slope' (slope must be >= min_slope)
            min_phase_duration: Minimum time to stay in phase to be considered valid

        Returns:
            DataFrame with phase transitions:
                - phase_name: Name of the phase
                - phase_number: Sequential phase number (0-indexed)
                - start: Phase start time
                - end: Phase end time
                - duration: Time spent in phase
                - next_phase: Name of the next phase (None for last phase)
                - completed: Whether full startup sequence completed
        """
        if self.series.empty or not phases:
            return pd.DataFrame(
                columns=[
                    "phase_name",
                    "phase_number",
                    "start",
                    "end",
                    "duration",
                    "next_phase",
                    "completed",
                ]
            )

        min_duration = pd.to_timedelta(min_phase_duration)
        s = self.series[[self.time_column, self.value_column]].copy()

        # Calculate slopes if needed
        s["dt_s"] = s[self.time_column].diff().dt.total_seconds()
        s["dv"] = s[self.value_column].diff()
        s["slope"] = s["dv"] / s["dt_s"]

        phase_results = []
        current_phase_idx = 0
        phase_start = None
        i = 0

        while i < len(s) and current_phase_idx < len(phases):
            row = s.iloc[i]
            phase = phases[current_phase_idx]

            # Check if current row satisfies phase condition
            in_phase = self._check_phase_condition(row, phase)

            if in_phase:
                if phase_start is None:
                    phase_start = row[self.time_column]

                # Check if we've been in this phase long enough
                if row[self.time_column] - phase_start >= min_duration:
                    # Check if we're transitioning to next phase
                    next_phase_idx = current_phase_idx + 1
                    if next_phase_idx < len(phases):
                        # Check if next phase condition is met
                        next_phase = phases[next_phase_idx]
                        if self._check_phase_condition(row, next_phase):
                            # Record completed phase
                            phase_results.append(
                                {
                                    "phase_name": phase["name"],
                                    "phase_number": current_phase_idx,
                                    "start": phase_start,
                                    "end": row[self.time_column],
                                    "duration": row[self.time_column] - phase_start,
                                    "next_phase": next_phase["name"],
                                    "completed": False,  # Will update at end
                                }
                            )
                            current_phase_idx = next_phase_idx
                            phase_start = row[self.time_column]
                    else:
                        # Last phase - check if it remains stable
                        remaining = s.iloc[i:]
                        if len(remaining) > 0:
                            # Check stability of last phase
                            stable = all(
                                self._check_phase_condition(s.iloc[j], phase)
                                for j in range(i, min(i + 10, len(s)))
                            )
                            if stable:
                                phase_results.append(
                                    {
                                        "phase_name": phase["name"],
                                        "phase_number": current_phase_idx,
                                        "start": phase_start,
                                        "end": row[self.time_column],
                                        "duration": row[self.time_column] - phase_start,
                                        "next_phase": None,
                                        "completed": True,
                                    }
                                )
                                break
            else:
                # Lost phase condition
                if (
                    phase_start is not None
                    and row[self.time_column] - phase_start >= min_duration
                ):
                    # Phase was valid but didn't progress - potential failed startup
                    phase_results.append(
                        {
                            "phase_name": phase["name"],
                            "phase_number": current_phase_idx,
                            "start": phase_start,
                            "end": row[self.time_column],
                            "duration": row[self.time_column] - phase_start,
                            "next_phase": None,
                            "completed": False,
                        }
                    )
                phase_start = None

            i += 1

        # Mark if full sequence completed
        if phase_results and phase_results[-1]["phase_number"] == len(phases) - 1:
            for result in phase_results:
                result["completed"] = True

        return pd.DataFrame(phase_results)

    def _check_phase_condition(self, row: pd.Series, phase: Dict[str, Any]) -> bool:
        """Helper method to check if a row satisfies a phase condition."""
        condition = phase.get("condition", "threshold")
        value = row[self.value_column]

        if condition == "threshold":
            min_val = phase.get("min_value", float("-inf"))
            return value >= min_val

        elif condition == "range":
            min_val = phase.get("min_value", float("-inf"))
            max_val = phase.get("max_value", float("inf"))
            return min_val <= value <= max_val

        elif condition == "slope":
            min_slope = phase.get("min_slope", 0)
            slope = row.get("slope", 0)
            return pd.notna(slope) and slope >= min_slope

        return False

    def detect_failed_startups(
        self,
        *,
        threshold: float,
        min_rise_duration: str = "5s",
        max_completion_time: str = "5m",
        completion_threshold: Optional[float] = None,
        required_stability: str = "10s",
    ) -> pd.DataFrame:
        """
        Detect failed or aborted startup attempts.

        A failed startup is identified when:
        1. Value rises above threshold for at least min_rise_duration
        2. But fails to reach completion_threshold within max_completion_time
        3. Or drops back below threshold before achieving required_stability

        Args:
            threshold: Initial threshold that must be crossed to begin startup
            min_rise_duration: Minimum time above threshold to consider it a startup attempt
            max_completion_time: Maximum time allowed to complete startup
            completion_threshold: Target threshold for successful completion (default: 2x threshold)
            required_stability: Time that must be maintained at completion level

        Returns:
            DataFrame with columns: start, end, uuid, is_delta, method, failure_reason,
                                    max_value_reached, time_to_failure
        """
        if self.series.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "uuid",
                    "is_delta",
                    "method",
                    "failure_reason",
                    "max_value_reached",
                    "time_to_failure",
                ]
            )

        if completion_threshold is None:
            completion_threshold = threshold * 2.0

        min_rise_td = pd.to_timedelta(min_rise_duration)
        max_completion_td = pd.to_timedelta(max_completion_time)
        stability_td = pd.to_timedelta(required_stability)

        s = self.series[[self.time_column, self.value_column]].copy()

        # Detect threshold crossings
        above_threshold = s[self.value_column] >= threshold
        rising = (~above_threshold.shift(fill_value=False)) & above_threshold
        rise_times = s.loc[rising, self.time_column]

        failed_events: List[Dict[str, Any]] = []

        for t0 in rise_times:
            # Get data for potential startup period
            startup_window = s[
                (s[self.time_column] >= t0)
                & (s[self.time_column] <= t0 + max_completion_td)
            ].copy()

            if len(startup_window) < 2:
                continue

            # Check if initially above threshold for min_rise_duration
            initial_window = startup_window[
                startup_window[self.time_column] <= t0 + min_rise_td
            ]

            if (
                initial_window.empty
                or not (initial_window[self.value_column] >= threshold).all()
            ):
                continue  # Not a valid startup attempt

            # Now check for failure modes
            max_value = startup_window[self.value_column].max()
            failure_detected = False
            failure_reason = None
            failure_time = None

            # Check if value drops back below threshold before completion
            above_in_window = startup_window[self.value_column] >= threshold
            if not above_in_window.all():
                # Find when it dropped below
                drop_idx = startup_window[~above_in_window].index[0]
                failure_time = startup_window.loc[drop_idx, self.time_column]

                # Check if this happened before reaching completion threshold
                if max_value < completion_threshold:
                    failure_detected = True
                    failure_reason = "dropped_below_threshold_before_completion"

            # Check if completion threshold was never reached
            if not failure_detected:
                reached_completion = (
                    startup_window[self.value_column] >= completion_threshold
                ).any()

                if not reached_completion:
                    failure_detected = True
                    failure_reason = "failed_to_reach_completion_threshold"
                    failure_time = startup_window[self.time_column].iloc[-1]
                else:
                    # Reached completion but check stability
                    completion_idx = startup_window[
                        startup_window[self.value_column] >= completion_threshold
                    ].index[0]
                    completion_time = startup_window.loc[
                        completion_idx, self.time_column
                    ]

                    stability_window = startup_window[
                        (startup_window[self.time_column] >= completion_time)
                        & (
                            startup_window[self.time_column]
                            <= completion_time + stability_td
                        )
                    ]

                    if not stability_window.empty:
                        if not (
                            stability_window[self.value_column] >= completion_threshold
                        ).all():
                            failure_detected = True
                            failure_reason = "insufficient_stability_at_completion"
                            failure_time = completion_time + stability_td

            if failure_detected:
                time_to_failure = (
                    (failure_time - t0).total_seconds() if failure_time else None
                )

                failed_events.append(
                    {
                        "start": t0,
                        "end": failure_time if failure_time else t0 + max_completion_td,
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "method": "failed_startup",
                        "failure_reason": failure_reason,
                        "max_value_reached": float(max_value),
                        "time_to_failure": time_to_failure,
                    }
                )

        return pd.DataFrame(failed_events)
