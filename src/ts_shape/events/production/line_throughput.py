import logging

import numpy as np
import pandas as pd

from ts_shape.events._windows import window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.counters import counter_increments

logger = logging.getLogger(__name__)


class LineThroughputEvents(Base):
    """Production: Line Throughput

    Methods:
    - count_parts: Part counts per fixed window from a monotonically increasing counter.
    - takt_adherence: Cycle time violations against a takt time from step/boolean triggers.
    - cycle_quality_check: Cycles from a trigger signal, flagged when outside the expected range.
    - throughput_oee: Availability, performance and OEE ratios (0-1) per window.
    - throughput_trends: Throughput trend with moving averages and degradation flags.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        event_uuid: str = "prod:throughput",
        time_column: str = "systime",
    ) -> None:
        super().__init__(dataframe, time_column=time_column)
        self.event_uuid = event_uuid
        self.time_column = time_column

    def count_parts(
        self,
        counter_uuid: str,
        *,
        value_column: str = "value_integer",
        window: str = "1min",
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Compute parts per window for a counter uuid.

        When ``handle_resets`` is False (default), the count per window is the
        window-over-window difference of the counter maximum, clipped at zero -
        so a window in which the counter resets to a lower value silently loses
        that production. When True, per-reading increments are computed on the
        raw counter (a decreasing counter is treated as a reset and the
        post-reset value counted as new production) and summed into each window,
        and a ``resets`` column reports the number of resets per window.

        Args:
            counter_uuid: UUID of the part counter signal.
            value_column: Column holding the counter value.
            window: Count window (pandas offset string, e.g. '1min', '1h').
            handle_resets: Count production across counter resets (see
                above) and add a ``resets`` column. Default False.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, is_delta, count (plus
            ``resets`` when ``handle_resets=True``).
        """
        base_columns = [
            "start",
            "end",
            "duration_seconds",
            "uuid",
            "source_uuid",
            "is_delta",
            "count",
        ]
        if handle_resets:
            base_columns.append("resets")
        c = (
            self.dataframe[self.dataframe["uuid"] == counter_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if c.empty:
            return pd.DataFrame(columns=base_columns)
        c[self.time_column] = pd.to_datetime(c[self.time_column])
        c = c.set_index(self.time_column)
        if handle_resets:
            series = c[value_column]
            increments = counter_increments(series)
            counts = increments.resample(window).sum()
            out = (
                counts.to_frame("count")
                .reset_index()
                .rename(columns={self.time_column: "start"})
            )
            is_reset = pd.to_numeric(series, errors="coerce").diff() < 0
            resets = is_reset.resample(window).sum().astype(int)
            out["resets"] = resets.values
        else:
            # take diff of last values within each window
            grp = c[value_column].resample(window)
            counts = grp.max().ffill().diff().fillna(0).clip(lower=0)
            out = (
                counts.to_frame("count")
                .reset_index()
                .rename(columns={self.time_column: "start"})
            )
        out["start"], out["end"] = window_bounds(out["start"], window)
        out["duration_seconds"] = (out["end"] - out["start"]).dt.total_seconds()
        out["uuid"] = self.event_uuid
        out["source_uuid"] = counter_uuid
        out["is_delta"] = True
        return out[base_columns]

    def takt_adherence(
        self,
        cycle_uuid: str,
        *,
        value_column: str = "value_bool",
        takt_time: str = "60s",
        min_violation: str = "0s",
    ) -> pd.DataFrame:
        """Flag cycles whose durations exceed the takt_time.

        For boolean triggers: detect True rising edges as cycle boundaries.
        For numeric counters/steps: detect increments as cycle boundaries; a
        decrease (a counter reset) is not a cycle boundary.

        Args:
            cycle_uuid: UUID of the cycle trigger signal.
            value_column: Column holding the trigger. ``"value_bool"`` uses
                rising edges as cycle boundaries; a numeric column uses every
                increase of the value (a reset or other decrease is not a
                boundary, so the cycle spanning it runs to the next
                increase); a non-numeric column (e.g. a string step name)
                uses every change of value.
            takt_time: Takt time (pandas offset string, e.g. '60s').
            min_violation: How far a cycle must exceed ``takt_time`` to be a
                violation: a cycle is a violation when it is strictly longer
                than ``takt_time + min_violation``. Default ``"0s"``: any
                cycle longer than the takt time counts; a cycle exactly equal
                to the takt time does not.

        Returns:
            DataFrame with columns: systime (at the boundary that closes the
            cycle), uuid, source_uuid, is_delta, cycle_time_seconds,
            violation.
        """
        s = (
            self.dataframe[self.dataframe["uuid"] == cycle_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if s.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "violation",
                ]
            )
        s[self.time_column] = pd.to_datetime(s[self.time_column])
        if value_column == "value_bool":
            # Cast first: ``~`` on an object column of Python bools is -2/-1.
            state = s[value_column].astype("boolean").fillna(False).astype(bool)
            edges = s[state & ~state.shift(fill_value=False)]
            times = edges[self.time_column].reset_index(drop=True)
        elif pd.api.types.is_numeric_dtype(s[value_column]):
            # Counter/step: a boundary is an increase; a reset is not a cycle.
            cur = s[value_column].fillna(0)
            prev = s[value_column].shift(1).fillna(0)
            edges = s[cur > prev]
            times = edges[self.time_column].reset_index(drop=True)
        else:
            s["prev"] = s[value_column].shift(1)
            edges = s[s[value_column].fillna(0) != s["prev"].fillna(0)]
            times = edges[self.time_column].reset_index(drop=True)
        if len(times) < 2:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "violation",
                ]
            )
        cycle_times = (times.diff().dt.total_seconds()).iloc[1:].reset_index(drop=True)
        min_td = pd.to_timedelta(min_violation).total_seconds()
        target = pd.to_timedelta(takt_time).total_seconds()
        viol = (cycle_times - target) > min_td
        out = pd.DataFrame(
            {
                "systime": times.iloc[1:].reset_index(drop=True),
                "uuid": self.event_uuid,
                "source_uuid": cycle_uuid,
                "is_delta": True,
                "cycle_time_seconds": cycle_times,
                "violation": viol,
            }
        )
        return out

    def throughput_oee(
        self,
        counter_uuid: str,
        *,
        value_column: str = "value_integer",
        window: str = "1h",
        target_rate: float | None = None,
        availability_threshold: float = 0.0,
        handle_resets: bool = False,
    ) -> pd.DataFrame:
        """Calculate Overall Equipment Effectiveness (OEE) metrics.

        OEE = Availability × Performance × Quality

        This counter-only estimate has no run-state signal, so availability is
        a production proxy: a window counts as available when its part count is
        at least ``availability_threshold`` × ``target_rate``. Quality is
        assumed to be 1.0.

        Args:
            counter_uuid: UUID for the part counter signal
            value_column: Column containing counter values
            window: Time window for aggregation
            target_rate: Target production rate (parts per window). If None, uses
                the 95th percentile of observed counts
            availability_threshold: Fraction of ``target_rate`` a window must
                reach to count as available (0.0-1.0). The default ``0.0`` keeps
                the historical behavior -- any window with at least one part is
                available. (Previously this argument was accepted but ignored.)
            handle_resets: When True, the part counter is treated as reset-aware
                (see :meth:`count_parts`), so a window where the counter resets
                is not dropped. Default False.

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, is_delta, actual_count, target_count, availability, performance, oee_score
        """
        parts_df = self.count_parts(
            counter_uuid,
            value_column=value_column,
            window=window,
            handle_resets=handle_resets,
        )

        if parts_df.empty:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration_seconds",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "actual_count",
                    "target_count",
                    "availability",
                    "performance",
                    "oee_score",
                ]
            )

        # Calculate target rate if not provided
        if target_rate is None:
            target_rate = parts_df["count"].quantile(0.95)

        parts_df["target_count"] = target_rate
        parts_df["actual_count"] = parts_df["count"]

        # Availability proxy: a window is available when it produced something
        # AND reached at least availability_threshold of the target rate. With
        # the default threshold of 0.0 this reduces to "produced anything",
        # matching the historical behavior.
        min_count = availability_threshold * (target_rate or 0.0)
        parts_df["availability"] = np.where(
            (parts_df["count"] > 0) & (parts_df["count"] >= min_count), 1.0, 0.0
        )

        # Performance: actual vs target rate. A non-positive target rate (flat
        # counter, long stoppage, or an explicit 0) would otherwise yield
        # inf/NaN performance; treat it as no performance instead.
        if target_rate and target_rate > 0:
            parts_df["performance"] = np.minimum(parts_df["count"] / target_rate, 1.0)
        else:
            parts_df["performance"] = 0.0

        # OEE score (simplified - assumes quality = 1.0)
        parts_df["oee_score"] = parts_df["availability"] * parts_df["performance"]

        return parts_df[
            [
                "start",
                "end",
                "duration_seconds",
                "uuid",
                "source_uuid",
                "is_delta",
                "actual_count",
                "target_count",
                "availability",
                "performance",
                "oee_score",
            ]
        ]

    def throughput_trends(
        self,
        counter_uuid: str,
        *,
        value_column: str = "value_integer",
        window: str = "1h",
        trend_window: int = 24,
    ) -> pd.DataFrame:
        """Analyze throughput trends with moving averages and degradation detection.

        Args:
            counter_uuid: UUID for the part counter signal
            value_column: Column containing counter values
            window: Time window for counting parts
            trend_window: Number of windows for trend calculation

        Returns:
            DataFrame with columns: start, end (start + window),
            duration_seconds, uuid, source_uuid, is_delta, count, moving_avg,
            trend_direction, degradation_detected
        """
        parts_df = self.count_parts(
            counter_uuid, value_column=value_column, window=window
        )

        if parts_df.empty or len(parts_df) < trend_window:
            return pd.DataFrame(
                columns=[
                    "start",
                    "end",
                    "duration_seconds",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "count",
                    "moving_avg",
                    "trend_direction",
                    "degradation_detected",
                ]
            )

        # Calculate moving average
        parts_df["moving_avg"] = (
            parts_df["count"].rolling(window=trend_window, min_periods=1).mean()
        )

        # Calculate trend (positive, negative, stable)
        parts_df["trend_slope"] = parts_df["moving_avg"].diff()
        parts_df["trend_direction"] = pd.cut(
            parts_df["trend_slope"],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=["decreasing", "stable", "increasing"],
        )

        # Detect degradation (current significantly below moving average)
        parts_df["degradation_detected"] = parts_df["count"] < (
            parts_df["moving_avg"] * 0.85
        )

        return parts_df[
            [
                "start",
                "end",
                "duration_seconds",
                "uuid",
                "source_uuid",
                "is_delta",
                "count",
                "moving_avg",
                "trend_direction",
                "degradation_detected",
            ]
        ]

    def cycle_quality_check(
        self,
        cycle_uuid: str,
        *,
        value_column: str = "value_bool",
        expected_cycle_time: float | None = None,
        tolerance_pct: float = 0.1,
    ) -> pd.DataFrame:
        """Enhanced cycle detection with quality validation.

        Args:
            cycle_uuid: UUID for the cycle trigger signal
            value_column: Column containing cycle trigger (bool/integer)
            expected_cycle_time: Expected cycle time in seconds. If None, uses median
            tolerance_pct: Tolerance percentage for cycle time validation

        Returns:
            DataFrame with cycle times, validation status, and quality flags
        """
        s = (
            self.dataframe[self.dataframe["uuid"] == cycle_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if s.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "expected_time",
                    "deviation_pct",
                    "is_valid",
                    "quality_flag",
                ]
            )

        s[self.time_column] = pd.to_datetime(s[self.time_column])

        # Detect cycle boundaries
        if value_column == "value_bool":
            # Cast first: ``~`` on an object column of Python bools is -2/-1.
            state = s[value_column].astype("boolean").fillna(False).astype(bool)
            edges = s[state & ~state.shift(fill_value=False)]
            times = edges[self.time_column].reset_index(drop=True)
        elif pd.api.types.is_numeric_dtype(s[value_column]):
            # Counter/step: a boundary is an increase; a reset is not a cycle.
            cur = s[value_column].fillna(0)
            prev = s[value_column].shift(1).fillna(0)
            edges = s[cur > prev]
            times = edges[self.time_column].reset_index(drop=True)
        else:
            s["prev"] = s[value_column].shift(1)
            edges = s[s[value_column].fillna(0) != s["prev"].fillna(0)]
            times = edges[self.time_column].reset_index(drop=True)

        if len(times) < 2:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "expected_time",
                    "deviation_pct",
                    "is_valid",
                    "quality_flag",
                ]
            )

        cycle_times = (times.diff().dt.total_seconds()).iloc[1:].reset_index(drop=True)

        # Calculate expected cycle time if not provided
        if expected_cycle_time is None:
            expected_cycle_time = cycle_times.median()

        # A non-positive (or NaN) reference has no meaningful deviation and
        # would divide by zero; return the typed empty frame instead.
        if pd.isna(expected_cycle_time) or expected_cycle_time <= 0:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "is_delta",
                    "cycle_time_seconds",
                    "expected_time",
                    "deviation_pct",
                    "is_valid",
                    "quality_flag",
                ]
            )

        # Calculate deviation
        deviation_pct = (
            (cycle_times - expected_cycle_time) / expected_cycle_time
        ).abs()

        # Validate cycles
        is_valid = deviation_pct <= tolerance_pct

        # Quality flags: good, warning, critical
        quality_flag = pd.cut(
            deviation_pct,
            bins=[-np.inf, 0.1, 0.25, np.inf],
            labels=["good", "warning", "critical"],
        )

        out = pd.DataFrame(
            {
                "systime": times.iloc[1:].reset_index(drop=True),
                "uuid": self.event_uuid,
                "source_uuid": cycle_uuid,
                "is_delta": True,
                "cycle_time_seconds": cycle_times,
                "expected_time": expected_cycle_time,
                "deviation_pct": deviation_pct,
                "is_valid": is_valid,
                "quality_flag": quality_flag,
            }
        )
        return out
