"""Cycle time tracking by part number.

Simple, practical module for cycle time analysis:
- Cycle time per part number
- Statistics (min, avg, max, std)
- Slow cycle detection
- Trend analysis
"""

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ts_shape.errors import DataQualityWarning
from ts_shape.events._output import twin_columns
from ts_shape.events._windows import floor_local
from ts_shape.utils.base import Base
from ts_shape.utils.coerce import run_bool
from ts_shape.utils.deprecation import caller_stacklevel, deprecated_class

if TYPE_CHECKING:
    from typing_extensions import deprecated

logger = logging.getLogger(__name__)


def _is_pulse(values: pd.Series) -> bool:
    """Whether ``values`` is a 0/1 pulse (numbers that are only 0 and 1)."""
    known = pd.to_numeric(values, errors="coerce").dropna()
    if known.empty or len(known) < len(values.dropna()):
        return False
    return set(known.unique().tolist()) <= {0.0, 1.0}


#: Share of pulses seen in a single sample above which a polled trigger is
#: taken to miss pulses.
SINGLE_SAMPLE_PULSE_SHARE = 0.8
#: Share of regular sample intervals that marks a polled (sampled) signal.
POLLED_REGULAR_SHARE = 0.6


def _warn_missed_pulses(times: pd.Series, trigger: pd.Series, uuid: str) -> bool:
    """Warn when a polled cycle trigger is likely to miss its pulses.

    A trigger read by polling every ``P`` seconds sees a pulse shorter than
    ``P`` only when a poll happens to fall on it. When the signal is polled
    (regular sample intervals) and most pulses are seen in exactly one
    sample, many pulses fall between the polls: every missed pulse merges
    two or more cycles into one, so the cycle times are too long.
    """
    from ts_shape.signals.infer import _interval_stats

    if len(trigger) < 4:
        return False
    on = trigger.to_numpy(dtype=bool)
    starts = on & ~np.concatenate(([False], on[:-1]))
    pulses = int(starts.sum())
    if pulses < 3:
        return False
    run_id = np.cumsum(starts)[on]
    lengths = np.bincount(run_id)[1:]
    single = float((lengths == 1).mean()) if len(lengths) else 0.0
    period, _, regular = _interval_stats(pd.to_datetime(times).reset_index(drop=True))
    if period is None or regular < POLLED_REGULAR_SHARE:
        return False  # change-encoded: one row per edge is normal
    if single < SINGLE_SAMPLE_PULSE_SHARE:
        return False
    warnings.warn(
        f"CycleTimeEvents: cycle trigger {uuid!r} is polled every "
        f"{period.total_seconds():g} s and {int((lengths == 1).sum())} of "
        f"{len(lengths)} pulses were seen in a single sample: pulses shorter "
        "than the polling interval are missed, and each missed pulse merges "
        "cycles, so the cycle times are too long. Use a cycle or part counter "
        "(or a change-encoded trigger) instead.",
        DataQualityWarning,
        stacklevel=caller_stacklevel(),
    )
    return True


class CycleTimeEvents(Base):
    """Track cycle times by part number.

    Each UUID represents one signal:
    - part_id_uuid: string signal with current part number
    - cycle_trigger_uuid: boolean/integer signal for cycle completion

    Example:
        >>> import pandas as pd, ts_shape
        >>> df = pd.concat([ts_shape.make_id_signal("part", ["A", "B"], hold=30),
        ...     ts_shape.make_timeseries(["cycle"], n_points=60, value_column="value_bool")])
        >>> stats = ts_shape.CycleTimeEvents(df).cycle_time_statistics("part", "cycle")
        >>> stats[["part_number", "cycle_count", "avg_seconds"]].round(1)
          part_number  cycle_count  avg_seconds
        0           A            7        111.4
        1           B            5        198.0
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        time_column: str = "systime",
        event_uuid: str = "prod:cycle_time",
    ) -> None:
        """Initialize cycle time tracker.

        Args:
            dataframe: Input DataFrame with timeseries data
            time_column: Name of timestamp column (default: 'systime')
            event_uuid: UUID tagged on derived cycle events (the ``uuid``
                column of point outputs; default: 'prod:cycle_time')
        """
        super().__init__(dataframe, time_column=time_column)
        self.time_column = time_column
        self.event_uuid = event_uuid

    def cycle_time_by_part(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate cycle time for each part number.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers
            max_cycle_gap: Optional cap (e.g. "5min"). Intervals between
                consecutive completions longer than this are dropped as
                inter-cycle downtime rather than reported as a cycle. Without
                it, a stoppage between completions (a break, a jam) becomes one
                giant "cycle" that inflates max/avg and is falsely flagged as a
                slow cycle.

        Returns:
            DataFrame with columns:
            - systime: Cycle completion time
            - uuid: Event UUID of this tracker
            - source_uuid: The cycle trigger UUID
            - part_number: Part number/ID
            - cycle_time_seconds: Cycle time in seconds

        Example:
            >>> import pandas as pd, ts_shape
            >>> df = pd.concat([ts_shape.make_id_signal("part", ["A", "B"], hold=30),
            ...     ts_shape.make_timeseries(["cycle"], n_points=60, value_column="value_bool")])
            >>> out = ts_shape.CycleTimeEvents(df).cycle_time_by_part("part", "cycle")
            >>> out[["part_number", "cycle_time_seconds"]].head(3)
              part_number  cycle_time_seconds
            0           A                60.0
            1           A               120.0
            2           A               120.0
        """
        # Get cycle completion times
        cycles = (
            self.dataframe[self.dataframe["uuid"] == cycle_trigger_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if cycles.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "part_number",
                    "cycle_time_seconds",
                ]
            )

        cycles[self.time_column] = pd.to_datetime(cycles[self.time_column])

        # Detect rising edges (cycle completion)
        if value_column_trigger == "value_bool" or _is_pulse(
            cycles[value_column_trigger]
        ):
            # A bool or a 0/1 pulse: a cycle ends on each rising edge (a 0/1
            # pulse in value_integer is not counted on both edges).
            trigger = run_bool(cycles[value_column_trigger], warn=False)
            _warn_missed_pulses(cycles[self.time_column], trigger, cycle_trigger_uuid)
            prev = trigger.shift(fill_value=False)
            cycle_ends = cycles[trigger & ~prev]
            times = cycle_ends[self.time_column].reset_index(drop=True)
        else:
            # For integer/counter-based cycles, use value changes
            cycles["prev"] = cycles[value_column_trigger].shift()
            cycle_ends = cycles[cycles[value_column_trigger] != cycles["prev"]]
            times = cycle_ends[self.time_column].reset_index(drop=True)

        if len(times) < 2:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "part_number",
                    "cycle_time_seconds",
                ]
            )

        # Calculate cycle times
        cycle_times = times.diff().dt.total_seconds()

        cap_seconds = (
            pd.to_timedelta(max_cycle_gap).total_seconds()
            if max_cycle_gap is not None
            else None
        )

        # Get part numbers at each cycle
        part_data = (
            self.dataframe[self.dataframe["uuid"] == part_id_uuid]
            .copy()
            .sort_values(self.time_column)
        )

        if part_data.empty:
            # Return cycles without part numbers
            no_part = pd.DataFrame(
                {
                    "systime": times.iloc[1:].reset_index(drop=True),
                    "part_number": "UNKNOWN",
                    "cycle_time_seconds": cycle_times.iloc[1:].reset_index(drop=True),
                }
            )
            if cap_seconds is not None:
                no_part = no_part[
                    no_part["cycle_time_seconds"] <= cap_seconds
                ].reset_index(drop=True)
            no_part.insert(1, "uuid", self.event_uuid)
            no_part.insert(2, "source_uuid", cycle_trigger_uuid)
            return no_part

        part_data[self.time_column] = pd.to_datetime(part_data[self.time_column])

        # Match part number to each cycle using merge_asof
        result_df = pd.DataFrame(
            {
                "systime": times.iloc[1:].reset_index(drop=True),
                "cycle_time_seconds": cycle_times.iloc[1:].reset_index(drop=True),
            }
        )

        # Merge with part data
        result_df = pd.merge_asof(
            result_df.sort_values("systime"),
            part_data[[self.time_column, value_column_part]].rename(
                columns={self.time_column: "systime"}
            ),
            on="systime",
            direction="backward",
        )

        result_df = result_df.rename(columns={value_column_part: "part_number"})
        result_df["part_number"] = result_df["part_number"].fillna("UNKNOWN")

        if cap_seconds is not None:
            result_df = result_df[
                result_df["cycle_time_seconds"] <= cap_seconds
            ].reset_index(drop=True)

        result_df = result_df[["systime", "part_number", "cycle_time_seconds"]].copy()
        result_df.insert(1, "uuid", self.event_uuid)
        result_df.insert(2, "source_uuid", cycle_trigger_uuid)
        return result_df

    @twin_columns({"count": "cycle_count"})
    def cycle_time_statistics(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Calculate statistics: min, avg, max, std cycle time by part.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers
            max_cycle_gap: Optional cap (e.g. "5min") forwarded to
                :meth:`cycle_time_by_part`: intervals between consecutive
                completions longer than this are dropped as inter-cycle
                downtime. Default None keeps every interval.

        Returns:
            DataFrame with statistics per part:
            - part_number: Part number/ID
            - count: Number of cycles (deprecated twin of ``cycle_count``)
            - cycle_count: Number of cycles
            - min_seconds: Minimum cycle time
            - avg_seconds: Average cycle time
            - max_seconds: Maximum cycle time
            - std_seconds: Standard deviation
            - median_seconds: Median cycle time
            - p10_seconds: 10th percentile, the best demonstrated cycle
                time (a target taken from the data, not from master data)
            - p90_seconds: 90th percentile
            - mad_seconds: Median absolute deviation from the median
                (robust spread; unscaled)

            Canonical twin columns: ``cycle_count`` (after ``count``); it
            holds the same values as the deprecated column right before it,
            which is removed in the next major release.

        Example:
            >>> import pandas as pd, ts_shape
            >>> df = pd.concat([ts_shape.make_id_signal("part", ["A", "B"], hold=30),
            ...     ts_shape.make_timeseries(["cycle"], n_points=60, value_column="value_bool")])
            >>> stats = ts_shape.CycleTimeEvents(df).cycle_time_statistics("part", "cycle")
            >>> stats[["part_number", "cycle_count", "median_seconds", "max_seconds"]]
              part_number  cycle_count  median_seconds  max_seconds
            0           A            7           120.0        180.0
            1           B            5           150.0        420.0
        """
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
            max_cycle_gap=max_cycle_gap,
        )

        if cycle_data.empty:
            return pd.DataFrame(
                columns=[
                    "part_number",
                    "count",
                    "min_seconds",
                    "avg_seconds",
                    "max_seconds",
                    "std_seconds",
                    "median_seconds",
                    "p10_seconds",
                    "p90_seconds",
                    "mad_seconds",
                ]
            )

        stats = (
            cycle_data.groupby("part_number")["cycle_time_seconds"]
            .agg(
                [
                    ("count", "count"),
                    ("min_seconds", "min"),
                    ("avg_seconds", "mean"),
                    ("max_seconds", "max"),
                    ("std_seconds", "std"),
                    ("median_seconds", "median"),
                    ("p10_seconds", lambda x: x.quantile(0.1)),
                    ("p90_seconds", lambda x: x.quantile(0.9)),
                    ("mad_seconds", lambda x: (x - x.median()).abs().median()),
                ]
            )
            .reset_index()
        )

        return stats

    def detect_slow_cycles(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        threshold_factor: float = 1.5,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Identify cycles that exceed normal time by threshold factor.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            threshold_factor: Cycles slower than median * factor are flagged (default: 1.5)
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers
            max_cycle_gap: Optional cap (e.g. "5min") forwarded to
                :meth:`cycle_time_by_part`: intervals between consecutive
                completions longer than this are dropped as inter-cycle
                downtime. Default None keeps every interval.

        Returns:
            DataFrame with slow cycles:
            - systime: Cycle completion time
            - uuid: Event UUID of this tracker
            - source_uuid: The cycle trigger UUID
            - part_number: Part number/ID
            - cycle_time_seconds: Actual cycle time
            - median_seconds: Expected (median) cycle time
            - deviation_factor: How many times slower than expected
            - is_slow: Flag (always True in returned data)

        Example:
            >>> import pandas as pd, ts_shape
            >>> df = pd.concat([ts_shape.make_id_signal("part", ["A", "B"], hold=30),
            ...     ts_shape.make_timeseries(["cycle"], n_points=60, value_column="value_bool")])
            >>> slow = ts_shape.CycleTimeEvents(df).detect_slow_cycles("part", "cycle")
            >>> slow[["part_number", "cycle_time_seconds", "deviation_factor"]]
              part_number  cycle_time_seconds  deviation_factor
            0           A               180.0               1.5
            1           B               240.0               1.6
            2           B               420.0               2.8
        """
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
            max_cycle_gap=max_cycle_gap,
        )

        if cycle_data.empty:
            return pd.DataFrame(
                columns=[
                    "systime",
                    "uuid",
                    "source_uuid",
                    "part_number",
                    "cycle_time_seconds",
                    "median_seconds",
                    "deviation_factor",
                    "is_slow",
                ]
            )

        # Calculate median per part
        medians = cycle_data.groupby("part_number")["cycle_time_seconds"].median()
        cycle_data["median_seconds"] = cycle_data["part_number"].map(medians)

        # Calculate deviation factor
        cycle_data["deviation_factor"] = (
            cycle_data["cycle_time_seconds"] / cycle_data["median_seconds"]
        )

        # Flag slow cycles
        cycle_data["is_slow"] = cycle_data["deviation_factor"] >= threshold_factor

        # Return only slow cycles
        return cycle_data[cycle_data["is_slow"]].reset_index(drop=True)

    def detect_unusual_cycles(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        n_sigma: float = 3.0,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Find cycles that are unusually long *or* short, per part.

        Unlike :meth:`detect_slow_cycles` (a fixed factor on the slow side
        only) the bounds are statistical and two-sided: a cycle is unusual
        when its log cycle time lies more than ``n_sigma`` robust sigmas
        (scaled median absolute deviation, at least 1 %) from the part's
        median log cycle time. Working on the log makes the bounds relative,
        so they transfer between a 5 s and a 300 s cycle. For pauses kept as
        their own class and a profile without signal mapping see
        :class:`~ts_shape.events.production.cycle_profiling.CycleProfileEvents`.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            n_sigma: Robust sigmas beyond which a cycle is unusual.
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers
            max_cycle_gap: Optional cap forwarded to
                :meth:`cycle_time_by_part`.

        Returns:
            DataFrame with the unusual cycles:
            - systime, uuid, source_uuid, part_number, cycle_time_seconds
            - median_seconds: The part's median cycle time
            - lower_limit_seconds / upper_limit_seconds: The robust bounds
            - z_score: Robust z-score of the log cycle time
            - cycle_class: ``"short"`` or ``"long"``
            - direction: ``"below"`` or ``"above"`` the robust band

        Example:
            >>> import pandas as pd, ts_shape
            >>> t = pd.Timestamp("2025-01-01") + pd.to_timedelta(
            ...     [0, 60, 121, 180, 242, 300, 330, 391, 450, 600], unit="s")
            >>> df = pd.concat([
            ...     pd.DataFrame({"systime": t, "uuid": "done", "value_integer": range(10)}),
            ...     pd.DataFrame({"systime": t[:1], "uuid": "part", "value_string": ["A"]})])
            >>> out = ts_shape.CycleTimeEvents(df).detect_unusual_cycles(
            ...     "part", "done", value_column_trigger="value_integer")
            >>> out[["cycle_time_seconds", "cycle_class"]]
               cycle_time_seconds cycle_class
            0                30.0       short
            1               150.0        long
        """
        columns = [
            "systime",
            "uuid",
            "source_uuid",
            "part_number",
            "cycle_time_seconds",
            "median_seconds",
            "lower_limit_seconds",
            "upper_limit_seconds",
            "z_score",
            "cycle_class",
            "direction",
        ]
        if n_sigma <= 0:
            raise ValueError("n_sigma must be positive")
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
            max_cycle_gap=max_cycle_gap,
        )
        cycle_data = cycle_data[cycle_data["cycle_time_seconds"] > 0]
        if cycle_data.empty:
            return pd.DataFrame(columns=columns)
        log_t = np.log(cycle_data["cycle_time_seconds"].astype(float))
        groups = log_t.groupby(cycle_data["part_number"])
        center = groups.transform("median")
        mad = (
            (log_t - center)
            .abs()
            .groupby(cycle_data["part_number"])
            .transform("median")
        )
        sigma = np.maximum(1.4826 * mad, 0.01)
        z = (log_t - center) / sigma
        out = cycle_data.assign(
            median_seconds=np.exp(center),
            lower_limit_seconds=np.exp(center - n_sigma * sigma),
            upper_limit_seconds=np.exp(center + n_sigma * sigma),
            z_score=z,
            cycle_class=np.where(z > 0, "long", "short"),
            direction=np.where(z > 0, "above", "below"),
        )
        out = out[z.abs() > n_sigma]
        return out[columns].reset_index(drop=True)

    @twin_columns({"moving_avg": "moving_avg_seconds"})
    def cycle_time_trend(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        part_number: str,
        *,
        window_size: int = 20,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
        max_cycle_gap: str | None = None,
        stable_band_ratio: float = 0.01,
    ) -> pd.DataFrame:
        """Analyze cycle time trends for a specific part.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            part_number: Specific part number to analyze
            window_size: Number of cycles for moving average (default: 20)
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers
            max_cycle_gap: Optional cap (e.g. "5min") forwarded to
                :meth:`cycle_time_by_part`: intervals between consecutive
                completions longer than this are dropped as inter-cycle
                downtime. Default None keeps every interval.
            stable_band_ratio: A change of the moving average within this
                fraction of the part's median cycle time is ``stable``
                (default 0.01 = 1 %). The band scales with the cycle, so the
                same setting fits a 5 s and a 300 s cycle; earlier versions
                used a fixed band of +-0.5 s.

        Returns:
            DataFrame with trend data:
            - systime: Cycle completion time
            - cycle_time_seconds: Actual cycle time
            - moving_avg: Moving average cycle time (deprecated twin)
            - moving_avg_seconds: Moving average cycle time in seconds
            - trend: 'improving', 'stable', or 'degrading'

        Example:
            >>> import pandas as pd, ts_shape
            >>> df = pd.concat([ts_shape.make_id_signal("part", ["A", "B"], hold=30),
            ...     ts_shape.make_timeseries(["cycle"], n_points=60, value_column="value_bool")])
            >>> trend = ts_shape.CycleTimeEvents(df).cycle_time_trend(
            ...     "part", "cycle", "A", window_size=3)
            >>> trend[["cycle_time_seconds", "moving_avg_seconds", "trend"]].tail(3)
               cycle_time_seconds  moving_avg_seconds      trend
            4                60.0               120.0  improving
            5               120.0               120.0     stable
            6               120.0               100.0  improving
        """
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
            max_cycle_gap=max_cycle_gap,
        )

        if cycle_data.empty:
            return pd.DataFrame(
                columns=["systime", "cycle_time_seconds", "moving_avg", "trend"]
            )

        # Filter for specific part
        part_cycles = cycle_data[cycle_data["part_number"] == part_number].copy()

        if part_cycles.empty or len(part_cycles) < window_size:
            return pd.DataFrame(
                columns=["systime", "cycle_time_seconds", "moving_avg", "trend"]
            )

        # Calculate moving average
        part_cycles["moving_avg"] = (
            part_cycles["cycle_time_seconds"]
            .rolling(window=window_size, min_periods=1)
            .mean()
        )

        # Calculate trend
        part_cycles["trend_slope"] = part_cycles["moving_avg"].diff()

        # Classify trend (improving = getting faster)
        if stable_band_ratio < 0:
            raise ValueError("stable_band_ratio must not be negative")
        band = stable_band_ratio * float(part_cycles["cycle_time_seconds"].median())
        slope = part_cycles["trend_slope"]
        # A missing slope (the first cycle) stays missing.
        labels = pd.Series(None, index=slope.index, dtype=object)
        labels[slope.notna()] = "stable"
        labels[slope < -band] = "improving"
        labels[slope > band] = "degrading"
        part_cycles["trend"] = pd.Categorical(
            labels, categories=["improving", "stable", "degrading"]
        )

        return part_cycles[["systime", "cycle_time_seconds", "moving_avg", "trend"]]

    @twin_columns(
        {
            "avg_cycle_time": "avg_cycle_time_seconds",
            "min_cycle_time": "min_cycle_time_seconds",
            "max_cycle_time": "max_cycle_time_seconds",
        }
    )
    def hourly_cycle_time_summary(
        self,
        part_id_uuid: str,
        cycle_trigger_uuid: str,
        *,
        value_column_part: str = "value_string",
        value_column_trigger: str = "value_bool",
        max_cycle_gap: str | None = None,
    ) -> pd.DataFrame:
        """Hourly summary of cycle times by part.

        Args:
            part_id_uuid: UUID for part number signal
            cycle_trigger_uuid: UUID for cycle completion trigger
            value_column_part: Column containing part numbers
            value_column_trigger: Column containing cycle triggers
            max_cycle_gap: Optional cap (e.g. "5min") forwarded to
                :meth:`cycle_time_by_part`: intervals between consecutive
                completions longer than this are dropped as inter-cycle
                downtime. Default None keeps every interval.

        Returns:
            DataFrame with hourly statistics:
            - hour: Hour timestamp
            - part_number: Part number/ID
            - cycles_completed: Number of cycles
            - avg_cycle_time / avg_cycle_time_seconds: Average cycle time
            - min_cycle_time / min_cycle_time_seconds: Fastest cycle
            - max_cycle_time / max_cycle_time_seconds: Slowest cycle

            All cycle times are in seconds. Canonical twin columns:
            ``avg_cycle_time_seconds``, ``min_cycle_time_seconds``,
            ``max_cycle_time_seconds``, each right after its deprecated
            unitless column (same values), which is removed in the next major
            release.

        Example:
            >>> import pandas as pd, ts_shape
            >>> df = pd.concat([ts_shape.make_id_signal("part", ["A", "B"], hold=30),
            ...     ts_shape.make_timeseries(["cycle"], n_points=60, value_column="value_bool")])
            >>> hourly = ts_shape.CycleTimeEvents(df).hourly_cycle_time_summary("part", "cycle")
            >>> hourly[["part_number", "cycles_completed", "avg_cycle_time_seconds"]]
              part_number  cycles_completed  avg_cycle_time_seconds
            0           A                 7              111.428571
            1           B                 5              198.000000
        """
        cycle_data = self.cycle_time_by_part(
            part_id_uuid,
            cycle_trigger_uuid,
            value_column_part=value_column_part,
            value_column_trigger=value_column_trigger,
            max_cycle_gap=max_cycle_gap,
        )

        if cycle_data.empty:
            return pd.DataFrame(
                columns=[
                    "hour",
                    "part_number",
                    "cycles_completed",
                    "avg_cycle_time",
                    "min_cycle_time",
                    "max_cycle_time",
                ]
            )

        # Add hour column
        cycle_data["hour"] = floor_local(cycle_data["systime"], "h")

        # Group by hour and part
        hourly = (
            cycle_data.groupby(["hour", "part_number"])["cycle_time_seconds"]
            .agg(
                [
                    ("cycles_completed", "count"),
                    ("avg_cycle_time", "mean"),
                    ("min_cycle_time", "min"),
                    ("max_cycle_time", "max"),
                ]
            )
            .reset_index()
        )

        return hourly


#: Deprecated name, removed in the next major release.
if TYPE_CHECKING:

    @deprecated("CycleTimeTracking is deprecated, use CycleTimeEvents")
    class CycleTimeTracking(CycleTimeEvents): ...

else:
    CycleTimeTracking = deprecated_class(CycleTimeEvents, "CycleTimeTracking")
