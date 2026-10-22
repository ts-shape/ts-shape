"""Demand pattern events for supply chain timeseries.

Aggregates demand by time window, detects demand spikes,
and summarizes seasonal demand patterns.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

from ts_shape.events._windows import with_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs
from ts_shape.utils.frames import window_offset

logger = logging.getLogger(__name__)

#: How :meth:`DemandPatternEvents.demand_by_period` treats a period without
#: orders: ``"zero"`` keeps it with zero demand, ``"drop"`` leaves it out.
EMPTY_PERIOD_MODES: tuple[str, ...] = ("zero", "drop")


def _check_empty_periods(empty_periods: str) -> None:
    if empty_periods not in EMPTY_PERIOD_MODES:
        raise ValueError(
            f"empty_periods must be one of {EMPTY_PERIOD_MODES}, "
            f"got {empty_periods!r}"
        )


class DemandPatternEvents(Base):
    """Analyze demand patterns from a timeseries demand signal.

    Each demand event is a numeric value representing demand quantity
    at a point in time.

    Example usage:
        analyzer = DemandPatternEvents(df, demand_uuid='order_demand')
        daily = analyzer.demand_by_period(window='1D')
        spikes = analyzer.detect_demand_spikes(threshold_factor=2.0)
        seasonal = analyzer.seasonality_summary(window='1D')
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        demand_uuid: str,
        *,
        event_uuid: str = "sc:demand",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize demand pattern analyzer.

        Args:
            dataframe: Input DataFrame with timeseries data.
            demand_uuid: UUID of the demand signal.
            event_uuid: UUID assigned to generated events.
            value_column: Column containing numeric demand values.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.demand_uuid = demand_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self._validate_uuid(self.dataframe, demand_uuid)

        # Isolate the demand series
        self.demand = (
            self.dataframe[self.dataframe["uuid"] == self.demand_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if not self.demand.empty:
            self.demand[self.time_column] = pd.to_datetime(
                self.demand[self.time_column]
            )

    @keyword_only(1)
    @renamed_kwargs(period="window")
    def demand_by_period(
        self,
        window: str = "1D",
        empty_periods: Literal["zero", "drop"] = "zero",
    ) -> pd.DataFrame:
        """Aggregate demand per time period.

        Each row of the demand signal is one order (or demand event), so a
        period without rows is a period without demand. With
        ``empty_periods="zero"`` (the default) every period between the
        first and the last period with data appears, the empty ones with
        ``total_demand = 0``, ``peak_demand = 0``, ``order_count = 0`` and
        ``avg_demand`` missing (it is the mean order size, undefined
        without orders). A cumulative counter is not event-encoded demand:
        difference it into per-event quantities first.

        Args:
            window: Pandas offset alias for grouping (e.g. ``'1D'``, ``'1h'``).
            empty_periods: ``"zero"`` keeps periods without orders as zero
                demand; ``"drop"`` leaves them out (only periods with at
                least one order).

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, total_demand, avg_demand, peak_demand,
            order_count. ``order_count`` is the number of orders (non-missing
            values) in the period.

        Raises:
            ValueError: If ``empty_periods`` is not ``"zero"`` or ``"drop"``.
        """
        _check_empty_periods(empty_periods)
        empty_cols = [
            "start",
            "uuid",
            "source_uuid",
            "is_delta",
            "total_demand",
            "avg_demand",
            "peak_demand",
            "order_count",
        ]
        if self.demand.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        dm = self.demand[[self.time_column, self.value_column]].copy()
        dm = dm.set_index(self.time_column)

        grouped = dm[self.value_column].resample(window)
        count = grouped.count()
        total = grouped.sum(min_count=1)
        avg = grouped.mean()
        peak = grouped.max()

        result = pd.DataFrame(
            {
                "start": total.index,
                "total_demand": total.values,
                "avg_demand": avg.values,
                "peak_demand": peak.values,
                "order_count": count.values.astype("int64"),
            }
        )

        if empty_periods == "drop":
            result = result[result["order_count"] > 0].copy()
        else:
            # The resample spans the first to the last period with data;
            # trim leading/trailing periods whose rows held no value.
            has = np.flatnonzero(result["order_count"].to_numpy() > 0)
            if has.size:
                result = result.iloc[has[0] : has[-1] + 1].copy()
            else:
                result = result.iloc[0:0].copy()
            empty = result["order_count"] == 0
            result.loc[empty, "total_demand"] = 0.0
            result.loc[empty, "peak_demand"] = 0.0

        if result.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        result["uuid"] = self.event_uuid
        result["source_uuid"] = self.demand_uuid
        result["is_delta"] = True

        return with_window_bounds(result[empty_cols].reset_index(drop=True), window)

    @keyword_only(1)
    def detect_demand_spikes(
        self,
        threshold_factor: float = 2.0,
        window: str = "1D",
        empty_periods: Literal["zero", "drop"] = "zero",
    ) -> pd.DataFrame:
        """Flag periods where demand exceeds mean + threshold_factor * std.

        The mean and std are taken over the period totals of
        :meth:`demand_by_period`, by default including the periods without
        orders as zero demand: a spike is measured against the demand over
        the whole span, not only against the periods that had orders.

        Args:
            threshold_factor: Number of standard deviations above the mean
                to flag as a spike.
            window: Pandas offset alias for aggregation period.
            empty_periods: Passed to :meth:`demand_by_period`; ``"drop"``
                gives the baseline over periods with orders only.

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, demand, baseline_mean, spike_magnitude.
            ``end`` is ``start`` plus ``window``.
        """
        empty_cols = [
            "start",
            "uuid",
            "source_uuid",
            "is_delta",
            "demand",
            "baseline_mean",
            "spike_magnitude",
        ]
        _check_empty_periods(empty_periods)
        if self.demand.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        period_data = self.demand_by_period(window=window, empty_periods=empty_periods)

        if period_data.empty or len(period_data) < 2:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        mean_demand = period_data["total_demand"].mean()
        std_demand = period_data["total_demand"].std()

        if std_demand == 0 or np.isnan(std_demand):
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        threshold = mean_demand + threshold_factor * std_demand
        spikes = period_data[period_data["total_demand"] > threshold].copy()

        if spikes.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        result = pd.DataFrame(
            {
                "start": spikes["start"].array,
                "uuid": self.event_uuid,
                "source_uuid": self.demand_uuid,
                "is_delta": True,
                "demand": spikes["total_demand"].values,
                "baseline_mean": mean_demand,
                "spike_magnitude": spikes["total_demand"].values - mean_demand,
            }
        )

        return with_window_bounds(result[empty_cols].reset_index(drop=True), window)

    @keyword_only(1)
    @renamed_kwargs(period="window")
    def seasonality_summary(
        self,
        window: str = "1D",
        empty_periods: Literal["zero", "drop"] = "zero",
    ) -> pd.DataFrame:
        """Compute demand patterns by day-of-week (for daily) or hour-of-day (for hourly).

        When ``window='1D'``, groups by day of week (Monday=0 .. Sunday=6).
        When ``window`` is sub-daily (e.g. ``'1h'``), groups by hour of day (0..23).
        The statistics are over the period totals of :meth:`demand_by_period`,
        by default including periods without orders as zero demand (a
        weekday that usually has no orders shows a low average).

        Args:
            window: Pandas offset alias. ``'1D'`` for day-of-week analysis,
                any sub-daily frequency for hour-of-day analysis.
            empty_periods: Passed to :meth:`demand_by_period`; ``"drop"``
                uses periods with orders only.

        Returns:
            DataFrame with columns: period_label, avg_demand, std_demand,
            min_demand, max_demand.
        """
        empty_cols = [
            "period_label",
            "avg_demand",
            "std_demand",
            "min_demand",
            "max_demand",
        ]
        _check_empty_periods(empty_periods)
        if self.demand.empty:
            return pd.DataFrame(columns=empty_cols)

        period_data = self.demand_by_period(window=window, empty_periods=empty_periods)

        if period_data.empty:
            return pd.DataFrame(columns=empty_cols)

        # Calendar periods ('MS', 'W', ...) have no fixed length and made
        # pd.to_timedelta raise; they are all at least a day long.
        offset = window_offset(window)
        try:
            daily_or_longer = pd.Timedelta(offset) >= pd.Timedelta("1D")
        except ValueError:
            daily_or_longer = True

        if daily_or_longer:
            # Day-of-week grouping
            day_names = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            period_data["label"] = period_data["start"].dt.dayofweek
            grouped = period_data.groupby("label")["total_demand"]
            stats = grouped.agg(["mean", "std", "min", "max"]).reset_index()
            stats.columns = [
                "label",
                "avg_demand",
                "std_demand",
                "min_demand",
                "max_demand",
            ]
            stats["period_label"] = stats["label"].apply(
                lambda x: day_names[int(x)] if 0 <= int(x) <= 6 else str(x)
            )
        else:
            # Hour-of-day grouping
            period_data["label"] = period_data["start"].dt.hour
            grouped = period_data.groupby("label")["total_demand"]
            stats = grouped.agg(["mean", "std", "min", "max"]).reset_index()
            stats.columns = [
                "label",
                "avg_demand",
                "std_demand",
                "min_demand",
                "max_demand",
            ]
            stats["period_label"] = stats["label"].apply(lambda x: f"{int(x):02d}:00")

        stats["std_demand"] = stats["std_demand"].fillna(0.0)

        return stats[empty_cols].reset_index(drop=True)
