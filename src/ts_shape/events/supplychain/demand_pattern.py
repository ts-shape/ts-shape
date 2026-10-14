"""Demand pattern events for supply chain timeseries.

Aggregates demand by time window, detects demand spikes,
and summarizes seasonal demand patterns.
"""

import logging

import numpy as np
import pandas as pd

from ts_shape.events._windows import with_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only, renamed_kwargs

logger = logging.getLogger(__name__)


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
    ) -> pd.DataFrame:
        """Aggregate demand per time period.

        Args:
            window: Pandas offset alias for grouping (e.g. ``'1D'``, ``'1h'``).

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, total_demand, avg_demand, peak_demand.
        """
        empty_cols = [
            "start",
            "uuid",
            "source_uuid",
            "is_delta",
            "total_demand",
            "avg_demand",
            "peak_demand",
        ]
        if self.demand.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        dm = self.demand[[self.time_column, self.value_column]].copy()
        dm = dm.set_index(self.time_column)

        grouped = dm[self.value_column].resample(window)
        # min_count=1: an empty window is "no data" (NaN, dropped below), not
        # zero demand -- zeros skewed the spike and seasonality statistics.
        total = grouped.sum(min_count=1)
        avg = grouped.mean()
        peak = grouped.max()

        result = pd.DataFrame(
            {
                "start": total.index,
                "total_demand": total.values,
                "avg_demand": avg.values,
                "peak_demand": peak.values,
            }
        )

        # Drop periods with NaN (no data)
        result = result.dropna(subset=["total_demand"]).copy()

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
    ) -> pd.DataFrame:
        """Flag periods where demand exceeds mean + threshold_factor * std.

        Args:
            threshold_factor: Number of standard deviations above the mean
                to flag as a spike.
            window: Pandas offset alias for aggregation period.

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
        if self.demand.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        period_data = self.demand_by_period(window=window)

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
    ) -> pd.DataFrame:
        """Compute demand patterns by day-of-week (for daily) or hour-of-day (for hourly).

        When ``window='1D'``, groups by day of week (Monday=0 .. Sunday=6).
        When ``window`` is sub-daily (e.g. ``'1h'``), groups by hour of day (0..23).

        Args:
            window: Pandas offset alias. ``'1D'`` for day-of-week analysis,
                any sub-daily frequency for hour-of-day analysis.

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
        if self.demand.empty:
            return pd.DataFrame(columns=empty_cols)

        period_data = self.demand_by_period(window=window)

        if period_data.empty:
            return pd.DataFrame(columns=empty_cols)

        # Calendar periods ('MS', 'W', ...) have no fixed length and made
        # pd.to_timedelta raise; they are all at least a day long.
        offset = pd.tseries.frequencies.to_offset(window)
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
