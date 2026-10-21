"""Inventory monitoring events for supply chain timeseries.

Detects low stock intervals, calculates consumption rates,
identifies reorder point breaches, and predicts stockouts.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from ts_shape.events._windows import window_bounds, with_window_bounds
from ts_shape.utils.base import Base
from ts_shape.utils.deprecation import keyword_only
from ts_shape.utils.intervals import interval_end_times
from ts_shape.utils.runs import Runs, run_agg, total_seconds, true_runs

logger = logging.getLogger(__name__)


class InventoryMonitoringEvents(Base):
    """Monitor inventory levels from a timeseries signal and detect supply chain events.

    Each inventory level is tracked via a UUID signal whose numeric value
    represents the current stock quantity.

    Example usage:
        tracker = InventoryMonitoringEvents(df, level_uuid='warehouse_a_level')
        low = tracker.detect_low_stock(min_level=100, hold='5min')
        rate = tracker.consumption_rate(window='1h')
        breach = tracker.reorder_point_breach(reorder_level=200, safety_stock=50)
        prediction = tracker.stockout_prediction(consumption_rate_window='4h')
    """

    #: Methods read only the constructor's signals (``Base`` keeps their rows).
    _reads_own_signals_only = True

    def __init__(
        self,
        dataframe: pd.DataFrame,
        level_uuid: str,
        *,
        event_uuid: str = "sc:inventory",
        value_column: str = "value_double",
        time_column: str = "systime",
    ) -> None:
        """Initialize inventory monitoring.

        Args:
            dataframe: Input DataFrame with timeseries data.
            level_uuid: UUID of the inventory level signal.
            event_uuid: UUID assigned to generated events.
            value_column: Column containing numeric inventory levels.
            time_column: Name of timestamp column.
        """
        super().__init__(dataframe, time_column=time_column)
        self.level_uuid = level_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column
        self._validate_uuid(self.dataframe, level_uuid)

        # Isolate the inventory level series
        self.levels = (
            self.dataframe[self.dataframe["uuid"] == self.level_uuid]
            .copy()
            .sort_values(self.time_column)
        )
        if not self.levels.empty:
            self.levels[self.time_column] = pd.to_datetime(
                self.levels[self.time_column]
            )

    @keyword_only(2)
    def detect_low_stock(
        self,
        min_level: float,
        hold: str = "0s",
        *,
        interval_end: str = "next_change",
    ) -> pd.DataFrame:
        """Flag intervals where inventory stays below *min_level* for at least *hold*.

        Args:
            min_level: Threshold below which stock is considered low.
            hold: Minimum duration the level must stay below threshold
                  (e.g. ``'0s'``, ``'5min'``, ``'1h'``).
            interval_end: ``"next_change"`` (default) ends an interval at the
                next sample (the transition), so the interval spans the time the
                condition held and change-encoded signals -- one row per change --
                keep their duration. ``"last_sample"`` ends it at its last flagged
                sample. See :func:`ts_shape.utils.intervals.interval_end_times`.

        Returns:
            DataFrame with columns: start, end, uuid, source_uuid, is_delta,
            min_value, avg_value, duration_seconds.
        """
        empty_cols = [
            "start",
            "end",
            "uuid",
            "source_uuid",
            "is_delta",
            "min_value",
            "avg_value",
            "duration_seconds",
        ]
        if self.levels.empty:
            return pd.DataFrame(columns=empty_cols)

        hold_td = pd.to_timedelta(hold)
        lv = (
            self.levels[[self.time_column, self.value_column]]
            .copy()
            .reset_index(drop=True)
        )
        below = (lv[self.value_column] < min_level).to_numpy(dtype=bool)

        # Contiguous below-threshold runs, kept when they last at least hold.
        runs = true_runs(below)
        start = lv[self.time_column].array[runs.starts]
        end = interval_end_times(lv[self.time_column], interval_end).array[runs.last]
        keep = ~((end - start) < hold_td)
        if not keep.any():
            return pd.DataFrame(columns=empty_cols)
        runs = Runs(runs.starts[keep], runs.ends[keep], runs.lengths[keep])
        start, end = start[keep], end[keep]
        values = lv[self.value_column]
        return pd.DataFrame(
            {
                "start": start,
                "end": end,
                "uuid": self.event_uuid,
                "source_uuid": self.level_uuid,
                "is_delta": True,
                "min_value": run_agg(values, runs, "nanmin"),
                "avg_value": run_agg(values, runs, "nanmean"),
                "duration_seconds": total_seconds(end - start),
            }
        )

    @keyword_only(1)
    def consumption_rate(
        self,
        window: str = "1h",
        *,
        handle_refills: bool = True,
    ) -> pd.DataFrame:
        """Calculate rolling consumption rate from inventory level decreases.

        The rate is expressed as units consumed per hour within each window.

        Args:
            window: Time-based window for grouping (e.g. ``'1h'``, ``'30min'``).
            handle_refills: When ``True`` (default) only level decreases are
                summed (a refill is treated like a counter reset), booked in
                the window of the reading that reports them. When ``False``
                consumption is ``first - last`` level of the window, so a
                refill inside the window cancels out the consumption
                (100 -> 40 -> 100 -> 40 reads as 0).

        Returns:
            DataFrame with columns: start, end, duration_seconds, uuid,
            source_uuid, is_delta, consumption_rate, level_start, level_end.
            ``end`` is ``start`` plus ``window`` (calendar-correct: a daily
            window on a DST change lasts 23 h or 25 h, and the rate is per
            hour of that real length).
        """
        empty_cols = [
            "start",
            "uuid",
            "source_uuid",
            "is_delta",
            "consumption_rate",
            "level_start",
            "level_end",
        ]
        if self.levels.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        lv = self.levels[[self.time_column, self.value_column]].copy()
        lv = lv.set_index(self.time_column)

        # Resample into windows and take first/last level
        resampled = lv[self.value_column].resample(window)
        first = resampled.first()
        last = resampled.last()

        result = pd.DataFrame(
            {
                "start": first.index,
                "level_start": first.values,
                "level_end": last.values,
            }
        )

        # Drop windows with no data
        result = result.dropna(subset=["level_start", "level_end"]).copy()

        if result.empty:
            return with_window_bounds(pd.DataFrame(columns=empty_cols), window)

        # Consumption is positive when level decreases
        # Per hour of the window's real length (a DST day lasts 23 h or 25 h).
        w_start, w_end = window_bounds(result["start"], window)
        window_hours = (w_end - w_start).dt.total_seconds().to_numpy() / 3600.0
        if handle_refills:
            consumed = (
                (-lv[self.value_column].diff()).clip(lower=0).resample(window).sum()
            )
            result["consumption_rate"] = (
                consumed.reindex(result["start"]).to_numpy() / window_hours
            )
        else:
            result["consumption_rate"] = (
                result["level_start"] - result["level_end"]
            ) / window_hours
        result["uuid"] = self.event_uuid
        result["source_uuid"] = self.level_uuid
        result["is_delta"] = True

        return with_window_bounds(result[empty_cols].reset_index(drop=True), window)

    @keyword_only(2)
    def reorder_point_breach(
        self,
        reorder_level: float,
        safety_stock: float = 0.0,
    ) -> pd.DataFrame:
        """Detect when inventory falls below reorder point or safety stock level.

        Args:
            reorder_level: The reorder point threshold.
            safety_stock: Safety stock threshold (must be <= reorder_level).

        Returns:
            DataFrame with columns: systime, uuid, source_uuid, is_delta,
            current_level,
            breach_type ('reorder' or 'safety_stock'), deficit.
        """
        empty_cols = [
            "systime",
            "uuid",
            "source_uuid",
            "is_delta",
            "current_level",
            "breach_type",
            "deficit",
        ]
        if self.levels.empty:
            return pd.DataFrame(columns=empty_cols)

        lv = self.levels[[self.time_column, self.value_column]].copy()

        # Detect transitions into breach (was above, now below)
        lv["prev_value"] = lv[self.value_column].shift()

        rows: list[dict[str, Any]] = []

        # Reorder point breaches: crossing below reorder_level
        reorder_mask = (lv[self.value_column] < reorder_level) & (
            (lv["prev_value"] >= reorder_level) | lv["prev_value"].isna()
        )
        for idx in lv[reorder_mask].index:
            row = lv.loc[idx]
            # Determine if it is also a safety stock breach
            if row[self.value_column] < safety_stock:
                breach_type = "safety_stock"
                deficit = safety_stock - row[self.value_column]
            else:
                breach_type = "reorder"
                deficit = reorder_level - row[self.value_column]
            rows.append(
                {
                    "systime": row[self.time_column],
                    "uuid": self.event_uuid,
                    "source_uuid": self.level_uuid,
                    "is_delta": True,
                    "current_level": float(row[self.value_column]),
                    "breach_type": breach_type,
                    "deficit": float(deficit),
                }
            )

        # Safety stock breaches that were not already caught (was above safety but below reorder)
        safety_mask = (lv[self.value_column] < safety_stock) & (
            (lv["prev_value"] >= safety_stock) & (lv["prev_value"] < reorder_level)
        )
        for idx in lv[safety_mask].index:
            row = lv.loc[idx]
            rows.append(
                {
                    "systime": row[self.time_column],
                    "uuid": self.event_uuid,
                    "source_uuid": self.level_uuid,
                    "is_delta": True,
                    "current_level": float(row[self.value_column]),
                    "breach_type": "safety_stock",
                    "deficit": float(safety_stock - row[self.value_column]),
                }
            )

        if not rows:
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame(rows)

    @keyword_only(1)
    def stockout_prediction(
        self,
        consumption_rate_window: str = "4h",
        *,
        handle_refills: bool = True,
    ) -> pd.DataFrame:
        """Estimate time until stockout based on recent consumption rate.

        For each data point, uses the consumption rate calculated over the
        preceding *consumption_rate_window* to project when inventory will
        reach zero.

        Args:
            consumption_rate_window: Lookback window for calculating
                consumption rate (e.g. ``'4h'``).
            handle_refills: When True (default) the consumption in the
                lookback is the sum of the level *decreases* between
                consecutive samples (the same refill-aware rule as
                :meth:`consumption_rate`), so a refill inside the window does
                not cancel the consumption around it. When False it is the
                level at the window's first sample minus the level at its
                last (a refill then hides consumption, often giving ``0``).

        Returns:
            DataFrame with columns: systime, uuid, source_uuid, is_delta,
            current_level,
            consumption_rate (units per hour), estimated_stockout_time_hours.
            ``estimated_stockout_time_hours`` is NaN where the level is not
            falling (no consumption in the window): no stockout is
            projected.
        """
        empty_cols = [
            "systime",
            "uuid",
            "source_uuid",
            "is_delta",
            "current_level",
            "consumption_rate",
            "estimated_stockout_time_hours",
        ]
        if self.levels.empty:
            return pd.DataFrame(columns=empty_cols)

        lv = self.levels[[self.time_column, self.value_column]].copy()
        window_td = pd.to_timedelta(consumption_rate_window)

        # Lookback window [t - window, t] (every sample at time t included) as
        # index bounds on the sorted levels; the rate is the consumption in
        # the window over the time between its first and last sample.
        times = lv[self.time_column]
        values = lv[self.value_column].to_numpy(dtype=float)
        lo = times.searchsorted(times - window_td, side="left")
        hi = times.searchsorted(times, side="right") - 1
        t_ns = times.to_numpy(dtype="datetime64[ns]")
        elapsed_hours = (t_ns[hi] - t_ns[lo]) / np.timedelta64(1, "h")
        if handle_refills:
            # Sum of the decreases between consecutive samples in the window:
            # a refill (an increase) adds nothing instead of cancelling the
            # consumption before it.
            drops = np.zeros(len(values))
            if len(values) > 1:
                drops[1:] = np.nan_to_num(np.clip(values[:-1] - values[1:], 0, None))
            cum = np.cumsum(drops)
            consumed = cum[hi] - cum[lo]
        else:
            consumed = values[lo] - values[hi]
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(elapsed_hours > 0, consumed / elapsed_hours, 0.0)
            rate = np.where(hi - lo + 1 < 2, 0.0, rate)
            est_hours = np.where(rate > 0, values / rate, np.nan)

        return pd.DataFrame(
            {
                "systime": times.to_numpy(),
                "uuid": self.event_uuid,
                "source_uuid": self.level_uuid,
                "is_delta": True,
                "current_level": values,
                "consumption_rate": rate.astype(float),
                "estimated_stockout_time_hours": est_hours.astype(float),
            }
        )
