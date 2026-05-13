"""Inventory monitoring events for supply chain timeseries.

Detects low stock intervals, calculates consumption rates,
identifies reorder point breaches, and predicts stockouts.
"""

import logging
import pandas as pd  # type: ignore
import numpy as np
from typing import List, Dict, Any

from ts_shape.utils.base import Base

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
        super().__init__(dataframe, column_name=time_column)
        self.level_uuid = level_uuid
        self.event_uuid = event_uuid
        self.value_column = value_column
        self.time_column = time_column

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

    def detect_low_stock(
        self,
        min_level: float,
        hold: str = "0s",
    ) -> pd.DataFrame:
        """Flag intervals where inventory stays below *min_level* for at least *hold*.

        Args:
            min_level: Threshold below which stock is considered low.
            hold: Minimum duration the level must stay below threshold
                  (e.g. ``'0s'``, ``'5min'``, ``'1h'``).

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
        lv = self.levels[[self.time_column, self.value_column]].copy()
        lv["below"] = lv[self.value_column] < min_level

        # Group contiguous below-threshold segments
        lv["group"] = (lv["below"] != lv["below"].shift()).cumsum()

        rows: List[Dict[str, Any]] = []
        for _, seg in lv[lv["below"]].groupby("group"):
            start = seg[self.time_column].iloc[0]
            end = seg[self.time_column].iloc[-1]
            duration = (end - start).total_seconds()
            if (end - start) < hold_td:
                continue
            rows.append(
                {
                    "start": start,
                    "end": end,
                    "uuid": self.event_uuid,
                    "source_uuid": self.level_uuid,
                    "is_delta": True,
                    "min_value": float(seg[self.value_column].min()),
                    "avg_value": float(seg[self.value_column].mean()),
                    "duration_seconds": duration,
                }
            )

        if not rows:
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame(rows)

    def consumption_rate(
        self,
        window: str = "1h",
    ) -> pd.DataFrame:
        """Calculate rolling consumption rate from inventory level decreases.

        Only considers intervals where the level decreased (consumption).
        The rate is expressed as units consumed per hour within each window.

        Args:
            window: Time-based window for grouping (e.g. ``'1h'``, ``'30min'``).

        Returns:
            DataFrame with columns: window_start, uuid, is_delta,
            consumption_rate, level_start, level_end.
        """
        empty_cols = [
            "window_start",
            "uuid",
            "is_delta",
            "consumption_rate",
            "level_start",
            "level_end",
        ]
        if self.levels.empty:
            return pd.DataFrame(columns=empty_cols)

        lv = self.levels[[self.time_column, self.value_column]].copy()
        lv = lv.set_index(self.time_column)

        # Resample into windows and take first/last level
        resampled = lv[self.value_column].resample(window)
        first = resampled.first()
        last = resampled.last()

        result = pd.DataFrame(
            {
                "window_start": first.index,
                "level_start": first.values,
                "level_end": last.values,
            }
        )

        # Drop windows with no data
        result = result.dropna(subset=["level_start", "level_end"]).copy()

        if result.empty:
            return pd.DataFrame(columns=empty_cols)

        # Consumption is positive when level decreases
        window_td = pd.to_timedelta(window)
        window_hours = window_td.total_seconds() / 3600.0
        result["consumption_rate"] = (
            result["level_start"] - result["level_end"]
        ) / window_hours
        result["uuid"] = self.event_uuid
        result["is_delta"] = True

        return result[empty_cols].reset_index(drop=True)

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
            DataFrame with columns: systime, uuid, is_delta, current_level,
            breach_type ('reorder' or 'safety_stock'), deficit.
        """
        empty_cols = [
            "systime",
            "uuid",
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

        rows: List[Dict[str, Any]] = []

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
                    "is_delta": True,
                    "current_level": float(row[self.value_column]),
                    "breach_type": "safety_stock",
                    "deficit": float(safety_stock - row[self.value_column]),
                }
            )

        if not rows:
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame(rows)

    def stockout_prediction(
        self,
        consumption_rate_window: str = "4h",
    ) -> pd.DataFrame:
        """Estimate time until stockout based on recent consumption rate.

        For each data point, uses the consumption rate calculated over the
        preceding *consumption_rate_window* to project when inventory will
        reach zero.

        Args:
            consumption_rate_window: Lookback window for calculating
                consumption rate (e.g. ``'4h'``).

        Returns:
            DataFrame with columns: systime, uuid, is_delta, current_level,
            consumption_rate, estimated_stockout_time_hours.
        """
        empty_cols = [
            "systime",
            "uuid",
            "is_delta",
            "current_level",
            "consumption_rate",
            "estimated_stockout_time_hours",
        ]
        if self.levels.empty:
            return pd.DataFrame(columns=empty_cols)

        lv = self.levels[[self.time_column, self.value_column]].copy()
        window_td = pd.to_timedelta(consumption_rate_window)

        rows: List[Dict[str, Any]] = []
        times = lv[self.time_column].values
        values = lv[self.value_column].values

        for i in range(len(lv)):
            current_time = times[i]
            current_level = float(values[i])
            # Lookback window
            window_start = current_time - window_td
            mask = (times >= window_start) & (times <= current_time)
            window_vals = values[mask]
            window_times = times[mask]

            if len(window_vals) < 2:
                rows.append(
                    {
                        "systime": pd.Timestamp(current_time),
                        "uuid": self.event_uuid,
                        "is_delta": True,
                        "current_level": current_level,
                        "consumption_rate": 0.0,
                        "estimated_stockout_time_hours": np.inf,
                    }
                )
                continue

            # Consumption = first - last in window (positive if consuming)
            first_val = float(window_vals[0])
            last_val = float(window_vals[-1])
            elapsed_hours = (
                pd.Timestamp(window_times[-1]) - pd.Timestamp(window_times[0])
            ).total_seconds() / 3600.0

            if elapsed_hours <= 0:
                rate = 0.0
            else:
                rate = (first_val - last_val) / elapsed_hours

            if rate > 0:
                est_hours = current_level / rate
            else:
                est_hours = np.inf

            rows.append(
                {
                    "systime": pd.Timestamp(current_time),
                    "uuid": self.event_uuid,
                    "is_delta": True,
                    "current_level": current_level,
                    "consumption_rate": float(rate),
                    "estimated_stockout_time_hours": float(est_hours),
                }
            )

        if not rows:
            return pd.DataFrame(columns=empty_cols)
        return pd.DataFrame(rows)
